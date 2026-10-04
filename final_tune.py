#!/usr/bin/env python3
"""r3w final tuning: test-time paraphrase averaging (TTA) + model-head x retrieval blend.

Everything (head temperature, retrieval temperatures, blend weights, blend
temperature) is fitted on the 1,993-row val slice through the SAME pipeline that
runs on test (TTA on both sides — symmetric calibration; retrieval view also
averaged over the three query variants). Test is scored once, at the very end.

Honest caveat: r3w was trained while data/train_aug.jsonl still contained
paraphrases of val parents, so val-side head numbers are mildly optimistic.
Test-side input was never in any training pool, and the retrieval reference
holds only labelled train rows.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
import make_aug
from boost import softmax, ece, coverage, knn_sim, fit_temp

torch.set_num_threads(2)
from transformers import AutoModel, AutoTokenizer

T0 = time.time()
card = json.load(open(HERE / "jev-pro-model-r3w" / "model_card.json"))
tok = AutoTokenizer.from_pretrained(HERE / "jev-pro-model-r3w")
base = AutoModel.from_pretrained(card["base_model"])
st = torch.load(HERE / "jev-pro-model-r3w" / "ft_state.pt", map_location="cpu", weights_only=True)
_, unexpected = base.load_state_dict({k[5:]: v for k, v in st.items() if k.startswith("base.")}, strict=False)
assert len(unexpected) == 0
W = st["head.weight"].numpy().astype(np.float32)
bias = st["head.bias"].numpy().astype(np.float32)
labels = card["labels"]
idx = {c: i for i, c in enumerate(labels)}
Cn = len(labels)
print(f"r3w loaded [{time.time()-T0:.0f}s]", flush=True)


@torch.inference_mode()
def pooled(texts, maxlen=48, bs=128):
    base.eval()
    out = np.zeros((len(texts), W.shape[1]), dtype=np.float32)
    order = np.argsort([len(t) for t in texts], kind="stable")
    for s in range(0, len(order), bs):
        sel = order[s : s + bs]
        e = tok([texts[i] for i in sel], padding=True, truncation=True, max_length=maxlen, return_tensors="pt")
        h = base(**e).last_hidden_state
        msk = e["attention_mask"].unsqueeze(-1).to(h.dtype)
        out[sel] = ((h * msk).sum(1) / msk.sum(1).clamp(min=1e-6)).numpy()
    return out


def stack(rows, seed):
    """original + 2 paraphrases -> averaged LOGITS and averaged unit pooled embedding."""
    txt = [r["text"] for r in rows]
    rng = random.Random(seed)
    views = [txt, [make_aug.variant1(t, rng) for t in txt], [make_aug.variant2(t, rng) for t in txt]]
    E = np.concatenate([pooled(v) for v in views], 0).reshape(3, len(txt), -1)
    L = (E.mean(0) @ W.T + bias)          # average embeddings, then head (linear == mean of per-view logits)
    Ea = E.mean(0)
    Ea = Ea / np.clip(np.linalg.norm(Ea, axis=1, keepdims=True), 1e-9, None)
    return L, Ea


acc = lambda P, y: float((P.argmax(1) == y).mean())

tr_all = [json.loads(l) for l in open(HERE / "data/train.jsonl")]
perm = np.random.default_rng(1234).permutation(len(tr_all))
tr_sh = [tr_all[i] for i in perm]
val, te = tr_sh[8000:], [json.loads(l) for l in open(HERE / "data/test.jsonl")]
yv = np.array([idx[r["label"]] for r in val])
yt = np.array([idx[r["label"]] for r in te])
ytr_all = np.array([idx[r["label"]] for r in tr_sh])

Lv, Ev = stack(val, 51)
Lt, Et = stack(te, 52)
print(f"TTA encoded [{time.time()-T0:.0f}s]", flush=True)

# head: temperature on val, then reuse
Tv = fit_temp(Lv, yv, lo=0.4, hi=3.0)
Phv, Pht = softmax(Lv / Tv), softmax(Lt / Tv)

# retrieval views: reference = fit-8000 for val, ALL 9993 labelled train rows for test
Xtr0 = np.load(HERE / "emb_train_ft.npy").astype(np.float32)[perm]
unit = lambda A: A / np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-9, None)
Rv, Rt = unit(Xtr0[:8000]), unit(Xtr0)
Yv_, Yt_ = ytr_all[:8000], ytr_all

def cent(R, Y):
    C = np.stack([R[Y == c].mean(0) for c in range(Cn)])
    return C / np.clip(np.linalg.norm(C, axis=1, keepdims=True), 1e-9, None)

Zk = {k: knn_sim(Ev, Rv, Yv_, Cn, k, 0.05) for k in (8, 16, 32)}
Tk = {k: fit_temp(Zk[k], yv) for k in Zk}
acc_k = {k: acc(softmax(Zk[k] / Tk[k]), yv) for k in Zk}
kk = max(acc_k, key=acc_k.get)
Cv = cent(Rv, Yv_)
Zcv = Ev @ Cv.T / 0.08
Tcv = fit_temp(Zcv, yv)
print(f"val acc: head-TTA {acc(Phv, yv):.4f} (T={Tv:.2f})  knn {kk}: {max(acc_k.values()):.4f} "
      f"(all k: { {k: round(v,4) for k,v in acc_k.items()} })  cent {acc(softmax(Zcv/Tcv), yv):.4f}", flush=True)

# blend weights on val NLL, then one blend temperature on val
Pk_v = softmax(Zk[kk] / Tk[kk])
Pc_v = softmax(Zcv / Tcv)
grid = np.arange(0, 1.0001, 0.05)
best = None
for w in grid:
    for v in grid[grid <= 1 - w + 1e-9]:
        P = (1 - w - v) * Phv + w * Pk_v + v * Pc_v
        P /= np.clip(P.sum(1, keepdims=True), 1e-12, None)
        s = -np.mean(np.log(np.clip(P[np.arange(len(yv)), yv], 1e-12, None)))
        if best is None or s < best[0]:
            best = (s, float(w), float(v))
_, wK, wC = best
Pb = (1 - wK - wC) * Phv + wK * Pk_v + wC * Pc_v
Pb /= Pb.sum(1, keepdims=True)
Tb = fit_temp(np.log(np.clip(Pb, 1e-9, None)), yv, lo=0.5, hi=2.0)
Pb_v = softmax(np.log(np.clip(Pb, 1e-9, None)) / Tb)
print(f"val blend: w_knn={wK:.2f} w_cent={wC:.2f} T={Tb:.2f}  val acc {acc(Pb_v, yv):.4f}", flush=True)

# ---- TEST, scored once ----
Zkt = knn_sim(Et, Rt, Yt_, Cn, kk, 0.05)
Ct = cent(Rt, Yt_)
Pb_t = (1 - wK - wC) * Pht + wK * softmax(Zkt / Tk[kk]) + wC * softmax((Et @ Ct.T / 0.08) / Tcv)
Pb_t /= np.clip(Pb_t.sum(1, keepdims=True), 1e-12, None)
Pb_t = softmax(np.log(np.clip(Pb_t, 1e-9, None)) / Tb)

P1 = softmax(np.load(HERE / "logits_test_ft.npy").astype(np.float64) / card["temperature"])
rows = []
print("\nofficial test split (3,076 rows, one measurement each):")
for name, P in [("r3w head single-pass (T=0.75)", P1), ("r3w head + TTA (val-T)", Pht),
                ("r3w TTA blend knn+centroid", Pb_t)]:
    a = acc(P, yt)
    top3 = float((np.argsort(-P, 1)[:, :3] == yt[:, None]).any(1).mean())
    top5 = float((np.argsort(-P, 1)[:, :5] == yt[:, None]).any(1).mean())
    ec = ece(P, yt)
    over = float(P.max(1).mean() * 100 - a * 100)
    c95, c99 = coverage(P, yt, 0.95), coverage(P, yt, 0.99)
    rows.append({"model": name, "acc": round(a, 4), "top3": round(top3, 4), "top5": round(top5, 4),
                 "ece": round(ec, 4), "conf_minus_acc_pts": round(over, 1), "cov95": round(c95, 2),
                 "cov99": round(c99, 2)})
    print(f"  {name:32s} acc {a:.4f}  top3 {top3:.4f}  top5 {top5:.4f}  ECE {ec:.4f}  "
          f"conf-acc {over:+.1f}p  cov@95 {c95:.2f}  cov@99 {c99:.2f}")
json.dump({"rows": rows, "Tv": Tv, "Tk": Tk[kk], "Tcv": Tcv, "Tb": Tb, "k": kk, "w_knn": wK, "w_cent": wC,
           "val": {"head_tta": acc(Phv, yv), "blend": acc(Pb_v, yv)}, "seconds": round(time.time()-T0, 1)},
          open(HERE / "final_tune_report.json", "w"), indent=1)
print(f"\nwrote final_tune_report.json [{time.time()-T0:.0f}s]")
print("targets: Banking77 SOTA 0.9383 | fine-tuned BERT 0.9366 | 22M+LR published 0.932")
