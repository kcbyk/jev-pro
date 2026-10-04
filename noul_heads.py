#!/usr/bin/env python3
"""Re-embed Banking77 with the FINE-TUNED encoder and fit the noul heads.

Why noul probes on top of the fine-tuned embedding: a 77-way Choice answer is
not the whole contract. The served endpoint also has to answer yes/no
propositions with a probability, and the honest way to get those is a small
calibrated head per proposition — not asking the classifier to "explain itself".

Propositions are derived from the label taxonomy, so they are auditable:
  is_card      "the message concerns a physical or virtual card"
  is_transfer  "the message concerns moving money to another account"
  is_topup     "the message concerns topping up the balance"
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROP = {
    "is_card": ("Does this message concern a card (physical or virtual)?", lambda l: "card" in l),
    "is_transfer": ("Does this message concern transferring money to another account?",
                    lambda l: "transfer" in l or "payment" in l),
    "is_topup": ("Does this message concern topping up the balance?",
                 lambda l: "top_up" in l or "topup" in l),
}


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def fit_logreg(X, y, iters=600, lr=0.1, wd=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.01, (X.shape[1],))
    b = 0.0
    mW = np.zeros_like(W); vW = np.zeros_like(W); mb = vb = 0.0
    n = len(X); batch = 256; t = 0
    for it in range(iters):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            ix = order[s : s + batch]
            xb, yb = X[ix], y[ix].astype(np.float64)
            z = xb @ W + b
            p = 1.0 / (1.0 + np.exp(-z))
            g = (p - yb) / len(yb)
            gW = xb.T @ g + wd * W
            gb = float(g.sum())
            t += 1
            mW = 0.9 * mW + 0.1 * gW; vW = 0.999 * vW + 0.001 * gW * gW
            mb = 0.9 * mb + 0.1 * gb; vb = 0.999 * vb + 0.001 * gb * gb
            W -= lr * (mW / (1 - 0.9**t)) / (np.sqrt(vW / (1 - 0.999**t)) + 1e-8)
            b -= lr * (mb / (1 - 0.9**t)) / (np.sqrt(vb / (1 - 0.999**t)) + 1e-8)
    return W, b


def ece_bin(p, y, bins=15):
    conf = np.maximum(p, 1 - p)
    cor = (p >= 0.5) == (y == 1)
    edges = np.linspace(0, 1, bins + 1)
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf <= hi) if lo == 0 else (conf > lo) & (conf <= hi)
        if m.any():
            tot += m.mean() * abs(cor[m].mean() - conf[m].mean())
    return float(tot)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="default: base model from model_card.json")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--outdir", default=str(HERE / "jev-pro-model"))
    ap.add_argument("--state", default=str(HERE / "ft_state.pt"))
    a = ap.parse_args()
    outdir = Path(a.outdir)
    torch.set_num_threads(a.threads)

    card = json.load(open(outdir / "model_card.json"))
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(outdir)
    base = AutoModel.from_pretrained(card["base_model"])
    state = torch.load(a.state, map_location="cpu", weights_only=True)
    head_state = {k: v for k, v in state.items() if k.startswith("head.")}
    W = head_state["head.weight"].numpy().astype(np.float32)
    bias = head_state["head.bias"].numpy().astype(np.float32)
    missing, unexpected = base.load_state_dict({k[len("base."):]: v for k, v in state.items()
                                                if k.startswith("base.")}, strict=False)
    print(f"loaded fine-tuned encoder ({len(state)} tensors, unexpected={len(unexpected)})", flush=True)

    labels = card["labels"]

    def rows(split):
        out = [json.loads(l) for l in open(HERE / "data" / f"{split}.jsonl")]
        return out

    @torch.inference_mode()
    def embed(txts, maxlen=48, bs=128):
        base.eval()
        out = np.zeros((len(txts), W.shape[1]), dtype=np.float32)
        order = np.argsort([len(t) for t in txts], kind="stable")
        for s in range(0, len(order), bs):
            sel = order[s : s + bs]
            e = tok([txts[i] for i in sel], padding=True, truncation=True, max_length=maxlen, return_tensors="pt")
            h = base(**e).last_hidden_state
            m = e["attention_mask"].unsqueeze(-1).to(h.dtype)
            out[sel] = ((h * m).sum(1) / m.sum(1).clamp(min=1e-6)).numpy()
        return out

    t0 = time.time()
    tr, te = rows("train"), rows("test")
    Xtr = embed([r["text"] for r in tr])
    Xte = embed([r["text"] for r in te])
    np.save(HERE / "emb_train_ft.npy", Xtr); np.save(HERE / "emb_test_ft.npy", Xte)
    # the choice answer, straight from the fine-tuned head (encoder output -> logits)
    Ltr, Lte = Xtr @ W.T + bias, Xte @ W.T + bias
    np.save(HERE / "logits_test_ft.npy", Lte)
    ytr = np.array([labels.index(r["label"]) for r in tr])
    yte = np.array([labels.index(r["label"]) for r in te])
    print(f"embedded {len(tr)}+{len(te)} in {time.time()-t0:.0f}s", flush=True)
    T = card["temperature"]
    Pc = softmax(Lte / T)
    print(f"choice head on test: acc {float((Pc.argmax(1)==yte).mean()):.4f}  "
          f"ECE {ece_choice(Pc, yte):.4f}  T={T:.2f}", flush=True)

    fit_n = 8000
    perm = np.random.default_rng(1234).permutation(len(tr))
    Xtr, ytr = Xtr[perm], ytr[perm]
    lbl_tr = [tr[i]["label"] for i in perm]
    lbl_te = [r["label"] for r in te]
    saved = {}
    print("\nnoul heads (logistic probe on the fine-tuned embedding, temperature on val):", flush=True)
    for name, (instr, pred) in PROP.items():
        ytr_b = np.array([int(bool(pred(lbl))) for lbl in lbl_tr])
        yte_b = np.array([int(bool(pred(lbl))) for lbl in lbl_te])
        Xf, yf, Xv, yv = Xtr[:fit_n], ytr_b[:fit_n], Xtr[fit_n:], ytr_b[fit_n:]
        Wc, bc = fit_logreg(Xf, yf)
        raw_v = Xv @ Wc + bc
        grid = np.linspace(0.5, 6.0, 56)
        nll = []
        for T2 in grid:
            q = np.clip(1 / (1 + np.exp(-raw_v / T2)), 1e-9, 1 - 1e-9)
            nll.append(-np.mean(yv * np.log(q) + (1 - yv) * np.log(1 - q)))
        T2 = float(grid[int(np.argmin(nll))])
        pt = 1 / (1 + np.exp(-(Xte @ Wc + bc) / T2))
        p_raw = 1 / (1 + np.exp(-(Xte @ Wc + bc)))
        acc = float(((pt >= 0.5).astype(int) == yte_b).mean())
        e0, e1 = ece_bin(p_raw, yte_b), ece_bin(pt, yte_b)
        base_rate = float(yte_b.mean())
        saved[name] = {"W": Wc.tolist(), "b": float(bc), "T": T2, "instructions": instr,
                       "pos_share": base_rate, "acc": acc, "ece_before": e0, "ece_after": e1}
        print(f"  {name:12s} pos_rate {base_rate:.3f}  acc {acc:.4f}  (majority-only acc {max(base_rate,1-base_rate):.4f})"
              f"  ECE {e0:.4f} -> {e1:.4f}  T={T2:.2f}", flush=True)

    json.dump(saved, open(outdir / "noul_heads.json", "w"))
    np.savez_compressed(outdir / "choice_head.npz", W=W, bias=bias, temperature=T, labels=np.array(labels))
    print(f"\nwrote {outdir}/noul_heads.json + choice_head.npz", flush=True)
    return 0


def ece_choice(P, y, bins=15):
    conf = P.max(1)
    cor = (P.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf <= hi) if lo == 0 else (conf > lo) & (conf <= hi)
        if m.any():
            tot += m.mean() * abs(cor[m].mean() - conf[m].mean())
    return float(tot)


if __name__ == "__main__":
    raise SystemExit(main())
