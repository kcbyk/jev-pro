#!/usr/bin/env python3
"""Frozen-encoder ladder: how far does a *trained* local decision layer get on
Banking77, measured against Jev's published numbers on the same split?

Five rungs, all on the same vectors, all honest about what they saw:

  zero_shot_names  label names only, no training data  <- the Jev comparison point
  logreg           softmax regression on 8k labelled examples
  mlp              one hidden layer, same labels
  knn              cosine kNN over the labelled examples (no weights at all)
  ensemble         probability average of the three trained heads

Splits: fit on train[:8000], every model-selection decision (temperature,
thresholds, which head) on train[8000:], and the 3,076-row official test split
touched once, at the end.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# published reference points (cited in README so nobody has to trust me)
JEV_REF = {
    "jevbench_accuracy": 0.803,   # jevbench.xyz, 3,080 cases, jev-1.13.0, zero-shot typed Choice
    "jevbench_p50_ms": 310.0,
    "jev_overconf_pts": 13.9,     # sotaaz Kev-vs-Jev: mean top prob exceeded accuracy by 13.9 pts
    "jev_coverage95": 0.0,        # and on BANKING77 no threshold reached 95% accuracy
    "kev9b_accuracy": 0.825,      # same source, 154-sample Jev comparison set
    "published_lr_head": 0.932,   # 22M encoder + logistic head, 8 ms CPU (mindstudio benchmark)
    "published_bert": 0.9366,     # fine-tuned BERT, original Banking77 paper
}


# ---------------------------------------------------------------- data
def load(shuffle_seed: int = 1234):
    """The mteb mirror ships the train split GROUPED BY LABEL (rows 0..133 are
    all card_arrival, then the next class, ...). Slicing that by index puts a few
    whole classes in "val" and none in "fit", and every metric silently collapses
    to chance. Shuffle deterministically before any split."""
    ytr = [json.loads(l)["label"] for l in open(HERE / "data/train.jsonl")]
    yte = [json.loads(l)["label"] for l in open(HERE / "data/test.jsonl")]
    labels = json.load(open(HERE / "data/labels.json"))
    idx = {c: i for i, c in enumerate(labels)}
    Xtr = np.load(HERE / "emb_train.npy")
    Xte = np.load(HERE / "emb_test.npy")
    ytr = np.asarray([idx[c] for c in ytr])
    yte = np.asarray([idx[c] for c in yte])
    perm = np.random.default_rng(shuffle_seed).permutation(len(Xtr))
    n0 = int((ytr[perm] == ytr[perm][:1]).sum()) if False else 0
    Xtr_raw, ytr_raw = Xtr[perm], ytr[perm]
    assert len(np.unique(ytr_raw[:2000])) > 60, "train split is not class-mixed after shuffle"
    texts = [json.loads(l)["text"] for l in open(HERE / "data/train.jsonl")]
    texts = [texts[i] for i in perm]
    return Xtr_raw, ytr_raw, Xte, yte, labels, texts


# ---------------------------------------------------------------- heads (numpy)
def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def fit_linear(X, y, C, Cn, iters=400, lr=0.2, wd=1e-4, seed=0, hidden=0):
    """Adam on softmax regression, or a 1-hidden-layer MLP when hidden>0."""
    rng = np.random.default_rng(seed)
    if hidden:
        W1 = rng.normal(0, 1.0 / np.sqrt(X.shape[1]), (X.shape[1], hidden))
        b1 = np.zeros(hidden)
        W2 = rng.normal(0, 1.0 / np.sqrt(hidden), (hidden, Cn))
        b2 = np.zeros(Cn)
        params = [W1, b1, W2, b2]
    else:
        params = [rng.normal(0, 0.01, (X.shape[1], Cn)), np.zeros(Cn)]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    n = len(X)
    batch = 256
    order_all = np.arange(n)
    for it in range(iters):
        order = rng.permutation(order_all)
        for s in range(0, n, batch):
            b = order[s : s + batch]
            xb, yb = X[b], y[b]
            T = np.full((len(yb), Cn), 0.02 / max(Cn - 1, 1))
            T[np.arange(len(yb)), yb] = 0.98
            if hidden:
                a = np.maximum(xb @ params[0] + params[1], 0.0)
                P = softmax(a @ params[2] + params[3])
                g = (P - T) / len(yb)
                gW2 = a.T @ g + wd * params[2]
                gb2 = g.sum(0)
                ga = (g @ params[2].T) * (a > 0)
                gW1 = xb.T @ ga + wd * params[0]
                gb1 = ga.sum(0)
                grads = [gW1, gb1, gW2, gb2]
            else:
                P = softmax(xb @ params[0] + params[1])
                g = (P - T) / len(yb)
                grads = [xb.T @ g + wd * params[0], g.sum(0)]
            t += 1
            b1m, b2m = 0.9, 0.999
            for i, gg in enumerate(grads):
                m[i] = b1m * m[i] + (1 - b1m) * gg
                v[i] = b2m * v[i] + (1 - b2m) * gg * gg
                params[i] -= lr * (m[i] / (1 - b1m**t)) / (np.sqrt(v[i] / (1 - b2m**t)) + 1e-8)
    return params


def apply_linear(params, X, hidden=0):
    if hidden:
        a = np.maximum(X @ params[0] + params[1], 0.0)
        return a @ params[2] + params[3]
    return X @ params[0] + params[1]


def knn_logits(Xq, Xr, Yr, Cn, k=8, tau=0.05):
    """Distance-weighted cosine kNN, fully vectorised (embeddings are unit norm,
    so Xq @ Xr.T already is the similarity)."""
    S = Xq @ Xr.T
    part = np.argpartition(-S, k, axis=1)[:, :k]              # (Q, k)
    sim = np.take_along_axis(S, part, axis=1)                 # (Q, k)
    w = np.exp((sim - sim.max(1, keepdims=True)) / tau)        # (Q, k)
    lab = Yr[part]                                             # (Q, k)
    mass = np.zeros((len(Xq), Cn), dtype=np.float64)
    np.add.at(mass, (np.arange(len(Xq))[:, None].repeat(k, 1).ravel(), lab.ravel()), w.ravel())
    return np.log(np.clip(mass, 1e-9, None))


def zero_shot_names(Xq, label_texts, tau=0.05):
    """The closest local analogue of "a general model that only got your option
    names and no training data": match each message to the embedded name."""
    L = EMBEDDER(label_texts)
    L = L / np.linalg.norm(L, axis=1, keepdims=True)
    return (Xq @ L.T) / tau


EMBEDDER = None  # set by main(): batched encoder for arbitrary text


# ---------------------------------------------------------------- metrics
def metrics(P, y, Cn):
    pred = P.argmax(1)
    acc = float((pred == y).mean())
    prec = np.zeros(Cn)
    rec = np.zeros(Cn)
    f1 = np.zeros(Cn)
    for c in range(Cn):
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        prec[c] = tp / max(tp + fp, 1)
        rec[c] = tp / max(tp + fn, 1)
        f1[c] = 2 * prec[c] * rec[c] / max(prec[c] + rec[c], 1e-9)
    top3 = float((np.argsort(-P, axis=1)[:, :3] == y[:, None]).any(1).mean())
    return {"acc": acc, "macro_f1": float(f1.mean()), "top3": top3,
            "per_class_f1": f1.tolist(), "n_wrong": int((pred != y).sum())}


def ece(P, y, bins=15):
    conf = P.max(1)
    cor = (P.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf <= hi) if lo == 0 else (conf > lo) & (conf <= hi)
        if m.any():
            tot += m.mean() * abs(cor[m].mean() - conf[m].mean())
    return float(tot)


def brier(P, y, Cn):
    T = np.zeros_like(P)
    T[np.arange(len(y)), y] = 1.0
    return float(np.mean(((P - T) ** 2).sum(1)))


def fit_T(logits, y, grid=np.linspace(0.5, 6.0, 111)):
    best = (np.inf, 1.0)
    for T in grid:
        P = softmax(logits / T)
        nll = -np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, None)))
        if nll < best[0]:
            best = (nll, float(T))
    return best[1]


def coverage(P, y, targets=(0.95, 0.99)):
    """Share of traffic you can auto-accept while *staying* at the target
    accuracy. This is the number the Jev-vs-Kev comparison hit hardest."""
    conf = P.max(1)
    cor = P.argmax(1) == y
    out = {}
    for tgt in targets:
        best = 0.0
        for thr in np.quantile(conf, np.linspace(0, 0.999, 200)):
            m = conf >= thr
            if m.mean() > 0 and cor[m].mean() >= tgt:
                best = max(best, float(m.mean()))
        out[tgt] = best
    return out


# ---------------------------------------------------------------- run
def main() -> int:
    global EMBEDDER
    ap = argparse.ArgumentParser()
    ap.add_argument("--knn-k", type=int, default=8)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--mlp-iters", type=int, default=350)
    ap.add_argument("--out", default=str(HERE / "decide_report.json"))
    a = ap.parse_args()

    # zero-shot rung needs the encoder for the label names
    import torch
    from transformers import AutoModel, AutoTokenizer

    meta = json.load(open(HERE / "embed_meta.json"))
    tok = AutoTokenizer.from_pretrained(meta["model"])
    enc = AutoModel.from_pretrained(meta["model"]).eval()
    torch.set_num_threads(2)

    @torch.inference_mode()
    def _embed_one_impl(texts):
        e = tok(texts, padding=True, truncation=True, max_length=32, return_tensors="pt")
        h = enc(**e).last_hidden_state
        m = e["attention_mask"].unsqueeze(-1).to(h.dtype)
        return ((h * m).sum(1) / m.sum(1).clamp(min=1e-6)).numpy()

    EMBEDDER = _embed_one_impl

    Xtr, ytr, Xte, yte, labels, ttr = load()
    Xtr, Xte = Xtr.astype(np.float32), Xte.astype(np.float32)
    Cn = len(labels)
    fit_n = 8000
    Xf, yf, tf = Xtr[:fit_n], ytr[:fit_n], ttr[:fit_n]
    Xv, yv = Xtr[fit_n:], ytr[fit_n:]
    assert len(np.unique(yf)) == len(np.unique(yv)) == Cn, "fit/val must both cover all 77 classes"
    lines = [f"Banking77: fit={fit_n}  val={len(Xv)}  test={len(Xte)}  classes={Cn}  "
             f"encoder={meta['model'].split('/')[-1]} ({meta['params']/1e6:.1f}M, frozen)  "
             f"train split shuffled with seed 1234"]
    print(lines[0], flush=True)

    t0 = time.time()
    logits_fit = {"logreg": apply_linear(fit_linear(Xf, yf, Cn, Cn, iters=a.iters), Xv), }
    logits_fit["mlp"] = apply_linear(fit_linear(Xf, yf, Cn, Cn, iters=a.mlp_iters, hidden=256), Xv, hidden=256)
    logits_fit["knn"] = knn_logits(Xv, Xf, yf, Cn, a.knn_k)
    logits_fit["zero_shot_names"] = zero_shot_names(Xv, [l.replace("_", " ") for l in labels])
    print(f"  heads fitted in {time.time()-t0:.1f}s", flush=True)

    # model selection on val only
    report = {}
    for name, lv in logits_fit.items():
        Tv = fit_T(lv, yv)
        Pv = softmax(lv / Tv)
        report[name] = {"val_acc": metrics(Pv, yv, Cn)["acc"], "val_ece": ece(Pv, yv), "T": Tv}
        print(f"  {name:18s} val acc {report[name]['val_acc']:.4f}  val ECE {report[name]['val_ece']:.4f}  T={Tv:.2f}", flush=True)

    ens_v = np.mean([softmax(logits_fit[n] / report[n]["T"]) for n in ("logreg", "mlp", "knn")], axis=0)
    ens_logits_v = np.log(np.clip(ens_v, 1e-9, None))
    logits_fit["ensemble"] = ens_logits_v
    report["ensemble"] = {"val_acc": metrics(softmax(ens_logits_v / 1.0), yv, Cn)["acc"],
                          "val_ece": ece(softmax(ens_logits_v), yv), "T": 1.0}
    print(f"  {'ensemble':18s} val acc {report['ensemble']['val_acc']:.4f}  (val-selected average of the 3 trained heads)", flush=True)

    # ONE pass at the end: refit trained heads on all train data, score the test split
    Xt = Xtr  # shuffled, same order as ytr
    full = {"logreg": fit_linear(Xt, ytr, Cn, Cn, iters=a.iters),
            "mlp": fit_linear(Xt, ytr, Cn, Cn, iters=a.mlp_iters, hidden=256)}
    test_logits = {"logreg": apply_linear(full["logreg"], Xte),
                   "mlp": apply_linear(full["mlp"], Xte, hidden=256),
                   "knn": knn_logits(Xte, Xt, ytr, Cn, a.knn_k),
                   "zero_shot_names": zero_shot_names(Xte, [l.replace("_", " ") for l in labels])}
    Ts = {}
    for n, lv in list(test_logits.items()):
        src = report[n]["T"] if n in report else 1.0
        Ts[n] = src
    test_logits["ensemble"] = np.log(np.clip(np.mean([softmax(test_logits[n] / Ts[n])
                                                     for n in ("logreg", "mlp", "knn")], axis=0), 1e-9, None))
    Ts["ensemble"] = 1.0

    rows = []
    for n in ("zero_shot_names", "logreg", "mlp", "knn", "ensemble"):
        P = softmax(test_logits[n] / Ts[n])
        m = metrics(P, yte, Cn)
        cov = coverage(P, yte)
        overconf = float(P.max(1).mean() * 100 - m["acc"] * 100)
        rows.append((n, m, ece(P, yte), brier(P, yte, Cn), Ts[n], overconf, cov))
        print(f"  TEST {n:18s} acc {m['acc']:.4f}  macroF1 {m['macro_f1']:.4f}  top3 {m['top3']:.3f}  "
              f"ECE {ece(P, yte):.4f}  conf-acc {overconf:+.1f}pt  cov@95 {cov[0.95]:.2f}  cov@99 {cov[0.99]:.2f}", flush=True)
    assert len(tf) == fit_n

    best = max(rows, key=lambda r: r[1]["acc"])
    f1s = np.asarray(best[1]["per_class_f1"])
    worst = sorted(zip(labels, f1s), key=lambda z: z[1])[:8]
    lines += [f"\n  {'head':18s} {'acc':>7s} {'macroF1':>8s} {'top3':>6s} {'ECE':>7s} {'conf-acc':>9s} {'cov@95':>7s} {'cov@99':>7s}",
              f"  {'-'*70}"]
    for n, m, ec, br, T, oc, cov in rows:
        lines.append(f"  {n:18s} {m['acc']:7.4f} {m['macro_f1']:8.4f} {m['top3']:6.3f} {ec:7.4f} "
                     f"{oc:+8.1f}p {cov[0.95]:7.2f} {cov[0.99]:7.2f}")
    lines.append("\n  published references for the same split:")
    lines.append(f"    Jev jev-1.13.0 (zero-shot typed Choice, 3,080 cases): {JEV_REF['jevbench_accuracy']:.3f}, p50 {JEV_REF['jevbench_p50_ms']:.0f} ms")
    lines.append(f"    Jev overconfidence on BANKING77: +{JEV_REF['jev_overconf_pts']:.1f} pts, coverage at 95% accuracy: {JEV_REF['jev_coverage95']:.0%}")
    lines.append(f"    published 22M encoder + logistic head: {JEV_REF['published_lr_head']:.3f} @ 8 ms; fine-tuned BERT: {JEV_REF['published_bert']:.4f}")
    lines.append(f"  worst classes of the best head: " + ", ".join(f"{l}({v:.2f})" for l, v in worst))

    # save the two trained heads for the server
    np.savez_compressed(HERE / "heads.npz", W_logreg=full["logreg"][0], b_logreg=full["logreg"][1],
                        W1=full["mlp"][0], b1=full["mlp"][1], W2=full["mlp"][2], b2=full["mlp"][3],
                        T_logreg=Ts["logreg"], T_mlp=Ts["mlp"])
    json.dump({"labels": labels, "report": report,
               "test": {n: {"acc": m["acc"], "macro_f1": m["macro_f1"], "ece": ec, "brier": br,
                            "T": T, "conf_minus_acc_pts": oc, "coverage": cov}
                        for n, m, ec, br, T, oc, cov in rows},
               "jev_reference": JEV_REF, "best_head": best[0]},
              open(a.out, "w"), indent=2)
    Path(HERE / "decide_report.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines[1:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
