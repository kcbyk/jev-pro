#!/usr/bin/env python3
"""Can the retrieval head beat or help the trained head? Blend, measured.

Jev's own best Banking77 number (92.40%) came from giving it 24 retrieved
labelled examples per class. So retrieval is a fair ingredient, not a cheat —
but it has to be scored against the same discipline: blend weight and
temperature fitted on val, test touched once.

Runs off cached embeddings (emb_*_ft.npy), so it costs seconds, not a training run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def softmax(z):
    z = np.asarray(z, np.float64)
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def ece(P, y, bins=15):
    conf, cor = P.max(1), (P.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf <= hi) if lo == 0 else (conf > lo) & (conf <= hi)
        if m.any():
            tot += m.mean() * abs(cor[m].mean() - conf[m].mean())
    return float(tot)


def coverage(P, y, tgt):
    conf, cor = P.max(1), P.argmax(1) == y
    best = 0.0
    for thr in np.quantile(conf, np.linspace(0, 0.999, 240)):
        m = conf >= thr
        if m.any() and cor[m].mean() >= tgt:
            best = max(best, float(m.mean()))
    return best


def knn_sim(Xq, Xr, yr, Cn, k, tau):
    S = Xq @ Xr.T
    part = np.argpartition(-S, k, axis=1)[:, :k]
    sim = np.take_along_axis(S, part, axis=1)
    w = np.exp((sim - sim.max(1, keepdims=True)) / tau)
    lab = yr[part]
    mass = np.zeros((len(Xq), Cn))
    np.add.at(mass, (np.arange(len(Xq))[:, None].repeat(k, 1).ravel(), lab.ravel()), w.ravel())
    return np.log(np.clip(mass, 1e-9, None))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-train", default=str(HERE / "emb_train_ft.npy"))
    ap.add_argument("--emb-test", default=str(HERE / "emb_test_ft.npy"))
    ap.add_argument("--head", default=str(HERE / "jev-pro-model/choice_head.npz"))
    ap.add_argument("--fit-n", type=int, default=8000)
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()

    nz = np.load(a.head, allow_pickle=True)
    W, bias, T0 = nz["W"], nz["bias"], float(nz["temperature"])
    labels = [str(x) for x in nz["labels"]]
    Cn = len(labels)
    idx = {c: i for i, c in enumerate(labels)}

    def read(split):
        rows = [json.loads(l) for l in open(HERE / "data" / f"{split}.jsonl")]
        return rows

    # the head was trained on the raw pooled embedding; cosine kNN needs unit
    # vectors. Two views, no shared array: feeding normalised vectors to the head
    # quietly rescales every logit and turns "calibrated" into garbage.
    Xtr0, Xte0 = np.load(a.emb_train).astype(np.float32), np.load(a.emb_test).astype(np.float32)
    Xtr = Xtr0 / np.clip(np.linalg.norm(Xtr0, axis=1, keepdims=True), 1e-9, None)
    Xte = Xte0 / np.clip(np.linalg.norm(Xte0, axis=1, keepdims=True), 1e-9, None)
    tr = read("train")
    perm = np.random.default_rng(1234).permutation(len(tr))
    tr = [tr[i] for i in perm]
    Xtr, Xtr0 = Xtr[perm], Xtr0[perm]   # embeddings and labels shuffled together
    ytr = np.array([idx[r["label"]] for r in tr])
    te = read("test")
    yte = np.array([idx[r["label"]] for r in te])

    Xf, yf, Xv, yv = Xtr[: a.fit_n], ytr[: a.fit_n], Xtr[a.fit_n :], ytr[a.fit_n :]
    hv, ht = Xtr0[a.fit_n :] @ W.T + bias, Xte0 @ W.T + bias
    kv, kt = knn_sim(Xv, Xf, yf, Cn, a.k, 0.05), knn_sim(Xte, Xtr, ytr, Cn, a.k, 0.05)

    def tv(lg, lo=0.4, hi=6.0):
        grid = np.linspace(lo, hi, 60)
        return min((( -np.mean(np.log(np.clip(softmax(lg / T)[np.arange(len(yv)), yv], 1e-12, None))), float(T))
                    for T in grid), key=lambda s: s[0])[1]

    Th, Tk = tv(hv), tv(kv)
    Ph, Pk = softmax(hv / Th), softmax(kv / Tk)
    Ph_t, Pk_t = softmax(ht / Th), softmax(kt / Tk)
    print(f"on val:  head acc {float((Ph.argmax(1)==yv).mean()):.4f} (T={Th:.2f})   "
          f"knn@{a.k} acc {float((Pk.argmax(1)==yv).mean()):.4f} (T={Tk:.2f})")

    grid = np.linspace(0.0, 1.0, 21)
    scored = []
    for w in grid:
        Pv = (1 - w) * Ph + w * Pk
        nll = -np.mean(np.log(np.clip(Pv[np.arange(len(yv)), yv], 1e-12, None)))
        scored.append((nll, float(w), float((Pv.argmax(1) == yv).mean()), ece(Pv, yv)))
    best = min(scored, key=lambda s: s[0])
    _, w_best, vacc, vece = best
    print(f"blend weight picked on val NLL: w={w_best:.2f}  val acc {vacc:.4f}  val ECE {vece:.4f}")
    print("  w    valNLL   val_acc   val_ECE")
    for nll, w, va, ve in scored[::4]:
        print(f"  {w:.2f}  {nll:.4f}   {va:.4f}    {ve:.4f}")

    blend_val_logits = np.log(np.clip((1 - w_best) * Ph + w_best * Pk, 1e-9, None))
    Tb = tv(blend_val_logits)          # one more scaling step, still val-only
    blend_test_logits = np.log(np.clip((1 - w_best) * Ph_t + w_best * Pk_t, 1e-9, None))
    Pb_t = softmax(blend_test_logits / Tb)
    rows = [("fine-tuned head", Ph_t, Th), ("retrieval kNN", Pk_t, Tk), ("blend (val-selected w)", Pb_t, Tb)]
    print("\non the official test split:")
    print(f"  {'model':24s} {'acc':>7s} {'top3':>7s} {'ECE':>8s} {'conf-acc':>9s} {'cov@95':>7s} {'cov@99':>7s}")
    out = {}
    for name, P, tt in rows:
        acc = float((P.argmax(1) == yte).mean())
        top3 = float((np.argsort(-P, 1)[:, :3] == yte[:, None]).any(1).mean())
        ec, over = ece(P, yte), float(P.max(1).mean() * 100 - acc * 100)
        c95, c99 = coverage(P, yte, 0.95), coverage(P, yte, 0.99)
        out[name] = {"acc": acc, "top3": top3, "ece": ec, "conf_minus_acc_pts": over,
                     "cov95": c95, "cov99": c99, "T": tt}
        print(f"  {name:24s} {acc:7.4f} {top3:7.4f} {ec:8.4f} {over:+8.1f}p {c95:7.2f} {c99:7.2f}")
    print("\nreference: Jev 1.13 zero-shot 0.803 / Jev + 24 retrieved examples 0.9240 "
          "/ fine-tuned BERT (2020) 0.9366")
    json.dump({"blend_weight": w_best, "test": out, "val": {"head": float((Ph.argmax(1)==yv).mean()),
               "knn": float((Pk.argmax(1)==yv).mean())}, "k": a.k}, open(HERE / "ensemble_report.json", "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
