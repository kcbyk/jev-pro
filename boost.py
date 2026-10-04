#!/usr/bin/env python3
"""Probability ensemble on the cached r2 embeddings — no encoder training here.

Ingredients (all choices made out-of-fold on the 9,993-row train split; the
official test split is touched exactly once, at the end):

  * multi-seed logistic heads + one small MLP head on standardized pooled
    embeddings, probability-averaged; optionally trained with rule-based
    paraphrase copies (--aug) whose parent row must live in the training part
    of every fold, so no augmented copy ever helps predict its own parent
  * similarity-weighted kNN retrieval on the fine-tuned unit embeddings
  * per-class centroid scores
  * label-description reranker (--desc): cosine of the query against one
    customer-voice sentence per class, temperature-fitted on OOF
  * blend weights + one blend temperature fitted on the OOF stack

Known traps, handled: raw vs normalized embeddings kept as separate views;
embeddings, labels and augmented parents shuffled with the same permutation;
MLP member added once per ensemble, not once per seed.
"""
from __future__ import annotations

import argparse
import json
import time
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


def fit_head(X, y, Cn, *, seed, hidden=0, epochs=25, bs=256, lr=0.08, wd=1e-5,
             smooth=0.05, mom=0.9):
    """Minibatch SGD multinomial logistic (hidden=0) or one-hidden-layer MLP."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    nl = np.full((n, Cn), smooth / Cn, np.float32)
    nl[np.arange(n), y] += 1.0 - smooth
    P = [rng.standard_normal((d, Cn)).astype(np.float32) * (0.05 if not hidden else 1.0 / np.sqrt(d)),
         np.zeros(Cn, np.float32)]
    if hidden:
        P += [rng.standard_normal((d, hidden)).astype(np.float32) * np.float32(1.0 / np.sqrt(d)),
              np.zeros(hidden, np.float32),
              rng.standard_normal((hidden, Cn)).astype(np.float32) * np.float32(1.0 / np.sqrt(hidden)),
              np.zeros(Cn, np.float32)]
    V = [np.zeros_like(p) for p in P]

    def logits(Xb):
        if hidden:
            h = np.maximum(Xb @ P[2] + P[3], 0.0)
            return h @ P[4] + P[5]
        return Xb @ P[0] + P[1]

    for ep in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, bs):
            ix = order[s : s + bs]
            Xb, yb = X[ix], nl[ix]
            Pr = softmax(logits(Xb)).astype(np.float32)
            g = (Pr - yb) / len(ix)
            if hidden:
                Hpre = np.maximum(Xb @ P[2] + P[3], 0.0)
                gh = g @ P[4].T * (Hpre > 0)
                Pgrad = [np.zeros_like(P[0]), np.zeros_like(P[1]), Xb.T @ gh, gh.mean(0), Hpre.T @ g, g.mean(0)]
            else:
                Pgrad = [Xb.T @ g, g.mean(0)]
            for i, gr in enumerate(Pgrad):
                V[i] = mom * V[i] - lr * (gr + wd * P[i])
                P[i] = P[i] + V[i]
    return P, hidden


def head_logits(P, hidden, X):
    if hidden:
        h = np.maximum(X @ P[2] + P[3], 0.0)
        return h @ P[4] + P[5]
    return X @ P[0] + P[1]


def knn_sim(Xq, Xr, yr, Cn, k, tau):
    S = Xq @ Xr.T
    part = np.argpartition(-S, k, axis=1)[:, :k]
    sim = np.take_along_axis(S, part, axis=1)
    w = np.exp((sim - sim.max(1, keepdims=True)) / tau).astype(np.float32)
    lab = yr[part]
    mass = np.zeros((len(Xq), Cn), np.float32)
    np.add.at(mass, (np.arange(len(Xq))[:, None].repeat(k, 1).ravel(), lab.ravel()), w.ravel())
    return np.log(np.clip(mass, 1e-9, None))


def fit_temp(Z, y, lo=0.3, hi=8.0):
    grid = np.linspace(lo, hi, 80)
    scored = [(-np.mean(np.log(np.clip(softmax(Z / T)[np.arange(len(y)), y], 1e-12, None))), float(T)) for T in grid]
    return min(scored, key=lambda s: s[0])[1]


def report(P, yte, name):
    acc = float((P.argmax(1) == yte).mean())
    top3 = float((np.argsort(-P, 1)[:, :3] == yte[:, None]).any(1).mean())
    top5 = float((np.argsort(-P, 1)[:, :5] == yte[:, None]).any(1).mean())
    ec = ece(P, yte)
    over = float(P.max(1).mean() * 100 - acc * 100)
    c95, c99 = coverage(P, yte, 0.95), coverage(P, yte, 0.99)
    return {"model": name, "acc": round(acc, 4), "top3": round(top3, 4), "top5": round(top5, 4),
            "ece": round(ec, 4), "conf_minus_acc_pts": round(over, 1), "cov95": round(c95, 2),
            "cov99": round(c99, 2)}, f"  {name:34s} acc {acc:.4f}  top3 {top3:.4f}  top5 {top5:.4f}  ECE {ec:.4f}  cov@95 {c95:.2f}  cov@99 {c99:.2f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nseeds", type=int, default=4)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--ks", default="8,16,32")
    ap.add_argument("--aug", type=int, default=1, help="use emb_aug_ft.npy if present")
    ap.add_argument("--desc", type=int, default=1, help="use emb_desc_ft.npy if present")
    ap.add_argument("--out", default=str(HERE / "boost_report.json"))
    a = ap.parse_args()
    ks = [int(x) for x in a.ks.split(",")]

    t0 = time.time()
    Xtr0 = np.load(HERE / "emb_train_ft.npy").astype(np.float32)
    Xte0 = np.load(HERE / "emb_test_ft.npy").astype(np.float32)
    labs = [json.loads(l) for l in open(HERE / "data/train.jsonl")]
    telabs = [json.loads(l) for l in open(HERE / "data/test.jsonl")]
    nz = np.load(HERE / "jev-pro-model-r2/choice_head.npz", allow_pickle=True)
    labels = [str(x) for x in nz["labels"]]
    idx = {c: i for i, c in enumerate(labels)}
    Cn = len(labels)

    perm = np.random.default_rng(1234).permutation(len(labs))
    labs = [labs[i] for i in perm]
    Xtr0 = Xtr0[perm]
    ytr = np.array([idx[r["label"]] for r in labs])
    yte = np.array([idx[r["label"]] for r in telabs])

    # augmented block, aligned to ORIGINAL train order; map parents to shuffled pos
    use_aug = bool(a.aug) and (HERE / "emb_aug_ft.npy").exists()
    if use_aug:
        Xau0 = np.load(HERE / "emb_aug_ft.npy").astype(np.float32)
        base = np.load(HERE / "aug_base.npy")            # original train index per aug row
        inv = np.argsort(perm)                            # orig index -> shuffled position
        parent_pos = inv[base]
        yaug_rows = [json.loads(l) for l in open(HERE / "data/train_aug.jsonl")]
        yau = np.array([idx[r["label"]] for r in yaug_rows])
    use_desc = bool(a.desc) and (HERE / "emb_desc_ft.npy").exists()
    if use_desc:
        D = np.load(HERE / "emb_desc_ft.npy").astype(np.float32)

    mu, sd = Xtr0.mean(0), Xtr0.std(0) + 1e-6
    Xtr, Xte = (Xtr0 - mu) / sd, (Xte0 - mu) / sd
    if use_aug:
        Xau = (Xau0 - mu) / sd
    Ntr = Xtr0 / np.clip(np.linalg.norm(Xtr0, axis=1, keepdims=True), 1e-9, None)
    Nte = Xte0 / np.clip(np.linalg.norm(Xte0, axis=1, keepdims=True), 1e-9, None)
    if use_aug:
        Nau = Xau0 / np.clip(np.linalg.norm(Xau0, axis=1, keepdims=True), 1e-9, None)
    if use_desc:
        Dn = D / np.clip(np.linalg.norm(D, axis=1, keepdims=True), 1e-9, None)
    print(f"data ready {time.time()-t0:.1f}s  train {Xtr.shape} test {Xte.shape}  aug={use_aug} desc={use_desc}")

    def heads_pred(Xq, Xs, ys, seeds):
        Ps = []
        for s in range(seeds):
            P, _ = fit_head(Xs, ys, Cn, seed=100 + s, hidden=0)
            Ps.append(softmax(head_logits(P, 0, Xq)))
            if s == 0:  # exactly one MLP member per ensemble call
                Pm, _ = fit_head(Xs, ys, Cn, seed=900, hidden=192, epochs=18, lr=0.03)
                Ps.append(softmax(head_logits(Pm, 1, Xq)))
        return np.mean(Ps, 0)

    def head_pred_fold(iq, it):
        """it: shuffled-train positions to train on; aug copies only if parent in it."""
        Xs, ys = Xtr[it], ytr[it]
        if use_aug:
            keep = np.isin(parent_pos, it)
            Xs, ys = np.vstack([Xs, Xau[keep]]), np.concatenate([ys, yau[keep]])
        return heads_pred(Xtr[iq], Xs, ys, a.nseeds)

    n = len(ytr)
    oof = np.random.default_rng(7).permutation(n)
    fold = np.zeros(n, np.int64)
    for f in range(a.folds):
        fold[oof[f :: a.folds]] = f
    OofH = np.zeros((n, Cn), np.float32)
    OofHa = np.zeros((n, Cn), np.float32) if use_aug else None
    OofK = {k: np.zeros((n, Cn), np.float32) for k in ks}
    OofC = np.zeros((n, Cn), np.float32)
    OofD = np.zeros((n, Cn), np.float32) if use_desc else None
    for f in range(a.folds):
        trp, vap = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        OofH[vap] = heads_pred(Xtr[vap], Xtr[trp], ytr[trp], a.nseeds)
        if use_aug:
            OofHa[vap] = head_pred_fold(vap, trp)
        for k in ks:
            OofK[k][vap] = softmax(knn_sim(Ntr[vap], Ntr[trp], ytr[trp], Cn, k, 0.05))
        C = np.stack([Ntr[trp][ytr[trp] == c].mean(0) for c in range(Cn)])
        C /= np.clip(np.linalg.norm(C, axis=1, keepdims=True), 1e-9, None)
        OofC[vap] = softmax(Ntr[vap] @ C.T / 0.08)
        if use_desc:
            OofD[vap] = Ntr[vap] @ Dn.T
        acc_c = float((OofH[vap].argmax(1) == ytr[vap]).mean())
        acc_a = float((OofHa[vap].argmax(1) == ytr[vap]).mean()) if use_aug else float("nan")
        print(f"fold {f}: head {acc_c:.4f}  head+aug {acc_a:.4f}  knn@{ks[1]} "
              f"{float((OofK[ks[1]][vap].argmax(1)==ytr[vap]).mean()):.4f}  desc "
              f"{float((OofD[vap].argmax(1)==ytr[vap]).mean()) if use_desc else float('nan'):.4f}  "
              f"[{time.time()-t0:.0f}s]", flush=True)

    def nll(P, yv):
        return -np.mean(np.log(np.clip(P[np.arange(len(yv)), yv], 1e-12, None)))

    Th = fit_temp(np.log(np.clip(OofH, 1e-9, None)), ytr)
    Ph_o = softmax(np.log(np.clip(OofH, 1e-9, None)) / Th)
    if use_aug:
        Tha = fit_temp(np.log(np.clip(OofHa, 1e-9, None)), ytr)
        Pha_o = softmax(np.log(np.clip(OofHa, 1e-9, None)) / Tha)
        print(f"OOF head {float((Ph_o.argmax(1)==ytr).mean()):.4f} (T={Th:.2f}) | "
              f"OOF head+aug {float((Pha_o.argmax(1)==ytr).mean()):.4f} (T={Tha:.2f})")
        Ppick = Pha_o if (Pha_o.argmax(1) == ytr).mean() >= (Ph_o.argmax(1) == ytr).mean() else Ph_o
    else:
        Ppick = Ph_o
    kk = max(ks, key=lambda k: (OofK[k].argmax(1) == ytr).mean())
    if use_desc:
        Td = fit_temp(OofD, ytr)
        Po = softmax(OofD / Td)
        print(f"OOF desc-only acc {float((Po.argmax(1)==ytr).mean()):.4f} (T={Td:.2f})")
    best = None
    grid = np.arange(0, 1.0001, 0.05)
    for w in grid:
        for v in grid[grid <= 1 - w + 1e-9]:
            for u in (grid[grid <= 1 - w - v + 1e-9] if use_desc else [0.0]):
                P = (1 - w - v - u) * Ppick + w * OofK[kk] + v * OofC + (u * Po if use_desc else 0)
                P = P / P.sum(1, keepdims=True)
                s = nll(P, ytr)
                if best is None or s < best[0]:
                    best = (s, float(w), float(v), float(u))
    _, wK, wC, wD = best
    Pb_o = (1 - wK - wC - wD) * Ppick + wK * OofK[kk] + wC * OofC + (wD * Po if use_desc else 0)
    Pb_o = Pb_o / Pb_o.sum(1, keepdims=True)
    Tb = fit_temp(np.log(np.clip(Pb_o, 1e-9, None)), ytr)
    Pb_o = softmax(np.log(np.clip(Pb_o, 1e-9, None)) / Tb)
    print(f"blend picked on OOF: w_knn={wK:.2f} w_cent={wC:.2f} w_desc={wD:.2f} T={Tb:.2f}  "
          f"OOF acc {float((Pb_o.argmax(1)==ytr).mean()):.4f}")

    # ---- final models on ALL train, measured on test ----
    aug_wins = use_aug and float((Pha_o.argmax(1) == ytr).mean()) >= float((Ph_o.argmax(1) == ytr).mean())
    T_use = Tha if aug_wins else Th
    Xs_full = np.vstack([Xtr, Xau]) if aug_wins else Xtr
    ys_full = np.concatenate([ytr, yau]) if aug_wins else ytr
    print(f"final head trained on {len(ys_full)} rows ({'with' if aug_wins else 'without'} aug), T={T_use:.2f}")
    Ph_t = softmax(np.log(np.clip(heads_pred(Xte, Xs_full, ys_full, a.nseeds + 4), 1e-9, None)) / T_use)
    kt = softmax(knn_sim(Nte, Ntr, ytr, Cn, kk, 0.05))
    C = np.stack([Ntr[ytr == c].mean(0) for c in range(Cn)])
    C /= np.clip(np.linalg.norm(C, axis=1, keepdims=True), 1e-9, None)
    Ct = softmax(Nte @ C.T / 0.08)
    Ph_tl = np.log(np.clip(Ph_t, 1e-9, None))
    def blend(*Ps):
        P = sum(w * p for w, p in zip((1 - wK - wC - wD, wK, wC, wD), Ps + (0,) * (4 - len(Ps))))
        return P / np.clip(P.sum(1, keepdims=True), 1e-12, None)
    if use_desc:
        Pd_t = softmax((Nte @ Dn.T) / Td)
        Pfin = softmax(np.log(np.clip(blend(Ph_t, kt, Ct, Pd_t), 1e-9, None)) / Tb)
        geo = softmax((1 - wK - wC - wD) * Ph_tl + wK * np.log(np.clip(kt, 1e-9, None)) +
                      wC * np.log(np.clip(Ct, 1e-9, None)) + wD * np.log(np.clip(Pd_t, 1e-9, None)))
    else:
        Pfin = softmax(np.log(np.clip(blend(Ph_t, kt, Ct), 1e-9, None)) / Tb)
        geo = softmax((1 - wK - wC - wD) * Ph_tl + wK * np.log(np.clip(kt, 1e-9, None)) +
                      wC * np.log(np.clip(Ct, 1e-9, None)))

    W, b0, T0 = nz["W"], nz["bias"], float(nz["temperature"])
    ref_path = HERE / "emb_test_ft_r2.npy"   # shipped head lives in r2 space; score it there
    Xref = np.load(ref_path).astype(np.float32) if ref_path.exists() else Xte0
    Pr2 = softmax((Xref @ W.T + b0) / T0)
    flip = int((Pfin.argmax(1) != Pr2.argmax(1)).sum())
    print(f"\nr2 single head -> boost flips on {flip} / {len(yte)} test rows")

    rows = []
    print("\non the official test split (3,076 rows, measured once):")
    cand = [("r2 single head (in r2 space)", Pr2), ("multi-seed head ensemble", Ph_t),
            (f"kNN k={kk} on ft embeddings", kt), ("boost blend (this run)", Pfin),
            ("boost blend, geometric mean", geo)]
    if use_desc:
        cand.append(("label-desc reranker alone", Pd_t))
    for name, P in cand:
        d, line = report(P, yte, name)
        rows.append(d)
        print(line)
    json.dump({"rows": rows, "knn_k": kk, "w_knn": wK, "w_centroid": wC, "w_desc": wD,
               "T_head": Th, "T_blend": Tb, "aug": use_aug, "desc": use_desc,
               "seconds": round(time.time() - t0, 1)}, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}   [{time.time()-t0:.0f}s total]")
    print("targets: Banking77 SOTA 0.9383 | fine-tuned BERT 0.9366 | published 22M+LR 0.932")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
