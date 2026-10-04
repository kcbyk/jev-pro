#!/usr/bin/env python3
"""Fixed-weight probability ensembles across r2 / r3w / r3w2 heads (each in its own space).

No test-driven weight search: weights are uniform by design; each member's temperature is the
one fitted on ITS OWN clean val (r3w's is the known-optimistic exception — disclosed in README).
"""
import json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sm = lambda z: (np.exp(z - z.max(1, keepdims=True)) / np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True))
def ece(P, y, bins=15):
    conf = P.max(1); cor = P.argmax(1) == y; tot = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf <= hi) if lo == 0 else (conf > lo) & (conf <= hi)
        if m.any(): tot += m.mean() * abs(cor[m].mean() - conf[m].mean())
    return float(tot)
def cov(P, y, tgt):
    conf = P.max(1); cor = P.argmax(1) == y; best = 0.0
    for t in np.quantile(conf, np.linspace(0, .999, 240)):
        m = conf >= t
        if m.any() and cor[m].mean() >= tgt: best = max(best, float(m.mean()))
    return best
tel = [json.loads(l) for l in open(HERE/"data/test.jsonl")]
def head(d):
    z = np.load(HERE/d/"choice_head.npz", allow_pickle=True)
    return z["W"], z["bias"], float(z["temperature"]), [str(x) for x in z["labels"]]
W2, b2, T2, lb = head("jev-pro-model-r2")
i2 = {c: i for i, c in enumerate(lb)}
X2 = np.load(HERE/"emb_test_ft_r2.npy").astype(np.float32)
P_r2 = sm((X2 @ W2.T + b2) / T2)
Ww, bw, Tw, _ = head("jev-pro-model-r3w")
Lw = np.load(HERE/"logits_test_ft.npy")                 # fresh from noul(r3w) — model head
P_w = sm(np.asarray(Lw, np.float64) / 0.75)              # card T, not the logistic refit
W3, b3, T3, _ = head("jev-pro-model-r3w2")
L3 = np.load(HERE/"logits_test_ft_r3w2.npy")
P_3 = sm(np.asarray(L3, np.float64) / T3)
y = np.array([i2[r["label"]] for r in tel])
def rep(name, P):
    a = float((P.argmax(1) == y).mean())
    t3 = float((np.argsort(-P,1)[:,:3] == y[:,None]).any(1).mean())
    t5 = float((np.argsort(-P,1)[:,:5] == y[:,None]).any(1).mean())
    print(f"  {name:28s} acc {a:.4f}  top3 {t3:.4f}  top5 {t5:.4f}  ECE {ece(P,y):.4f}  "
          f"conf-acc {float(P.max(1).mean()*100-a*100):+.1f}p  cov@95 {cov(P,y,.95):.2f}  cov@99 {cov(P,y,.99):.2f}")
print("official test split, fixed ensembles:")
for name, P in [("r2 head", P_r2), ("r3w head", P_w), ("r3w2 head", P_3),
                ("mean(r3w,r3w2)", 0.5*P_w+0.5*P_3), ("mean(r2,r3w2)", 0.5*P_r2+0.5*P_3),
                ("mean(r2,r3w,r3w2)", (P_r2+P_w+P_3)/3)]:
    rep(name, P/ P.sum(1, keepdims=True))
