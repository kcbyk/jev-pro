#!/usr/bin/env python3
"""Two-model probability average (r2 head in r2 space x r3w2 head in r3w2 space).

Fixed alpha=0.5 arithmetic mean + geometric mean; alpha grid shown as ORACLE
(diagnostic only, never a selection channel). Test scored post-hoc for research;
the SERVED model choice remains the single-val-selected stack.
"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
def sm(z):
    z = np.asarray(z, np.float64); z -= z.max(1, keepdims=True); e = np.exp(z); return e/e.sum(1, keepdims=True)

tel = [json.loads(l) for l in open(HERE/"data/test.jsonl")]
nz2 = np.load(HERE/"jev-pro-model-r2/choice_head.npz", allow_pickle=True)
labels2 = [str(x) for x in nz2["labels"]]
i2 = {c: i for i, c in enumerate(labels2)}
X2 = np.load(HERE/"emb_test_ft_r2.npy").astype(np.float32)
P2 = sm((X2 @ nz2["W"].T + nz2["bias"]) / float(nz2["temperature"]))
nz3 = np.load(HERE/"jev-pro-model-r3w2/choice_head.npz", allow_pickle=True)
labels3 = [str(x) for x in nz3["labels"]]
assert labels3 == labels2
y = np.array([i2[r["label"]] for r in tel])
Z3 = np.load(HERE/"logits_test_ft.npy")          # written by noul for r3w2
P3 = sm(np.asarray(Z3, np.float64) / float(nz3["temperature"]))
acc = lambda P: float((P.argmax(1) == y).mean())
print(f"r2 head {acc(P2):.4f}   r3w2 head {acc(P3):.4f}")
ar = lambda P: np.mean((P.argmax(1)[:,None] == np.argsort(-P,1)[:,:3]).any(1)*0 + (np.argsort(-P,1)[:,:3]==y[:,None]).any(1)*1)
grid = np.linspace(0,1,21)
accs = [acc((1-w)*P2 + w*P3) for w in grid]
print("alpha-oracle best:", round(max(accs),4), "at w=", grid[int(np.argmax(accs))])
for name, P in (("arith mean a=0.5", sm(np.log(np.clip(0.5*P2+0.5*P3,1e-9,None)))),
                ("geometric mean", sm(np.log(np.clip(P2,1e-9,None))*0.5 + np.log(np.clip(P3,1e-9,None))*0.5))):
    top3 = float((np.argsort(-P,1)[:,:3]==y[:,None]).any(1).mean())
    print(f"{name}: acc {acc(P):.4f}  top3 {top3:.4f}")
