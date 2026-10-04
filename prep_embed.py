#!/usr/bin/env python3
"""Encode the augmented train rows and the 77 label descriptions with the
r2 fine-tuned encoder (same pooled space as emb_*_ft.npy).

Outputs: emb_aug_ft.npy  (19986, 384)  aligned with data/train_aug.jsonl
         emb_desc_ft.npy (77, 384)     aligned with data/labels.json order
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


import numpy as np
import torch

HERE = Path(__file__).resolve().parent
AP = argparse.ArgumentParser()
AP.add_argument("--outdir", default=str(HERE / "jev-pro-model-r2"), help="dir with model_card.json + tokenizer")
AP.add_argument("--state", default=str(HERE / "ft_state.pt"))
ARGS, _ = AP.parse_known_args()


@torch.inference_mode()
def embed(base, tok, txts, maxlen=48, bs=128):
    base.eval()
    out = np.zeros((len(txts), 384), dtype=np.float32)
    order = np.argsort([len(t) for t in txts], kind="stable")
    for s in range(0, len(order), bs):
        sel = order[s : s + bs]
        e = tok([txts[i] for i in sel], padding=True, truncation=True, max_length=maxlen, return_tensors="pt")
        h = base(**e).last_hidden_state
        m = e["attention_mask"].unsqueeze(-1).to(h.dtype)
        out[sel] = ((h * m).sum(1) / m.sum(1).clamp(min=1e-6)).numpy()
        if s and s % 1280 == 0:
            print(f"  {s}/{len(txts)} rows [{time.time()-T0:.0f}s]", flush=True)
    return out


T0 = time.time()
torch.set_num_threads(2)
card = json.load(open(Path(ARGS.outdir) / "model_card.json"))
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained(ARGS.outdir)
base = AutoModel.from_pretrained(card["base_model"])
state = torch.load(ARGS.state, map_location="cpu", weights_only=True)
missing, unexpected = base.load_state_dict({k[5:]: v for k, v in state.items() if k.startswith("base.")}, strict=False)
print(f"loaded r2 encoder ({len(state)} tensors, unexpected={len(unexpected)})", flush=True)

aug = [json.loads(l) for l in open(HERE / "data/train_aug.jsonl")]
E = embed(base, tok, [r["text"] for r in aug])
np.save(HERE / "emb_aug_ft.npy", E)
np.save(HERE / "aug_base.npy", np.array([r["base"] for r in aug], np.int32))
print(f"emb_aug_ft.npy {E.shape}  [{time.time()-T0:.0f}s]", flush=True)

labels = json.load(open(HERE / "data/labels.json"))
desc = json.load(open(HERE / "data/label_desc.json"))
D = embed(base, tok, [desc[c] for c in labels])
np.save(HERE / "emb_desc_ft.npy", D)
print(f"emb_desc_ft.npy {D.shape}  [{time.time()-T0:.0f}s]", flush=True)

# sanity: augmented copy should sit closer to its own class centroid than to the
# global mean; report the fraction of aug rows whose nearest TRAIN row (cosine)
# shares the label (on a 2000-row sample, against unit-normalized spaces)
Xtr = np.load(HERE / "emb_train_ft.npy").astype(np.float32)
tr = [json.loads(l) for l in open(HERE / "data/train.jsonl")]
perm = np.random.default_rng(1234).permutation(len(tr))
tr = [tr[i] for i in perm]
Xtr = Xtr[perm]
u = lambda A: A / np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-9, None)
Un, Ue = u(Xtr), u(E)
sample = np.random.default_rng(3).choice(len(E), 2000, replace=False)
S = Ue[sample] @ Un.T
nn = S.argmax(1)
same = np.mean([aug[i]["label"] == tr[j]["label"] for i, j in zip(sample, nn)])
base_same = np.mean([aug[i]["base"] == j for i, j in zip(sample, nn)])
print(f"aug-row nearest train row shares label: {same:.4f}  (exact parent row: {base_same:.4f})")
