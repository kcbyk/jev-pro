#!/usr/bin/env python3
"""Frozen DistilBERT-base embeddings for the head/ensemble ladder (no optimizer -> fits RAM)."""
import json, time
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
torch.set_num_threads(2)
from transformers import AutoModel, AutoTokenizer
T0 = time.time()
tok = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
base = AutoModel.from_pretrained("distilbert/distilbert-base-uncased").eval()

@torch.inference_mode()
def pooled(texts, maxlen=48, bs=96):
    out = np.zeros((len(texts), 768), np.float32)
    order = np.argsort([len(t) for t in texts], kind="stable")
    for s in range(0, len(order), bs):
        sel = order[s:s+bs]
        e = tok([texts[i] for i in sel], padding=True, truncation=True, max_length=maxlen, return_tensors="pt")
        h = base(**e).last_hidden_state
        mk = e["attention_mask"].unsqueeze(-1).to(h.dtype)
        out[sel] = ((h*mk).sum(1)/mk.sum(1).clamp(min=1e-6)).numpy()
        if s and s % 2880 == 0:
            print(f"  {s}/{len(texts)} [{time.time()-T0:.0f}s]", flush=True)
    return out

tr = [json.loads(l) for l in open(HERE/"data/train.jsonl")]
te = [json.loads(l) for l in open(HERE/"data/test.jsonl")]
au = [json.loads(l) for l in open(HERE/"data/train_aug.jsonl")]
np.save(HERE/"emb_train_ft.npy", pooled([r["text"] for r in tr]))
print(f"train done [{time.time()-T0:.0f}s]", flush=True)
np.save(HERE/"emb_test_ft.npy", pooled([r["text"] for r in te]))
np.save(HERE/"emb_aug_ft.npy", pooled([r["text"] for r in au]))
np.save(HERE/"aug_base.npy", np.array([r["base"] for r in au], np.int32))
desc = json.load(open(HERE/"data/label_desc.json")); labs = json.load(open(HERE/"data/labels.json"))
np.save(HERE/"emb_desc_ft.npy", pooled([desc[c] for c in labs]))
print(f"ALL DONE [{time.time()-T0:.0f}s]", flush=True)
