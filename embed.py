#!/usr/bin/env python3
"""Embed Banking77 with a small real encoder, once, on CPU.

This is the step that turns the toy into a real system: instead of hand-written
templates, we use a pretrained 22M-parameter sentence encoder (the same size
class the published "beat Jev locally" results used) and reuse its embeddings
for every downstream experiment. Embedding 13k short messages once and then
solving 77-way decision problems on the frozen vectors costs seconds per run,
which is the only way this is tractable on 2 CPU cores.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--maxlen", type=int, default=48)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    from transformers import AutoModel, AutoTokenizer  # deferred: slow import

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModel.from_pretrained(a.model)
    model.eval()
    nparam = sum(p.numel() for p in model.parameters())
    print(f"model {a.model}  {nparam/1e6:.1f}M params  dtype fp32  threads={a.threads}", flush=True)

    for split in ("train", "test"):
        rows = [json.loads(l) for l in open(HERE / "data" / f"{split}.jsonl")]
        # length-bucketed batches so we pad to ~14 tokens instead of 48
        order = np.argsort([len(r["text"]) for r in rows], kind="stable")
        out = np.zeros((len(rows), model.config.hidden_size), dtype=np.float32)
        t0 = time.time()
        done = 0
        with torch.inference_mode():
            for s in range(0, len(order), a.batch):
                idx = order[s : s + a.batch]
                enc = tok([rows[i]["text"] for i in idx], padding=True, truncation=True,
                          max_length=a.maxlen, return_tensors="pt")
                hid = model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).to(hid.dtype)
                emb = (hid * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
                emb = torch.nn.functional.normalize(emb, dim=-1)
                out[idx] = emb.numpy().astype(np.float32)
                done += len(idx)
                if done % 1999 < a.batch:
                    rate = done / (time.time() - t0)
                    print(f"  {split}: {done}/{len(order)}  {rate:.0f} examples/s", flush=True)
        np.save(HERE / f"emb_{split}.npy", out)
        dt = time.time() - t0
        print(f"wrote emb_{split}.npy {out.shape} in {dt:.1f}s  ({len(order)/dt:.0f} ex/s, "
              f"{dt / len(order) * 1000:.2f} ms/example encode time)", flush=True)
    (HERE / "embed_meta.json").write_text(json.dumps({"model": a.model, "params": nparam,
                                                      "maxlen": a.maxlen, "pooling": "mean+l2"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
