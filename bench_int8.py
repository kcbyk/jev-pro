#!/usr/bin/env python3
"""fp32 vs int8 accuracy + latency for the served encoder path (single request).

Accuracy: re-embed the FULL test split with the quantized encoder and score the
shipped choice head — so any quantization damage is measured, not assumed.
"""
import argparse, json, time
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", default=str(HERE / "jev-pro-model-r3w2"))
ap.add_argument("--state", default=str(HERE / "jev-pro-model-r3w2" / "ft_state_fp16.pt"))
ap.add_argument("--n-lat", type=int, default=200)
a = ap.parse_args()
torch.set_num_threads(2)
from transformers import AutoModel, AutoTokenizer

card = json.load(open(Path(a.model_dir) / "model_card.json"))
nz = np.load(Path(a.model_dir) / "choice_head.npz", allow_pickle=True)
W, bias, T = nz["W"], nz["bias"], float(nz["temperature"])
labels = [str(x) for x in nz["labels"]]
idx = {c: i for i, c in enumerate(labels)}
tok = AutoTokenizer.from_pretrained(a.model_dir)

def build(q):
    m = AutoModel.from_pretrained(card["base_model"])
    st = torch.load(a.state, map_location="cpu", weights_only=True)
    m.load_state_dict({k[5:]: (v.float() if v.is_floating_point() else v) for k, v in st.items() if k.startswith("base.")}, strict=False)
    if q:
        m = torch.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8)
    return m.eval()

@torch.inference_mode()
def pooled(m, texts, maxlen=48, bs=128):
    out = np.zeros((len(texts), W.shape[1]), np.float32)
    order = np.argsort([len(t) for t in texts], kind="stable")
    for s in range(0, len(order), bs):
        sel = order[s:s+bs]
        e = tok([texts[i] for i in sel], padding=True, truncation=True, max_length=maxlen, return_tensors="pt")
        h = m(**e).last_hidden_state
        mk = e["attention_mask"].unsqueeze(-1).to(h.dtype)
        out[sel] = ((h*mk).sum(1)/mk.sum(1).clamp(min=1e-6)).numpy()
    return out

te = [json.loads(l) for l in open(HERE / "data/test.jsonl")]
txt = [r["text"] for r in te]
yt = np.array([idx[r["label"]] for r in te])

def sm(z):
    z = np.asarray(z, np.float64); z -= z.max(1, keepdims=True); e = np.exp(z); return e/e.sum(1, keepdims=True)

for tag, q in (("fp32", False), ("int8", True)):
    m = build(q)
    t0 = time.perf_counter(); P = sm((pooled(m, txt) @ W.T + bias)/T); enc = time.perf_counter()-t0
    acc = float((P.argmax(1) == yt).mean())
    conf = P.max(1); ece = 0.0
    edges = np.linspace(0,1,16)
    for lo, hi in zip(edges[:-1], edges[1:]):
        msk = (conf >= lo) & (conf <= hi) if lo == 0 else (conf > lo) & (conf <= hi)
        if msk.any(): ece += msk.mean()*abs((P.argmax(1)==yt)[msk].mean()-conf[msk].mean())
    lats = []
    for r in te[: a.n_lat]:
        s0 = time.perf_counter()
        with torch.inference_mode():
            e = tok([r["text"]], padding=False, truncation=True, max_length=48, return_tensors="pt")
            h = m(**e).last_hidden_state
            mk = e["attention_mask"].unsqueeze(-1).to(h.dtype)
            v = ((h*mk).sum(1)/mk.sum(1).clamp(min=1e-6)).numpy()
            (v @ W.T + bias)
        lats.append((time.perf_counter()-s0)*1000)
    print(f"{tag}: test acc {acc:.4f}  ECE {ece:.4f}  full-split embed {enc:.0f}s  "
          f"single-req p50 {np.median(lats):.1f} ms  p95 {np.quantile(lats,0.95):.1f} ms")
