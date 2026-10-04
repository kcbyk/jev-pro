#!/usr/bin/env python3
"""Export jev-pro to a browser-ONNX bundle (web/): model + tokenizer + head as data.

Graph: input_ids/attention_mask -> mean-pooled embedding(384). The 77-label choice
head + noul heads stay OUTSIDE the graph (float32 .bin + JSON) — in JS a 30k-flop
matmul is cheaper + exact than baking them in. Steps: fp32 export -> quantize int8
-> parity vs the torch server path on the full test set (acc, top1 match, softmax
max|Δ|, latency). This is the "no server at all" deploy from the README discussion.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
MD = HERE / "jev-pro-model-r3w2"
WEB = HERE / "web"


class Enc(torch.nn.Module):
    """tokenizer-outputs -> embedding (the exact math of serve_pro.Engine.embed)."""

    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, input_ids, attention_mask):
        h = self.base(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        emb = (h * m).sum(1) / m.sum(1).clamp(min=1e-6)
        return torch.nn.functional.normalize(emb, dim=-1) if False else emb


def main():
    WEB.mkdir(exist_ok=True)
    from transformers import AutoModel, AutoTokenizer
    from serve_pro import Engine
    eng = Engine(MD, MD / "ft_state_fp16.pt")
    tok = eng.tok
    model = Enc(eng.base).float().eval()

    demo = tok(["parity check sentence"], padding="max_length", truncation=True,
               max_length=48, return_tensors="pt")
    fp32_path = WEB / "_enc_fp32.onnx"
    torch.onnx.export(model, (demo["input_ids"], demo["attention_mask"]), str(fp32_path),
                      input_names=["input_ids", "attention_mask"], output_names=["embedding"],
                      dynamic_axes={"input_ids": {0: "b", 1: "s"},
                                    "attention_mask": {0: "b", 1: "s"},
                                    "embedding": {0: "b"}},
                      opset_version=17, dynamo=False)

    from onnxruntime.quantization import QuantType, quantize_dynamic
    int8_path = WEB / "encoder_int8.onnx"
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QUInt8)
    fp32_path.unlink()   # 91 MB — do not keep; workspace budget is tight

    # ---- head/noul/tokenizer artifacts --------------------------------------
    nz = np.load(MD / "choice_head.npz", allow_pickle=True)
    W, bias, T = nz["W"], nz["bias"], float(nz["temperature"])
    labels = [str(x) for x in nz["labels"]]
    (WEB / "head.bin").write_bytes(np.asarray(W, np.float32).tobytes())
    json.dump({"labels": labels, "bias": [float(x) for x in bias], "T": T,
               "noul": json.load(open(MD / "noul_heads.json"))},
              open(WEB / "head.json", "w"))
    (WEB / "tokenizer.json").write_bytes((MD / "tokenizer.json").read_bytes())
    (WEB / "tokenizer_config.json").write_bytes((MD / "tokenizer_config.json").read_bytes())

    # ---- parity: ORT int8 web path vs the torch server path (test set) -----
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    sess = ort.InferenceSession(str(int8_path), sess_options=so, providers=["CPUExecutionProvider"])

    rows = [json.loads(l) for l in open(HERE / "data" / "test.jsonl")]
    texts = [r["text"] for r in rows]
    golds = [labels.index(r["label"]) for r in rows]   # labels are strings here

    n = 500
    hit_ort = hit_ref = agree = 0
    dmax = 0.0
    lat = []
    for i in range(0, min(n, len(texts))):
        t = texts[i]
        e = tok([t], padding="max_length", truncation=True, max_length=48, return_tensors="np")
        t0 = time.perf_counter()
        emb = sess.run(None, {"input_ids": e["input_ids"].astype(np.int64),
                               "attention_mask": e["attention_mask"].astype(np.int64)})[0][0]
        lat.append((time.perf_counter() - t0) * 1000)
        lg = emb @ W.T + bias
        p = np.exp(lg - lg.max()); p /= p.sum()
        lg2 = eng.logits([t])[0]
        p2 = np.exp(lg2 - lg2.max()); p2 /= p2.sum()
        hit_ort += int(p.argmax() == (golds[i] if golds else -1))
        hit_ref += int(p2.argmax() == (golds[i] if golds else -1))
        agree += int(p.argmax() == p2.argmax())
        dmax = max(dmax, float(np.abs(p - p2).max()))
    lat.sort()
    print(f"parity on {n} test texts:")
    print(f"  gold acc:  ort-int8 {hit_ort/n:.4f}  vs  torch-server {hit_ref/n:.4f}")
    print(f"  top1 agreement int8 vs server: {agree/n:.4f}   max|Δsoftmax| = {dmax:.5f}")
    print(f"  latency (2 threads): p50 {lat[len(lat)//2]:.1f} ms · p95 {lat[int(len(lat)*.95)]:.1f} ms")
    sz = int8_path.stat().st_size / 1e6
    print(f"  web/ bundle: encoder_int8.onnx {sz:.1f} MB + head.bin {(WEB/'head.bin').stat().st_size/1e3:.0f} KB"
          f" + head.json + tokenizer.json {(WEB/'tokenizer.json').stat().st_size/1e3:.0f} KB"
          f" | web/parity_test.mjs: node ile kütüphane E2E (ref için: dump web/_ref_ids.json)")


if __name__ == "__main__":
    main()
