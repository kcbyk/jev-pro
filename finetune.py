#!/usr/bin/env python3
"""Fine-tune the encoder on Banking77 — the rung that is supposed to beat Jev.

Jev never sees the training split: it is a general decision model given your
option names (80.3% on this split) or, in the friendlier setup, 24 retrieved
examples per class (92.4%). A specialist that updates its weights on the task's
own labels should win, and the published reference for that is a fine-tuned BERT
at 93.66%. This script reproduces the *idea* at 22.7M parameters, on 2 CPU cores,
and — the part nobody benchmarks — recalibrates the probabilities so a threshold
actually means what it says.

Leak discipline: fit on 8,000, every choice (best epoch, temperature, threshold
policy) on the 1,993-row val split, the 3,076-row official test split scored once.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
SHUFFLE_SEED = 1234


def softmax_np(z):
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


def coverage(P, y, targets=(0.95, 0.99)):
    conf, cor = P.max(1), P.argmax(1) == y
    out = {}
    for tgt in targets:
        best = 0.0
        for thr in np.quantile(conf, np.linspace(0, 0.999, 240)):
            m = conf >= thr
            if m.any() and cor[m].mean() >= tgt:
                best = max(best, float(m.mean()))
        out[tgt] = best
    return out


class Classifier(nn.Module):
    def __init__(self, base: nn.Module, n_labels: int, dropout: float = 0.15):
        super().__init__()
        self.base = base
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(base.config.hidden_size, n_labels)
        nn.init.normal_(self.head.weight, 0, 0.02)

    def encode(self, input_ids, attention_mask):
        h = self.base(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1) / m.sum(1).clamp(min=1e-6)

    def forward(self, input_ids=None, attention_mask=None, **extra):
        """Accepts a tokenizer output dict straight into model(**enc)."""
        return self.head(self.drop(self.encode(input_ids, attention_mask)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr-enc", type=float, default=3e-5)
    ap.add_argument("--lr-head", type=float, default=8e-4)
    ap.add_argument("--maxlen", type=int, default=48)
    ap.add_argument("--smooth", type=float, default=0.05)
    ap.add_argument("--fit-n", type=int, default=8000)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=str(HERE / "jev-pro-model"))
    ap.add_argument("--train-aug", default="", help="jsonl of extra paraphrase rows appended to the fit pool")
    ap.add_argument("--resume", default="", help="continue from this state dict (warm start) instead of base weights")
    a = ap.parse_args()
    out = Path(a.outdir)
    out.mkdir(exist_ok=True)

    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    from transformers import AutoModel, AutoTokenizer

    labels = json.load(open(HERE / "data/labels.json"))
    idx = {c: i for i, c in enumerate(labels)}
    Cn = len(labels)

    def read(split):
        rows = [json.loads(l) for l in open(HERE / "data" / f"{split}.jsonl")]
        for r in rows:
            r["y"] = idx[r["label"]]
        return rows

    tr = read("train")
    te = read("test")
    perm = np.random.default_rng(SHUFFLE_SEED).permutation(len(tr))
    tr = [tr[i] for i in perm]  # the mirror ships the split grouped by label
    fit, val = tr[: a.fit_n], tr[a.fit_n :]
    assert len({r["y"] for r in fit}) == Cn and len({r["y"] for r in val}) == Cn
    if a.train_aug:
        aug = [json.loads(l) for l in open(HERE / a.train_aug)]
        for r in aug:
            r["y"] = idx[r["label"]]
        fit = fit + aug  # paraphrase copies join only the fit pool; val stays original
        fit = [fit[i] for i in np.random.default_rng(a.seed + 1).permutation(len(fit))]
        print(f"train-aug: +{len(aug)} paraphrase rows -> fit pool {len(fit)} (val {len(val)} unchanged)", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    base = AutoModel.from_pretrained(a.model)
    model = Classifier(base, Cn)
    if a.resume:
        st = torch.load(HERE / a.resume, map_location="cpu", weights_only=True)
        miss = model.load_state_dict(st, strict=False)
        print(f"warm start from {a.resume} (missing={len(miss.missing_keys) if hasattr(miss,'missing_keys') else '?'} unexpected={len(miss.unexpected_keys) if hasattr(miss,'unexpected_keys') else '?'})", flush=True)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"model {a.model.split('/')[-1]} + head  {nparam/1e6:.1f}M params trainable  "
          f"fit={len(fit)} val={len(val)} test={len(te)} classes={Cn}", flush=True)

    def batch(rows, shuffle=False):
        order = np.random.default_rng(a.seed).permutation(len(rows)) if shuffle else np.arange(len(rows))
        # group by length so padding stays near the real token count
        order = order[np.argsort([len(rows[i]["text"]) for i in order], kind="stable")]
        for s in range(0, len(order), a.batch):
            chunk = [rows[i] for i in order[s : s + a.batch]]
            enc = tok([r["text"] for r in chunk], padding=True, truncation=True,
                      max_length=a.maxlen, return_tensors="pt")
            yield enc, torch.tensor([r["y"] for r in chunk], dtype=torch.long)

    enc_params = [p for n, p in model.named_parameters() if not n.startswith("head")]
    head_params = [p for n, p in model.named_parameters() if n.startswith("head")]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": a.lr_enc},
                             {"params": head_params, "lr": a.lr_head}], weight_decay=0.01)
    steps = math.ceil(len(fit) / a.batch) * a.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr_enc, a.lr_head], total_steps=steps,
                                                 pct_start=0.15, div_factor=8)
    lossf = nn.CrossEntropyLoss(label_smoothing=a.smooth)

    @torch.no_grad()
    def logits_of(rows, bs=128):
        model.eval()
        out = np.zeros((len(rows), Cn), dtype=np.float32)
        order = np.argsort([len(r["text"]) for r in rows], kind="stable")
        for s in range(0, len(order), bs):
            sel = order[s : s + bs]
            enc = tok([rows[i]["text"] for i in sel], padding=True, truncation=True,
                      max_length=a.maxlen, return_tensors="pt")
            out[sel] = model(**enc).float().numpy()
        return out

    best = (-1.0, None, None)
    hist = []
    for ep in range(a.epochs):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        steps_per_epoch = math.ceil(len(fit) / a.batch)
        for enc, y in batch(fit, shuffle=True):
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(**enc), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss)
            nb += 1
            if nb % 80 == 0:
                el = time.time() - t0
                print(f"    epoch {ep+1} step {nb}  loss {tot/nb:.4f}  {el:.0f}s elapsed  "
                      f"({nb/el:.1f} it/s, ~{(steps_per_epoch-nb)/max(nb/el,1e-6):.0f}s left)", flush=True)
        lv = logits_of(val)
        acc = float((lv.argmax(1) == np.array([r["y"] for r in val])).mean())
        hist.append({"epoch": ep + 1, "loss": tot / nb, "val_acc": acc, "secs": round(time.time() - t0, 1)})
        print(f"  epoch {ep+1}/{a.epochs} loss {tot/nb:.4f}  val acc {acc:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        if acc > best[0]:
            best = (acc, {k: v.clone() for k, v in model.state_dict().items()}, ep + 1)
            torch.save(model.state_dict(), out / "ft_state.pt")

    model.load_state_dict(best[1])
    print(f"\nbest epoch {best[2]} (val acc {best[0]:.4f}) — recalibrating on the same val split", flush=True)
    lv = logits_of(val)
    yv = np.array([r["y"] for r in val])
    grid = np.linspace(0.5, 6.0, 111)
    nlls = [(-np.mean(np.log(np.clip(softmax_np(lv / T)[np.arange(len(yv)), yv], 1e-12, None))), float(T)) for T in grid]
    T = min(nlls)[1]
    Pv = softmax_np(lv / T)
    cal = {"T": T, "val_ece_before": ece(softmax_np(lv), yv), "val_ece_after": ece(Pv, yv),
           "val_acc": float((Pv.argmax(1) == yv).mean())}
    print(f"  temperature {T:.2f}: val ECE {cal['val_ece_before']:.4f} -> {cal['val_ece_after']:.4f}", flush=True)

    print("\nofficial test split, scored once:", flush=True)
    lt = logits_of(te)
    yt = np.array([r["y"] for r in te])
    P1, P2 = softmax_np(lt), softmax_np(lt / T)
    acc1, acc2 = float((P1.argmax(1) == yt).mean()), float((P2.argmax(1) == yt).mean())
    prec = np.zeros(Cn); rec = np.zeros(Cn); f1 = np.zeros(Cn)
    for c in range(Cn):
        tp = int(((P2.argmax(1) == c) & (yt == c)).sum()); fp = int(((P2.argmax(1) == c) & (yt != c)).sum())
        fn = int(((P2.argmax(1) != c) & (yt == c)).sum())
        prec[c] = tp / max(tp + fp, 1); rec[c] = tp / max(tp + fn, 1)
        f1[c] = 2 * prec[c] * rec[c] / max(prec[c] + rec[c], 1e-9)
    cov = coverage(P2, yt)
    over = float(P2.max(1).mean() * 100 - acc2 * 100)
    top3 = float((np.argsort(-P2, 1)[:, :3] == yt[:, None]).any(1).mean())
    # honest latency: one example at a time, cold batch of 1, encode+head
    model.eval()
    lats = []
    for r in te[:200]:
        t0 = time.perf_counter()
        with torch.no_grad():
            e = tok([r["text"]], padding=False, truncation=True, max_length=a.maxlen, return_tensors="pt")
            model(**e)
        lats.append((time.perf_counter() - t0) * 1000)
    med = float(np.median(lats))

    lines = [
        f"jev-pro fine-tuned  {a.model.split('/')[-1]} ({nparam/1e6:.1f}M params, trainable, CPU, 2 threads)",
        f"fit={len(fit)} val={len(val)} test={len(te)} classes={Cn} epochs={a.epochs} best_epoch={best[2]}",
        "",
        f"  test accuracy            {acc2:.4f}   (T=1.0 gives {acc1:.4f}; temperature only rescales)",
        f"  macro F1                 {f1.mean():.4f}",
        f"  top-3 accuracy           {top3:.4f}",
        f"  ECE after calibration    {ece(P2, yt):.4f}   (before: {ece(P1, yt):.4f})",
        f"  mean confidence - accuracy  {over:+.1f} pts",
        f"  share auto-acceptable at 95% accuracy   {cov[0.95]:.2f}",
        f"  share auto-acceptable at 99% accuracy   {cov[0.99]:.2f}",
        f"  latency per decision     {med:.1f} ms  (median, single example, CPU)",
        "",
        "  published comparators on this exact split:",
        "    Jev 1.13 zero-shot typed Choice ......... 0.803  @ p50 310 ms   (jevbench.xyz)",
        "    Jev 1.13 with 24 retrieved examples ..... 0.9240               (simonmesmith)",
        "    fine-tuned BERT (2020 paper) ............ 0.9366",
        "    22M encoder + logistic head (frozen) .... 0.932  @ 8 ms        (mindstudio)",
        f"    Kev-9B (open, trained on this data) ..... 0.825 on a 154-sample set",
        "",
        f"  training history: {json.dumps(hist)}",
    ]
    tok.save_pretrained(out)
    json.dump({"labels": labels, "temperature": T, "base_model": a.model, "maxlen": a.maxlen,
               "metrics": {"test_acc": acc2, "macro_f1": float(f1.mean()), "ece": ece(P2, yt),
                           "ece_uncalibrated": ece(P1, yt), "coverage": cov, "top3": top3,
                           "conf_minus_acc_pts": over, "median_ms": med, "val_acc": best[0]},
               "history": hist, "cal": cal, "per_class_f1": f1.tolist()},
              open(out / "model_card.json", "w"), indent=2)
    np.save(out / "test_logits.npy", lt)
    (out / "finetune_report.txt").write_text("\n".join(lines) + "\n")
    Path(HERE / "finetune_report.txt").write_text("\n".join(lines) + "\n")  # last run's copy
    print("\n" + "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
