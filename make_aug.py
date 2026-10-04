#!/usr/bin/env python3
"""Rule-based paraphrase augmentation for the Banking77 train split.

No external model, no labels touched: two deterministic variants per row.
The encoder never saw these phrasings, so re-training heads (or the encoder)
on train+aug acts like mild distribution smoothing toward real chat.

Every emitted row keeps `base = <index of the original train row>` so the fold
logic downstream can guarantee an augmented copy never trains against the fold
that holds its parent row.
"""
from __future__ import annotations

import argparse
import json
import random
import re

import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent

LEADS = [
    ("how do i ", "how can i "), ("how can i ", "how do i "),
    ("i want to ", "i would like to "), ("i would like to ", "i want to "),
    ("i need to ", "i have to "), ("what do i do if ", "what can i do if "),
    ("what can i do if ", "what do i do if "), ("why is ", "why is it that "),
    ("i can't ", "i am unable to "), ("i can ", "i am able to "),
    ("my ", "there is an issue with my "), ("can i ", "is it possible to "),
]
TAILS = ["", " please", " today", ", please help", " asap"]
EXPAND = {"i'm ": "i am ", "can't ": "cannot ", "won't ": "will not ",
          "doesn't ": "does not ", "didn't ": "did not ", "it's ": "it is ",
          "i've ": "i have ", "what's ": "what is ", "isn't ": "is not ",
          "that's ": "that is ", "couldn't ": "could not ", "haven't ": "have not ",
          "don't ": "do not "}


def variant1(t: str, rng: random.Random) -> str:
    s = " " + t.strip().lower() + " "
    for k, v in EXPAND.items():
        if k in s and rng.random() < 0.6:
            s = s.replace(k, v, 1)
            break
    for k, v in LEADS:
        if s.strip().startswith(k) and rng.random() < 0.7:
            s = " " + v + s.strip()[len(k):] + " "
            break
    s = re.sub(r"\s+", " ", s).strip()
    if rng.random() < 0.25:
        s = "i was hoping you could help. " + s
    s = s[0].upper() + s[1:]
    return s


def variant2(t: str, rng: random.Random) -> str:
    s = t.strip()
    q = s.endswith("?")
    s = s.rstrip("?").strip()
    tail = rng.choice(TAILS)
    s = s + (tail + "?" if q or tail else "")
    if rng.random() < 0.3:
        s = "hello, " + s[0].lower() + s[1:]
    if rng.random() < 0.3:
        s = s + " can you help me with that?"
    return re.sub(r"\s+", " ", s).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-row", type=int, default=2, choices=(1, 2))
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--fit-n", type=int, default=0, help="restrict parents to the first N shuffled train rows (0 = all)")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rows = [json.loads(l) for l in open(HERE / "data/train.jsonl")]
    if a.fit_n:  # keep only the rows that a finetune run would put in the fit pool
        perm = np.random.default_rng(1234).permutation(len(rows))
        keep = set(int(i) for i in perm[: a.fit_n])
        rows = [r for i, r in enumerate(rows) if i in keep]
        base_offset = list(keep)  # base indices refer to original order
    else:
        base_offset = list(range(len(rows)))
    out = []
    for i, r in enumerate(rows):
        b = base_offset[i]
        v1 = variant1(r["text"], rng)
        out.append({"text": v1, "label": r["label"], "base": b})
        if a.per_row == 2:
            v2 = variant2(r["text"], rng)
            out.append({"text": v2, "label": r["label"], "base": b})
    with open(HERE / "data/train_aug.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(out)} augmented rows for {len(rows)} originals -> data/train_aug.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
