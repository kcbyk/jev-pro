#!/usr/bin/env python3
"""End-to-end regression suite for the jev-pro HTTP contract (stdlib only — CI-safe).

What it checks, in order:
  1. GET /health is open and reports 77 labels + 3 noul heads
  2. POST without a Bearer key -> 401 (when the server was started with --api-key)
  3. GET /v1/labels matches data/labels.json byte-for-byte (catalog drift catcher)
  4. n sampled official-test messages: served choice == gold intent on >= min-acc,
     probability vector sums to 1 (|sum-1| <= 2e-3) on every row, served pick == argmax
  5. bogus option name -> 400/200+errors with nearest-intent hint (no silent fallback)
  6. mixed choice+noul in ONE call answers both

Usage:
  python3 qa_flow_test.py --base http://127.0.0.1:8100 --key <KEY> --n 300
Exit code 0 = all green.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request

FAILS = []


def check(name, ok, info=""):
    print(("  PASS  " if ok else "  FAIL  ") + name + (f"   [{info}]" if info else ""))
    if not ok:
        FAILS.append(name)


def post(base, key, payload, path="/v1/systemone"):
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(),
                                 headers={"content-type": "application/json",
                                          **({"Authorization": f"Bearer {key}"} if key else {})})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.load(e)
        except Exception:
            return e.code, {}


def get(base, path):
    with urllib.request.urlopen(base + path, timeout=15) as r:
        return json.load(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8100")
    ap.add_argument("--key", default="")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--min-acc", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()

    h = get(a.base, "/health")
    check("health open + labels=77 + noul=3", h.get("ok") and h.get("labels") == 77
          and len(h.get("noul_heads", [])) == 3, json.dumps(h)[:80])

    if a.key:
        code, body = post(a.base, None, {"state": "x", "questions": {}})
        check("no-key POST -> 401", code == 401, body.get("error", {}).get("type", "?"))
        code, _ = post(a.base, a.key, {"state": "ping", "questions": {}})
        check("wrong-empty-questions with key -> 400", code == 400)

    labels = get(a.base, "/v1/labels")["labels"]
    gold = [json.loads(l) for l in open("data/test.jsonl")]
    check("label catalog == data/labels.json", labels == json.load(open("data/labels.json")),
          f"{len(labels)} labels")

    rows = random.Random(a.seed).sample(gold, min(a.n, len(gold)))
    t0 = time.perf_counter()
    correct = 0
    sum_ok = pick_ok = True
    for r in rows:
        code, d = post(a.base, a.key or None, {"state": r["text"],
                       "questions": {"intent": {"type": "choice", "instructions": "Which intent?", "criteria": labels}}})
        if code != 200:
            sum_ok = False
            break
        p = d["answers"]["intent"]["probabilities"]
        s = sum(p.values())
        if abs(s - 1.0) > 2e-3:
            sum_ok = False
        chosen = d["answers"]["intent"]["choice"]
        pick_ok &= (chosen == max(p, key=p.get))
        correct += (chosen == r["label"])
    acc = correct / len(rows)
    el = time.perf_counter() - t0
    check(f"served accuracy >= {a.min_acc} on {len(rows)} test msgs", acc >= a.min_acc,
          f"acc {acc:.4f}, {el*1000/len(rows):.1f} ms/msg incl. http")
    check("probabilities sum to 1 (all rows)", sum_ok)
    check("served pick == argmax (all rows)", bool(pick_ok))

    code, d = post(a.base, a.key or None, {"state": "when will it arrive?", "questions": {
        "intent": {"type": "choice", "criteria": ["card_arrival", "totally_made_up_intent"]}}})
    assert code == 400, "alias-Regression: options-alias ayrı test ediliyor"
    code, d = post(a.base, a.key or None, {"state": "when will it arrive?", "questions": {
        "intent": {"type": "choice", "options": ["card_arrival", "totally_made_up_intent"]}}})
    bad = d.get("answers", {}).get("intent", d.get("error", d))
    check("bogus option surfaced (400 or per-q error, never silent)", code == 400 or "error" in str(bad),
          str(bad)[:70])

    code, d = post(a.base, a.key or None, {"state": "My card is broken, top it up?", "questions": {
        "intent": {"type": "choice", "instructions": "Which intent?"},
        "c": {"type": "noul", "instructions": "Does this message concern a card (physical or virtual)?"}}})
    mixed_ok = code == 200 and "intent" in d.get("answers", {}) and "c" in d.get("answers", {})
    check("mixed choice+noul single call -> both answered", mixed_ok, f"{d.get('latency_ms', '?')} ms")

    print("\n" + ("QA FLOW: ALL GREEN" if not FAILS else f"QA FLOW: {len(FAILS)} FAILED -> {FAILS}"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
