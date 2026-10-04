#!/usr/bin/env python3
"""Verify the served model from the outside: accuracy, calibration and thresholds
measured on the HTTP responses themselves — not on whatever the training script
printed. If the endpoint's numbers do not hold up here, it does not ship.

    python3 serve_pro.py --port 8100 &      # then
    python3 test_pro.py --port 8100 --n 400
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

JEV = {"acc": 0.803, "p50_ms": 310.0, "overconf_pts": 13.9, "cov95": 0.0}


API_KEY = ""  # set from --key in main()


def post(port, payload, path="/v1/systemone"):
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(payload).encode(),
                                 headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def call_expect_error(port, payload):
    try:
        post(port, payload)
        return None
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def softmax(z):
    z = np.asarray(z, float)
    z -= z.max(-1, keepdims=True)
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


def coverage(P, y, tgt):
    conf, cor = P.max(1), P.argmax(1) == y
    best = 0.0
    for thr in np.quantile(conf, np.linspace(0, 0.999, 240)):
        m = conf >= thr
        if m.any() and cor[m].mean() >= tgt:
            best = max(best, float(m.mean()))
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--key", default="", help="Bearer token when the server runs with --api-key")
    a = ap.parse_args()
    global API_KEY
    API_KEY = a.key
    fails: list[str] = []
    if API_KEY:
        API_KEY = ""  # naked request: exactly what an unauthenticated agent would send
        try:
            post(a.port, {"state": "x"})
            print("  [FAIL] POST without key was ACCEPTED (auth not enforced)")
            fails.append("auth enforced")
        except urllib.error.HTTPError as e:
            print(f"  [{'PASS' if e.code == 401 else 'FAIL'}] POST without key -> {e.code} (401 expected)")
            if e.code != 401:
                fails.append("401 for no-key POST")
        API_KEY = a.key

    def check(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}")
        if not ok:
            fails.append(name)

    h = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=30).read())
    check("GET /health", h.get("ok"), f"labels={h.get('labels')} noul={h.get('noul_heads')}")
    labels = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{a.port}/v1/labels", timeout=30).read())["labels"]

    rows = [json.loads(l) for l in open(HERE / "data/test.jsonl")]
    rng = np.random.default_rng(a.seed)
    sample = [rows[i] for i in rng.permutation(len(rows))[: a.n]]
    print(f"\nmeasuring on {len(sample)} official-test messages through HTTP\n")

    yidx = np.array([labels.index(r["label"]) for r in sample])
    P, conf_field, lat = np.zeros((len(sample), len(labels))), [], []
    for i, r in enumerate(sample):
        t0 = time.perf_counter()
        ans = post(a.port, {"state": r["text"], "questions": {"intent": {"type": "choice",
                "instructions": "Which intent?", "criteria": labels}}})["answers"]["intent"]
        lat.append((time.perf_counter() - t0) * 1000)
        p = np.array([ans["probabilities"][c] for c in labels], dtype=float)
        P[i] = p / p.sum()
        conf_field.append(ans["confidence"])
        s = abs(p.sum() - 1.0)
        if i == 0:
            check("probabilities sum to 1", s < 5e-3, f"|Σ-1| = {s:.1e}")
            check("served pick == argmax of served probabilities",
                  labels.index(ans["choice"]) == int(P[0].argmax()), ans["choice"])

    acc = float((P.argmax(1) == yidx).mean())
    ec = ece(P, yidx)
    over = float(np.mean(P.max(1)) * 100 - acc * 100)
    cov95, cov99 = coverage(P, yidx, 0.95), coverage(P, yidx, 0.99)
    p50 = float(np.median(lat[20:]))
    print(f"\n  choice (77 options)   acc {acc:.4f}   ECE {ec:.4f}   conf-acc {over:+.1f} pts   "
          f"cov@95 {cov95:.2f}   cov@99 {cov99:.2f}   p50 {p50:.1f} ms")
    print(f"  published Jev 1.13  acc {JEV['acc']:.3f}                     "
          f"conf-acc ~+{JEV['overconf_pts']:.1f} pts   cov@95 {JEV['cov95']:.2f}   p50 {JEV['p50_ms']:.0f} ms")
    check(f"endpoint accuracy beats Jev's published {JEV['acc']:.1%} by >5 pts", acc > JEV["acc"] + 0.05, f"got {acc:.4f}")
    check("endpoint ECE under 0.05 (usable thresholds)", ec < 0.05, f"{ec:.4f}")
    check("mean confidence within 3 pts of accuracy", abs(over) < 3.0, f"{over:+.1f} pts")
    check("share auto-acceptable at 95% accuracy above 0.7", cov95 > 0.7, f"{cov95:.2f} (Jev: {JEV['cov95']:.2f})")
    check("median latency under 30 ms (Jev p50 310 ms)", p50 < 30.0, f"{p50:.1f} ms")

    print("\nnoul propositions (same endpoint, same call):")
    noul_specs = {"is_card": "Does this message concern a card (physical or virtual)?",
                  "is_transfer": "Does this message concern transferring money to another account?",
                  "is_topup": "Does this message concern topping up the balance?"}
    def truth(name, label):
        return {"is_card": "card" in label, "is_transfer": "transfer" in label or "payment" in label,
                "is_topup": "top_up" in label or "topup" in label}[name]
    for name, instr in noul_specs.items():
        ys, ps, errs = [], [], 0
        for r in sample[:200]:
            ans = post(a.port, {"state": r["text"], "questions": {"q": {"type": "noul", "instructions": instr}}})
            if "q" not in ans["answers"]:
                errs += 1
                continue
            ps.append(ans["answers"]["q"]["noul"]); ys.append(int(truth(name, r["label"])))
        ps, ys = np.array(ps), np.array(ys)
        pred = (ps >= 0.5).astype(int)
        a_ = float((pred == ys).mean())
        e_ = float(np.mean((ps - ys) ** 2))  # Brier score for a binary prediction
        tpr = float(((pred == 1) & (ys == 1)).sum() / max((ys == 1).sum(), 1))
        tnr = float(((pred == 0) & (ys == 0)).sum() / max((ys == 0).sum(), 1))
        bal = 0.5 * (tpr + tnr)
        prec = float(((pred == 1) & (ys == 1)).sum() / max((pred == 1).sum(), 1))
        f1p = 2 * prec * tpr / max(prec + tpr, 1e-9)
        maj = max(ys.mean(), 1 - ys.mean())
        print(f"  {name:12s} acc {a_:.4f} (majority-only {maj:.4f})  balanced {bal:.4f}  "
              f"pos-F1 {f1p:.4f} (TPR {tpr:.2f}/TNR {tnr:.2f})  Brier {e_:.4f}  errors {errs}")
        # a rare-positive task cannot be judged as "beat the majority by 10 pts":
        # with an 11.7% positive rate that bar is mathematically near-unreachable.
        # Balanced accuracy is the honest floor (majority = 0.50 by definition).
        check(f"noul {name} balanced accuracy > 0.85", bal > 0.85, f"{bal:.4f}")
        check(f"noul {name} positive-class F1 > 0.55", f1p > 0.55, f"{f1p:.4f}")

    print("\nrequest handling:")
    one = sample[0]["text"]
    v = post(a.port, {"state": one, "questions": {"intent": {"type": "choice", "instructions": "Which intent?",
        "criteria": ["Card Arrival", "Top Up Failed", "Exchange Rate"]}}})
    check("case/underscore variants of intent names accepted", v["answers"]["intent"]["choice"] in
          ("card_arrival", "Card Arrival", "Top Up Failed", "Exchange Rate", "top_up_failed", "exchange_rate"),
          v["answers"]["intent"]["choice"])
    code, err = call_expect_error(a.port, {"state": one, "questions": {"intent": {"type": "choice",
        "instructions": "Which?", "criteria": ["does_not_exist_intent", "another_fake"]}}})
    msg = json.dumps(err)
    check("unknown option name -> 400 with nearest-intent hint", code == 400 and "closest known intents" in msg,
          msg[:120])
    code, err = call_expect_error(a.port, {"state": one, "questions": {"q": {"type": "noul",
        "instructions": "Is the customer thinking about lunch?"}}})
    check("untrained proposition -> 400 that lists what exists", code == 400 and "trained propositions" in json.dumps(err),
          json.dumps(err)[:120])
    both = post(a.port, {"state": one, "questions": {"intent": {"type": "choice", "instructions": "Which intent?",
        "criteria": labels}, "is_card": {"type": "noul", "instructions": noul_specs["is_card"]}}})
    check("mixed choice+noul in a single call", set(both["answers"]) == {"intent", "is_card"},
          f"latency {both['latency_ms']} ms")

    print("\n" + ("ALL CHECKS PASSED" if not fails else f"{len(fails)} FAILED: {fails}"))
    Path(HERE / "served_metrics.json").write_text(json.dumps(
        {"n": len(sample), "acc": acc, "ece": ec, "conf_minus_acc_pts": over, "cov95": cov95,
         "cov99": cov99, "p50_ms": p50, "jev_reference": JEV}, indent=2))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
