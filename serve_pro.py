#!/usr/bin/env python3
"""jev-pro as a service — the Jev request shape, answered by a local 22.7M model.

    python3 serve_pro.py --port 8100

Supports:
  * choice over Banking77 intent names (exact name, or underscore/case variants)
  * noul for the propositions the model has heads for
  * a scored "nearest known intent" hint when you send a name it does not know

Unknown option names are not silently guessed: you get a 400 that lists the
closest known intents. Silent fallback is how a routing model becomes a
production incident.
"""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "jev-pro-model"


def softmax(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


class Engine:
    def __init__(self, model_dir: Path = MODEL_DIR, state: Path | None = None, device_threads: int = 2,
                 int8: bool = False):
        torch.set_num_threads(device_threads)
        from transformers import AutoModel, AutoTokenizer

        card = json.load(open(model_dir / "model_card.json"))
        nz = np.load(model_dir / "choice_head.npz", allow_pickle=True)
        self.labels = [str(x) for x in nz["labels"]]
        self.W, self.bias = nz["W"], nz["bias"]
        self.T = float(nz["temperature"])
        self.noul = json.load(open(model_dir / "noul_heads.json"))
        self.lookup = {l.lower().replace("_", " "): i for i, l in enumerate(self.labels)}
        self.noul_lookup = {v["instructions"].lower().rstrip("?").strip(): k for k, v in self.noul.items()}
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        if (model_dir / "config.json").exists():
            # offline-safe: architecture from local config, weights always overwritten
            # by the fine-tuned state below — the hub is never a startup dependency
            from transformers import AutoConfig
            self.base = AutoModel.from_config(AutoConfig.from_pretrained(model_dir)).eval()
        else:
            self.base = AutoModel.from_pretrained(card["base_model"]).eval()
        st = torch.load(state or (HERE / "ft_state.pt"), map_location="cpu", weights_only=True)
        self.base.load_state_dict({k[len("base."):]: (v.float() if v.is_floating_point() else v)
                                   for k, v in st.items() if k.startswith("base.")}, strict=False)
        if int8:
            # per-tensor dynamic quant on every Linear; head/noul layers stay fp32 numpy on purpose
            self.base = torch.quantization.quantize_dynamic(self.base, {torch.nn.Linear}, dtype=torch.qint8)

    @torch.inference_mode()
    def embed(self, texts):
        e = self.tok(texts, padding=True, truncation=True, max_length=48, return_tensors="pt")
        h = self.base(**e).last_hidden_state
        m = e["attention_mask"].unsqueeze(-1).to(h.dtype)
        return ((h * m).sum(1) / m.sum(1).clamp(min=1e-6)).numpy()

    def logits(self, texts):
        return self.embed(texts) @ self.W.T + self.bias

    def choice(self, text: str, wanted: list[str]) -> dict:
        idx, missing = [], []
        for name in wanted:
            key = name.lower().replace("_", " ").strip()
            if key in self.lookup:
                idx.append(self.lookup[key])
            else:
                missing.append(name)
        if missing:
            near = self.nearest(missing[0], k=6)
            raise KeyError(f"unknown option(s) {missing}; closest known intents: {near}")
        lg = self.logits([text])[0]
        P = softmax(lg[idx] / self.T)
        pick = int(P.argmax())
        srt = np.sort(P)[::-1]
        conf = float(min(0.995, max(0.005, P[pick] - 0.5 * (srt[1] if len(srt) > 1 else 0.0))))
        return {
            "type": "choice",
            "choice": wanted[pick],
            "probabilities": {n: round(float(p), 4) for n, p in zip(wanted, P)},
            "confidence": round(conf, 3),
            "expected_value_index": pick,
            "calibration": f"temperature {self.T:.2f}, fitted on 1,993 held-out examples",
        }

    def noul_answer(self, text: str, question: str) -> dict:
        key = question.lower().rstrip("?").strip()
        name = self.noul_lookup.get(key) or (key if key in self.noul else None)  # başlık metni VEYA head adı (is_card…)
        if name is None:
            raise KeyError(f"no noul head for {question!r}; trained propositions: "
                           + ", ".join(f"{k} ({v['instructions']})" for k, v in self.noul.items()))
        h = self.noul[name]
        z = float(self.embed([text])[0] @ np.asarray(h["W"]) + h["b"])
        p = 1.0 / (1.0 + np.exp(-z / h["T"]))
        return {"type": "noul", "noul": round(p, 4), "confidence": round(max(p, 1 - p), 3),
                "proposition": name, "calibration": f"temperature {h['T']:.2f}"}

    def nearest(self, text: str, k: int = 5) -> list[str]:
        v = self.embed([text])[0]
        V = self.embed(self.labels)
        s = V @ v / max(np.linalg.norm(v), 1e-9)
        return [self.labels[i] for i in np.argsort(-s)[:k]]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    engine: Engine = None  # type: ignore[assignment]
    keys: dict[str, str] | None = None   # key -> etiket; None = auth kapalı (--api-key/--keys yoksa)
    usage: dict[str, int] = {}           # etiket -> işlenmiş istek sayısı (/health'te görünür)

    def _authorized(self) -> str | None:
        """Return the key's label when authorized ("*" when auth is off), else None."""
        if not Handler.keys:
            return "*"
        got = self.headers.get("Authorization", "").strip()
        if not got.startswith("Bearer "):
            return None
        import secrets as _sec
        tok = got[7:].strip()
        for k, label in Handler.keys.items():
            if _sec.compare_digest(tok, k):
                return label
        return None

    def log_message(self, fmt, *args):  # silence default request log
        pass

    allow_origin: str = "*"   # set from --allow-origin

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", Handler.allow_origin)
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _send(self, code: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):  # noqa: N802  (browser preflight)
        self.send_response(204)
        self._cors()
        self.send_header("Access-Control-Max-Age", "600")
        self.end_headers()

    def do_GET(self):  # noqa: N802
        if self.path in ("/health", "/"):
            h = {"ok": True, "model": "jev-pro-1", "labels": len(self.engine.labels),
                 "noul_heads": list(self.engine.noul)}
            if Handler.keys:
                h["keys"] = {lbl: Handler.usage.get(lbl, 0) for lbl in Handler.keys.values()}
            self._send(200, h)
        elif self.path == "/v1/labels":
            self._send(200, {"labels": self.engine.labels})
        else:
            self._send(404, {"error": {"message": f"no route {self.path}"}})

    def do_POST(self):  # noqa: N802
        who = self._authorized()
        if who is None:
            return self._send(401, {"error": {"type": "AuthenticationError",
                                              "message": "missing or invalid API key",
                                              "details": {"hint": "send header: Authorization: Bearer <key>"}}})
        if who != "*":
            Handler.usage[who] = Handler.usage.get(who, 0) + 1
        if self.path not in ("/v1/systemone", "/v1/classifier", "/v1/evaluate"):
            return self._send(404, {"error": {"message": f"no route {self.path}"}})
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            state = req.get("state")
            text = state if isinstance(state, str) else json.dumps(state, sort_keys=True, default=str)
            t0 = time.perf_counter()
            answers, errors = {}, {}
            for qid, spec in (req.get("questions") or {}).items():
                qtype = spec.get("type")
                try:
                    if qtype == "choice":
                        crit = spec.get("criteria") or spec.get("options")  # agent-friendly alias
                        wanted = list(crit) if isinstance(crit, dict) else list(crit or self.engine.labels)
                        answers[qid] = self.engine.choice(text, wanted)
                    elif qtype == "noul":
                        answers[qid] = self.engine.noul_answer(text, spec.get("instructions", ""))
                    else:
                        raise ValueError(f"unsupported question type {qtype!r}")
                except (KeyError, ValueError) as e:
                    errors[qid] = str(e.args[0] if e.args else e)
            ms = (time.perf_counter() - t0) * 1000
            if not answers:
                return self._send(400, {"error": {"message": "no answerable question", "details": errors}})
            payload = {"model": req.get("model", "jev-pro-1"), "answers": answers,
                       "latency_ms": round(ms, 2), "usage": {"input_tokens": max(1, len(text) // 4),
                                                              "output_tokens": 0}}
            if errors:
                payload["errors"] = errors
            self._send(200, payload)
        except Exception as e:  # noqa: BLE001
            self._send(400, {"error": {"type": type(e).__name__, "message": str(e)}})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(__import__("os").environ.get("PORT", 8100)))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--model-dir", default=str(MODEL_DIR))
    ap.add_argument("--state", default=None, help="fine-tuned state_dict; defaults to ft_state.pt")
    ap.add_argument("--int8", action="store_true", help="dynamic int8 quantize the encoder (latency mode)")
    ap.add_argument("--api-key", default=None, help="require Authorization: Bearer <key> on POST routes")
    ap.add_argument("--keys", default=None, help='JSON dosyası {"<key>": "etiket", ...} — proje başına anahtar, /health kullanım sayacı')
    ap.add_argument("--allow-origin", default="*", help="CORS origin for browser callers (use your site URL in production)")
    a = ap.parse_args()
    Handler.engine = Engine(Path(a.model_dir), Path(a.state) if a.state else None, int8=a.int8)
    if a.keys:
        Handler.keys = json.loads(Path(a.keys).read_text())
    if a.api_key:
        Handler.keys = {**(Handler.keys or {}), a.api_key: "api-key"}
    Handler.allow_origin = a.allow_origin
    print(f"jev-pro-1 ready: {len(Handler.engine.labels)} intents, noul heads "
          f"{list(Handler.engine.noul)}, T={Handler.engine.T:.2f}", flush=True)
    print(f"POST http://{a.host}:{a.port}/v1/systemone"
          + (f"  [Bearer auth ON: {len(Handler.keys)} key]" if Handler.keys else "  [no auth]"), flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    raise SystemExit(main())
