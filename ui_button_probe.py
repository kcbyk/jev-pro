#!/usr/bin/env python3
"""UI button probe — ISKELET (bu sandbox'ta KOŞURULMADI; browser kurulu değil).

Amaç: "arayüzde çalışmayan butonları bulma" işi jev-pro'nun işi DEĞİL; bu onun
Playwright tarafıdır. AI ajanına bunu scaffold olarak ver: kendi app URL'inle
koşsun, raporu JSON olarak okusun.

Kurulum (ajanının çalıştığı makinede):
    pip install playwright && playwright install chromium
Koşum:
    python3 ui_button_probe.py --url https://localhost:3000 --out probe.json
Mantık: her buton/link/submit için ->
  click öncesi/sonrası DOM imzası, URL değişimi, console/pageerror, başarısız
  istek (>=400 ya da net::ERR) toplar.
  "ölü": hata YOK ve DOM/URL hiç değişmedi (tıklama hiçbir şey yapmıyor olabilir —
  sessiz butonlar kasti de olabilir, rapor öneridir, mahkeme kararı değil)
  "bozuk": exception/console-error/failed-request varsa kesin bulgu.
Opsiyonel köprü: yakalanan hata metinlerini jev-pro'ya göndermek istersen
probe_to_intents.py benzeri bir adım ekle (state=error text -> intent/noul).
"""
from __future__ import annotations

import argparse
import hashlib
import json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", default="probe.json")
    ap.add_argument("--selector", default="button, a[href], [role=button], input[type=submit]")
    ap.add_argument("--headful", action="store_true")
    a = ap.parse_args()

    from playwright.sync_api import sync_playwright  # ImportError = kurulum adımı atlanmış

    results = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not a.headful)
        page = browser.new_page()
        errors, failed = [], []
        page.on("pageerror", lambda e: errors.append(str(e)[:300]))
        page.on("console", lambda m: errors.append(m.text[:300]) if m.type == "error" else None)
        page.on("requestfailed", lambda r: failed.append(f"{r.method} {r.url[:160]} :: {r.failure}"))
        page.on("response", lambda r: failed.append(f"{r.status} {r.url[:160]}") if r.status >= 400 else None)

        page.goto(a.url, wait_until="networkidle")
        handles = page.query_selector_all(a.selector)
        print(f"{len(handles)} tıklanabilir unsur bulundu")
        for i, el in enumerate(handles):
            label = (el.inner_text().strip() or el.get_attribute("aria-label")
                     or el.get_attribute("id") or f"[untitled #{i}]")[:80]
            errs_before, fails_before = len(errors), len(failed)
            snap0 = hashlib.md5((page.content() + page.url).encode()).hexdigest()
            try:
                el.click(timeout=4000)
                page.wait_for_timeout(400)  # async UI güncellemelerine nefes payı
            except Exception as e:
                results.append({"label": label, "status": "error", "detail": str(e)[:200]})
                page.reload(wait_until="networkidle")
                continue
            snap1 = hashlib.md5((page.content() + page.url).encode()).hexdigest()
            new_errs = errors[errs_before:]
            new_fails = failed[fails_before:]
            if new_errs or new_fails:
                status = "broken"
            elif snap0 == snap1:
                status = "dead?"        # hiçbir gözle görülür etki yok — elle bakılmalı
            else:
                status = "ok"
            results.append({"label": label, "status": status,
                            "errors": new_errs[:3], "network": new_fails[:3]})
        browser.close()

    summary = {"url": a.url, "probed": len(results),
               "broken": [r for r in results if r["status"] == "broken"],
               "dead": [r for r in results if r["status"] == "dead?"]}
    json.dump({"summary": summary, "results": results}, open(a.out, "w"), indent=2)
    print(f"broken={len(summary['broken'])} dead={len(summary['dead'])} -> {a.out}")
    return 0 if not summary["broken"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
