// parity_test.mjs — jevpro.js uçtan uca doğrulama (Node + onnxruntime-node,
// tarayıcıdaki aynı kod yolu). Testler:
//  1) tokenizer vs HF referans ids (web/_ref_ids.json varsa; export_web.py üretir)
//  2) kütüphane batch classify vs fp32 test_logits.npy: top-1 uyumu + altın doğruluk
//  3) tekli == batchli (sayısal eşdeğerlik, ~1e-6)
// Koşum: node parity_test.mjs  (npm i onnxruntime-node gerekir — dev-only)
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.dirname(HERE);
const ort = await import('onnxruntime-node').then(m => m.default ?? m);

// --- fetcher: base dosya yolundan okur (tarayıcıda gerçek fetch) -----------
const fileFetch = (u) => {
  const p = u.startsWith('file://') ? new URL(u).pathname : u;
  return Promise.resolve({
    ok: true,
    json: async () => JSON.parse(fs.readFileSync(p, 'utf8')),
    arrayBuffer: async () => { const b = fs.readFileSync(p); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); },
  });
};
const { loadJevPro } = await import('./jevpro.js');
const jev = await loadJevPro({ base: HERE + '/', ort, fetchImpl: fileFetch });

// --- npy okuyucu (v1, '<f4') -----------------------------------------------
function readNpy(p) {
  const buf = fs.readFileSync(p);
  if (buf.slice(1, 6).toString() !== 'NUMPY') throw new Error('npy değil');
  const off = buf[6] === 1 ? 8 : 12;
  const hdrEnd = buf.indexOf(0x0a, off);
  const hdr = buf.slice(off, hdrEnd).toString();
  if (!hdr.includes("'<f4'")) throw new Error('beklenen dtype <f4: ' + hdr.slice(0, 80));
  const shape = JSON.parse('[' + hdr.match(/'shape': \(([^)]*)\),/)[1].replace(/,\s*$/, '') + ']');
  const data = new Float32Array(buf.buffer.slice(buf.byteOffset + hdrEnd + 1, buf.byteOffset + buf.length));
  return { data, shape };
}
const ref = readNpy(path.join(REPO, 'jev-pro-model-r3w2/test_logits.npy'));
const [N, C] = ref.shape;
function softmaxRow(i) {
  const o = i * C; let m = -1e30;
  for (let j = 0; j < C; j++) m = Math.max(m, ref.data[o + j]);
  let s = 0; const e = new Float64Array(C);
  for (let j = 0; j < C; j++) { e[j] = Math.exp(ref.data[o + j] - m); s += e[j]; }
  return e.map(v => v / s);
}
const rows = fs.readFileSync(path.join(REPO, 'data/test.jsonl'), 'utf8').trim().split('\n').map(l => JSON.parse(l));
const labels = jev.labels;
const M = Math.min(400, N);
const texts = rows.slice(0, M).map(r => r.text);
const gold = rows.slice(0, M).map(r => labels.indexOf(r.label));

let fails = 0;
const ok = (cond, name, extra = '') => { console.log(`${cond ? 'PASS' : 'FAIL'} · ${name}${extra ? ' · ' + extra : ''}`); if (!cond) fails++; };

// 1) tokenizer vs HF
if (fs.existsSync(HERE + '/_ref_ids.json')) {
  const refs = JSON.parse(fs.readFileSync(HERE + '/_ref_ids.json', 'utf8'));
  let bad = 0;
  for (let i = 0; i < refs.length; i++) {
    const mine = Array.from(jev.tokenizer.encode(texts[i]).input_ids, Number);
    if (JSON.stringify(mine) !== JSON.stringify(refs[i])) bad++;
  }
  ok(bad === 0, `tokenizer == HF (${refs.length - bad}/${refs.length})`);
} else console.log('SKIP · tokenizer referansı yok (_ref_ids.json)');

// 2) batch classify vs fp32 referans
const t0 = Date.now();
const out = await jev.classifyBatch(texts);
const ms = (Date.now() - t0) / M;
let agree = 0, acc = 0, accRef = 0, dpMax = 0;
for (let i = 0; i < M; i++) {
  const p = softmaxRow(i);
  let bi = 0; for (let j = 1; j < C; j++) if (p[j] > p[bi]) bi = j;
  agree += out[i].choice.pick === labels[bi];
  acc += out[i].choice.pick === labels[gold[i]];
  accRef += labels[bi] === labels[gold[i]];
  for (let j = 0; j < C; j++) { const d = Math.abs((out[i].choice.probs[labels[j]] ?? 0) - p[j]); if (d > dpMax) dpMax = d; }
}
ok(agree / M > 0.97, `top-1 uyum ${agree}/${M} (${(100 * agree / M).toFixed(1)}%)`);
ok(acc / M >= accRef / M - 0.01, `altın doğruluk int8 ${(100 * acc / M).toFixed(2)}% ≥ fp32 ${(100 * accRef / M).toFixed(2)}% − 1p`);
console.log(`     batch gecikme ${(ms).toFixed(2)} ms/metin (ort-node, ${M} metin) · Δmedyan dışı en büyük Δp ${dpMax.toFixed(4)} (quantize farkı, eşit-vaka)`);

// 3) tekli == batchli
let singleBad = 0, sMax = 0;
for (let i = 0; i < 60; i++) {
  const one = await jev.classify(texts[i]);
  if (one.pick !== out[i].choice.pick) singleBad++;
  sMax = Math.max(sMax, Math.abs(one.confidence - out[i].choice.confidence));
}
ok(singleBad === 0 && sMax < 1e-6, `tekli==batchli (${60 - singleBad}/60, Δconf ${sMax.toExponential(1)})`);

console.log(fails ? `\n${fails} FAIL` : '\nTÜM YEŞİL');
process.exit(fails ? 1 : 0);
