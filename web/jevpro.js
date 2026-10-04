// jevpro.js — jev-pro model API'si, tek dosya, sıfır build adımı.
//
// Asset'ler (hepsi `base` yolunda; site kökü varsayılan):
//   encoder_int8.onnx · tokenizer.json · head.json · head.bin
//
// Kullanım (ESM):
//   import { loadJevPro } from 'https://SİTENİZ/jevpro.js';
//   const jev = await loadJevPro();                 // window.ort varsa onu kullanır
//   const r = await jev.classify('why was my card payment declined?');
//   // r = { pick:'declined_card_payment', confidence:0.995, probs:{...77...} }
//   const s = await jev.classify(text, ['card_arrival','lost_or_stolen_card']); // subset
//   const many = await jev.classifyBatch(t100);     // tek batch koşum, oyunlar için
//   const n = await jev.noul(text);                 // {is_card, is_transfer, is_topup}
//   const emb = await jev.embed(text);              // Float32Array(384) — kendi kafan için
//
// Script-tag (build'siz oyunlar):
//   <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.min.js"></script>
//   <script type="module">
//     import { loadJevPro } from 'https://SİTENİZ/jevpro.js';
//     window.JevPro = await loadJevPro({ base: 'https://SİTENİZ/' });  // JevPro.classify(...)
//   </script>
//
// Sunucu formüllerinin birebir aynısı (serve_pro.py): mean-pool embedding → W@emb+bias,
// softmax(istenen-altkümesi / T), confidence = clip(top1 − 0.5·top2, 0.005, 0.995),
// noul = sigmoid((W@emb+b)/T). Tok: BertWordPiece, max 48 truncation — HF ile 3084/3084 birebir.
// ort'u dışarıdan verebilirsin (test/bundler): loadJevPro({ ort: require('onnxruntime-node') }).

export class BertTokenizer {
  constructor(tokJson) {
    this.vocab = tokJson.model.vocab;
    this.unk = this.vocab["[UNK]"] ?? 0;
    this.cls = this.vocab["[CLS]"];
    this.sep = this.vocab["[SEP]"];
    this.pad = tokJson.post_processor?.pad_id ?? this.vocab["[PAD]"] ?? 0;
    this.maxInputCharsPerWord = tokJson.model.max_input_chars_per_word ?? 100;
    const n = tokJson.normalizer || {};
    this.lowercase = n.lowercase !== false;
    this.stripAccents = n.strip_accents ?? this.lowercase;
    this.idsCap = 48;
  }
  clean(s) {
    let out = "";
    for (const ch of s) {
      const c = ch.codePointAt(0);
      if (c === 0 || c === 0xfffd || (c >= 0xd800 && c <= 0xdfff)) continue;
      if (c <= 0x1f && c !== 9 && c !== 10 && c !== 13) continue;
      if (c === 0x7f || (c >= 0x80 && c <= 0x9f)) continue;
      out += ch;
    }
    return out.replace(/[\t\n\r ]+/g, " ");
  }
  normalize(s) {
    if (this.lowercase) s = s.toLowerCase();
    if (this.stripAccents) s = s.normalize("NFD").replace(/\p{Mn}/gu, "").normalize("NFC");
    return this.clean(s);
  }
  isPunct(ch) {
    // exact HF BasicTokenizer rule: ASCII ranges + Unicode P-category (NOT S!)
    const cp = ch.codePointAt(0);
    if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126)) return true;
    return /\p{P}/u.test(ch);
  }
  tokenize(text) {
    let punctSpaced = "";
    for (const ch of this.normalize(text)) punctSpaced += this.isPunct(ch) ? ` ${ch} ` : ch;
    const spaced = [...punctSpaced].map(ch => {
      const c = ch.codePointAt(0);
      return (c >= 0x4e00 && c <= 0x9fff) || (c >= 0x3400 && c <= 0x4dbf) ? ` ${ch} ` : ch;
    }).join("");
    const words = spaced.split(" ").filter(Boolean);
    const out = [];
    for (const w of words) {
      if (w.length > this.maxInputCharsPerWord) { out.push(this.unk); continue; }
      const chars = [...w];
      const pieces = [];
      let i = 0, bad = false;
      while (i < chars.length) {
        let best = null;
        for (let j = chars.length; j > i; j--) {
          const sub = (i > 0 ? "##" : "") + chars.slice(i, j).join("");
          if (this.vocab[sub] !== undefined) { best = [this.vocab[sub], j]; break; }
        }
        if (best === null) { bad = true; break; }
        pieces.push(best[0]); i = best[1];
      }
      out.push(...(bad ? [this.unk] : pieces));
    }
    return out;
  }
  encode(text, maxLen = this.idsCap) {
    const pieces = this.tokenize(text).slice(0, maxLen - 2);
    const ids = [this.cls, ...pieces, this.sep];
    return { input_ids: BigInt64Array.from(ids, x => BigInt(x)),
             attention_mask: BigInt64Array.from(ids.map(() => 1n)), len: ids.length };
  }
}

// head + noul matematiği — serve_pro.py ile birebir formüller
export function makeHeads(labels, bias, T, W, noulHeads) {
  return {
    classify(emb, wanted) {
      const lg = new Float64Array(labels.length);
      for (let i = 0; i < labels.length; i++) {
        let s = bias[i], o = i * 384;
        for (let j = 0; j < 384; j++) s += W[o + j] * emb[j];
        lg[i] = s;
      }
      const a = wanted.map(i => lg[i] / T);
      const m = Math.max(...a);
      const e = a.map(x => Math.exp(x - m));
      const sum = e.reduce((p, c) => p + c, 0);
      const probs = Object.fromEntries(wanted.map((i, k) => [labels[i], e[k] / sum]));
      const pairs = wanted.map((i, k) => [labels[i], e[k] / sum]).sort((x, y) => y[1] - x[1]);
      const conf = Math.min(0.995, Math.max(0.005, pairs[0][1] - (pairs[1] ? 0.5 * pairs[1][1] : 0)));
      return { pick: pairs[0][0], confidence: conf, probs };
    },
    logits(emb) {
      const lg = new Float32Array(labels.length);
      for (let i = 0; i < labels.length; i++) {
        let s = bias[i], o = i * 384;
        for (let j = 0; j < 384; j++) s += W[o + j] * emb[j];
        lg[i] = s;
      }
      return lg;
    },
    noul(emb) {
      const o = {};
      for (const [name, h] of noulHeads) {
        let z = h.b;
        for (let j = 0; j < 384; j++) z += h.W[j] * emb[j];
        o[name] = 1 / (1 + Math.exp(-(z / h.T)));
      }
      return o;
    },
  };
}

export async function loadJevPro({ base = "", ort, fetchImpl } = {}) {
  if (!ort && typeof globalThis.ort !== "undefined") ort = globalThis.ort;
  if (!ort) throw new Error("jevpro: onnxruntime-web yok — <script src=ort.min.js> ekle ya da loadJevPro({ort}) ver");
  const F = fetchImpl ?? ((u) => fetch(u));
  const b = base ? (base.endsWith("/") ? base : base + "/") : "";
  const [tokJson, headJson, headBuf] = await Promise.all([
    F(b + "tokenizer.json").then(r => r.json()),
    F(b + "head.json").then(r => r.json()),
    F(b + "head.bin").then(r => r.arrayBuffer()),
  ]);
  const W = new Float32Array(headBuf);
  if (W.length !== headJson.labels.length * 384)
    throw new Error(`jevpro: head.bin boyutu bozuk (${W.length}; ${headJson.labels.length * 384} olmalı) — Float32Array tamlığı`);
  const heads = makeHeads(headJson.labels, headJson.bias, headJson.T, W, Object.entries(headJson.noul));
  const tok = new BertTokenizer(tokJson);
  const sess = await ort.InferenceSession.create(b + "encoder_int8.onnx", { graphOptimizationLevel: "all" });
  const allIdx = headJson.labels.map((_, i) => i);

  async function runEmbs(texts, padTo = 48) {
    const encs = texts.map(t => tok.encode(t));
    // varsayılan 48 = serve_pro'nun padding='max_length' ile BIT-BAZLI aynı girdi;
    // 'longest' daha hızlı ama int8 kuantalama sekans-uzunluğuna duyarlı → ~0.3 emb sapma
    const L = padTo === 'longest' ? Math.max(...encs.map(e => e.len)) : Math.max(48, ...encs.map(e => e.len));
    const ids = new BigInt64Array(texts.length * L), mask = new BigInt64Array(texts.length * L);
    encs.forEach((e, i) => {
      ids.set(e.input_ids, i * L);
      mask.set(e.attention_mask, i * L);   // sağdaki pad'ler 0 kalır
    });
    const dims = [texts.length, L];
    const out = await sess.run({ input_ids: new ort.Tensor("int64", ids, dims),
                                 attention_mask: new ort.Tensor("int64", mask, dims) });
    const data = out.embedding.data, E = 384;
    return texts.map((_, i) => data.subarray(i * E, (i + 1) * E));
  }
  const want = (labels) => labels && labels.length
    ? labels.map(l => headJson.labels.indexOf(l)).filter(i => i >= 0) : allIdx;

  return {
    labels: headJson.labels,
    tokenizer: tok, heads, session: sess, ort,
    embed: async (t) => (await runEmbs([t]))[0],
    // embedBatch: tek koşumda toplu — HIZLI ama satırlar arası quantize etkileşimi
    // sonucu ~0.3 kaydırabilir; yalnızca GPU/heuristik tarama için, kesin cevap için classify* kullan.
    embedBatch: (texts, padTo) => runEmbs(texts, padTo),
    classify: async (t, labels) => heads.classify((await runEmbs([t]))[0], want(labels)),
    noul: async (t) => heads.noul((await runEmbs([t]))[0]),
    // Deterministik sözleşme: her satır [1,48] olarak koşar → classify ile BIT aynı.
    // (ORT int8 per-tensor aktivasyon ölçeği: gerçek batch'te satırlar birbirini ~0.3
    // kaydırıyor — oyun/benchmark tekrarı için bu kabul edilemez.)
    async classifyBatch(texts, labels) {
      const w = want(labels), out = [];
      for (const t of texts) { const e = (await runEmbs([t]))[0];
        out.push({ emb: e, choice: heads.classify(e, w), noul: heads.noul(e) }); }
      return out;
    },
    // ham logit — jeovs kalıbı / kendi eşiklerin için (77,)
    logits: async (t) => heads.logits((await runEmbs([t]))[0]),
  };
}

// script-tag tüketicileri için global (module olarak yüklense de kurulur)
if (typeof window !== "undefined") window.JevProLib = { loadJevPro, BertTokenizer, makeHeads };
