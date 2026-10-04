// jev-pro HTTP API istemcisi — tarayıcı ya da Node, tek dosya, bağımlılıksız.
// Kullanım:
//   import { JevClient } from './client.mjs';
//   const jev = new JevClient('https://SUNUCU:8100', 'jev_xxx');   // keys.json'daki anahtar
//   const a = await jev.ask('How do I locate my card?', { card: ['lost_or_stolen_card','card_arrival'] });
//   a.card.choice /* -> 'lost_or_stolen_card' */, a.card.confidence, a.card.probabilities
//   await jev.noul('I added money but it is not on my balance')   // {is_card, is_transfer, is_topup}
//   await jev.labels()                                           // 77 etiket
//
// Anahtar uyarısı: tarayıcıda inline anahtar HERKESİN görüşüne açıktır (CORS * olduğu
// için siteler de kullanabilir) — hobi oyunları için sorun değil; ciddi işte kendi
// proxy'nden geçir ya da --allow-origin'i site URL'ne sabitle.
export class JevClient {
  constructor(baseUrl = '', key) {
    this.base = baseUrl.replace(/\/$/, "");
    this.h = { "Content-Type": "application/json" };
    if (key) this.h["Authorization"] = `Bearer ${key}`;
  }
  async #post(path, body) {
    const r = await fetch(this.base + path, { method: "POST", headers: this.h, body: JSON.stringify(body) });
    const j = await r.json().catch(() => ({}));
    if (!r.ok) { const e = new Error(j?.error?.message || `HTTP ${r.status}`); e.status = r.status; e.api = j; throw e; }
    return j;
  }
  // state: string ya da nesne; questions: {id:{type:'choice',criteria?}} — criteria boşsa 77'i de skorlar
  async ask(state, options = {}, extra = {}) {
    const questions = {};
    for (const [id, crit] of Object.entries(options))
      questions[id] = Array.isArray(crit) ? { type: "choice", criteria: crit } : { type: "choice" };
    Object.assign(questions, extra);
    const j = await this.#post("/v1/systemone", { state, questions });
    return j.answers;
  }
  async noul(text) {
    const j = await this.#post("/v1/systemone", {
      state: text,
      questions: Object.fromEntries(["is_card", "is_transfer", "is_topup"].map(k => [k, { type: "noul", instructions: k }])),
    });
    return Object.fromEntries(Object.entries(j.answers).map(([k, v]) => [k, v.noul ?? v.score ?? v]));
  }
  async labels() {
    const r = await fetch(this.base + "/v1/labels");
    return (await r.json()).labels;
  }
  async health() { return (await fetch(this.base + "/health")).json(); }
}
