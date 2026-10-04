# jev-pro · tarayıcı modülü (sunucusuz)

Modelin kendisi, sunucusuz: tek statik klasör. API anahtarı yok, CORS yok, kuyruk yok,
barındırma ücreti yok (Cloudflare Pages free). Tarayıcı 23 MB'lık int8 ONNX'u bir kez
indirir, sonrası tamamen yerel — offline/PWA olarak da çalışabilir.

## Dosyalar

| dosya | boyut | ne |
|---|---|---|
| `jevpro.js` | ~9 KB | **model API'si** — tokenizer + head matematiği + yükleici, tek dosya |
| `encoder_int8.onnx` | 22.9 MB | MiniLM kodlayıcı + mean-pool; int8 dinamik kuantalama |
| `tokenizer.json` | 711 KB | HF tokenizer verisi |
| `head.bin` | 118 KB | 77×384 choice head ağırlıkları (float32, row-major) |
| `head.json` | 28 KB | etiket adları + bias + T + 3 noul head'i |
| `index.html` | — | demo sayfa |
| `_headers` | — | Cloudflare Pages: CORS `*` + uzun cache |
| `parity_test.mjs` | — | Node doğrulaması (`npm i` ile onnxruntime-node kurulur) |

## Diğer projelerden kullanım (API budur)

```html
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.min.js"></script>
<script type="module">
  import { loadJevPro } from 'https://SENİN-SİTE/jevpro.js';
  const jev = await loadJevPro({ base: 'https://SENİN-SİTE/' });
  const r = await jev.classify('why was my card payment declined?');
  // { pick: 'declined_card_payment', confidence: 0.995, probs: {...} }
  await jev.classify(text, ['card_arrival', 'lost_or_stolen_card']); // subset (sunucu mantığı)
  await jev.noul(text);                                              // {is_card, is_transfer, is_topup}
  await jev.classifyBatch(textler);                                  // oyunlar: deterministik döngü
</script>
```

`base` vermezsen site kökünden çeker. Asset'ler CORS açık (`_headers`) — başka origin'den
import serbest. Kendi hosting'ine gömerken klasörün 5 dosyasını birlikte taşı.

## Doğrulama (bu depoda ölçüldü — `node parity_test.mjs`)

- **Tokenizer**: 3084/3084 (parity_test'te 400/400 canlı koşulur) metinde HF `AutoTokenizer` ile birebir aynı input_ids
  (3076 test satırı + noktalama/CJK/120-char kelime gibi 8 uç durum).
- **Model grafiği** (python onnxruntime, tam test seti n=3076): altın doğruluk
  **0.9400 → 0.9400** (int8'de düşüş yok), sunucuyla top-1 uyumu **%98.6**;
  Δsoftmax medyan 0.0025, ~%8 belirsiz-eşit vakada >0.05 (quantize gürültüsü).
- **Kütüphane E2E** (`jevpro.js` + onnxruntime-node): 400 metinde altın doğruluk
  int8 0.9350 vs fp32 0.9375, top-1 uyum %99.0; **classify == classifyBatch bit-bazlı**
  (Δ 0.0) — ORT int8 per-tensor aktivasyon ölçeği gerçek batch'te satırları ~0.3
  kaydırdığı için batch içerde tek-tekir koşar; toplu hız gerekirse `embedBatch` var.
- **Gecikme**: int8, 2 thread, pad-48: p50 12.8 ms. Tarayıcıda `padding="longest"`
  (~12 token) sayesinde tipik metinde 5-10 ms beklenir; WASM, VPS'teki torch
  servisinin (~656 MB RSS, 12 thread) yerini alır.

## Kullanım

1. **Yerel önizleme**: bu klasörde `python3 -m http.server 8080` → `http://localhost:8080`.
   (`file://` açılmaz — fetch çalışmaz; sayfa bunu uyarı olarak gösterir.)
2. **Cloudflare Pages**: klasörü olduğu gibi sürümle (her dosya < 25 MiB sınırında).
   Build ayarı: framework "None", build command boş, output `/`. Sayfa kökü = bu klasör;
   `parity_test.mjs`/`package.json`'ı atabilirsin (zararsız).
3. **Kendi oyununa gömme**: `index.html`'i oku — üç satır: `BertTokenizer.encode` →
   ort session `run` → `heads.classify(emb, istenenLabelIndexleri)`. İstersen bu adımı
   `jeovs` kalıbında olduğu gibi aday-sıralamalı güven vektörüne çevir.

## Yeniden üretme

`../export_web.py` her şeyi kurar: torch motorundan ONNX export → int8 quantize →
head/tokenizer dosyaları → doğrulama. CDN'siz/airgap hedefin varsa `onnxruntime-web`
paketini (`ort.min.js` + `.wasm` dosyaları, ~12 MB) klasöre indir, `index.html`'deki
iki CDN URL'sini yerel dosyayla değiştir.

## Sınırlar (dürüst kayıt)

- int8 kuantalama, eşit-ağırlıklı (~%8) vakada sunucuya göre fark edilebilir olasılık
  kayması yapar; seçimler aynı doğrulukta kalır ama 0.995 eşikli otomasyon kuracaksan
  bandı genişlet.
- ORT-WASM'nin float toplama sırası python'dan farklı → ~1e-7'lik gürültü (RH-9 dersi:
  eşitlik testleri kaba toleranslı olsun).
- Model herkese açık olur: bundle'ı indiren ağırlıkları da indirir (MIT/Apache taban —
  sorun değil ama bilerek paylaş).
- `test_pro`/`qa_flow` kapılarından geçmiş gerçek referans **sunucu** yollarıdır;
  web yolu onların int8-izinli türevidir.
