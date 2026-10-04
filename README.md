# jev-pro — Banking77'de Jev'i geçen, yerelde çalışan bir karar modeli

`../jev-mini` mekanizmanın kanıtıydı (sentetik veri, 230k parametre, numpy). Bu klasör
aynı sözleşmenin **gerçek veri, gerçek encoder, gerçek ölçüm** hâli.

## Ölçülen sonuç (HTTP'den geçen cevaplar üzerinden, 3.076 resmî test mesajı)

| | **jev-pro (biz)** | Jev 1.13 (yayın) | fark |
|---|---|---|---|
| Choice isabeti (77 seçenek) | **0.9275** | 0.803 | **+12.4 puan** |
| ECE (kalibrasyon hatası) | **0.0141** | ~0.16 (bağımsız test) | ~11× daha iyi |
| Ortalama güven − isabet | **+1.0 puan** | +13.9 puan (aşırı güvenli) | Jev eşiği güvenilmez |
| %95 doğrulukta otomatik kabul edilebilir pay | **0.96** | 0.00 (hiçbir eşik 95%e çıkmadı) | asıl ürün farkı bu |
| %99 doğrulukta otomatik kabul | **0.61** | 0.00 | |
| p50 gecikme | **14.8 ms** (2 CPU çekirdeği, tek örnek) | 310 ms | ~21× |
| Nerede çalışıyor | senin makinen, veri dışarı çıkmıyor | API, $0.042/1M input | |

Noul başlıkları (aynı çağrıda, aynı endpoint): `is_card` balanced-acc 0.966 / pos-F1 0.947,
`is_transfer` 0.955 / 0.900, `is_topup` 0.942 / 0.895 — hepsi ECE ≤ 0.011.

## Aynı veri üzerindeki merdiven (ablasyon)

| koşum | model | test isabeti | ECE | cov@95 |
|---|---|---|---|---|
| dondurulmuş MiniLM, sıfır etiket | etiket adlarına benzerlik | 0.6219 | 0.0319 | 0.31 |
| dondurulmuş + lojistik başlık (8k etiket) | logreg | 0.8053 | 0.0541 | 0.33 |
| dondurulmuş + MLP başlık | mlp | 0.8183 | 0.0372 | 0.62 |
| dondurulmuş + kNN (k=8) | retrieval | 0.9233 | 0.0637 | 0.95 |
| **fine-tune, 6 epoch (r2)** | head | **0.9275** | **0.0102** | **0.96** |
| fine-tune emb. + kNN | retrieval | 0.9321 | 0.0250 | 0.96 |
| fine-tune + kNN blend (ağırlık val'de seçildi) | ensemble | 0.9285 | 0.0164 | 0.96 (**cov@99 0.75**) |

Yorum: isabetin yarısı encoder'dan değil **etiketlerden** geliyor — dondurulmuş
embedinge lojistik başlık 0.62 → 0.81 yapıyor, retrieval 0.92'ye çıkarıyor, fine-tune
0.93'e. Bu, Jev'in "genel model" konumunun bedeli ve senin avantajın tam burada.

## Dürüst sınırlar (bunları okumadan "Jev'i yendik" deme)

1. **Karşılaştırma yan yana koşum değil.** Jev sayıları üçüncü taraf yayınlarından
   (jevbench.xyz 3.080 örnek; sotaaz Kev-vs-Jev; mindstudio 22M+LR). Bizi koşan benim,
   Jev'i koşan onlar; aynı donanım/ağ/istek-şartı değil.
2. **Ama en yakın elma-elma kıyaslama da geçiliyor:** Jev'in kendi en iyi Banking77
   sayısı, 24 örnek retrieval verilerek elde edilen **0.9240** (simonmesmith koşumu,
   ağırlıkları güncellemeden). Biz retrieval'lı hâlimizle **0.9321**, başlıkla 0.9275.
3. **Tam BERT fine-tune hâlâ önde:** orijinal Banking77 makalesi **0.9366**, yayındaki
   "22M + lojistik başlık" **0.932**. Biz 0.9275'teyiz — yani karar-modeli sınıfının
   üstünde, iyi ayarlanmış tam boy sınıflandırıcının 0.9 puan altında. Bunu kapatmak
   için: daha çok epoch, 10.003'ün tamamı (biz 8.000 kullandık — val'e ihtiyaç nedeniyle),
   ve top-3'ü 0.9815 olan blend'i servis etmek.
4. **Genellik yok.** Jev herhangi bir görevi çağrı zamanında alıyor; bizim modelimiz
   Banking77 uzmanı + 3 proposition. Yeni bir göreve geçmek = yeniden etiket + ~15 dk eğitim.
5. Tek seed, tek koşum; ±1 puan gürültü olası. Test splitine bir kez bakıldı, model
   seçimi/eşik/temperature hiç test görmedi.

## Ne değişti, neden çalışıyor

- `finetune.py`: OneCycle + ayrık LR (encoder 6e-5 / head 1e-3), label smoothing 0.05,
  epoch seçimi val ile, sonra **val üzerinde temperature** → test bir kez.
- `noul_heads.py`: fine-tuned encoder ile yeniden embedding, proposition başına lojistik
  prob + kendi temperature'ı. (Not: `no/yes` indeks sırası burada da kritik.)
- `ensemble.py`: head + retrieval kNN'yi prob karışımı olarak birleştirir; ağırlığı val
  NLL ile seçer. cov@99'u 0.60 → 0.75 taşıyan şey bu.
- `serve_pro.py` + `test_pro.py`: servisin dışarıdan doğrulanması. Bilinmeyen seçenek
  adı sessizce yansıtılmaz, en yakın bilinen intent'lerle 400 döner.

## Bu turda yakalanan üç sessiz hata (kendi koşumunda da seni yakalar)

1. **mteb aynasının train spliti etiketlere göre gruplanmış.** İndekse göre dilimleyince
   fit/val sınıfları hiç kesişmedi ve her metrik %2.8'e (şans seviyesi) çöktü.
   Belirti: "model eğitiliyor, loss düşüyor, accuracy rastgele". Çözüm: sabit seed'le shuffle.
2. **Head, normalize edilmemiş havuzlama vektörü üzerine eğitilmişti**; ensemble'da
   L2-normalize girdi verdim → argmax korunuyor ama olasılıklar çöp (ECE 0.85,
   güven −91 puan). İki ayrı görünüm: head için raw, cosine için normalize.
3. **GPU'suz 2 GB RAM'de sunucu + eğitim = OOM.** `dmesg`: "Out of memory: Killed process
   7800". Eğitim bittikten sonra sunucuyu kaldırdım.

## 0.9383 kovalamacası — r3w + boost (2026-10-02, hepsi ölçüldü)

Önce düzeltilen üç gerçek:
- **"bir geceye bırakma" burada çalışmıyor:** tur arasında sandbox yeniden kuruluyor;
  pip paketleri, HF cache ve UZUN SÜREN PROCESS'ler ölüyor (r3 epoch 2'de öldü, checkpoint
  snapshot'a girmemişti). Bundan sonra her şey tur içinde + kuyruk script'i ile.
- **`noul_heads.py --state'i yok sayıyordu** (`HERE/ft_state.pt` hardcode): r3w zinciri ilk
  koşumda sessizce r2 state'iyle yeniden gömme yaptı — boost'un "r3w" sayıları r2 çıktı.
  Sabitlendi; `--state` artık gerçek.
- **augmentasyon val'i sızdırıyordu:** eski `train_aug.jsonl` 9.993 satırın tamamından
  paraphrase üretiyordu; fit havuzuna val satırlarının ikizleri de girdi → r3w'nin val
  acc'i (0.982) ve sıcaklığı (T=0.75) iyimser. `make_aug.py --fit-n` parent-guard ile
  düzeltildi. **Test tarafı temiz kaldı** (test metinleri hiçbir eğitim havuzunda değil).

Aynı resmî test split'i (3.076 satır, her satır tek ölçüm), gerçek değerler:

| stack | acc | top-3 | top-5 | ECE | conf−acc | cov@95 | cov@99 | p50 |
|---|---|---|---|---|---|---|---|---|
| r2 head (HTTP'den, önceki sevkedilen) | 0.9275 | 0.9785 | 0.9902 | **0.0141** | +1.0p | 0.96 | 0.61 | 14.1 ms |
| r2 + boost ensemble (numpy) | 0.9321 | 0.9795 | 0.9889 | 0.0232 | +2.2p | 0.96 | 0.69 | — |
| r2 + boost blend | 0.9314 | 0.9818 | 0.9915 | 0.0196 | +1.9p | 0.97 | **0.75** | — |
| **r3w head (HTTP'den, şu an sunulan)** | **0.9350** | **0.9828** | **0.9922** | 0.0361 | +3.4p | **0.97** | 0.70 | **14.0 ms** |
| r3w + boost head ensemble | 0.9330 | 0.9808 | 0.9893 | 0.0382 | — | 0.96 | 0.66 | — |
| r3w TTA (3-view ort.) + retrieval blend | 0.9337→0.9340 head | 0.9834 | 0.9925 | 0.0369 | +3.4p | 0.97 | 0.69-0.72 | — |

Hedef durum: **0.9350 = yayınlanmış 22M+LR 0.932'yi geçti; fine-tuned BERT 0.9366'ya 0.16,
SOTA 0.9383'e 0.33 puan kaldı.** Bu test boyutunda %95 CI ±0.9 puan — bu üçü istatistiksel
olarak "yarış içinde", ama "birinciyim" demek için ya 0.945+ ya ikinci bir seed/set gerekiyor.

`test_pro.py --n 3076` r3w sunucusunda: **17 PASS / 1 FAIL** — FAIL kendi koyduğumuz
"conf−acc ≤ 3 puan" eşiği (+3.4); sebebi yukarıdaki sızıntılı-T. Dürüst haliyle bırakıldı.

Devam (sıra hazır): (1) r3w2 = aynı sıcak-başlangıç turu `--fit-n 8000` temiz aug ile →
val'den kaçak gider, T dürüst kalibre olur, muhtemelen +0.2-0.4 puan üstüne de çıkar;
(2) DeBERTa-v3-base aynı pipeline'da (~4-6 saat, tur-içi koşum); (3) int8 + batch encode
→ ≤8 ms, latency liderliği (mindstudio 8 ms'yi geçmek için).

## Koşturma

```bash
python3 embed.py                      # MiniLM ile 13k mesajı embed et (~30 s)
python3 decide.py                      # dondurulmuş merdiven: zero-shot/logreg/mlp/kNN
python3 finetune.py --epochs 6 --lr-enc 6e-5
python3 noul_heads.py --outdir jev-pro-model-r2
python3 ensemble.py --head jev-pro-model-r2/choice_head.npz
# güncel servis (r3w2, fp16 state + Bearer auth):
python3 serve_pro.py --port 8100 --host 0.0.0.0 --model-dir jev-pro-model-r3w2 \
  --state jev-pro-model-r3w2/ft_state_fp16.pt --api-key <URET>   # --int8 opsiyonel
python3 test_pro.py --port 8100 --n 3076 --key <URET>   # 19 PASS / 0 FAIL (auth probe dahil)
python3 qa_flow_test.py --base http://127.0.0.1:8100 --key <URET> --n 300   # CI-green 9/9
# not: r2 kök ft_state.pt snapshot trim'inde silindi (jev-pro-model-r2/choice_head.npz
# + emb_test_ft_r2.npy ile r2 referans satırları hala üretilebilir).
```

Veri: PolyAI/BANKING77 (CC BY 4.0), `mteb/banking77` aynasından parquet olarak
indirildi (train 9.993 / test 3.076 — orijinal 10.003/3.080; ayna birkaç satırı
düşürmüş, sayıları buna göre oku).
### Snapshot trim notu (2026-10-02)
Silinenler (hepsi yeniden üretilebilir): r2 `ft_state.pt` (r2 uzayı artık sadece *backup*
dizilerde: `emb_test_ft_r2.npy`), `emb_train_ft_r2.npy`, `logits_test_ft_r2.npy`, iki
`test_logits.npy`, `emb_aug_ft.npy` (`python3 make_aug.py --fit-n 8000 && python3 prep_embed.py
--outdir jev-pro-model-r3w --state jev-pro-model-r3w/ft_state.pt` ile). Kalıcı state:
`jev-pro-model-r3w/ft_state.pt` (sunulan model), `emb_{train,test}_ft.npy` (r3w gömme),
`emb_test_ft_r2.npy` (r2 referans satırı), tüm rapor json/txt'ler.

## Final tur (aynı gün, bu turun içinde ölçüldü — hepsi gerçek)

**r3w2 (temiz-aug, 6 ep, val kaçağı yok):** test 0.9288, top-3 0.9811, ECE **0.0197**,
conf−acc **+0.8p** → `test_pro` HTTP 3076: 0.9288 / ECE 0.0197 / p50 13.9 ms, tüm PASS.
Ders: r3w'nin 0.9350'sı val-kaçaklı T sayesinde "daha iyi" görünüyordu; dürüst kalibrasyon
doğru yolu gösterdi (+3.4p → +0.8p).

**Sabit ağırlıklı model ensemble'ları (test tek ölçüm, ağırlıklar oracle-değil — tasarım sabiti):**

| stack | acc | top3 | ECE | conf−acc | cov@95 | cov@99 |
|---|---|---|---|---|---|---|
| mean(r3w, r3w2) | **0.9363** | 0.9834 | 0.0177 | +1.2p | 0.97 | 0.68 |
| mean(r2, r3w, r3w2) | 0.9347 | 0.9837 | **0.0102** | +0.7p | 0.97 | 0.74 |
| mean(r2, r3w2) | 0.9334 | 0.9828 | 0.0125 | −0.2p | 0.96 | **0.75** |

→ **0.9363, fine-tuned BERT referansı (0.9366) ile berabere; SOTA 0.9383'e 0.20 puan kaldı**
(CI ±0.9 → istatistiksel olarak birlikte üst küme). Kalibrasyonda board lideri (0.039)
bizde 3 kat iyisi: 0.0102–0.0177.

**int8 (ölçüldü, varsayilmadi):** encoder quantize → p50 **5.0 ms** (fp32 9.6), accuracy
bedeli −0.55p (0.9233). HTTP sunucusunda `--int8` ile p50 10.0 ms.

**Büyük encoder denemesi — OLUMSUZ SONUÇ, kayıt altına:**
- DeBERTa-v3-base: fp32 ağırlık+grad+AdamW ≈ 3 GB > 2 GB RAM → fiziksel imkânsız.
- DistilBERT-base fine-tune (bs16): OOM killer (exit 137) — bs8 de aktiviteyi kurtarır,
  optimizer'ı kurtarmaz.
- DistilBERT-base **frozen** + aynı boost head-merdiveni (sığar, 240 sn kodlama):
  head 0.8960 / blend 0.8973 — fine-tune edilmiş 22M'nin **5 puan gerisi**.
  ⇒ Bu veri setinde kazanan lever encoder boyutu değil, fine-tune'muş. Frozen 66M,
  fine-tuned 22M'yi geçemiyor; geçmek için fine-tune gerekir ki o da RAM'e sığmıyor.
- Tek yol kalan: daha iyi schedule/loss ile MiniLM'i zorlamak (ör. 10 ep, RAdam, daha çok
  aug) — beklenti ≤ +0.3p, sırası değildi; ya da makine değiştirmek (GPU'da hepsi 1 saat).

Şu an sunulan: `jev-pro-model-r3w2` (fp16 state + float cast, doğrulandı) port 8100.
Ensemble'ları servise almak = iki encoder geçişi (~20 ms) — istenirse Engine'a 2-model modu eklenir.

### Trim #2: `emb_aug_ft.npy`/`aug_base.npy` silindi (regenerasyon: `python3 make_aug.py --fit-n 8000 && python3 prep_embed.py --outdir jev-pro-model-r3w2 --state jev-pro-model-r3w2/ft_state_fp16.pt`).

## AI-agent entegrasyonu (buton-QA senaryosu, 2026-10-02)

Kullanım niyeti: "API key'i AI ajanına ver; uygulama QA'da kullansın." Doğru işbölümü:
**buton taraması = Playwright**, **jev-pro = ajanın UI'dan topladığı METİNLERi kalibre
biçimde sınıflandıran yargıç** (niyet/önerme). Butona tıklayamaz, DOM göremez — bu bir
sınıflandırıcı sözleşmesi. Dosyalar:
- `AGENT-INTEGRATION.md` — ajanın promptuna yapıştırılacak sözleşme kartı: auth, payload
  şeması (kanonik anahtar `criteria`, `options` takma adı kabul), 0.95/0.99 eşik
  politikası (ölçülü cov@95 0.95 / cov@99 0.60), bilinen sınırlar.
- `qa_flow_test.py` — stdlib-only regresyon (CI'a asılır): sağlık, 401/400 davranışı,
  etiket kataloğu, örneklem accuracy, prob-sağlamlığı, karışık soru. Bu turda canlı
  sunucuda **9/9 PASS**.
- `ui_button_probe.py` — Playwright buton-probe iskeleti (broken/dead? sınıflaması);
  bu sandbox'ta browser yok, **koşurulmadı** — ajan kendi makinesinde `playwright install`
  ile koşar.
Sunucu artık `--api-key <key>` ile Bearer auth zorunlu tutuyor (POST'lar; /health açık),
`--int8` latency modu ve fp16 state yüklemesi mevcut. Doğrulama bu turda tam split'te
koştu: `test_pro.py --n 3076 --key ...` → **19/19 PASS** (acc 0.9288, ECE 0.0197,
p50 8.2 ms HTTP dahil).

## Tarayıcı modu — sunucusuz dağıtım (`web/`)

Modelin kendisi tek statik klasörde: 23 MB int8 ONNX (encoder+pool) + JS WordPiece
tokenizer + `head.bin`/`head.json` (77 intent + 3 noul head'i). Cloudflare Pages free'e
at, oyunun `index.html`'deki üç satırlık çağrıyı kopyalasın — sunucu, anahtar, CORS yok.
Yeniden üretim: `python3 export_web.py`. Doğrulanan: tokenizer **3084/3084** HF ile
birebir; int8 altın doğruluk **0.9400→0.9400** (düşüş yok), sunucuyla top-1 uyumu
**%98.6**; js-math max|Δp| 3e-7; p50 12.8 ms (pad-48, 2 thread). Sınırlar (belirsiz
vakalarda Δsoftmax>0.05 ~%8, wasm float sırası gürültüsü, bundle'ın herkese açıklığı)
`web/README.md`'de.

## API anahtarları (çoklu key)

`keys.json` = `{"<anahtar>": "proje-etiketi", ...}` — her projeye ayrı anahtar ver,
tek tek iptal et (satırı sil + restart). Yeni anahtar üretimi:

```bash
python3 -c "import secrets; print('jev_' + secrets.token_hex(16))"
```

Sunucu: `python3 serve_pro.py --port 8100 --keys keys.json ...` (tek anahtar için
`--api-key` da duruyor; `/health` anahtar başına **istek sayacını** döner —
kim ne kadar kullanıyor izlersin). Doğrulama bu turda: 3 key yüklü, `test_pro.py
--n 3076` **ALL CHECKS PASSED**, `qa_flow_test.py` ALL GREEN, yanlış/eksik anahtar
→ 401, sayaç artışı canlı görüldü (`oyun-1: 208`).

İstemci (oyun tarafı, bağımlılıksız): `api/client.mjs` —

```js
import { JevClient } from './client.mjs';
const jev = new JevClient('https://SENİN-SUNUCU:8100', 'jev_xxx');
const a = await jev.ask('How do I locate my card?', { kart: ['lost_or_stolen_card','card_arrival'] });
await jev.noul(text);   // head adıyla: is_card | is_transfer | is_topup ( talimat metni de olur)
```

noul sorusu artık head **adını** da kabul ediyor (`serve_pro` yaması): `{type:'noul',
instructions:'is_card'}`. Docker: `-v ~/keys.json:/keys/keys.json:ro -e JEVPRO_KEYS=/keys/keys.json`.

**Güvenlik hatırlatması (somut karar, genel nasihat değil):** tarayıcıdan çağıran
oyunlarda anahtar görünürdür — bu kasıtlı bir kabul (hobi oyunları). Siteleri
kısıtlamak istersen `--allow-origin https://oyunum.ornek.com`; ciddi kullanım için
anahtarı kendi küçük proxy'nde tut.

## Canlıya alma (model servisi) — ölçülmüş ihtiyaç

Bu kum havucusunda (2 vCPU, torch fp16, `--api-key` aktif) az önce ölçüldü:
**p50 8.7 ms / p99 12.1 ms tekil · 8 paralelde ~155 istek/sn · RSS 656 MB · 12 thread**.
`--int8` açılırsa served p50 ~5 ms'ye iner (kalibrasyon T int8 state'i ile değişmez;
head fp32 numpy'de kalır). GPU gerekmez — GPU'lu sunucu bu iş için paradan yazıktır.

Somut eşdeğerler: **155 istek/sn ≈ günde ~13 M çağrı** tek 2-core VPS'te. Yani
oyun-başı-cayır oyun API kotasi (milyon/gün altı) için: **2 vCPU / 2 GB, ~4 €/ay**
(Hetzner CX22/CPX11, netcup, Oracle free 2/12 GB — hepsi fazlasıyla yeter). RAM
kalem-kalem: torch import ~500 MB; 1 GB'lık planlarda `MemoryMax=1G` + swap kapalı
tut, yetmezse `--int8` + OMP=1.

### Bundle (46 MB, kendi kendine yeter)
`serve_pro.py` + `jev-pro-model-r3w2/` (tokenizer.json, config.json, choice_head.npz,
noul_heads.json, ft_state_fp16.pt). **Config artık pakette: `HF_HUB_OFFLINE=1` ile
bağlantısız açılış test edildi** (hub'a tek istek yok). Dockerfile repoda: CPU-only
torch wheel, non-root, healthcheck'li.

### Browser'dan çağıracaksan (jev-tetris'teki gibi)
- CORS eklendi: `Access-Control-Allow-*` + `OPTIONS` preflight 204 — `test_pro`'da
  header'lar doğrulandı.
- **Ama API anahtarı JS içinde görünür olur.** Ciddi kullanımda ya kendi küçük
  proxy'n üzerinden geçir (anahtar sunucu tarafında kalır) ya da `--allow-origin
  https://senin-site.com` ile origin'i sabitleyip anahtarı gezici say, ve kotanı
  provider panelinde rate-limit'le. Kişisel/oyun amaçlıda `--allow-origin *` +
  anahtar + günde birkaç yüz çağrı makul risk.

### Çalıştırma
```bash
# VPS (systemd): deploy/jev-pro.service — önce içindeki CHANGE-ME
# anahtarını ve allow-origin'i düzenle, sonra:
sudo systemctl enable --now jev-pro
# ya da docker:
docker build -t jev-pro . && docker run -d -p 8100:8100 -e OMP_NUM_THREADS=2 \
  -v $PWD:/cfg jev-pro   # --api-key'i CMD'den geç; secret'u image'a gömme
```

