# jev-pro API — AI-agent entegrasyon kartı

Bu dosya, bir kodlama ajanına (ya da CI pipeline'ına) "bu API'yi nasıl kullanırsın" diye
olduğu gibi yapıştırılacak referanstır. Örnek çıktıların hepsi bu makinede canlı ölçüldü.

## 0) Önce sınır netliği — bu API NE, NE DEĞİL

jev-pro bir **metin-anlayıcı sınıflandırıcıdır**: bir metni verirsin, 77 bankacılık
niyetinden birine kalibre olasılıkla atar; ayrıca eğitili evet/hayir önermelerine
(is_card, is_transfer, is_topup) olasılık verir. 14 ms'de cevaplar, eşik politikası
kurulabilir (aşağıda).

Bu API **şunları YAPMAZ** (amaç listesinden çıkan buton-testi işleri dahil):
- DOM'a/butona dokunmaz, tıklama yapmaz, tarayıcı açmaz;
- "çalışmayan buton" bulmaz — o bir **browser-automation** işidir (Playwright/Selenium);
- UI göremez, ekran çözemez, ekrandan metin okuyamaz.

Sağlıklı işbölümü (ajanla kullanırken):
```
UI testi / buton taraması   -> Playwright (ajenin yazdığı script)  ── üretime metin:
                                log satırları, hata mesajları, toast'lar,
                                chat akışı mesajları, ticket başlıkları
                                        │
                                        ▼
                                jev-pro /v1/systemone  ── semantik yargı:
                                "bu metin hangi niyet/akışa ait? güvenim %97"
                                        │
                                        ▼
                                eşik politikası: auto-accept / escalation / test-fail
```
Yani jev-pro, ajanın UI'dan **topladığı metinleri** sınıflandıran pahalı-judge ucuz
versiyonudur; butonların kendisini test eden şey Playwright'tır. (Repo kökündeki
`ui_button_probe.py` o iş için iskelettir.)

## 1) Bağlantı + auth

```
BASE   : http://<host>:8100
AUTH   : POST'lar için  Authorization: Bearer <api-key>   (--api-key ile açılır)
GET    : /health (açık)  |  /v1/labels (77 etiketin kanonik listesi — payload
         yazmadan ÖNCE buradan çek; uydurma etiket 400 döner, en yakın önerisiyle)
```

## 2) Tek karar çağrısı — `POST /v1/systemone`

```jsonc
// REQUEST
{
  "state": "My card hasnt arrived after two weeks",      // str veya obj (obj -> json'lanır)
  "questions": {
    "intent": { "type": "choice",                         // qid'ler senin anahtarların
      "instructions": "Which intent?",
      "criteria": ["card_arrival", "card_delivery_estimate", "lost_or_stolen_card", "change_pin"]
      // kanonik anahtar "criteria"; "options" takma adı da kabul edilir.
      // İkisi de yoksa 77 etiketin TAMAMI üzerinden puanlar.
      // Geçersiz isim => cevap ÜRETMEZ: qid hata olarak döner (sessiz fallback yok).
    },
    "card": { "type": "noul",
      "instructions": "Does this message concern a card (physical or virtual)?" }
  }
}
// RESPONSE (canlı ölçülmüş biçim)
{
  "model": "jev-pro-1",
  "answers": {
    "intent": { "type": "choice", "choice": "card_arrival",
                "probabilities": { "card_arrival": 0.9676, "...": 0.0 } },
    "card":   { "type": "noul", "noul": 0.9731, "confidence": 0.973,
                "proposition": "is_card", "calibration": "temperature 3.50" }
  },
  "latency_ms": 13.9,
  "usage": { "input_tokens": 12, "output_tokens": 0 },
  "errors": { "qid": "noul head yoksa/seçenek hatalıysa mesaj + en yakın niyet önerisi" }
}
```

Hatalar: key yok/yanlış → `401 AuthenticationError`; tüm sorular geçersiz → `400
"no answerable question"` + detayda her qid'nin hatası (cevap varsa soru-bazında `errors`
dolu gelir, 200 bozulmaz — kısmi cevap felsefesi).

## 3) Eşik politikası — sayılarla, sezgiyle değil

Bu model resmî test split'inde (3.076 satır) HTTP'den ölçüldü:

| politika | eşik | ölçülen |
|---|---|---|
| auto-accept (doğru muhatap/aksiyon) | p ≥ 0.95 | akıllarda **%97 otomatik** işlenir, seçilenlerde doğruluk ≥ %95 |
| sıkı otomasyon | p ≥ 0.99 | %60 kapsama, doğruluk ≥ %99 |
| escalation (insana/ajana sor) | p < 0.95 | kalan %3-5 |

Kullanım kalıbı: `choice.probabilities[chosen] >= 0.95` → güvenle devam; değilse metni
ajanın kendisine/devreye taşı. Kümülatif olasılık bire bir eder (`|Σ−1| ≤ 1e-3`).

## 4) Regresyon testi — CI'a asılacak şey

```
python3 qa_flow_test.py --base http://127.0.0.1:8100 --key <KEY> --n 300 [--min-acc 0.90]
```
Banking77 test satırlarından örneklemeyle uçtan uca gider; accuracy, 401/400 davranışı,
prob-sağlam-lığı kontrol eder; exit-code 0 = geçiyor. Model dosyası değişince, fine-tune
yapılınca, sunucu taşınınca bunu koş — `README.md`'deki tam ölçüm (n=3076) referanstır.

## 5) Agent'a verirken söylenecek 3 cümle (promptuna ekle)

1. "Her kullanıcı/uygulama metnini tek çağrıda sınıflandır; `state` alanına ham mesajı,
   `questions.intent.criteria`'ya **GEÇERLİ akış etiketlerini** (önce GET /v1/labels) koy."
2. "`confidence ≥ 0.95` ise aksiyonu otomatik uygula; altında her şey escalate."
3. "Buton/UI testi senin işin (Playwright), bu API'nin işi değil — bu API sadece
   UI'dan/loktan akan METİNLERİ anlamlandırır."

## 6) Bilinen sınırlar (kısa ve acımasız)

- 77 niyet **bankacılık** alanı; başka alanda etiket setini `finetune.py` ile
  kendi verinle yeniden eğitmeden kullanma (uydurma seçenek → 400, doğru davranış bu).
- Model metni anlar; bağlam/konuşma geçmişi anlamaz (state'e son mesajı ya da
  birkaç turu birleştirilmiş metni ver, ikincisini eğitmedik — dene, ölç).
- Sunucuda TLS/rate-limit/çok-kullanıcılı key yönetimi yok; ağa açacaksan bir
  reverse-proxy (nginx + https + IP kısıtı) önüne koy.
