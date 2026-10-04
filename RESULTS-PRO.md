# jev-pro — ham çıktılar (bu oturumda üretildi, 2026-10-02, sandbox: 2 CPU çekirdeği / 2 GB RAM / GPU yok)

## 1) dondurulmuş encoder merdiveni — decide.py
```
Banking77: fit=8000  val=1993  test=3076  classes=77  encoder=all-MiniLM-L6-v2 (22.7M, frozen)  train split shuffled with seed 1234

  head                   acc  macroF1   top3     ECE  conf-acc  cov@95  cov@99
  ensemble            0.9151   0.9150  0.978  0.1283    -12.8p    0.92    0.63

  published references for the same split:
    Jev jev-1.13.0 (zero-shot typed Choice, 3,080 cases): 0.803, p50 310 ms
    Jev overconfidence on BANKING77: +13.9 pts, coverage at 95% accuracy: 0%
    published 22M encoder + logistic head: 0.932 @ 8 ms; fine-tuned BERT: 0.9366
  worst classes of the best head: balance_not_updated_after_bank_transfer(0.75), topping_up_by_card(0.79), card_payment_not_recognised(0.80), transfer_not_received_by_recipient(0.81), pending_transfer(0.81), card_arrival(0.85), top_up_failed(0.85), lost_or_stolen_card(0.85)
```

## 2) fine-tune koşumu r2 — finetune_report.txt
```
jev-pro fine-tuned  all-MiniLM-L6-v2 (22.7M params, trainable, CPU, 2 threads)
fit=8000 val=1993 test=3076 classes=77 epochs=6 best_epoch=5

  test accuracy            0.9275   (T=1.0 gives 0.9275; temperature only rescales)
  macro F1                 0.9278
  top-3 accuracy           0.9785
  ECE after calibration    0.0137   (before: 0.0155)
  mean confidence - accuracy  +0.9 pts
  share auto-acceptable at 95% accuracy   0.96
  share auto-acceptable at 99% accuracy   0.61
  latency per decision     12.4 ms  (median, single example, CPU)

  published comparators on this exact split:
    Jev 1.13 zero-shot typed Choice ......... 0.803  @ p50 310 ms   (jevbench.xyz)
    Jev 1.13 with 24 retrieved examples ..... 0.9240               (simonmesmith)
    fine-tuned BERT (2020 paper) ............ 0.9366
    22M encoder + logistic head (frozen) .... 0.932  @ 8 ms        (mindstudio)
    Kev-9B (open, trained on this data) ..... 0.825 on a 154-sample set

  training history: [{"epoch": 1, "loss": 2.643514274597168, "val_acc": 0.6818866031108881, "secs": 110.5}, {"epoch": 2, "loss": 0.8687113356590271, "val_acc": 0.868038133467135, "secs": 111.9}, {"epoch": 3, "loss": 0.6514314345121384, "val_acc": 0.8986452584044154, "secs": 101.6}, {"epoch": 4, "loss": 0.5667312765121459, "val_acc": 0.9056698444555946, "secs": 103.7}, {"epoch": 5, "loss": 0.530103231549263, "val_acc": 0.9212242849974912, "secs": 105.9}, {"epoch": 6, "loss": 0.5145565286874771, "val_acc": 0.91921726041144, "secs": 99.7}]
```

## 3) noul başlıkları (fine-tuned embedding üzerinde)
```
is_card      acc 0.9672  ECE 0.0265 -> 0.0106  T=3.50  pos_rate=0.351
is_transfer  acc 0.9720  ECE 0.0226 -> 0.0074  T=2.80  pos_rate=0.220
is_topup     acc 0.9889  ECE 0.0111 -> 0.0062  T=3.00  pos_rate=0.117
```

## 4) head + retrieval blend — ensemble.py
```

on the official test split:
  model                        acc    top3      ECE  conf-acc  cov@95  cov@99
  fine-tuned head           0.9275  0.9785   0.0102     -0.5p    0.96    0.60
  retrieval kNN             0.9321  0.9652   0.0250     -2.5p    0.96    0.35
  blend (val-selected w)    0.9285  0.9815   0.0164     -0.3p    0.96    0.75

reference: Jev 1.13 zero-shot 0.803 / Jev + 24 retrieved examples 0.9240 / fine-tuned BERT (2020) 0.9366
```

## 5) servisin dışarıdan doğrulanması — test_pro.py (n=200 hızlı koşum, 18 PASS / 0 FAIL)
```
  [PASS] GET /health: labels=77 noul=['is_card', 'is_transfer', 'is_topup']

measuring on 200 official-test messages through HTTP

  [PASS] probabilities sum to 1: |Σ-1| = 6.0e-04
  [PASS] served pick == argmax of served probabilities: card_about_to_expire

  choice (77 options)   acc 0.9250   ECE 0.0396   conf-acc +2.0 pts   cov@95 0.95   cov@99 0.72   p50 14.2 ms
  published Jev 1.13  acc 0.803                     conf-acc ~+13.9 pts   cov@95 0.00   p50 310 ms
  [PASS] endpoint accuracy beats Jev's published 80.3% by >5 pts: got 0.9250
  [PASS] endpoint ECE under 0.05 (usable thresholds): 0.0396
  [PASS] mean confidence within 3 pts of accuracy: +2.0 pts
  [PASS] share auto-acceptable at 95% accuracy above 0.7: 0.95 (Jev: 0.00)
  [PASS] median latency under 30 ms (Jev p50 310 ms): 14.2 ms

noul propositions (same endpoint, same call):
  is_card      acc 0.9650 (majority-only 0.6800)  balanced 0.9660  pos-F1 0.9466 (TPR 0.97/TNR 0.96)  Brier 0.0328  errors 0
  [PASS] noul is_card balanced accuracy > 0.85: 0.9660
  [PASS] noul is_card positive-class F1 > 0.55: 0.9466
  is_transfer  acc 0.9600 (majority-only 0.8100)  balanced 0.9552  pos-F1 0.9000 (TPR 0.95/TNR 0.96)  Brier 0.0326  errors 0
  [PASS] noul is_transfer balanced accuracy > 0.85: 0.9552
  [PASS] noul is_transfer positive-class F1 > 0.55: 0.9000
  is_topup     acc 0.9800 (majority-only 0.9050)  balanced 0.9418  pos-F1 0.8947 (TPR 0.89/TNR 0.99)  Brier 0.0154  errors 0
  [PASS] noul is_topup balanced accuracy > 0.85: 0.9418
  [PASS] noul is_topup positive-class F1 > 0.55: 0.8947

request handling:
  [PASS] case/underscore variants of intent names accepted: Exchange Rate
  [PASS] unknown option name -> 400 with nearest-intent hint: {"error": {"message": "no answerable question", "details": {"intent": "unknown option(s) ['does_not_exist_intent', 'anot
  [PASS] untrained proposition -> 400 that lists what exists: {"error": {"message": "no answerable question", "details": {"q": "no noul head for 'Is the customer thinking about lunch
  [PASS] mixed choice+noul in a single call: latency 34.18 ms

ALL CHECKS PASSED
```

## 6) tam test spliti üzerinden HTTP ölçümü — test_pro.py --n 3076
```
measuring on 3076 official-test messages through HTTP

  [PASS] probabilities sum to 1: |Σ-1| = 6.0e-04
  [PASS] served pick == argmax of served probabilities: card_about_to_expire

  choice (77 options)   acc 0.9275   ECE 0.0141   conf-acc +1.0 pts   cov@95 0.96   cov@99 0.61   p50 14.8 ms
  published Jev 1.13  acc 0.803                     conf-acc ~+13.9 pts   cov@95 0.00   p50 310 ms
  [PASS] endpoint accuracy beats Jev's published 80.3% by >5 pts: got 0.9275
  [PASS] endpoint ECE under 0.05 (usable thresholds): 0.0141
  [PASS] mean confidence within 3 pts of accuracy: +1.0 pts
  [PASS] share auto-acceptable at 95% accuracy above 0.7: 0.96 (Jev: 0.00)
  [PASS] median latency under 30 ms (Jev p50 310 ms): 14.8 ms

noul propositions (same endpoint, same call):
```

=== r3w finetune raporu (chase_r3w.sh, 07:05 UTC) ===
jev-pro fine-tuned  all-MiniLM-L6-v2 (22.7M params, trainable, CPU, 2 threads)
fit=27986 val=1993 test=3076 classes=77 epochs=3 best_epoch=3

  test accuracy            0.9350   (T=1.0 gives 0.9350; temperature only rescales)
  macro F1                 0.9350
  top-3 accuracy           0.9828
  ECE after calibration    0.0355   (before: 0.0120)
  mean confidence - accuracy  +3.4 pts
  share auto-acceptable at 95% accuracy   0.97
  share auto-acceptable at 99% accuracy   0.70
  latency per decision     10.4 ms  (median, single example, CPU)

  published comparators on this exact split:
    Jev 1.13 zero-shot typed Choice ......... 0.803  @ p50 310 ms   (jevbench.xyz)
    Jev 1.13 with 24 retrieved examples ..... 0.9240               (simonmesmith)
    fine-tuned BERT (2020 paper) ............ 0.9366
    22M encoder + logistic head (frozen) .... 0.932  @ 8 ms        (mindstudio)
    Kev-9B (open, trained on this data) ..... 0.825 on a 154-sample set

  training history: [{"epoch": 1, "loss": 0.5343969776630402, "val_acc": 0.958354239839438, "secs": 370.5}, {"epoch": 2, "loss": 0.4928916313648224, "val_acc": 0.9809332664325138, "secs": 355.7}, {"epoch": 3, "loss": 0.471234849521092, "val_acc": 0.9824385348720521, "secs": 352.1}]

=== boost r3w (chase2.log kuyruğu) ===
on the official test split (3,076 rows, measured once):
  r2 single head (in r2 space)       acc 0.9275  top3 0.9785  top5 0.9902  ECE 0.0137  cov@95 0.96  cov@99 0.61
  multi-seed head ensemble           acc 0.9330  top3 0.9808  top5 0.9893  ECE 0.0382  cov@95 0.96  cov@99 0.66
  kNN k=32 on ft embeddings          acc 0.9330  top3 0.9678  top5 0.9698  ECE 0.0462  cov@95 0.97  cov@99 0.36
  boost blend (this run)             acc 0.9334  top3 0.9828  top5 0.9915  ECE 0.0376  cov@95 0.96  cov@99 0.74
  boost blend, geometric mean        acc 0.9330  top3 0.9818  top5 0.9893  ECE 0.0387  cov@95 0.97  cov@99 0.67
  label-desc reranker alone          acc 0.8843  top3 0.9509  top5 0.9711  ECE 0.7356  cov@95 0.81  cov@99 0.13

wrote boost_report_r3w.json   [86s total]
targets: Banking77 SOTA 0.9383 | fine-tuned BERT 0.9366 | published 22M+LR 0.932

=== final_tune.py (TTA+blend) ===
{
 "rows": [
  {
   "model": "r3w head single-pass (T=0.75)",
   "acc": 0.935,
   "top3": 0.9828,
   "top5": 0.9922,
   "ece": 0.0355,
   "conf_minus_acc_pts": 3.4,
   "cov95": 0.97,
   "cov99": 0.7
  },
  {
   "model": "r3w head + TTA (val-T)",
   "acc": 0.934,
   "top3": 0.9831,
   "top5": 0.9925,
   "ece": 0.038,
   "conf_minus_acc_pts": 3.3,
   "cov95": 0.97,
   "cov99": 0.72
  },
  {
   "model": "r3w TTA blend knn+centroid",
   "acc": 0.9337,
   "top3": 0.9834,
   "top5": 0.9925,
   "ece": 0.0369,
   "conf_minus_acc_pts": 3.4,
   "cov95": 0.97,
   "cov99": 0.69
  }
 ],
 "Tv": 0.7620253164556963,
 "Tk": 2.4443037974683546,
 "Tcv": 0.9822784810126584,
 "Tb": 0.9746835443037976,
 "k": 32,
 "w_knn": 0.05,
 "w_cent": 0.0,
 "val": {
  "head_tta": 0.9814350225790266,
  "blend": 0.9814350225790266
 },
 "seconds": 39.1
}

=== test_pro r3w sunucusu (3076, HTTP) ===
  [PASS] GET /health: labels=77 noul=['is_card', 'is_transfer', 'is_topup']

measuring on 3076 official-test messages through HTTP

  [PASS] probabilities sum to 1: |Σ-1| = 9.0e-04
  [PASS] served pick == argmax of served probabilities: card_about_to_expire

  choice (77 options)   acc 0.9350   ECE 0.0361   conf-acc +3.4 pts   cov@95 0.97   cov@99 0.70   p50 14.0 ms
  published Jev 1.13  acc 0.803                     conf-acc ~+13.9 pts   cov@95 0.00   p50 310 ms
  [PASS] endpoint accuracy beats Jev's published 80.3% by >5 pts: got 0.9350
  [PASS] endpoint ECE under 0.05 (usable thresholds): 0.0361
  [FAIL] mean confidence within 3 pts of accuracy: +3.4 pts
  [PASS] share auto-acceptable at 95% accuracy above 0.7: 0.97 (Jev: 0.00)
  [PASS] median latency under 30 ms (Jev p50 310 ms): 14.0 ms

noul propositions (same endpoint, same call):
  is_card      acc 0.9550 (majority-only 0.6800)  balanced 0.9545  pos-F1 0.9313 (TPR 0.95/TNR 0.96)  Brier 0.0357  errors 0
  [PASS] noul is_card balanced accuracy > 0.85: 0.9545
  [PASS] noul is_card positive-class F1 > 0.55: 0.9313
  is_transfer  acc 0.9700 (majority-only 0.8100)  balanced 0.9613  pos-F1 0.9231 (TPR 0.95/TNR 0.98)  Brier 0.0284  errors 0
  [PASS] noul is_transfer balanced accuracy > 0.85: 0.9613
  [PASS] noul is_transfer positive-class F1 > 0.55: 0.9231
  is_topup     acc 0.9900 (majority-only 0.9050)  balanced 0.9474  pos-F1 0.9444 (TPR 0.89/TNR 1.00)  Brier 0.0100  errors 0
  [PASS] noul is_topup balanced accuracy > 0.85: 0.9474
  [PASS] noul is_topup positive-class F1 > 0.55: 0.9444

request handling:
  [PASS] case/underscore variants of intent names accepted: Exchange Rate
  [PASS] unknown option name -> 400 with nearest-intent hint: {"error": {"message": "no answerable question", "details": {"intent": "unknown option(s) ['does_not_exist_intent', 'anot
  [PASS] untrained proposition -> 400 that lists what exists: {"error": {"message": "no answerable question", "details": {"q": "no noul head for 'Is the customer thinking about lunch
  [PASS] mixed choice+noul in a single call: latency 28.6 ms

1 FAILED: ['mean confidence within 3 pts of accuracy']

=== r3w2_ft.log ===
  epoch 1/6 loss 1.8661  val acc 0.8209  (319s)
  epoch 2/6 loss 0.6105  val acc 0.8926  (303s)
  epoch 3/6 loss 0.4974  val acc 0.9087  (289s)
  epoch 4/6 loss 0.4550  val acc 0.9157  (297s)
  epoch 5/6 loss 0.4353  val acc 0.9187  (292s)
  epoch 6/6 loss 0.4298  val acc 0.9197  (311s)
best epoch 6 (val acc 0.9197) — recalibrating on the same val split
official test split, scored once:
  test accuracy            0.9288   (T=1.0 gives 0.9288; temperature only rescales)
  ECE after calibration    0.0197   (before: 0.0197)
  mean confidence - accuracy  +0.8 pts
  share auto-acceptable at 95% accuracy   0.95
  share auto-acceptable at 99% accuracy   0.60
  latency per decision     9.2 ms  (median, single example, CPU)
  training history: [{"epoch": 1, "loss": 1.8660690867900849, "val_acc": 0.8208730556949323, "secs": 318.6}, {"epoch": 2, "loss": 0.6105189478000005, "val_acc": 0.8926241846462619, "secs": 302.9}, {"epoch": 3, "loss": 0.4974182930787404, "val_acc": 0.9086803813346713, "secs": 289.5}, {"epoch": 4, "loss": 0.45498445467154186, "val_acc": 0.9157049673858505, "secs": 296.7}, {"epoch": 5, "loss": 0.43530560688177744, "val_acc": 0.9187155042649272, "secs": 291.9}, {"epoch": 6, "loss": 0.4297503776152929, "val_acc": 0.9197190165579529, "secs": 310.6}]


=== chase3.log ===
  training history: [{"epoch": 1, "loss": 1.8660690867900849, "val_acc": 0.8208730556949323, "secs": 318.6}, {"epoch": 2, "loss": 0.6105189478000005, "val_acc": 0.8926241846462619, "secs": 302.9}, {"epoch": 3, "loss": 0.4974182930787404, "val_acc": 0.9086803813346713, "secs": 289.5}, {"epoch": 4, "loss": 0.45498445467154186, "val_acc": 0.9157049673858505, "secs": 296.7}, {"epoch": 5, "loss": 0.43530560688177744, "val_acc": 0.9187155042649272, "secs": 291.9}, {"epoch": 6, "loss": 0.4297503776152929, "val_acc": 0.9197190165579529, "secs": 310.6}]
choice head on test: acc 0.9288  ECE 0.0197  T=1.00
  is_card      pos_rate 0.351  acc 0.9685  (majority-only acc 0.6489)  ECE 0.0273 -> 0.0133  T=3.70
  is_transfer  pos_rate 0.220  acc 0.9785  (majority-only acc 0.7796)  ECE 0.0174 -> 0.0053  T=2.50
  is_topup     pos_rate 0.117  acc 0.9915  (majority-only acc 0.8830)  ECE 0.0088 -> 0.0042  T=2.60
=== boost on r3w2 embeddings ===
OOF desc-only acc 0.9393 (T=0.30)
blend picked on OOF: w_knn=0.20 w_cent=0.30 w_desc=0.00 T=1.08  OOF acc 0.9834
r2 single head -> boost flips on 126 / 3076 test rows
on the official test split (3,076 rows, measured once):
  r2 single head (in r2 space)       acc 0.9275  top3 0.9785  top5 0.9902  ECE 0.0137  cov@95 0.96  cov@99 0.61
  multi-seed head ensemble           acc 0.9288  top3 0.9808  top5 0.9886  ECE 0.0378  cov@95 0.96  cov@99 0.66
  kNN k=32 on ft embeddings          acc 0.9278  top3 0.9620  top5 0.9646  ECE 0.0540  cov@95 0.95  cov@99 0.46
  boost blend (this run)             acc 0.9295  top3 0.9821  top5 0.9896  ECE 0.0336  cov@95 0.96  cov@99 0.73
  boost blend, geometric mean        acc 0.9298  top3 0.9815  top5 0.9880  ECE 0.0523  cov@95 0.96  cov@99 0.72
  label-desc reranker alone          acc 0.8765  top3 0.9470  top5 0.9626  ECE 0.7095  cov@95 0.82  cov@99 0.05
wrote boost_report_r3w2.json   [88s total]


=== served_r3w2_fp32.txt ===
  choice (77 options)   acc 0.9288   ECE 0.0197   conf-acc +0.8 pts   cov@95 0.95   cov@99 0.60   p50 13.9 ms
  published Jev 1.13  acc 0.803                     conf-acc ~+13.9 pts   cov@95 0.00   p50 310 ms
  [PASS] mean confidence within 3 pts of accuracy: +0.8 pts
  [PASS] share auto-acceptable at 95% accuracy above 0.7: 0.95 (Jev: 0.00)

