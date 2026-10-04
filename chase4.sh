#!/bin/bash
cd /home/user/jev-pro
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
{ echo "=== d1: distilbert-base-uncased fine-tune (66M, no aug, 4 ep, bs16) ==="; date
  python3 finetune.py --model distilbert/distilbert-base-uncased --epochs 4 --batch 16 --lr-enc 5e-5 \
    --lr-head 5e-4 --outdir jev-pro-model-d1 > d1_ft.log 2>&1
  echo D1_EXIT $?
  tail -1 d1_ft.log
} > chase4.log 2>&1
