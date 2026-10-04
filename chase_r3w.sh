#!/bin/bash
cd /home/user/jev-pro
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
echo "=== r3w warm continuation (resume r2, 3 epochs, aug, lr-enc 2e-5) ==="; date
python3 finetune.py --epochs 3 --batch 32 --lr-enc 2e-5 --lr-head 4e-4 \
  --train-aug data/train_aug.jsonl --resume ft_state.pt --outdir jev-pro-model-r3w
echo "FT_EXIT $?"
if [ -f jev-pro-model-r3w/ft_state.pt ]; then
  echo "=== noul + head on r3w embeddings ==="; date
  python3 noul_heads.py --outdir jev-pro-model-r3w --state jev-pro-model-r3w/ft_state.pt
  echo "=== re-embed aug+desc with r3w encoder ==="; date
  python3 prep_embed.py --outdir jev-pro-model-r3w --state jev-pro-model-r3w/ft_state.pt
  echo "=== boost on r3w embeddings ==="; date
  python3 boost.py --out boost_report_r3w.json
fi
echo "CHASE_DONE"; date
