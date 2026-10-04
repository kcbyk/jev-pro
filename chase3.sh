#!/bin/bash
cd /home/user/jev-pro
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
{ echo "=== r3w2: clean-aug full recipe (base -> 6 ep, fit 8000 + 16000 clean aug, val untouched) ==="; date
  python3 finetune.py --epochs 6 --batch 32 --lr-enc 6e-5 --lr-head 8e-4 \
    --train-aug data/train_aug.jsonl --outdir jev-pro-model-r3w2 > r3w2_ft.log 2>&1
  echo FT_EXIT $?; tail -1 r3w2_ft.log
  if [ -f jev-pro-model-r3w2/ft_state.pt ]; then
    echo "=== compress r3w2 state to fp16 (snapshot budget) ==="
    python3 - <<'EOF'
import torch
st = torch.load("jev-pro-model-r3w2/ft_state.pt", map_location="cpu", weights_only=True)
torch.save({k: (v.half() if v.dtype==torch.float32 else v) for k, v in st.items()}, "jev-pro-model-r3w2/ft_state_fp16.pt")
print("fp16 written")
EOF
    echo "=== noul + r3w2 embeddings (state: fp32 best kept until chain end) ==="; date
    python3 noul_heads.py --outdir jev-pro-model-r3w2 --state jev-pro-model-r3w2/ft_state.pt
    echo NOUL_EXIT $?
    echo "=== prep_embed with r3w2 encoder ==="; date
    python3 prep_embed.py --outdir jev-pro-model-r3w2 --state jev-pro-model-r3w2/ft_state.pt
    echo PREP_EXIT $?
    echo "=== boost on r3w2 embeddings ==="; date
    python3 boost.py --out boost_report_r3w2.json
    echo BOOST_EXIT $?
    echo CHASE3_DONE; date
  fi
} > chase3.log 2>&1
