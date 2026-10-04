#!/bin/bash
cd /home/user/jev-pro
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
{ echo "=== noul on r3w (state bug fixed) ==="; date
  python3 noul_heads.py --outdir jev-pro-model-r3w --state jev-pro-model-r3w/ft_state.pt
  echo NOUL_EXIT $?
  echo "=== prep_embed with r3w encoder ==="; date
  python3 prep_embed.py --outdir jev-pro-model-r3w --state jev-pro-model-r3w/ft_state.pt
  echo PREP_EXIT $?
  echo "=== boost on r3w embeddings ==="; date
  python3 boost.py --out boost_report_r3w.json
  echo BOOST_EXIT $?
  echo CHASE2_DONE; date
} > chase2.log 2>&1
