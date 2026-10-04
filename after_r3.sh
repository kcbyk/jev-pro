#!/bin/bash
cd /home/user/jev-pro
while ! grep -q "^EXIT " r3.log 2>/dev/null; do sleep 20; done
if grep -q "^EXIT 0" r3.log; then
  export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
  { echo "=== noul_heads on r3 ==="; date
    python3 noul_heads.py --outdir jev-pro-model-r3 --state jev-pro-model-r3/ft_state.pt
    echo "=== prep_embed with r3 encoder (aug + desc in r3 space) ==="; date
    python3 prep_embed.py --outdir jev-pro-model-r3 --state jev-pro-model-r3/ft_state.pt
    echo "=== boost on r3 embeddings ==="; date
    python3 boost.py --out boost_report_r3.json
    echo "AFTER_R3_DONE"; date
  } > r3_post.log 2>&1
else
  echo "AFTER_R3_SKIPPED: r3 exited nonzero" > r3_post.log
fi
