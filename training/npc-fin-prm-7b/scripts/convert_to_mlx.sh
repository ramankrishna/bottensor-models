#!/usr/bin/env bash
# Convert the merged fp16 PRM to MLX 4-bit format.
# Runs after merge_and_convert.py completes successfully.
set -euo pipefail

cd /tmp/finprm-mlx

if [ ! -f merged/config.json ]; then
  echo "ERROR: /tmp/finprm-mlx/merged not ready yet — run merge_and_convert.py first"
  exit 1
fi

if [ -d mlx-q4 ] && [ -f mlx-q4/config.json ]; then
  echo "MLX 4-bit dir already exists at /tmp/finprm-mlx/mlx-q4 — skipping convert"
  exit 0
fi

# Convert via mlx_lm.convert (group_size 64, 4-bit, ~5GB output)
python3 -m mlx_lm convert \
  --hf-path ./merged \
  --mlx-path ./mlx-q4 \
  --quantize \
  --q-bits 4 \
  --q-group-size 64

echo
echo "DONE."
du -sh mlx-q4
ls -lh mlx-q4
