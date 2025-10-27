#!/usr/bin/env bash

# set -euo pipefail

ACT_SCALES="act_scales/opt-1.3b.pt"
alpha=0.85
model="facebook/opt-1.3b"

if [[ ! -f "$ACT_SCALES" ]]; then
  mkdir -p "$(dirname "$ACT_SCALES")"
  python examples/generate_act_scales.py \
    --model-name $model \
    --output-path "$ACT_SCALES" \
    # --n-samples 2048
fi

python smoothquant/ppl_eval.py \
  --alpha $alpha \
  --model_path $model \
  --act_scales_path "$ACT_SCALES" \
  --hardware_yaml hardwareconfig/hardware4.yaml \
  --search_mode grid \
  --wquantization fmadse \
  --datatype fmadsedontcare \
  --group_size 128 \
  --results_path results_mod/demo_results4.txt \
  --results_db results_mod/quant_results4.db \
  --quantize \
  --smooth \
  --random_seed 42
