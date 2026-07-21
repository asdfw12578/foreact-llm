#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/autodl-tmp/ForeAct-LLM}
DATA_ROOT=${DATA_ROOT:-$ROOT/data}
LLAMA_PATH=${LLAMA_PATH:-$ROOT/data/weights/7B}
RUN_DIR=${RUN_DIR:-$ROOT/result/assembly101}
GPU=${GPU:-0}
START_EPOCH=${START_EPOCH:-0}
END_EPOCH=${END_EPOCH:-24}
SAMPLE_RATE=${SAMPLE_RATE:-8}
N_QUERY=${N_QUERY:-12}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-3072}

cd "$ROOT"
mkdir -p "$RUN_DIR/eval_logs"

CUDA_VISIBLE_DEVICES="$GPU" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONUNBUFFERED=1 \
python -u eval.py \
  --device cuda \
  --dataset assembly101 \
  --data_root "$DATA_ROOT" \
  --text_feature "$DATA_ROOT/text_feature" \
  --llama_model_path "$LLAMA_PATH" \
  --checkpoint_dir "$RUN_DIR/weights" \
  --eval_start_epoch "$START_EPOCH" \
  --eval_end_epoch "$END_EPOCH" \
  --eval_csv "$RUN_DIR/eval_logs/moc.csv" \
  --sample_rate "$SAMPLE_RATE" \
  --n_query "$N_QUERY" \
  --max_seq_len "$MAX_SEQ_LEN" \
  --batch_size 1 \
  --num_workers 4 \
  --pin_mem False \
  --adapter_dim 4 \
  --multi_hidden_proj 128 \
  --bits 16bit \
  --disable_text_features \
  --gtda_enable \
  --gtda_layers 4 \
  --gtda_kernel_size 3 \
  --gtda_dropout 0.1 \
  --gtda_res_scale_init 1e-3 \
  --gtda_causal \
  --local_aux_enable \
  --local_aux_action_weight 0.05 \
  --local_aux_boundary_weight 0.0 \
  --local_aux_tail_ratio 0.3 \
  --sdr_resampler_enable \
  --sdr_resampler_num_tokens 8 \
  --sdr_resampler_layers 2 \
  --sdr_res_scale_init 1e-3 \
  2>&1 | tee -a "$RUN_DIR/eval_logs/eval.log"
