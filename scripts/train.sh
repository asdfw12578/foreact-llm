#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/autodl-tmp/ForeAct-LLM}
DATA_ROOT=${DATA_ROOT:-$ROOT/data}
LLAMA_PATH=${LLAMA_PATH:-$ROOT/data/weights/7B}
GPU=${GPU:-0}
EPOCHS=${EPOCHS:-25}
SAMPLE_RATE=${SAMPLE_RATE:-6}
N_QUERY=${N_QUERY:-20}
BATCH_SIZE=${BATCH_SIZE:-1}
ACCUM_ITER=${ACCUM_ITER:-4}
NUM_WORKERS=${NUM_WORKERS:-4}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-4096}
LR=${LR:-1e-4}
RUN_DIR=${RUN_DIR:-$ROOT/result/assembly101_$(date +%m%d_%H%M%S)}
RESUME=${RESUME:-}

cd "$ROOT"
mkdir -p "$RUN_DIR/weights" "$RUN_DIR/logs"

echo "[run_dir] $RUN_DIR"
echo "[gpu] $GPU"
echo "[epochs] $EPOCHS"
if [[ -n "$RESUME" ]]; then
  echo "[resume] $RESUME"
fi

RESUME_ARGS=()
if [[ -n "$RESUME" ]]; then
  RESUME_ARGS=(--resume "$RESUME")
fi

CUDA_VISIBLE_DEVICES="$GPU" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONUNBUFFERED=1 \
python -u train.py \
  --device cuda \
  --dataset assembly101 \
  --data_root "$DATA_ROOT" \
  --text_feature "$DATA_ROOT/text_feature" \
  --llama_model_path "$LLAMA_PATH" \
  --output_dir "$RUN_DIR/weights" \
  --log_dir "$RUN_DIR/logs" \
  --epochs "$EPOCHS" \
  --split 1 \
  --sample_rate "$SAMPLE_RATE" \
  --batch_size "$BATCH_SIZE" \
  --accum_iter "$ACCUM_ITER" \
  --num_workers "$NUM_WORKERS" \
  --pin_mem False \
  --adapter_dim 4 \
  --n_query "$N_QUERY" \
  --lr "$LR" \
  --clip_grad 1.0 \
  --multi_hidden_proj 128 \
  --max_seq_len "$MAX_SEQ_LEN" \
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
  --local_aux_boundary_pos_weight 5.0 \
  --local_aux_warmup_epochs 5 \
  --local_aux_start_epoch 0 \
  --local_aux_tail_ratio 0.3 \
  --sdr_resampler_enable \
  --sdr_resampler_num_tokens 8 \
  --sdr_resampler_layers 2 \
  --sdr_res_scale_init 1e-3 \
  "${RESUME_ARGS[@]}" \
  2>&1 | tee -a "$RUN_DIR/train.log"

echo "[done] log: $RUN_DIR/train.log"


