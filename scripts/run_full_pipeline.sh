#!/usr/bin/env bash
set -euo pipefail

# Linux GPU server wrapper for the full EDU-CHEMC pipeline.
# Override defaults with env vars, and pass extra args through to the Python launcher.

STAGES=${STAGES:-all}
DATASET_ID=${DATASET_ID:-ConstantHao/EDU-CHEMC_MM23}
DATASET_DIR=${DATASET_DIR:-data/processed/edu_chemc_normed}
TARGET_FIELD=${TARGET_FIELD:-ssml_normed}
CONFIG=${CONFIG:-configs/train_edu_chemc.yaml}
PRETRAINED_MODEL=${PRETRAINED_MODEL:-OleehyO/TexTeller}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/runs/edu_chemc_texteller_normed_len768_r32_all_lora_balanced_30ep}
EVAL_OUTPUT_CSV=${EVAL_OUTPUT_CSV:-outputs/eval_normed_len768_r32_all_lora_balanced_30ep_test_greedy.csv}
GRAPH_MATCHING_TOOL_DIR=${GRAPH_MATCHING_TOOL_DIR:-external/GraphMatchingTool}
GRAPH_NUM_WORKERS=${GRAPH_NUM_WORKERS:-8}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
DTYPE=${DTYPE:-bf16}

uv run python scripts/run_edu_chemc_pipeline.py \
  --stages ${STAGES} \
  --dataset_id "${DATASET_ID}" \
  --dataset_dir "${DATASET_DIR}" \
  --target_field "${TARGET_FIELD}" \
  --config "${CONFIG}" \
  --pretrained_model_name_or_path "${PRETRAINED_MODEL}" \
  --output_dir "${OUTPUT_DIR}" \
  --eval_output_csv "${EVAL_OUTPUT_CSV}" \
  --graph_matching_tool_dir "${GRAPH_MATCHING_TOOL_DIR}" \
  --graph_num_workers "${GRAPH_NUM_WORKERS}" \
  --mixed_precision "${MIXED_PRECISION}" \
  --dtype "${DTYPE}" \
  "$@"
