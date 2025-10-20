#!/bin/bash
# This script is used to extract image features.

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      GPU="$2"
      shift 2;;
    --dataset)
      DATASET="$2"
      shift 2;;
    --clip)
      CLIP_MODEL="$2"
      shift 2;;
    --subset_ratio)
      if [[ -z "$2" || "$2" == --* ]]; then echo "--subset_ratio requires a value"; exit 2; fi
      SUBSET_RATIO="$2"
      shift 2;;
    --subset_count)
      if [[ -z "$2" || "$2" == --* ]]; then echo "--subset_count requires a value"; exit 2; fi
      SUBSET_COUNT="$2"
      shift 2;;
    *)
      echo "Unknown argument: $1"
      exit 1;;
  esac
done

DATASET=${DATASET:-ok} # dataset name, one of ['ok', 'aok'], default 'ok'
GPU=${GPU:-0} # GPU id(s) you want to use, default '0'
CLIP_MODEL=${CLIP_MODEL:-RN50x64} # clip model name or path, default 'RN50x64'
SUBSET_RATIO=${SUBSET_RATIO:-}
SUBSET_COUNT=${SUBSET_COUNT:-}

# Enforce mutual exclusivity in shell (optional safety):
if [[ -n "$SUBSET_RATIO" && -n "$SUBSET_COUNT" ]]; then
  echo "Provide only one of --subset_ratio or --subset_count"; exit 2;
fi

# CUDA_VISIBLE_DEVICES=$GPU \
python tools/extract_img_feats.py \
    --dataset $DATASET --gpu $GPU \
  --clip_model $CLIP_MODEL \
  ${SUBSET_RATIO:+--subset_ratio $SUBSET_RATIO} \
  ${SUBSET_COUNT:+--subset_count $SUBSET_COUNT}