#!/bin/bash
# This script is used to prompt GPT-3 to generate final answers.
# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --task)
      TASK="$2"
      shift 2;;
    --version)
      VERSION="$2"
      shift 2;;
    --examples_path)
      EXAMPLES_PATH="$2"
      shift 2;;
    --candidates_path)
      CANDIDATES_PATH="$2"
      shift 2;;
    --captions_path)
      CAPTIONS_PATH="$2"
      shift 2;;
    # --openai_key)
    #   OPENAI_KEY="$2"
    #   shift 2;;
    --qa_caption_path)
      QA_CAPTION_PATH="$2"
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

TASK=${TASK:-ok} # task name, one of ['ok', 'aok_val', 'aok_test'], default 'ok'
VERSION=${VERSION:-"prompt_okvqa"} # version name, default 'prompt_for_$TASK'
EXAMPLES_PATH=${EXAMPLES_PATH:-"assets/answer_aware_examples_okvqa.json"} # path to the examples, default is the result from our experiments
CANDIDATES_PATH=${CANDIDATES_PATH:-"assets/candidates_okvqa.json"} # path to the candidates, default is the result from our experiments
CAPTIONS_PATH=${CAPTIONS_PATH:-"assets/captions_okvqa.json"} # path to the captions, default is the result from our experiments
# OPENAI_KEY=${OPENAI_KEY:-""} # path to the captions
# SUBSET_RATIO=${SUBSET_RATIO:-}
# SUBSET_COUNT=${SUBSET_COUNT:-}
QA_CAPTION_PATH=${QA_CAPTION_PATH:-""} # path to the captions


python main.py \
    --task $TASK --run_mode prompt \
    --version $VERSION \
    --cfg configs/prompt.yml \
    --examples_path $EXAMPLES_PATH \
    --candidates_path $CANDIDATES_PATH \
    --captions_path $CAPTIONS_PATH \
    --qa_caption_path $QA_CAPTION_PATH \

    # --openai_key $OPENAI_KEY