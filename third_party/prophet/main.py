import sys
import os

# Ensure the local 'prophet' package is used, not an installed one.
# This is crucial for development to ensure code changes are reflected.
# We add the parent directory of 'prophet' to the path.
# The structure is assumed to be /path/to/fca/third_party/prophet/main.py
# So we want to add /path/to/fca/third_party to sys.path
prophet_dir = os.path.dirname(os.path.abspath(__file__))
third_party_dir = os.path.dirname(prophet_dir)
if third_party_dir not in sys.path:
    sys.path.insert(0, third_party_dir)


import argparse
import yaml
import torch

from evaluation.okvqa_evaluate import OKEvaluater
from evaluation.aokvqa_evaluate import AOKEvaluater
from configs.task_cfgs import Cfgs
from prophet import get_args, get_runner

# parse cfgs and args
args = get_args()
__C = Cfgs(args)
with open(args.cfg_file, 'r') as f:
    yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
__C.override_from_dict(yaml_dict)
print(__C)

# build runner
if __C.RUN_MODE == 'pretrain':
    evaluater = None
elif 'aok' in __C.TASK:
    evaluater = AOKEvaluater(
        __C.EVAL_ANSWER_PATH,
        __C.EVAL_QUESTION_PATH,
    )
else:
    evaluater = OKEvaluater(
        __C.EVAL_ANSWER_PATH,
        __C.EVAL_QUESTION_PATH,
    )

runner = get_runner(__C, evaluater)
# run
runner.run()
