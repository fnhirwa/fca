# ------------------------------------------------------------------------------ #
# Author: Zhenwei Shao (https://github.com/ParadoxZW)
# Description: Runner that handles the prompting process
# ------------------------------------------------------------------------------ #

import os, sys
# sys.path.append(os.getcwd())

import pickle
import json, time
import math
import random
import argparse
from datetime import datetime
from copy import deepcopy
import yaml
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
import torch


from .utils.fancy_pbar import progress, info_column
from .utils.data_utils import Qid2Data
from configs.task_cfgs import Cfgs

token = os.getenv("HUGGINGFACE_TOKEN")

class StopOnNewline(StoppingCriteria):
        def __init__(self, tokenizer):
            self.newline_id = tokenizer.encode('\n', add_special_tokens=False)[0]
        
        def __call__(self, input_ids, scores, **kwargs):
            return input_ids[0, -1] == self.newline_id

class Runner:
    def __init__(self, __C, evaluater):
        self.__C = __C
        self.evaluater = evaluater

        # LLaMA model
        self.model = None
        self.tokenizer = None
        self.initialize_llama_model(__C.MODEL_PATH)
    
    def initialize_llama_model(self, model_path):
        """Loads the LLaMA model and tokenizer."""
        print(f"Loading LLaMA model from: {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_auth_token=token,
            padding_side='left',  # ← CRITICAL FIX
        )
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,  # Extra compression
        )
        # pad token if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8 else torch.float16
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            torch_dtype=torch_dtype,
            device_map="auto",
            use_auth_token=token,
        )
        
        self.model.eval()
        print("LLaMA model loaded successfully.")

    def llama_infer(self, prompt_texts):
        """
        Batched inference for a list of prompts.
        """
        stopping_criteria = StoppingCriteriaList([StopOnNewline(self.tokenizer)])
        if isinstance(prompt_texts, str):
            prompt_texts = [prompt_texts]

        if self.__C.DEBUG:
            return ['debug_answer' for _ in prompt_texts]

        inputs = self.tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,  # Greedy decoding for deterministic results
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2,
                stopping_criteria=stopping_criteria, 
            )
        gen_start = inputs["input_ids"].shape[1]
        new_tokens = outputs[:, gen_start:]
        decoded = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        cleaned = []
        for text in decoded:
            first_line = text.strip().split('\n')[0]
            first_line = first_line.split('(')[0]
            # first word
            answer = first_line.strip().split()[0] if first_line.strip() else ""
            # trailing punct
            answer = answer.rstrip('.,;:!?')
            cleaned.append(answer)
        
        if len(cleaned) > 0:
            print(f"Sample output: {cleaned[0]}")
    
        return cleaned

    def gpt3_infer(self, prompt_text, _retry=0):
        # print(prompt_text)
        # exponential backoff
        if _retry > 0:
            print('retrying...')
            st = 2 ** _retry
            time.sleep(st)
        
        if self.__C.DEBUG:
            # print(prompt_text)
            time.sleep(0.05)
            return 0, 0

        try:
            # print('calling gpt3...')
            response = openai.Completion.create(
                engine=self.__C.MODEL,
                prompt=prompt_text,
                temperature=self.__C.TEMPERATURE,
                max_tokens=self.__C.MAX_TOKENS,
                logprobs=1,
                stop=["\n", "<|endoftext|>"],
                # timeout=20,
            )
            # print('gpt3 called.')
        except Exception as e:
            print(type(e), e)
            if str(e) == 'You exceeded your current quota, please check your plan and billing details.':
                exit(1)
            return self.gpt3_infer(prompt_text, _retry + 1)

        response_txt = response.choices[0].text.strip()
        # print(response_txt)
        plist = []
        for ii in range(len(response['choices'][0]['logprobs']['tokens'])):
            if response['choices'][0]['logprobs']['tokens'][ii] in ["\n", "<|endoftext|>"]:
                break
            plist.append(response['choices'][0]['logprobs']['token_logprobs'][ii])
        prob = math.exp(sum(plist))
        
        return response_txt, prob
    
    def sample_make(self, ques, capt, cands, ans=None):
        line_prefix = self.__C.LINE_PREFIX
        cands = cands[:self.__C.K_CANDIDATES]
        prompt_text = line_prefix + f'Context: {capt}\n'
        prompt_text += line_prefix + f'Question: {ques}\n'
        cands_with_conf = [f'{cand["answer"]}({cand["confidence"]:.2f})' for cand in cands]
        cands = ', '.join(cands_with_conf)
        prompt_text += line_prefix + f'Candidates: {cands}\n'
        prompt_text += line_prefix + 'Answer:'
        if ans is not None:
            prompt_text += f' {ans}'
        return prompt_text

    def get_context(self, example_qids):
        # making context text for one testing input
        prompt_text = self.__C.PROMPT_HEAD
        examples = []
        for key in example_qids:
            ques = self.trainset.get_question(key)
            caption = self.trainset.get_caption(key)
            cands = self.trainset.get_topk_candidates(key)
            gt_ans = self.trainset.get_most_answer(key)
            examples.append((ques, caption, cands, gt_ans))
            prompt_text += self.sample_make(ques, caption, cands, ans=gt_ans)
            prompt_text += '\n\n'
        return prompt_text
    
    def run(self):
        ## where logs will be saved
        Path(self.__C.LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(self.__C.LOG_PATH, 'w') as f:
            f.write(str(self.__C) + '\n')
        ## where results will be saved
        Path(self.__C.RESULT_DIR).mkdir(parents=True, exist_ok=True)
        
        self.cache = {}
        self.cache_file_path = os.path.join(
            self.__C.RESULT_DIR,
            'cache.json'
        )
        if self.__C.RESUME:
            self.cache = json.load(open(self.cache_file_path, 'r'))
        
        print('Note that the accuracies printed before final evaluation (the last printed one) are rough, just for checking if the process is normal!!!\n')
        self.trainset = Qid2Data(
            self.__C, 
            self.__C.TRAIN_SPLITS,
            True
        )
        self.valset = Qid2Data(
            self.__C, 
            self.__C.EVAL_SPLITS,
            self.__C.EVAL_NOW,
            json.load(open(self.__C.EXAMPLES_PATH, 'r'))
        )

        # if 'aok' in self.__C.TASK:
        #     from evaluation.aokvqa_evaluate import AOKEvaluater as Evaluater
        # else:
        #     from evaluation.okvqa_evaluate import OKEvaluater as Evaluater
        # evaluater = Evaluater(
        #     self.valset.annotation_path,
        #     self.valset.question_path
        # )

        infer_times = self.__C.T_INFER
        N_inctx = self.__C.N_EXAMPLES
        
        print()

        batch_size = getattr(self.__C, "BATCH_SIZE", 64)
        prompt_batch, meta_batch = [], []

        for qid in progress.track(self.valset.qid_to_data, description="Working...  "):
            if qid in self.cache:
                continue

            ques = self.valset.get_question(qid)
            caption = self.valset.get_caption(qid)
            cands = self.valset.get_topk_candidates(qid, self.__C.K_CANDIDATES)
            prompt_query = self.sample_make(ques, caption, cands)
            example_qids = self.valset.get_similar_qids(qid, k=infer_times * N_inctx)
            random.shuffle(example_qids)

            # collect multiple inference variants for this qid
            for t in range(infer_times):
                prompt_in_ctx = self.get_context(example_qids[(N_inctx * t):(N_inctx * t + N_inctx)])
                prompt_text = prompt_in_ctx + prompt_query
                prompt_batch.append(prompt_text)
                meta_batch.append((qid, t))

                # run batched inference when batch full
                if len(prompt_batch) >= batch_size:
                    responses = self.llama_infer(prompt_batch)
                    for (qid_b, t_b), gen_text in zip(meta_batch, responses):
                        ans = self.evaluater.prep_ans(gen_text)
                        gen_prob = 1.0
                        prompt_info = {
                            "prompt": prompt_batch[t_b % batch_size],
                            "answer": gen_text,
                            "confidence": gen_prob,
                        }

                        # initialize if not exists
                        if qid_b not in self.cache:
                            self.cache[qid_b] = {
                                "question_id": qid_b,
                                "answer": "",
                                "prompt_info": [],
                            }
                        self.cache[qid_b]["prompt_info"].append(prompt_info)

                    # flush
                    prompt_batch, meta_batch = [], []

        # flush any remaining prompts
        if prompt_batch:
            responses = self.llama_infer(prompt_batch)
            for (qid_b, t_b), gen_text in zip(meta_batch, responses):
                ans = self.evaluater.prep_ans(gen_text)
                gen_prob = 1.0
                prompt_info = {
                    "prompt": prompt_batch[t_b % batch_size],
                    "answer": gen_text,
                    "confidence": gen_prob,
                }
                if qid_b not in self.cache:
                    self.cache[qid_b] = {
                        "question_id": qid_b,
                        "answer": "",
                        "prompt_info": [],
                    }
                self.cache[qid_b]["prompt_info"].append(prompt_info)

        # vote
        for qid, record in self.cache.items():
            ans_pool = {}
            for pinfo in record["prompt_info"]:
                ans = self.evaluater.prep_ans(pinfo["answer"])
                if ans:
                    ans_pool[ans] = ans_pool.get(ans, 0.0) + pinfo["confidence"]

            if len(ans_pool) == 0:
                answer = self.valset.get_topk_candidates(qid, 1)[0]["answer"]
            else:
                answer = sorted(ans_pool.items(), key=lambda x: x[1], reverse=True)[0][0]

            record["answer"] = answer
            self.evaluater.add(qid, answer)

        json.dump(self.cache, open(self.cache_file_path, "w"))
        self.evaluater.save(self.__C.RESULT_PATH)

        if self.__C.EVAL_NOW:
            with open(self.__C.LOG_PATH, 'a+') as logfile:
                self.evaluater.evaluate(logfile)
        
def prompt_login_args(parser):
    parser.add_argument('--debug', dest='DEBUG', help='debug mode', action='store_true')
    parser.add_argument('--resume', dest='RESUME', help='resume previous run', action='store_true')
    parser.add_argument('--task', dest='TASK', help='task name, e.g., ok, aok_val, aok_test', type=str, required=True)
    parser.add_argument('--version', dest='VERSION', help='version name', type=str, required=True)
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', type=str, default='configs/prompt.yml')
    parser.add_argument('--examples_path', dest='EXAMPLES_PATH', help='answer-aware example file path, default: "assets/answer_aware_examples_for_ok.json"', type=str, default=None)
    parser.add_argument('--candidates_path', dest='CANDIDATES_PATH', help='candidates file path, default: "assets/candidates_for_ok.json"', type=str, default=None)
    parser.add_argument('--captions_path', dest='CAPTIONS_PATH', help='captions file path, default: "assets/captions_for_ok.json"', type=str, default=None)
    # parser.add_argument('--openai_key', dest='OPENAI_KEY', help='openai api key', type=str, default=None)
    parser.add_argument('--model_path', dest='MODEL_PATH', 
                        help='path or Hugging Face ID for LLaMA model', 
                        type=str, default=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Heuristics-enhanced Prompting')
    prompt_login_args(parser)
    args = parser.parse_args()
    __C = Cfgs(args)
    with open(args.cfg_file, 'r') as f:
        yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
    __C.override_from_dict(yaml_dict)
    print(__C)

    runner = Runner(__C)
    runner.run()
