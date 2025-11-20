# ------------------------------------------------------------------------------ #
# Author: Zhenwei Shao (https://github.com/ParadoxZW)
# Description: dataset utils for stage2
# ------------------------------------------------------------------------------ #

import json
from typing import Dict
import pickle
from collections import Counter
import sys
import os

# allow imports from fca.src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../..')))
from fca.src.fuse.fuse import fuse_caption_with_heuristics

# following two score is rough, and only for print accuracies during inferring.
def ok_score(gt_answers):
    gt_answers = [a['answer'] for a in gt_answers]
    ans2cnt = Counter(gt_answers)
    # sort
    ans2cnt = sorted(ans2cnt.items(), key=lambda x: x[1], reverse=True)
    ans2score = {}
    for ans, cnt in ans2cnt:
        # ans2score[ans] = min(1.0, cnt / 3.0)
        if cnt == 1:
            ans2score[ans] = 0.3
        elif cnt == 2:
            ans2score[ans] = 0.6
        elif cnt == 3:
            ans2score[ans] = 0.9
        else:
            ans2score[ans] = 1.0
    return ans2score

def aok_score(gt_answers):
    gt_answers = [a for a in gt_answers]
    ans2cnt = Counter(gt_answers)
    # sort
    ans2cnt = sorted(ans2cnt.items(), key=lambda x: x[1], reverse=True)
    ans2score = {}
    for ans, cnt in ans2cnt:
        # ans2score[ans] = min(1.0, cnt / 3.0)
        if cnt == 1:
            ans2score[ans] = 1 / 3.
        elif cnt == 2:
            ans2score[ans] = 2 / 3.
        else:
            ans2score[ans] = 1.
    return ans2score


class Qid2Data(Dict):
    def __init__(self, __C, splits, annotated=False, similar_examples=None):
        super().__init__()

        self.__C = __C
        self.annotated = annotated
        # assert(0==1), f"Using QA caption: {getattr(self.__C, 'USE_QACAP', False)}"
        
        ques_set = []
        for split in splits:
            split_path = self.__C.QUESTION_PATH[split]
            _ques_set = json.load(open(split_path, 'r'))
            if 'questions' in _ques_set:
                _ques_set = _ques_set['questions']
            ques_set += _ques_set
        qid_to_ques = {str(q['question_id']): q for q in ques_set}

        if annotated:
            anno_set = []
            for split in splits:
                split_path = self.__C.ANSWER_PATH[split]
                _anno_set = json.load(open(split_path, 'r'))
                if 'annotations' in _anno_set:
                    _anno_set = _anno_set['annotations']
                anno_set += _anno_set
            qid_to_anno = {str(a['question_id']): a for a in anno_set}
        
        qid_to_topk = json.load(open(self.__C.CANDIDATES_PATH))
        # qid_to_topk = {t['question_id']: t for t in topk}

        iid_to_capt = json.load(open(self.__C.CAPTIONS_PATH))
        
        # Load question-aware captions if enabled
        qid_to_qacapt = {}
        # manually set the fca root path
        self.__C.FCA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..'))
        if getattr(self.__C, 'USE_QACAP', False) and getattr(self.__C, 'QA_CAPTION_PATH', None):
            qa_path = getattr(self.__C, 'QA_CAPTION_PATH', None)
            if qa_path and not os.path.isabs(qa_path):
                qa_path = os.path.join(self.__C.FCA_PATH, qa_path)
            try:
                with open(qa_path, 'r') as f:
                   qid_to_qacapt = json.load(f)
                print(f"Successfully loaded {len(qid_to_qacapt)} question-aware captions.")
            except FileNotFoundError:
                print(f"Warning: QA Captions file not found at {qa_path}. Proceeding without them.")
        _score = aok_score if 'aok' in __C.TASK else ok_score
        
        qid_to_data = {}
        # ques_set = ques_set['questions']
        # anno_set = anno_set['annotations']
        for qid in qid_to_ques:
            q_item = qid_to_ques[qid]
            t_item = qid_to_topk[qid]

            iid = str(q_item['image_id'])
            caption = iid_to_capt[iid].strip()
            if caption and caption[-1] != '.':
                caption += '.'

            # If requested, fuse the QA-aware caption into the original caption.
            if getattr(__C, 'USE_QACAP', False):
                qa_caption = qid_to_qacapt.get(qid, '')
                # assert 0==1, f"Fusing QA captions original caption: {caption} with QA caption for qid {qa_caption}"
                caption = fuse_caption_with_heuristics(
                    original_caption=caption,
                    qa_caption=qa_caption,
                    strategy=getattr(__C, 'QACAP_FUSION_STRATEGY', 'prepend')
                )
            
            qid_to_data[qid] = {
                'question_id': qid,
                'image_id': iid,
                'question': q_item['question'],
                # 'most_answer': most_answer,
                # 'gt_scores': ans2score,
                'topk_candidates': t_item,
                'caption': caption,
            }
            if annotated:
                a_item = qid_to_anno[qid]
                if 'answers' in a_item:
                    answers = a_item['answers']
                else:
                    answers = a_item['direct_answers']

                ans2score = _score(answers)

                most_answer = list(ans2score.keys())[0]
                if most_answer == '':
                    most_answer = list(ans2score.keys())[1]
                
                qid_to_data[qid]['most_answer'] = most_answer
                qid_to_data[qid]['gt_scores'] = ans2score

        self.qid_to_data = qid_to_data

        k = self.__C.K_CANDIDATES
        if annotated:
            print(f'Loaded dataset size: {len(self.qid_to_data)}, top{k} accuracy: {self.topk_accuracy(k)*100:.2f}, top1 accuracy: {self.topk_accuracy(1)*100:.2f}')
        
        if similar_examples:
            for qid in similar_examples:
                qid_to_data[qid]['similar_qids'] = similar_examples[qid]
            
            # check if all items have similar_qids
            for qid, item in self.items():
                if 'similar_qids' not in item:
                    raise ValueError(f'qid {qid} does not have similar_qids')
        
        

    def __getitem__(self, __key):
        return self.qid_to_data[__key]
    

    def get_caption(self, qid):
        caption = self[qid]['caption']
        # if with_tag:
        #     tags = self.get_tags(qid, k_tags)
        #     caption += ' ' + ', '.join(tags) + '.'
        return caption
    
    def get_question(self, qid):
        return self[qid]['question']
    
    
    def get_gt_answers(self, qid):
        if not self.annotated:
            return None
        return self[qid]['gt_scores']
    
    def get_most_answer(self, qid):
        if not self.annotated:
            return None
        return self[qid]['most_answer']

    def get_topk_candidates(self, qid, k=None):
        if k is None:
            return self[qid]['topk_candidates']
        else:
            return self[qid]['topk_candidates'][:k]
    
    def get_similar_qids(self, qid, k=None):
        similar_qids = self[qid]['similar_qids']
        if k is not None:
            similar_qids = similar_qids[:k]
        return similar_qids
    
    def evaluate_by_threshold(self, ans_set, threshold=1.0):
        if not self.annotated:
            return -1
        
        total_score = 0.0
        for item in ans_set:
            qid = item['question_id']
            topk_candidates = self.get_topk_candidates(qid)
            top1_confid = topk_candidates[0]['confidence']
            if top1_confid > threshold:
                answer = topk_candidates[0]['answer']
            else:
                answer = item['answer']
            gt_answers = self.get_gt_answers(qid)
            if answer in gt_answers:
                total_score += gt_answers[answer]
        return total_score / len(ans_set)
    
    def topk_accuracy(self, k=1, sub_set=None):
        if not self.annotated:
            return -1
        
        total_score = 0.0
        if sub_set is not None:
            qids = sub_set
        else:
            qids = list(self.qid_to_data.keys())
        for qid in qids:
            topk_candidates = self.get_topk_candidates(qid)[:k]
            gt_answers = self.get_gt_answers(qid)
            score_list = [gt_answers.get(a['answer'], 0.0) for a in topk_candidates]
            total_score += max(score_list)
        return total_score / len(qids)
    
    def rt_evaluate(self, answer_set):
        if not self.annotated:
            return ''
        score1 = self.evaluate_by_threshold(answer_set, 1.0) * 100
        score2 = self.evaluate_by_threshold(answer_set, 0.0) * 100
        score_string = f'{score2:.2f}->{score1:.2f}'
        return score_string
