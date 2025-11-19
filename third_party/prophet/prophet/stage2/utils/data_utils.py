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
        
        qid_to_topk = json.load(open(__C.CANDIDATES_PATH))
        # qid_to_topk = {t['question_id']: t for t in topk}

        iid_to_capt = json.load(open(__C.CAPTIONS_PATH))
        
        # Load question-aware captions if enabled
        qid_to_qacapt = {}
        if getattr(__C, 'USE_QACAP', False) and getattr(__C, 'QA_CAPTIONS_PATH', None):
            try:
                with open(__C.QA_CAPTIONS_PATH, 'r') as f:
                    qid_to_qacapt = json.load(f)
                print(f"Successfully loaded {len(qid_to_qacapt)} question-aware captions.")
            except FileNotFoundError:
                print(f"Warning: QA Captions file not found at {__C.QA_CAPTIONS_PATH}. Proceeding without them.")
        
        _score = aok_score if 'aok' in __C.TASK else ok_score
        
        # Filter to only process questions that have all required data
        available_qids = set(qid_to_ques.keys()) & set(qid_to_topk.keys())
        # Also filter by image IDs that have captions
        available_iids = set(iid_to_capt.keys())
        
        orig_ques_count = len(qid_to_ques)
        qid_to_ques_filtered = {
            qid: q for qid, q in qid_to_ques.items() 
            if qid in available_qids and str(q['image_id']) in available_iids
        }
        dropped = orig_ques_count - len(qid_to_ques_filtered)
        if dropped > 0:
            print(f"Warning: Dropped {dropped} questions due to missing captions or candidates.")
        
        qid_to_data = {}
        # ques_set = ques_set['questions']
        # anno_set = anno_set['annotations']
        for qid in qid_to_ques_filtered:
            ques = qid_to_ques[qid]
            iid = str(ques['image_id'])
            
            # Fuse captions if USE_QACAP is True
            original_caption = iid_to_capt.get(iid, {}).get('caption', '')
            qa_caption = qid_to_qacapt.get(qid, '')
            
            if getattr(__C, 'USE_QACAP', False):
                fused_caption = fuse_caption_with_heuristics(
                    original_caption=original_caption,
                    qa_caption=qa_caption,
                    strategy=getattr(__C, 'QACAP_FUSION_STRATEGY', 'prepend')
                )
            else:
                fused_caption = original_caption

            data = {
                'question': ques['question'],
                'image_id': ques['image_id'],
                'caption': fused_caption,
                'topk_cand': qid_to_topk[qid],
            }
            if annotated:
                data['gt_answers'] = qid_to_anno[qid]['answers']
                data['gt_scores'] = _score(data['gt_answers'])
                data['most_answer'] = max(data['gt_scores'], key=data['gt_scores'].get)
            
            if similar_examples:
                data['similar_qids'] = similar_examples[qid]

            qid_to_data[qid] = data

        self.qid_to_data = qid_to_data

        # Optional subsetting for quick workflow testing
        subset_count = getattr(__C, 'SUBSET_COUNT', None)
        subset_ratio = getattr(__C, 'SUBSET_RATIO', None)
        qid_list = list(self.qid_to_data.keys())
        orig_size = len(qid_list)
        
        if subset_count is None and subset_ratio is not None:
            if subset_ratio <= 0 or subset_ratio > 1:
                raise ValueError('SUBSET_RATIO must be in (0, 1].')
            subset_count = max(1, int(len(qid_list) * subset_ratio))
        if subset_count is not None:
            subset_count = min(subset_count, len(qid_list))
            qid_list = qid_list[:subset_count]
            # filter qid_to_data to only include subset
            self.qid_to_data = {qid: self.qid_to_data[qid] for qid in qid_list}
            print(f'== Applied subsetting: {len(self.qid_to_data)}/{orig_size} samples')

        k = __C.K_CANDIDATES
        if annotated:
            print(f'Loaded dataset size: {len(self.qid_to_data)}, top{k} accuracy: {self.topk_accuracy(k)*100:.2f}, top1 accuracy: {self.topk_accuracy(1)*100:.2f}')
        
        if similar_examples:
            # Filter similar_examples to only include qids present in qid_to_data
            valid_qids = set(self.qid_to_data.keys())
            filtered_count = 0
            
            for qid in similar_examples:
                if qid in valid_qids:
                    # Also filter the referenced similar_qids to only those present in qid_to_data
                    raw_similar = similar_examples[qid]
                    filtered_similar = [sim_qid for sim_qid in raw_similar if sim_qid in valid_qids]
                    if len(filtered_similar) < len(raw_similar):
                        filtered_count += 1
                    self.qid_to_data[qid]['similar_qids'] = filtered_similar
            
            if filtered_count > 0:
                print(f'== Filtered similar_qids for {filtered_count} questions due to subsetting/missing data')
            
            # Warn about items without similar_qids (but don't fail)
            missing_similar = [qid for qid in self.qid_to_data if 'similar_qids' not in self.qid_to_data[qid]]
            if missing_similar:
                print(f'== Warning: {len(missing_similar)} questions have no similar_qids (example: {missing_similar[0]})')
                # Set empty list for those without similar examples
                for qid in missing_similar:
                    self.qid_to_data[qid]['similar_qids'] = []
        
        

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
