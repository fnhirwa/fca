# FCA / Prophet Pipeline Guide (Stage1 & Stage2)

This guide explains the key code paths and data structures needed to understand how to integrate **question‑aware captioning** into the existing Prophet-based pipeline. It focuses on:

1. Configuration object `__C` (`configs/task_cfgs.py` + YAML overrides)
2. Stage1 data loading and heuristics generation
3. Stage2 prompting data assembly
4. Subsetting & filtering logic (recent additions)
5. Safe integration points for custom caption logic
6. Extension checklist

---
## 1. Configuration Object (`__C`)

Source: `third_party/prophet/configs/task_cfgs.py` (class `Cfgs` + path helpers in `path_cfgs.py`, splits in `task_to_split.py`).

`__C` centralizes runtime parameters. Construction flow:
1. Parse CLI args (top-level `prophet/__init__.py` or script local arg parser)
2. Instantiate `Cfgs(args)` → sets core attributes, GPU context, task/run mode.
3. Load YAML (`finetune.yml`, `pretrain.yml`, `prompt.yml`, etc.) and call `__C.override_from_dict()`.

Important attribute:

| Category | Examples |
|----------|----------|
| Tasking & Mode | `TASK`, `RUN_MODE`, `TRAIN_SPLITS`, `EVAL_SPLITS`, `FEATURE_SPLIT` |
| Paths | `CKPTS_DIR`, `RESULTS_ROOT`, `CANDIDATE_FILE_PATH`, `EXAMPLE_FILE_PATH`, `ANSWER_LATENTS_DIR` |
| Data Sizes | `IMG_FEAT_GRID`, `IMG_FEAT_SIZE`, `MAX_TOKEN` |
| Heuristics / Prompt | `CANDIDATE_NUM`, `EXAMPLE_NUM`, `K_CANDIDATES`, `N_EXAMPLES`, `T_INFER` |
| Subsetting | `SUBSET_RATIO`, `SUBSET_COUNT` (mutually exclusive CLI flags) |
| Model / Inference | `MODEL`, `TEMPERATURE`, `MAX_TOKENS`, `SLEEP_PER_INFER` |

Derivation helpers:
- `TRAIN_SPLITS` / `EVAL_SPLITS`: resolved via `TASK_TO_SPLIT` table.
- `FEATURE_SPLIT`: union of image splits needed by train + eval.

When adding new behavior (e.g., dynamic captioning), prefer adding a **new config key** in YAML + CLI flag instead of hard-coding.

---
## 2. Stage1: Data Loading & Heuristics

Key files:
- `stage1/utils/load_data.py` → `CommonData`, `DataSet`
- `stage1/heuristics.py` → `Runner.eval()` + `Runner.run()`

### 2.1 CommonData
Builds shared resources:
1. Image feature index: glob all `*.npz` from paths in `__C.FEATS_DIR[split]` for each split in `FEATURE_SPLIT`. Populates `imgid_to_path`.
2. Tokenizer (HuggingFace): loaded from `__C.BERT_VERSION`.
3. Answer vocabulary: loads `ans_to_ix` / `ix_to_ans` from JSON chosen by `__C.DATA_TAG`.

### 2.2 DataSet
For each requested split list (`TRAIN_SPLITS` or `EVAL_SPLITS`):
1. Load question JSON(s) (`QUESTION_PATH`), flatten `questions` if present.
2. Load annotations if available (`ANSWER_PATH`).
3. Derive `self.qids` from available questions or answers (ensures string IDs).
4. Filter out qids whose `image_id` lacks a feature file (recent patch).
5. Apply subsetting (ratio or count) if provided.
6. `__getitem__` returns `(img_feat, ques_ids, ans_vec)`:
   - Image features reshaped `[GRID^2, FEAT_SIZE]`.
   - Tokenized question (CLS + tokens + SEP padded to `MAX_TOKEN`).
   - Soft target vector across answer vocabulary if annotated.

### 2.3 Heuristics Generation Flow
`Runner.eval(dataset)`:
1. Lazy-load `MCANForFinetune` checkpoint on first call.
2. Iterate DataLoader: forward pass returns raw logits + latent answer features.
3. For each sample:
   - Sort answers by confidence, keep top `k = __C.CANDIDATE_NUM`.
   - Save per-question latent vector to `ANSWER_LATENTS_DIR`.
4. Return mapping: `qid -> [ {answer, confidence}, ... ]`, plus latent list.

`Runner.run()` post-processing:
1. Run eval on train and eval sets → produce union `candidates.json`.
2. Compute similarity: normalize latent vectors and build cosine scores from test to train.
3. Produce `examples.json`: `qid -> [similar_train_qids]` (top `__C.EXAMPLE_NUM`).
4. Empty split guard: if train or eval latent list is empty, skip similarity stage (recent patch).

Artifact expectations:
- `candidates.json` required downstream by stage2.
- `examples.json` optional (empty lists are accepted after filtering).

---
## 3. Stage2: Prompting Data Assembly

Key files:
- `stage2/utils/data_utils.py` → `Qid2Data`
- `stage2/prompt.py` → `Runner` (now using LLAMA or HF backends)

### 3.1 Qid2Data Construction
Given splits + paths in `__C`:
1. Load questions and (optional) annotations.
2. Load `candidates.json` (qid → candidate answers).
3. Load `captions.json` (image_id → plain caption strings).
4. Filter to qids present in both question set and candidates and whose `image_id` has a caption.
5. Apply subsetting (ratio or count) after filtering.
6. Attach ground-truth scoring fields if annotated (`most_answer`, `gt_scores`).
7. Integrate similar examples:
   - Filter `similar_qids` to existing qids.
   - Replace missing with empty list (no failure).

Each `qid_to_data[qid]` contains:
```json
{
  "question_id": <str>,
  "image_id": <str>,
  "question": <str>,
  "topk_candidates": [ {"answer": str, "confidence": float}, ... ],
  "caption": <str>,
  "similar_qids": [<qid>, ...],
  "most_answer": <str>,        // annotated only
  "gt_scores": {answer: score} // annotated only
}
```

### 3.2 Prompt Assembly
Inside `prompt.py`:
1. Build few-shot context from `similar_qids` (chunks of size `N_EXAMPLES`).
2. For each target qid:
   - Compose context + query with `sample_make()`.
   - Call `gpt3_infer()` (or patched local model) multiple times `T_INFER`.
   - Aggregate logprobs → a voting mechanism over generated answers (`ans_pool`).
3. Fallback: if no answers, use top candidate from candidates list.
4. Real-time accuracy logging if evaluation enabled.
5. Cache intermediate prompts to `cache.json` for resume support.

### 3.3 Temperature & Decoding
Config in `prompt.yml` sets `TEMPERATURE`. For local LLaMA/HF models ensure:
- Use small positive value (e.g. 0.01) for near-greedy sampling **OR** set `do_sample=False` in your backend.

---
## 4. Subsetting & Filtering Summary

Motivation: fast iteration for new captioning logic without full dataset cost.

Points:
- **Stage1 `DataSet`**: filters missing image features → applies subset.
- **Heuristics Runner**: guards empty splits, writes empty `examples.json` gracefully.
- **Stage2 `Qid2Data`**: filters missing candidates/captions → applies subset → prunes similar_qids.
- **Feature Extraction** (`tools/extract_img_feats.py`): per-split subset applied before saving `.npz` files.

Rules:
1. Pass **only one** of `--subset_ratio` or `--subset_count` (enforced by mutually exclusive argparse group and shell script checks).
2. Subsetting always occurs **after** filtering to avoid referencing missing data.

---
## 5. Integrating Question‑Aware Captioning

Goal: Generate or refine captions conditioned on the **current question** (and optionally candidates or similar examples) before prompting.

### 5.1 Potential Hook Points
| Hook | Location | When | Pros | Cons |
|------|----------|------|------|------|
| A. Pre-Stage1 Caption Generation | After feature extraction (new script) | Before heuristics | Consistent across pipeline | Requires recomputing captions when logic changes |
| B. Dynamic Caption Override in Stage2 | In `Qid2Data` after load | Prompt-time | Easy iteration; no extra files | Must ensure deterministic caching if needed |
| C. Context-Augmented Captions | In `Runner.get_context()` | Per few-shot example | Can fuse similar examples + candidates | More compute per inference |
| D. Candidate-Aware Caption Refinement | Before calling `sample_make()` | Per target qid | Tailors caption to answer space | Risk of leakage/bias in evaluation |

### 5.2 Minimal Implementation (Option B)
Add a callable `caption_enhancer(question:str, base_caption:str, candidates:list, similar_qids:list, __C)` and patch `Qid2Data` after base caption assignment:

```python
# pseudo inside Qid2Data loop
enh = getattr(__C, 'CAPTION_ENHANCER', None)
if enh:
    enhanced = enh(q_item['question'], caption, t_item, [])
    if enhanced:
        caption = enhanced
```

Then in a custom module, define:
```python
def question_aware_caption(question, base_caption, candidates, similar_qids, cfg):
    # candidate nouns
    answers = [c['answer'] for c in candidates]
    # Heuristic merge
    hint = ', '.join(answers[:3])
    if hint:
        return f"{base_caption} Relevant concepts: {hint}."
    return base_caption
```
Inject via:
```python
from my_module import question_aware_caption as qa_cap
__C.CAPTION_ENHANCER = qa_cap
```


### 5.4 Cautions
- Avoid leaking ground-truth answers into captions unless intentionally training.
- Maintain reproducibility: log new caption strings to result logs.
- If captions influence candidate ranking upstream, document the dependency.
- Keep temperature/search parameters stable when benchmarking.

### 5.5 Evaluation Considerations
- Compare baseline vs enhanced captions using identical seeds (`__C.SEED`).
- Track impact on: top1 accuracy, confidence distribution, variance across runs.
- Optionally add a JSONL trace: per-qid {question, base_caption, enhanced_caption, chosen_answer}.

---
## 6. Extension Checklist

- [ ] Decide hook strategy (A/B/C above).
- [ ] Add config flags: e.g. `USE_QA_CAPTION`, `QA_CAPTION_MODE`.
- [ ] Implement enhancer function or override method.
- [ ] Log enhanced captions (for audit & reproducibility).
- [ ] Preserve original caption for ablation comparison.
- [ ] Validate small subset run (`--subset_count 50`) end-to-end.
- [ ] Scale to full splits; monitor latency impact.
- [ ] Add unit test for enhancer (input shapes / null outputs).

---
## 7. Symbols & Files

| Symbol / File | Purpose |
|---------------|---------|
| `__C` | Global experiment configuration object |
| `CommonData` | Shared image features, tokenizer, answer vocab |
| `DataSet.__getitem__` | Returns image feature grid, token ids, answer vector |
| `heuristics.Runner.eval` | Generates top-k answer candidates + latent vectors |
| `examples.json` | Similar question IDs per eval qid |
| `Qid2Data` | Assembles per-qid data for prompting |
| `prompt.Runner.gpt3_infer` | LLM interaction with retries |
| `sample_make` | Formats one prompt example block |
| `SUBSET_RATIO / SUBSET_COUNT` | CLI-controlled subsetting (mutually exclusive) |
| `CAPTION_ENHANCER` (proposed) | Injected function to build question-aware captions |

---
## 8. Next Steps
1. Implement a prototype `CAPTION_ENHANCER` using heuristic noun extraction.
2. Add logging of enhanced vs base caption in `cache.json`.
3. Run subset benchmark; compare accuracy deltas.
4. Iterate with linguistic enrichment (e.g., external knowledge snippets) once stability verified.


---
## 9. QACap + Heuristic Fusion + LLaMA 3 Prompting (Proposed Wiring)

This maps the provided flow diagram into concrete steps and switches you can adopt without breaking the current pipeline. The goal is to optionally use a Question‑Aware Captioner (InstructBLIP) and fuse its caption with Prophet’s heuristics to construct a stronger prompt for LLaMA 3, controlled by a single toggle for clean A/B benchmarking.

### 9.1 Modules and data flow

- QACap (InstructBLIP): consumes (image, question) → returns a question‑conditioned caption.
- Prophet Stage1 (Heuristics): generates candidate answers (+ confidences) and similar training examples.
- Heuristic Fusion Layer (new): merges QACap caption with baseline caption and heuristics according to a policy.
- Stage2 Prompt Composer: formats the final prompt for LLaMA 3‑Instruct.
- LLM: LLaMA 3 (local or API) returns the final answer.

### 9.2 Config toggles (YAML + CLI)

- USE_QACAP: true|false — master toggle for enabling QACap. When false, baseline behavior is preserved.
- QACAP_MODE: "offline"|"online" — precompute to JSON (offline) or compute at prompt‑time (online).
- QACAP_MODEL: e.g., instructblip‑vicuna7b (or your exact HF model id).
- QACAP_DEVICE: cuda:0 | cpu; QACAP_FP16: true|false.
- QACAP_MAX_NEW_TOKENS: 32; QACAP_TEMPERATURE: 0.01 (or set do_sample=false in the wrapper).
- FUSION_POLICY: "append" | "prepend" | "replace_if_confident" | "ensemble".
- FUSION_WEIGHT: 0.7 (used by replace_if_confident or as a confidence threshold).
- PROMPT_TEMPLATE: "llama3" — selects formatting compatible with LLaMA 3‑Instruct.

CLI example for results with QACap enabled and ensemble fusion:

```
--use_qacap true --qacap_model instructblip-vicuna7b --fusion_policy ensemble
```

### 9.3 QACap interface contract

Input:
- image: path or preloaded tensor
- question: string

Output:
- { "caption": str, "confidence": float (optional) }

Errors/timeouts should fall back to the baseline caption and log a warning; never crash the run.

### 9.4 Where to compute QACap

Option A — Offline (reproducible): write `assets/qacap_<task>.json` mapping qid→{caption, confidence}. Stage2 loads this file when `USE_QACAP=true` and fuses.

Option B — Online (fast iteration): call QACap inside Stage2 (e.g., in `Qid2Data` or `Runner.sample_make`) and cache to `outputs/logs/qacap_cache.json`.

JSON sketch (offline):
```
{
  "123456": {"caption": "the boy holding a red frisbee", "confidence": 0.82, "image_id": "COCO_val2014_000000123456"},
  ...
}
```

### 9.5 Heuristic Fusion Layer

Policy options:
- append: final = base_caption + " " + qacap
- prepend: final = qacap + " " + base_caption
- replace_if_confident: if qacap.confidence ≥ FUSION_WEIGHT then use qacap else base
- ensemble: final = f"{qacap}. Relevant concepts: {top_k_candidates}."

Pseudo‑code:
```python
def fuse_caption(base_caption, qacap_obj, policy, weight=0.7, topk=None):
   if not qacap_obj or not qacap_obj.get('caption'):
      return base_caption
   qc = qacap_obj['caption']
   if policy == 'append':
      return f"{base_caption} {qc}".strip()
   if policy == 'prepend':
      return f"{qc} {base_caption}".strip()
   if policy == 'replace_if_confident':
      conf = qacap_obj.get('confidence', 1.0)
      return qc if conf >= weight else base_caption
   if policy == 'ensemble':
      add = ''
      if topk:
         answers = ', '.join(a['answer'] for a in topk[:3])
         add = f" Relevant concepts: {answers}."
      return f"{qc}{add}"
   return base_caption
```

Use the fused value wherever `caption` is read during prompt construction in Stage2.

### 9.6 Prompt Composer for LLaMA 3

Keep prompts short and structured. A simple, LLaMA 3‑friendly template:

```
System: You are a helpful VQA assistant. Use the context and candidate answers to answer concisely.

User:
Context: {fused_caption}
Question: {question}
Candidates: {answer1(conf), answer2(conf), ...}
Answer:
```

### 9.7 A/B benchmarking toggle

- Baseline Prophet captions: `--use_qacap false`
- QACap enabled: `--use_qacap true` (plus model and policy args)

Record: Same results structure as before, with added fields in logs for `qacap_caption` and `fusion_policy` used.
---