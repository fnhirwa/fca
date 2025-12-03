# fca
From Captions to Answers: Enhancing Prophet with Question-Aware Captioning for Knowledge-Based VQA

## Third-party code: Prophet (vendored)

We vendor the upstream Prophet repository under `third_party/prophet` to reuse most of its implementation and extend it with question‑aware captioning.

- Upstream: https://github.com/MILVLG/prophet (default branch: `main`)
- Vendored path: `third_party/prophet`
- License: see `third_party/prophet/LICENSE` (ensure attribution and license notices are preserved when distributing)

### Update Prophet to latest upstream

Use Git subtree to pull updates while keeping all code in this repository.

Option A: one‑off command

```
git subtree pull --prefix third_party/prophet https://github.com/MILVLG/prophet main --squash
```

Option B: via helper script

```
scripts/vendor_sync_prophet.sh # pulls from main by default
```

You can specify a different branch or tag:

```
scripts/vendor_sync_prophet.sh v1.0.0
scripts/vendor_sync_prophet.sh prophet++
```

Notes:

- Squashing keeps the repo small while still allowing future pulls.
- If you make local modifications to `third_party/prophet`, document them clearly in commit messages to ease future merges.


## Running Experiment for OKVQA

- Download all dataset and pretrained models as mentioned in [Prophet](third_party/prophet/README.md)

- Run Feature extraction 
```bash
bash scripts/extract_img_feats.sh --dataset ok --gpu 
```
- Generate Heuristics
```bash
bash scripts/heuristics_gen.sh --task ok --version okvqa_heuristics_1 --gpu 0 --ckpt_path outputs/ckpts/mcan_ft_okvqa.pkl --candidate_num 5 --example_num 10 
```
- Prompting
```bash
bash scripts/prompt.sh --task ok --version okvqa_prompt_1 --examples_path outputs/results/okvqa_heuristics_1/examples.json --candidates_path outputs/results/okvqa_heuristics_1/candidates.json 
```


## Understanding project structure
- `third_party/prophet/`: Vendored Prophet codebase.
- `scripts/`: Helper scripts for data processing, training, and evaluation.
- `src/`: Source code for our extensions and modifications to Prophet.

### third_party/prophet/
Refer to `third_party/prophet/README.md` for detailed documentation on the Prophet codebase.

#### prophet is made of two stages:
1. Heuristic Generation: Generate question-aware captions using a pretrained VQA model and captioning model.
2. Prompting: Use the generated captions to prompt a large language model for final answer

### QACap Integration

The core extension of this project is the integration of a **Question-Aware Captioning (QACap)** module. This module generates captions that are conditioned on the question, providing more relevant context to the downstream language model.

The implementation is in `src/qacap/qacap.py`, which provides a `QACaptioner` class wrapping an InstructBLIP model.

#### Workflow

1.  **QACap Module**: The `QACaptioner` takes an image and a question and generates a descriptive caption focused on details relevant to the question. It can be used in two modes:
    *   **Offline**: Pre-generate captions for the entire dataset and save them as a JSON file.
    *   **Online**: Generate captions at runtime during the prompting stage.

2.  **Configuration**: The QACap integration is controlled via configuration in the YAML files and can be toggled with CLI flags. Key settings include:
    *   `USE_QACAP`: A master switch to enable or disable the QACap module.
    *   `QACAP_MODE`: "offline" or "online" generation.
    *   `FUSION_POLICY`: A policy to combine the generated question-aware caption with the baseline caption from the original dataset. Policies include:
        *   `append`: Add the new caption after the original.
        *   `prepend`: Add the new caption before the original.
        *   `replace_if_confident`: Replace the original caption if the QACap model is confident.
        *   `ensemble`: Combine the QACap caption with heuristic concepts from Prophet's stage 1.

3.  **Prompting**: In Stage 2 (prompting), the fused caption is used to construct the final prompt for the large language model (e.g., LLaMA 3), providing richer, more focused context for answer generation.

This approach allows for flexible A/B testing between the baseline Prophet pipeline and the enhanced version with question-aware captioning.