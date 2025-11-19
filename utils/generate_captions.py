"""
This script generates question-aware captions for a given dataset using the QACaptioner.
It is designed to be used as a preliminary step before running the main Prophet pipeline,
to generate heuristic captions that can be used for prompt construction.

This script mirrors the argument parsing and configuration handling of the main Prophet
scripts to ensure seamless integration.
"""
import os
import sys
import argparse
import yaml
import json

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.qacap.qacap import QACaptioner, QACaptionerConfig
from src.qacap.qacap_dataset import create_caption_dataloader
from third_party.prophet.configs.task_cfgs import Cfgs

def main():
    # --- Argument Parsing and Configuration ---
    parser = argparse.ArgumentParser(description="Generate Question-Aware Captions")
    parser.add_argument('--task', dest='TASK', help='task name, e.g., ok, aok_val, aok_test', type=str, required=True)
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', type=str, required=True)
    parser.add_argument('--gpu', dest='GPU', help='gpu id', type=str, default='0')
    parser.add_argument('--output_path', dest='OUTPUT_PATH', help='path to save generated captions', type=str, default='generated_captions.json')
    parser.add_argument('--batch_size', dest='BATCH_SIZE', help='batch size for caption generation', type=int, default=8)
    # Subset options for quick workflow tests
    parser.add_argument('--subset_ratio', dest='SUBSET_RATIO', help='use only this fraction of data (0-1]', type=float, default=None)
    parser.add_argument('--subset_count', dest='SUBSET_COUNT', help='use only this many samples', type=int, default=None)

    args = parser.parse_args()
    __C = Cfgs(args)
    with open(args.cfg_file, 'r') as f:
        yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
    __C.override_from_dict(yaml_dict)
    
    # Manually set the prophet path to resolve relative paths in the config
    __C.PROPHET_PATH = 'fca/third_party/prophet'

    print("Loaded configuration:")
    print(__C)

    # --- Dataloader Creation ---
    # The splits to generate captions for are typically the same as the eval splits for heuristics
    splits_to_process = __C.EVAL_SPLITS
    print(f"Creating dataloader for splits: {splits_to_process}")

    # Note: The transform is handled internally by the QACaptioner's processor,
    # so we don't need to specify one for the dataloader.
    dataloader = create_caption_dataloader(
        __C,
        split_name_list=splits_to_process,
        batch_size=args.BATCH_SIZE,
        shuffle=False
    )

    # --- Captioner Initialization ---
    qacap_config = QACaptionerConfig(
        device=f'cuda:{args.GPU}' if torch.cuda.is_available() else 'cpu'
    )
    captioner = QACaptioner(config=qacap_config)

    # --- Caption Generation ---
    if dataloader.dataset.data_size > 0:
        generated_captions = captioner.generate_from_loader(dataloader)

        # --- Save Results ---
        # Construct the output path relative to the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        output_dir = os.path.join(parent_dir, "capres")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, args.OUTPUT_PATH)

        print(f"\nSaving {len(generated_captions)} generated captions to {output_path}...")
        with open(output_path, 'w') as f:
            json.dump(generated_captions, f, indent=4)
        print("Done.")
    else:
        print("Dataset is empty, no captions to generate.")

if __name__ == '__main__':
    # Need to import torch here for the cuda check in QACaptionerConfig
    """
    Usage:
    python fca/utils/generate_captions.py --task <task_name> --cfg <config_file> --gpu <gpu_id> --output_path <output_file> --batch_size <batch_size>

    python generate_captions.py \
    --task ok \
    --cfg fca/third_party/prophet/configs/finetune.yml \
    --subset_count 10 \
    --output_path generated_captions_test.json
    """
    import torch
    main()
