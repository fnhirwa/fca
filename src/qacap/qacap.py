"""Question-Aware Captioner.

`QACaptioner` exposes a minimal interface for generating question-conditioned
captions. It wraps InstructBLIP internally, but this is abstracted away from
the caller.

This file intentionally avoids importing anything from the vendored Prophet
code to keep separation. Stage2 can import and call this module without
side-effects. Fusion policies remain implemented inside the Prophet prompt
code (see docs section 9).
"""
import os
import json
from typing import Optional, Union, List, Dict, Any
from dataclasses import dataclass

from PIL import Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from tqdm import tqdm


token = os.getenv("HUGGINGFACE_TOKEN")

def load_instructblip_model(
        model_name: str,
        device: str = 'cpu',
        fp16: bool = False
    ) -> tuple[Optional[InstructBlipProcessor], Optional[InstructBlipForConditionalGeneration]]:
    """Load InstructBLIP model and processor.

    Parameters
    ----------
    model_name : str
        huggingface model identifier or local path.
    device : str, optional
        torch device string, by default 'cpu'
    fp16 : bool, optional
        whether to load the model in fp16 precision, by default False

    Returns
    -------
    tuple[Optional[InstructBlipProcessor], Optional[InstructBlipForConditionalGeneration]]
        processor and model instances, or (None, None) if loading fails.
    """
    try:
        dtype = torch.float16 if (device.startswith('cuda') and fp16) else torch.float32
        processor = InstructBlipProcessor.from_pretrained(model_name, token=token)
        model = InstructBlipForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype, token=token).to(device)
        return processor, model
    except Exception as e:
        print(f"[QACap] Error loading model {model_name}: {e}")
        return None, None

@dataclass
class QACaptionerConfig:
    """Configuration for QACaptioner."""
    model_name_or_path: str = 'Salesforce/instruct-blip-flan-t5-xl'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    fp16: bool = True
    max_new_tokens: int = 32
    temperature: float = 0.01
    include_question_in_fallback: bool = True # Whether to include the question in the fallback caption.


class QACaptioner:
    """Runtime wrapper for question-aware caption generation."""
    def __init__(
        self,
        config: Optional[QACaptionerConfig] = None,
        **kwargs
    ):
        if config is None:
            config = QACaptionerConfig(**kwargs)

        self.cfg = config
        self.processor, self.model = load_instructblip_model(
            self.cfg.model_name_or_path,
            device=self.cfg.device,
            fp16=self.cfg.fp16
        )

        if self.processor is None or self.model is None:
            self.using_stub = True
            print("[QACap] Warning: InstructBLIP model could not be loaded. Using stub implementation.")
        else:
            self.using_stub = False
            print(f"[QACap] Loaded InstructBLIP model: {self.cfg.model_name_or_path} on device {self.cfg.device}")

    def generate(
        self,
        image: Optional[Union[str, Image.Image]],
        question: str,
        base_caption: Optional[str] = None,
    ) -> dict[str, any]:
        """Generate a question-aware caption for a single image."""
        if self.using_stub or image is None:
            return self._heuristics(question=question, base_caption=base_caption)
        return self._inference(image=image, question=question, base_caption=base_caption)

    def generate_from_loader(self, dataloader) -> Dict[str, Dict[str, Any]]:
        """
        Generates captions for all samples in a dataloader and returns a dictionary
        mapping question_id to the caption and confidence.
        """
        results = {}
        print("Generating captions from dataloader...")
        for batch in tqdm(dataloader):
            if batch is None:
                continue
            
            # The dataloader from caption_dataset provides PIL images directly
            # when torch is not available or if not transformed.
            # For this captioner, we need the PIL image.
            # Let's assume the dataloader can provide it.
            # A robust implementation would fetch images from paths if not available.
            
            for i in range(len(batch['question'])):
                question_id = batch['question_id'][i].item()
                question = batch['question'][i]
                
                # To get the PIL image, we can reload it from the path,
                # which is inefficient but reliable.
                image_path = batch['image_path'][i]
                if not os.path.exists(image_path):
                    print(f"Warning: Image not found at {image_path}, skipping.")
                    continue
                
                image = Image.open(image_path).convert('RGB')

                result = self.generate(image=image, question=question)
                results[str(question_id)] = result
        
        return results

    def _inference(
        self,
        image: Union[str, Image.Image],
        question: str,
        base_caption: Optional[str] = None,
    ) -> dict[str, any]:
        """Run InstructBLIP inference to generate question-aware caption."""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')

        prompt = question
        if base_caption:
            prompt = f"Based on the fact that '{base_caption}', {question}"

        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.cfg.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                temperature=self.cfg.temperature,
                do_sample=False
            )
        caption = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        
        confidence = self._estimate_confidence(caption)
        return {'caption': caption, 'confidence': confidence}

    def _heuristics(
        self,
        question: Optional[str] = None,
        base_caption: Optional[str] = None,
    ) -> dict[str, any]:
        """Fallback heuristic caption generation."""
        if base_caption:
            caption = base_caption
            if self.cfg.include_question_in_fallback and question:
                caption += " | Question: " + question
        else:
            caption = "No caption available."
        confidence = 0.5  # Lower confidence for heuristic fallback
        return {'caption': caption, 'confidence': confidence}

    def _estimate_confidence(self, caption: str) -> float:
        """Estimate confidence score for the generated caption."""
        # Simple heuristic: longer captions are considered more confident
        length = len(caption.split())
        return min(1.0, max(0.1, length / 20.0))
