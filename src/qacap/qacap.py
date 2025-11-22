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
from typing import Optional, Union, Any
from dataclasses import dataclass

from PIL import Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration, GenerationConfig
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
    model_name_or_path: str = 'Salesforce/instructblip-flan-t5-xxl'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    fp16: bool = True
    max_new_tokens: int = 256
    include_question_in_fallback: bool = True # Whether to include the question in the fallback caption.
    do_sample=False,
    num_beams=5,
    max_length=256,
    min_length=1,
    top_p=0.9,
    repetition_penalty=1.5,
    length_penalty=1.0,
    temperature=1,


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

    def _get_prompt(self, question: str) -> str:
        """Constructs the prompt"""
        return f"Generate a descriptive image caption. The caption should focus only on the visual details in the image that are relevant to the following topic: {question}"

    def generate(
        self,
        image: Optional[Union[str, Image.Image]],
        question: str,
        generation_config: Optional[GenerationConfig] = None
    ) -> str:
        """Generate a question-aware caption for a single image."""
        if self.using_stub or image is None:
            return self._heuristics(question=question)
        return self._inference(image=image, question=question, generation_config=generation_config)

    def generate_from_loader(self, dataloader) -> dict[str, str]:
        """
        Generates captions for all samples in a dataloader and returns a dictionary
        mapping question_id to the caption string.
        """
        results = {}
        print("Generating captions from dataloader...")
        for batch in tqdm(dataloader):
            if batch is None:
                continue

            images = batch['image']
            questions = batch['question']
            question_ids = batch['question_id']

            batch_captions = self.generate_batch(
                images=images,
                questions=questions,
            )
            
            for i, caption in enumerate(batch_captions):
                question_id = question_ids[i].item()
                results[str(question_id)] = caption
        
        return results

    def _inference(
        self,
        image: Union[str, Image.Image],
        question: str,
        generation_config: Optional[GenerationConfig] = None
    ) -> str:
        """Run InstructBLIP inference to generate question-aware caption."""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')

        prompt = self._get_prompt(question)

        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.cfg.device)
        
        if generation_config is None:
            generation_config = GenerationConfig(
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=False,
                num_beams=self.cfg.num_beams,
                max_length=self.cfg.max_length,
                min_length=self.cfg.min_length,
                top_p=self.cfg.top_p,
                repetition_penalty=self.cfg.repetition_penalty,
                length_penalty=self.cfg.length_penalty,
                temperature=self.cfg.temperature,
            )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config
            )
        caption = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        
        return caption

    def generate_batch(
        self,
        images: Union[list[str], list[Image.Image], torch.Tensor],
        questions: list[str],
    ) -> list[str]:
        """Generate question-aware captions for a batch of images."""
        if self.using_stub:
            return [self._heuristics(q) for q in questions]

        prompts = [self._get_prompt(q) for q in questions]

        inputs = self.processor(images=images, text=prompts, return_tensors="pt", padding=True).to(self.cfg.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                temperature=self.cfg.temperature,
                do_sample=False
            )
        
        captions = self.processor.batch_decode(outputs, skip_special_tokens=True)
        
        return [caption.strip() for caption in captions]

    def _heuristics(
        self,
        question: Optional[str] = None,
    ) -> str:
        """Fallback heuristic caption generation."""
        caption = "No caption available."
        if self.cfg.include_question_in_fallback and question:
            caption += " | Question: " + question
        return caption
