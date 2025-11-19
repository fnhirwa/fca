"""Question-Aware Caption Generation

Provides a light wrapper (`QACaptioner`) around an optional InstructBLIP
model. The goal is to expose a stable interface that Stage2 (Prophet prompt
construction) can call when `USE_QACAP` is enabled.

High-level usage:
    captioner = QACaptioner(model_name_or_path, device)
    result = captioner.generate_captions(image=img_or_path, questions=questions, batch_size=batch_size)
    captions = result['captions']
    confidence = result.get('confidence', 1.0)
"""

from .qacap import QACaptioner
from .qacap_dataset import create_caption_dataloader

__all__ = ['QACaptioner', 'create_caption_dataloader']