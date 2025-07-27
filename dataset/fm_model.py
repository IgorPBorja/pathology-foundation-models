from dataclasses import dataclass
from torch import nn
from typing import Literal


@dataclass
class FoundationModel:
    model_id: str
    """Model identifier. Format depends on the source (e.g., Hugging Face model ID)."""
    model_source: Literal["hf"]
    """Model source. Currently only supports 'hf' for Hugging Face."""
    model: nn.Module
    processor: nn.Module
    """Preprocessing transform"""
    device: str
