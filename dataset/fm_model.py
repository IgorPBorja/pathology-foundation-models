from dataclasses import dataclass
from torch import nn
from typing import Literal

from enum import Enum


class FoundationModelEnum(Enum):
    """
    Enum for foundation model types.
    """

    UNI = "uni"
    UNI2H = "uni2h"
    PHIKON = "phikon"
    PHIKON_V2 = "phikon_v2"

    @property
    def embedding_dim(self) -> int:
        """
        Returns the embedding dimension for the model type.
        """
        if self == FoundationModelEnum.UNI:
            return 1024
        elif self == FoundationModelEnum.UNI2H:
            return 1536
        elif self == FoundationModelEnum.PHIKON:
            return 768
        elif self == FoundationModelEnum.PHIKON_V2:
            return 1024
        else:
            raise ValueError(f"Unknown model type: {self.value}")


@dataclass
class FoundationModel:
    model_type: FoundationModelEnum
    """Model identifier. Format depends on the source (e.g., Hugging Face model ID)."""
    model_source: Literal["hf"]
    """Model source. Currently only supports 'hf' for Hugging Face."""
    model: nn.Module
    processor: nn.Module
    """Preprocessing transform"""
    device: str
