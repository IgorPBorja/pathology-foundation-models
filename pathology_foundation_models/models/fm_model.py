"""
Module for abstracting model loading logic and providing a unified interface
"""

import logging

from huggingface_hub import login
from dataclasses import dataclass
from torch import nn
from typing import Literal

from pathology_foundation_models.models.config import (
    FoundationModelEnum,
    get_embedding_dim,
    get_loader_fn,
)

class FoundationModel(nn.Module):
    def __init__(
        self,
        model_type: FoundationModelEnum,
        model_source: Literal["hf"],
        model: nn.Module,
        processor: nn.Module,
        device: str = 'cuda'
    ):
        """
        Foundation model wrapper class.

        :param model_type: Model identifier. Format depends on the source (e.g., Hugging Face model ID).
        :param model_source: Model source. Currently only supports 'hf' for Hugging Face.
        :param model: Model object (nn.Module). 
        :param processor: Preprocessing transform.
        :param device: Device where the FM is going to be loaded to. Default: 'cuda'.
        """
        super().__init__()
        self.model_type = model_type
        self.model_source = model_source
        self.model = model
        self.processor = processor
        self.device = device

    @property
    def embedding_dim(self) -> int:
        """Returns the embedding dimension of the model."""
        return get_embedding_dim(self.model_type)

    @property
    def name(self) -> str:
        """Returns the HF name of the model."""
        return self.model_type.value


def load_foundation_model(
    model_type: FoundationModelEnum | str,
    device: str | None = None,
    token: str | None = None,
) -> FoundationModel:
    """
    Loads model specified by type. Agnostic to storage location (huggingface, etc.)

    Returns a pair (model, transform)

    :param model_type: model type, either enum instance or its string representation. See docs for list.
    :param device: device to load the model on (e.g. "cuda" or "cpu"). If None, will not move the model to any device
    :param token: access token (e.g Hugging Face access token). Might be None
        (if None, will try to get from environment variables if necessary. For HF, this env var is `HF_TOKEN`. If this variable is not set, will prompt for it)
        HF's User Access Token can be found at https://huggingface.co/settings/tokens
    :return model: nn.Module
    :return transform: nn.Module
    """
    model_type: FoundationModelEnum = FoundationModelEnum(model_type)
    if not device or not device.startswith("cuda"):
        logging.warning(
            "Model will be loaded on CPU. If you want to use GPU, please specify `device='cuda'`"
        )
    login(token)
    source = "hf"  # for now only supports Hugging Face

    loader = get_loader_fn(model_type)
    model, transform = loader()
    model = model.to(device) if device else model
    return FoundationModel(
        model_type=model_type,
        model_source=source,
        model=model,
        processor=transform,
        device=device or "cpu",
    )
