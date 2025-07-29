"""
Module for abstracting model loading logic
"""

import logging

from huggingface_hub import login

from fm_model import FoundationModel, FoundationModelEnum
from models.loader import (
    __load_uni,
    __load_uni2h,
    __load_phikon,
    __load_phikon_v2,
)


def load_foundation_model(
    model_type: FoundationModelEnum, device: str | None = None, token: str | None = None
) -> FoundationModel:
    """
    Loads model specified by type. Agnostic to storage location (huggingface, etc.)

    Returns a pair (model, transform)

    :param model_type: FoundationModelEnum
    :param device: device to load the model on (e.g. "cuda" or "cpu"). If None, will not move the model to any device
    :param token: access token (e.g Hugging Face access token). Might be None
        (if None, will try to get from environment variables if necessary. For HF, this env var is `HF_TOKEN`)
        HF's User Access Token can be found at https://huggingface.co/settings/tokens
    :return model: nn.Module
    :return transform: nn.Module
    """
    if not device.startswith("cuda"):
        logging.warning(
            "Model will be loaded on CPU. If you want to use GPU, please specify `device='cuda'`"
        )
    login(token)
    source = "hf"  # for now only supports Hugging Face

    if model_type == FoundationModelEnum.UNI:
        model, transform = __load_uni()
    elif model_type == FoundationModelEnum.UNI2H:
        model, transform = __load_uni2h()
    elif model_type == FoundationModelEnum.PHIKON:
        model, transform = __load_phikon()
    elif model_type == FoundationModelEnum.PHIKON_V2:
        model, transform = __load_phikon_v2()
    else:
        raise NotImplementedError(
            f"Model '{model_type.value}' does not exist or was not implemented"
        )

    model = model.to(device) if device else model
    return FoundationModel(
        model_type=model_type,
        model_source=source,
        model=model,
        processor=transform,
        device=device or "cpu",
    )
