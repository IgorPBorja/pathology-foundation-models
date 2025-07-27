"""
Module for abstracting model loading logic
"""

import timm
import torch

from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login
from transformers import AutoModel, AutoImageProcessor

from dataset.fm_model import FoundationModel


def load_foundation_model(
    model_id: str, device: str | None = None, token: str | None = None
) -> FoundationModel:
    """
    Loads model from :model_id:. Agnostic to storage location (huggingface, etc.)

    Returns a pair (model, transform)

    :param model_id: str
    :param device: device to load the model on (e.g. "cuda" or "cpu"). If None, will not move the model to any device
    :param token: access token (e.g Hugging Face access token). Might be None
        (if None, will try to get from environment variables if necessary. For HF, this env var is `HF_TOKEN`)
        HF's User Access Token can be found at https://huggingface.co/settings/tokens
    :return model: nn.Module
    :return transform: nn.Module
    """
    if model_id == "MahmoodLab/UNI":
        # --> See https://huggingface.co/MahmoodLab/UNI
        login(token)
        # pretrained=True needed to load UNI weights (and download weights for the first time)
        # init_values need to be passed in to successfully load LayerScale parameters (e.g. - block.0.ls1.gamma)
        model = timm.create_model(
            "hf-hub:MahmoodLab/uni",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True,
        )
        transform = create_transform(
            **resolve_data_config(model.pretrained_cfg, model=model)
        )
        model.eval()
        source = "hf"
    elif model_id == "MahmoodLab/UNI2-h":
        # --> See https://huggingface.co/MahmoodLab/UNI2-h
        login(token)
        # pretrained=True needed to load UNI2-h weights (and download weights for the first time)
        timm_kwargs = {
            "img_size": 224,
            "patch_size": 14,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-5,
            "embed_dim": 1536,
            "mlp_ratio": 2.66667 * 2,
            "num_classes": 0,
            "no_embed_class": True,
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
            "reg_tokens": 8,
            "dynamic_img_size": True,
        }
        model = timm.create_model(
            "hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs
        )
        transform = create_transform(
            **resolve_data_config(model.pretrained_cfg, model=model)
        )
        model.eval()
        source = "hf"
    elif model_id == "owkin/phikon-v2":
        # --> See https://huggingface.co/owkin/phikon-v2
        transform = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
        model = AutoModel.from_pretrained("owkin/phikon-v2")
        model.eval()
        source = "hf"
    else:
        raise NotImplementedError(
            f"Model '{model_id}' does not exist or was not implemented"
        )

    model = model.to(device) if device else model
    return FoundationModel(
        model_id=model_id,
        model_source=source,
        model=model,
        processor=transform,
        device=device or "cpu",
    )
