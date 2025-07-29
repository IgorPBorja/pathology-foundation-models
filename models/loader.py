"""
Implements model-specific loading functions for foundation models.
"""

import timm
import torch

from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch import nn
from transformers import AutoModel, AutoImageProcessor, ViTModel


def __load_uni() -> tuple[nn.Module, nn.Module]:
    """
    --> See https://huggingface.co/MahmoodLab/UNI

    Loads the UNI model from Hugging Face to CPU.

    DON'T use this function directly

    :return: model, transform
    """
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
    return model, transform


def __load_uni2h() -> tuple[nn.Module, nn.Module]:
    """
    --> See https://huggingface.co/MahmoodLab/UNI2-h

    Loads the UNI2-h model from Hugging Face to CPU.

    DON'T use this function directly

    :return: model, transform
    """
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
    return model, transform


def __load_phikon() -> tuple[nn.Module, nn.Module]:
    """
    --> See https://github.com/owkin/HistoSSLscaling/

    Loads the Phikon model from Hugging Face to CPU.

    DON'T use this function directly

    :return: model, transform
    """
    image_processor = AutoImageProcessor.from_pretrained("owkin/phikon")
    model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
    model.eval()
    return model, image_processor


def __load_phikon_v2() -> tuple[nn.Module, nn.Module]:
    """
    --> See https://huggingface.co/owkin/phikon-v2

    Loads the Phikon v2 model from Hugging Face to CPU.

    DON'T use this function directly
    """
    transform = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
    model = AutoModel.from_pretrained("owkin/phikon-v2")
    model.eval()
    return model, transform
