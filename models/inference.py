"""
Implement model-specific inference functions for foundation models.
"""

import torch

from torch import nn


def __extract_features_uni(
    images: torch.Tensor,
    model: nn.Module,
    transform: nn.Module,
) -> torch.Tensor:
    """
    --> See https://huggingface.co/MahmoodLab/UNI

    Extracts features from images using the UNI model.

    DON'T use this function directly

    :param images: Batch tensor of shape (N, 3, H, W)
    :param model: The UNI model
    :return: Extracted features
    """
    image_tensor = transform(images)
    with torch.inference_mode():
        features = model(image_tensor)
    return features


def __extract_features_uni2h(
    images: torch.Tensor,
    model: nn.Module,
    transform: nn.Module,
) -> torch.Tensor:
    """
    --> See https://huggingface.co/MahmoodLab/UNI2-h

    Extracts features from images using the UNI2-h model.

    DON'T use this function directly

    :param images: Batch tensor of shape (N, 3, H, W)
    :param model: The UNI2-h model
    :return: Extracted features
    """
    image_tensor = transform(images)
    with torch.inference_mode():
        features = model(image_tensor)
    return features


def __extract_features_phikon(
    images: torch.Tensor,
    model: nn.Module,
    transform: nn.Module,
) -> torch.Tensor:
    """
    --> See https://github.com/owkin/HistoSSLscaling/

    Extracts features from images using the Phikon model.

    DON'T use this function directly

    :param images: Batch tensor of shape (N, 3, H, W)
    :param model: The Phikon model
    :return: Extracted features
    """
    # process the image
    inputs = transform(images, return_tensors="pt")
    # cast back to original device
    inputs = {k: v.to(images.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        features = outputs.last_hidden_state[:, 0, :]
    return features


def __extract_features_phikon_v2(
    images: torch.Tensor,
    model: nn.Module,
    transform: nn.Module,
) -> torch.Tensor:
    """
    --> See https://huggingface.co/owkin/phikon-v2

    Extracts features from images using the Phikon v2 model.

    DON'T use this function directly

    :param images: Batch tensor of shape (N, 3, H, W)
    :param model: The Phikon v2 model
    :return: Extracted features
    """
    inputs = transform(images, return_tensors="pt")
    # cast back to original device
    inputs = {k: v.to(images.device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)
        features = outputs.last_hidden_state[:, 0, :]
    return features
