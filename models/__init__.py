import torch


from enum import Enum
from torch import nn
from typing import Callable

from .loader import (
    __load_uni,
    __load_uni2h,
    __load_phikon,
    __load_phikon_v2,
    __load_h_optimus_0,
    __load_hibou_b,
    __load_hibou_L,
)
from .inference import (
    __extract_features_uni,
    __extract_features_uni2h,
    __extract_features_phikon,
    __extract_features_phikon_v2,
    __extract_features_h_optimus_0,
    __extract_features_hibou_b,
    __extract_features_hibou_L,
)


class FoundationModelEnum(Enum):
    """
    Enum for foundation model types.
    """

    # TODO review those enum values (strs). Should they be the HF path?
    UNI = "uni"
    UNI2H = "uni2h"
    PHIKON = "phikon"
    PHIKON_V2 = "phikon_v2"
    H_OPTIMUS_0 = "h-optimus-0"
    HIBOU_B = "hibou-b"
    HIBOU_L = "hibou-L"


_embedding_dims = {
    FoundationModelEnum.UNI: 1024,
    FoundationModelEnum.UNI2H: 1536,
    FoundationModelEnum.PHIKON: 768,
    FoundationModelEnum.PHIKON_V2: 1024,
    FoundationModelEnum.H_OPTIMUS_0: 1536,
    FoundationModelEnum.HIBOU_B: 768,
    FoundationModelEnum.HIBOU_L: 1024,
}

_loader_fns = {
    FoundationModelEnum.UNI: __load_uni,
    FoundationModelEnum.UNI2H: __load_uni2h,
    FoundationModelEnum.PHIKON: __load_phikon,
    FoundationModelEnum.PHIKON_V2: __load_phikon_v2,
    FoundationModelEnum.H_OPTIMUS_0: __load_h_optimus_0,
    FoundationModelEnum.HIBOU_B: __load_hibou_b,
    FoundationModelEnum.HIBOU_L: __load_hibou_L,
}

_inference_fns = {
    FoundationModelEnum.UNI: __extract_features_uni,
    FoundationModelEnum.UNI2H: __extract_features_uni2h,
    FoundationModelEnum.PHIKON: __extract_features_phikon,
    FoundationModelEnum.PHIKON_V2: __extract_features_phikon_v2,
    FoundationModelEnum.H_OPTIMUS_0: __extract_features_h_optimus_0,
    FoundationModelEnum.HIBOU_B: __extract_features_hibou_b,
    FoundationModelEnum.HIBOU_L: __extract_features_hibou_L,
}


def get_embedding_dim(model_type: FoundationModelEnum) -> int:
    """
    Returns the embedding dimension for the model type.
    """
    try:
        return _embedding_dims[model_type]
    except KeyError:
        raise NotImplementedError(f"Unknown model type: {model_type.value}")


def get_loader_fn(
    model_type: FoundationModelEnum,
) -> Callable[[], tuple[nn.Module, nn.Module]]:
    """
    Returns the model loading function for the model type.
    """
    try:
        return _loader_fns[model_type]
    except KeyError:
        raise NotImplementedError(f"Unknown model type: {model_type.value}")


def get_inference_fn(
    model_type: FoundationModelEnum,
) -> Callable[[torch.Tensor, nn.Module, nn.Module], torch.Tensor]:
    """
    Returns the inference function for the model type.
    """
    try:
        return _inference_fns[model_type]
    except KeyError:
        raise NotImplementedError(f"Unknown model type: {model_type.value}")
