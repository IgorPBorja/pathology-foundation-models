import torch
import torchvision.transforms as T

from PIL.Image import Image

from .dataset.fm_model import FoundationModel


def convert_to_batch_tensor(image: Image | torch.Tensor) -> torch.Tensor:
    """
    Converts a single PIL image or a torch tensor of shape (3, H, W) to a batch tensor of shape (1, 3, H, W).

    If the input is already a batch tensor of shape (N, 3, H, W), it will be returned as is.

    :param image: PIL Image or torch.Tensor of shape (3, H, W) or torch Tensor of shape (N, 3, H, W)
    :return: Batch tensor of shape (1, 3, H, W)
    """
    if isinstance(image, Image):
        transform = T.ToTensor()
        image = transform(image).unsqueeze(0)  # Add batch dimension
    elif isinstance(image, torch.Tensor):
        if image.dim() == 3:
            image = image.unsqueeze(0)  # Add batch dimension
        elif image.dim() != 4:
            raise ValueError("Input tensor must be of shape (3, H, W) or (N, 3, H, W)")
    else:
        raise TypeError("Input must be a PIL Image or a torch Tensor")

    assert (
        image.dim() == 4
    ), "Unexpected return shape, expected (1, 3, H, W) or (N, 3, H, W)"
    return image


def extract_features(
    image: Image | torch.Tensor, model: FoundationModel
) -> torch.Tensor:
    """
    Extracts features from single PIL image using the specified model.

    :param image: PIL Image or torch.Tensor of shape (3, H, W) or torch Tensor of shape (N, 3, H, W)
    :param format: either 'pil' or 'torch'
    :param model: FoundationModel instance containing the model and processor
    :return: Extracted features as a torch.Tensor of shape (1, N)
    """
    if not isinstance(image, Image) and not isinstance(image, torch.Tensor):
        raise TypeError(
            f"Input must be a PIL Image or a torch Tensor, got {type(image)}"
        )

    image_tensor = convert_to_batch_tensor(image)

    if model.model_id == "MahmoodLab/UNI" or model.model_id == "MahmoodLab/UNI2-h":
        # For UNI and UNI2-h, we use the transform directly
        image_tensor = model.processor(image_tensor)
        image_tensor = image_tensor.to(model.device)
        with torch.inference_mode():
            features = model.model(image_tensor)
        return features
    elif model.model_id == "owkin/phikon-v2":
        inputs = model.processor(image_tensor, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        # Get the features
        with torch.inference_mode():
            outputs = model.model(**inputs)
            features = outputs.last_hidden_state[:, 0, :]  # (1, 1024) shape
            assert features.shape == (1, 1024), "Expected feature shape (1, 1024)"
        return features
