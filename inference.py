import torch
import torchvision.transforms as T

from PIL.Image import Image

from dataset.fm_model import FoundationModel


def convert_to_batch_tensor(images: Image | list[Image] | torch.Tensor) -> torch.Tensor:
    """
    Converts a single PIL image or a torch tensor of shape (3, H, W) to a batch tensor of shape (1, 3, H, W).

    If the input is already a batch tensor of shape (N, 3, H, W), it will be returned as is.

    :param image: PIL Image or torch.Tensor of shape (3, H, W) or torch Tensor of shape (N, 3, H, W)
    :return: Batch tensor of shape (1, 3, H, W)
    """
    if isinstance(images, list):
        if not all(isinstance(img, Image) for img in images):
            raise TypeError("All items in the list must be PIL Images")
    elif not isinstance(images, Image) and not isinstance(images, torch.Tensor):
        raise TypeError(
            f"Input must be a PIL Image or a torch Tensor, got {type(images)}"
        )

    if isinstance(images, list):
        transform = T.ToTensor()
        image_list = [transform(img) for img in images]
        images = torch.stack(image_list, dim=0)
    if isinstance(images, Image):
        transform = T.ToTensor()
        images = transform(images).unsqueeze(0)  # Add batch dimension
    elif isinstance(images, torch.Tensor):
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension
        elif images.dim() != 4:
            raise ValueError(
                f"Input tensor must be of shape (3, H, W) or (N, 3, H, W). Got {images.shape}"
            )
    else:
        raise TypeError("Input must be a PIL Image or a torch Tensor")

    assert (
        images.dim() == 4
    ), "Unexpected return shape, expected (1, 3, H, W) or (N, 3, H, W)"
    return images


def extract_features(
    images: Image | list[Image] | torch.Tensor, model: FoundationModel
) -> torch.Tensor:
    """
    Extracts features from single PIL image, list of PIL images or tensor using the specified model.

    **Note: images must be of the same size, since inference is performed in a single pass**

    :param image: PIL Image or torch.Tensor of shape (3, H, W) or torch Tensor of shape (N, 3, H, W)
    :param format: either 'pil' or 'torch'
    :param model: FoundationModel instance containing the model and processor
    :return: Extracted features as a torch.Tensor of shape (1, N)
    """
    image_tensor = convert_to_batch_tensor(images)

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


def extract_features_batched(
    images: Image | list[Image] | torch.Tensor,
    model: FoundationModel,
    batch_size: int,
) -> torch.Tensor:
    """
    Extracts features from a collection images using the specified model.
    Performs batching to handle large datasets efficiently.

    **Note: images must be of the same size, since images are batched before each inference pass**
    """
    N = len(images)
    _embeddings = []
    for i in range(0, N, batch_size):
        image_batch = [images[i] for i in range(i, min(i + batch_size, N))]
        embedding_batch = extract_features(image_batch, model)
        _embeddings.append(embedding_batch)
    return torch.cat(_embeddings, dim=0)
