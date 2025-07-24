import torch
import PIL

from .loader import FoundationModel


def extract_features_from_pil(image: PIL.Image, model: FoundationModel) -> torch.Tensor:
    """
    Extracts features from single PIL image using the specified model.

    :param image: PIL Image
    :param model: FoundationModel instance containing the model and processor
    :return: Extracted features as a torch.Tensor of shape (1, N)
    """
    if model.model_id == "MahmoodLab/UNI" or model.model_id == "MahmoodLab/UNI2-h":
        # For UNI and UNI2-h, we use the transform directly
        transform = model.processor
        image = transform(image).unsqueeze(0)  # Add batch dimension
        with torch.inference_mode():
            features = model.model(image)
        return features
    elif model.model_id == "owkin/phikon-v2":
        # Process the image
        inputs = model.processor(image, return_tensors="pt")
        # Get the features
        with torch.inference_mode():
            outputs = model(**inputs)
            features = outputs.last_hidden_state[:, 0, :]  # (1, 1024) shape
            assert features.shape == (1, 1024), "Expected feature shape (1, 1024)"
        return features
