import PIL
import torch
import os

from ..inference import extract_features_from_pil
from ..loader import load_foundation_model

TOKEN = os.getenv("HF_TOKEN")


def test_inference_uni():
    image = PIL.Image.new("RGB", (224, 224), color="red")
    model = load_foundation_model("MahmoodLab/UNI", device="cuda", token=TOKEN)
    features = extract_features_from_pil(image, model)

    assert isinstance(features, torch.Tensor)
    assert features.shape == (1, 1024)


def test_inference_uni2():
    image = PIL.Image.new("RGB", (224, 224), color="red")
    model = load_foundation_model("MahmoodLab/UNI2-h", device="cuda", token=TOKEN)
    features = extract_features_from_pil(image, model)

    assert isinstance(features, torch.Tensor)
    assert features.shape == (1, 1536)


def test_inference_phikon_v2():
    image = PIL.Image.new("RGB", (224, 224), color="red")
    model = load_foundation_model("owkin/phikon-v2", device="cuda", token=TOKEN)
    features = extract_features_from_pil(image, model)

    assert isinstance(features, torch.Tensor)
    assert features.shape == (1, 1024)
