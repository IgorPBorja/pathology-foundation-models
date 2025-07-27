import PIL
import torch

from inference import (
    extract_features,
    extract_features_from_dataset,
    convert_to_batch_tensor,
)
from dataset.loader import load_foundation_model
from tests.fixtures import hf_token, image_dataset


def test_inference_uni(hf_token):
    print(hf_token)
    image = PIL.Image.new("RGB", (224, 224), color="red")
    model = load_foundation_model("MahmoodLab/UNI", device="cuda", token=hf_token)
    features = extract_features(image, model)

    assert isinstance(features, torch.Tensor)
    assert features.shape == (1, 1024)


def test_inference_uni2(hf_token):
    print(hf_token)
    image = PIL.Image.new("RGB", (224, 224), color="red")
    model = load_foundation_model("MahmoodLab/UNI2-h", device="cuda", token=hf_token)
    features = extract_features(image, model)

    assert isinstance(features, torch.Tensor)
    assert features.shape == (1, 1536)


def test_inference_phikon_v2(hf_token):
    print(hf_token)
    image = PIL.Image.new("RGB", (224, 224), color="red")
    model = load_foundation_model("owkin/phikon-v2", device="cuda", token=hf_token)
    features = extract_features(image, model)

    assert isinstance(features, torch.Tensor)
    assert features.shape == (1, 1024)


def test_convert_to_batch_tensor_pil():
    # PIL format is (W, H)
    image = PIL.Image.new("RGB", (200, 300), color="red")
    batch_tensor = convert_to_batch_tensor(image)

    assert isinstance(batch_tensor, torch.Tensor)
    assert batch_tensor.shape == (1, 3, 300, 200)


def test_batch_inference(hf_token, image_dataset):
    model = load_foundation_model("MahmoodLab/UNI", device="cuda", token=hf_token)
    batch_tensor = extract_features_from_dataset(
        image_dataset, model, batch_size=3, num_workers=2, display_progress=True
    )

    assert isinstance(batch_tensor, torch.Tensor)
    assert batch_tensor.shape == (len(image_dataset), 1024)
