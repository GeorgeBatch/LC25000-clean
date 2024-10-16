import pytest
import torch
from torchvision.io import read_image
from torch import nn

from source.constants import FEATURE_EXTRACTOR_2_ORIGINAL_TRANSFORM, ALL_EXTRACTOR_MODELS
from source.feature_extraction.data import get_original_image_transform
from source.feature_extraction.get_model import get_feature_extractor


@pytest.mark.parametrize("model_name", FEATURE_EXTRACTOR_2_ORIGINAL_TRANSFORM.keys())
def test_get_original_image_transform(model_name):
    transform = get_original_image_transform(model_name)
    assert transform is not None, f"Transform is None for model: {model_name}"
    assert callable(transform), f"Transform is not callable for model: {model_name}"


@pytest.mark.parametrize("model_name", ALL_EXTRACTOR_MODELS)
def test_model_with_transformed_image(model_name, sample_image_path):
    # Load the model
    model = get_feature_extractor(model_name)
    assert isinstance(model, nn.Module), f"Model is not an instance of nn.Module for model: {model_name}"

    # Load the transform
    transform = get_original_image_transform(model_name)
    assert transform is not None, f"Transform is None for model: {model_name}"
    assert callable(transform), f"Transform is not callable for model: {model_name}"

    # Read a sample image
    image = read_image(sample_image_path)
    assert image is not None, "Failed to read the sample image"

    # Apply the transform
    transformed_image = transform(image)
    assert transformed_image is not None, "Transform returned None"
    assert isinstance(transformed_image, torch.Tensor), "Transformed image is not a tensor"

    # Pass the transformed image through the model
    model.eval()
    with torch.no_grad():
        output = model(transformed_image.unsqueeze(0))  # Add batch dimension
    assert output is not None, "Model output is None"
    assert isinstance(output, torch.Tensor), "Model output is not a tensor"
    # print(f"Model output shape: {output.shape}")
