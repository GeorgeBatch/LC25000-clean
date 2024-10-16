import pytest

from source.constants import FEATURE_EXTRACTOR_2_ORIGINAL_TRANSFORM
from source.feature_extraction.data import get_original_image_transform

@pytest.mark.parametrize("model_name", FEATURE_EXTRACTOR_2_ORIGINAL_TRANSFORM.keys())
def test_get_original_image_transform(model_name):
    transform = get_original_image_transform(model_name)
    assert transform is not None, f"Transform is None for model: {model_name}"
    assert callable(transform), f"Transform is not callable for model: {model_name}"
