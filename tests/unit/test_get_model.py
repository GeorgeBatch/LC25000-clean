import pytest
from torch import nn

from source.constants import ALL_EXTRACTOR_MODELS
from source.feature_extraction.get_model import get_feature_extractor


@pytest.mark.parametrize("model_name", ALL_EXTRACTOR_MODELS)
def test_get_feature_extractor(model_name):
    extractor = get_feature_extractor(model_name)
    assert isinstance(extractor, nn.Module)

    if model_name == 'imagenet_resnet18-last-layer':
        assert isinstance(extractor.fc, nn.Identity)
