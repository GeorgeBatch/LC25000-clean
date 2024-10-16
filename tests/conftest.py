import os
import pytest

# -------------------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def mock_img_dir():
    return "assets/test_images"


@pytest.fixture(scope="session")
def sample_image_path(mock_img_dir):
    return os.path.join(mock_img_dir, "lungaca1.jpeg")
