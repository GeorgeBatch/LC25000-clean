import pytest

# -------------------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------------------

@pytest.fixture
def mock_img_dir():
    return "assets/test_images"

@pytest.fixture(scope="session")
def sample_image_path():
    return f"{mock_img_dir()}/lungaca1.jpeg"
