import pytest

# -------------------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_image_path():
    return "assets/lungaca1.jpeg"