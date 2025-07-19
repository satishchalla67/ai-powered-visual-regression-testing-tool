import pytest
import os
from PIL import Image
from unittest.mock import MagicMock
import io

@pytest.fixture(autouse=True)
def setup_environment():
    """Setup test environment variables"""
    os.environ['TEST_MODE'] = 'True'
    yield
    os.environ.pop('TEST_MODE', None)

@pytest.fixture
def test_image():
    """Create a test image"""
    return Image.new('RGB', (100, 100), color='white')

@pytest.fixture
def mock_webdriver():
    """Create a mock webdriver"""
    mock = MagicMock()
    # Create a fake screenshot
    test_image = Image.new('RGB', (100, 100), color='white')
    img_byte_arr = io.BytesIO()
    test_image.save(img_byte_arr, format='PNG')
    mock.get_screenshot_as_png.return_value = img_byte_arr.getvalue()
    return mock

@pytest.fixture
def test_directories(tmp_path):
    """Create test directories"""
    baseline_dir = tmp_path / "baseline_images"
    results_dir = tmp_path / "test_images"
    baseline_dir.mkdir()
    results_dir.mkdir()
    return {
        'baseline_dir': baseline_dir,
        'results_dir': results_dir
    }