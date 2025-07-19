import pytest
import os
from PIL import Image
import io
import json
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from app import AIVisualTester

@pytest.fixture
def mock_image():
    """Create a simple test image"""
    img = Image.new('RGB', (100, 100), color='white')
    return img

@pytest.fixture
def mock_driver():
    """Create a mock webdriver"""
    mock = MagicMock()
    # Create a fake screenshot
    test_image = Image.new('RGB', (100, 100), color='white')
    img_byte_arr = io.BytesIO()
    test_image.save(img_byte_arr, format='PNG')
    mock.get_screenshot_as_png.return_value = img_byte_arr.getvalue()
    return mock

@pytest.fixture
def tester(mock_driver):
    """Create an instance of AIVisualTester with mocked driver"""
    with patch('selenium.webdriver.Chrome', return_value=mock_driver):
        tester = AIVisualTester("mock-api-key")
        tester.driver = mock_driver
        yield tester

@pytest.fixture
def setup_test_dirs(tmp_path):
    """Setup temporary test directories"""
    baseline_dir = tmp_path / "baseline_images"
    test_dir = tmp_path / "test_images"
    baseline_dir.mkdir()
    test_dir.mkdir()
    return baseline_dir, test_dir

def test_init(setup_test_dirs):
    """Test initialization of AIVisualTester"""
    with patch('selenium.webdriver.Chrome') as mock_chrome:
        tester = AIVisualTester("test-key")
        assert tester.openai_api_key == "test-key"
        assert os.path.exists(tester.baseline_dir)
        assert os.path.exists(tester.results_dir)

def test_setup_webdriver():
    """Test webdriver setup"""
    with patch('selenium.webdriver.Chrome') as mock_chrome:
        tester = AIVisualTester("test-key")
        assert tester.driver is not None

def test_capture_screenshot(tester, mock_image):
    """Test screenshot capture functionality"""
    # Mock the driver's get method
    tester.driver.get.return_value = None
    
    # Create a test image and convert to bytes
    img_byte_arr = io.BytesIO()
    mock_image.save(img_byte_arr, format='PNG')
    img_bytes = img_byte_arr.getvalue()
    
    # Set the mock return value for screenshot
    tester.driver.get_screenshot_as_png.return_value = img_bytes
    
    # Capture screenshot
    image, filepath = tester.capture_screenshot("http://example.com", "test")
    
    # Verify the results
    assert isinstance(image, Image.Image)
    assert filepath.endswith(".png")
    assert os.path.dirname(filepath).endswith("test_images")
    assert os.path.exists(filepath)
    
    # Verify the mock was called
    tester.driver.get.assert_called_once_with("http://example.com")

def test_save_baseline(tester, mock_image, tmp_path):
    """Test baseline image saving"""
    with patch.object(tester, 'baseline_dir', str(tmp_path)):
        image, path = tester.save_baseline(mock_image, "test")
        assert os.path.exists(path)
        assert path.endswith("test_baseline.png")
        assert isinstance(image, Image.Image)

def test_calculate_similarity(tester, mock_image):
    """Test similarity calculation"""
    # Create two identical images
    img1 = mock_image
    img2 = mock_image
    
    similarity = tester.calculate_similarity(img1, img2)
    assert similarity == pytest.approx(1.0, rel=1e-3)

    # Create different image
    img3 = Image.new('RGB', (100, 100), color='black')
    similarity = tester.calculate_similarity(img1, img3)
    assert similarity < 1.0

@patch('requests.post')
def test_analyze_with_gpt4(mock_post, tester, mock_image):
    """Test GPT-4 analysis"""
    # Mock successful API response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "Analysis of the differences between images"
                }
            }
        ]
    }
    mock_post.return_value = mock_response

    result = tester.analyze_with_gpt4(mock_image, mock_image)
    assert "choices" in result
    assert result["choices"][0]["message"]["content"] == "Analysis of the differences between images"

@patch('requests.post')
def test_analyze_with_gpt4_error(mock_post, tester, mock_image):
    """Test GPT-4 analysis error handling"""
    mock_response = Mock()
    mock_response.status_code = 404
    mock_response.json.return_value = {
        "error": {
            "message": "Model not found"
        }
    }
    mock_post.return_value = mock_response

    result = tester.analyze_with_gpt4(mock_image, mock_image)
    assert "error" in result
    assert "API Error 404" in result["error"]

def test_cleanup(tester):
    """Test cleanup when object is destroyed"""
    with patch.object(tester.driver, 'quit') as mock_quit:
        tester.__del__()
        mock_quit.assert_called_once()

@pytest.mark.integration
def test_full_workflow(mock_driver, mock_image, tmp_path):
    """Integration test for full workflow"""
    with patch('selenium.webdriver.Chrome', return_value=mock_driver):
        # Initialize tester with temp directories
        tester = AIVisualTester("test-key")
        tester.baseline_dir = str(tmp_path / "baseline_images")
        tester.results_dir = str(tmp_path / "test_images")
        
        # Setup directories
        os.makedirs(tester.baseline_dir, exist_ok=True)
        os.makedirs(tester.results_dir, exist_ok=True)
        
        # Mock screenshot capture
        img_byte_arr = io.BytesIO()
        mock_image.save(img_byte_arr, format='PNG')
        mock_driver.get_screenshot_as_png.return_value = img_byte_arr.getvalue()
        
        # Test capturing and saving images
        image1, path1 = tester.capture_screenshot("http://example.com", "test")
        assert os.path.exists(path1)
        
        # Test saving baseline
        baseline_image, baseline_path = tester.save_baseline(image1, "test")
        assert os.path.exists(baseline_path)
        
        # Test similarity calculation
        similarity = tester.calculate_similarity(image1, baseline_image)
        assert 0 <= similarity <= 1