import streamlit as st
import os
import base64
from PIL import Image
import io
import json
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import requests
import logging
from typing import Dict, Tuple, Optional, List
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from datetime import datetime

class AIVisualTester:
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        self.baseline_dir = os.path.join(os.getcwd(), "baseline_images")
        self.results_dir = os.path.join(os.getcwd(), "test_images")
        self.logger = self._setup_logging()
        
        # Create directories
        for directory in [self.baseline_dir, self.results_dir]:
            os.makedirs(directory, exist_ok=True)
            self.logger.info(f"Directory created/verified: {directory}")
        
        self.setup_webdriver()

    def _setup_logging(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        if not logger.handlers:
            logger.addHandler(handler)
        return logger

    def setup_webdriver(self):
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless=new")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--window-size=1920,1080")
            
            service = Service()
            self.driver = webdriver.Chrome(options=chrome_options)
            self.logger.info("WebDriver setup completed")
        except Exception as e:
            self.logger.error(f"WebDriver setup error: {str(e)}")
            raise

    def capture_screenshot(self, url: str, test_name: str, css_selector: Optional[str] = None) -> Tuple[Image.Image, str]:
        try:
            self.driver.get(url)
            time.sleep(2)  # Wait for page load
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{test_name}_{timestamp}.png"
            filepath = os.path.join(self.results_dir, filename)
            
            if css_selector:
                element = self.driver.find_element("css selector", css_selector)
                screenshot = element.screenshot_as_png
            else:
                screenshot = self.driver.get_screenshot_as_png()
            
            image = Image.open(io.BytesIO(screenshot))
            os.makedirs(self.results_dir, exist_ok=True)
            image.save(filepath)
            
            self.logger.info(f"Screenshot saved to: {filepath}")
            return image, filepath
            
        except Exception as e:
            self.logger.error(f"Screenshot capture error: {str(e)}")
            raise

    def save_baseline(self, image: Image.Image, test_name: str) -> Tuple[Image.Image, str]:
        try:
            baseline_path = os.path.join(self.baseline_dir, f"{test_name}_baseline.png")
            os.makedirs(self.baseline_dir, exist_ok=True)
            image.save(baseline_path)
            self.logger.info(f"Baseline saved to: {baseline_path}")
            return image, baseline_path
        except Exception as e:
            self.logger.error(f"Error saving baseline: {str(e)}")
            raise

    def calculate_similarity(self, baseline_img: Image.Image, current_img: Image.Image) -> float:
        try:
            baseline = np.array(baseline_img.convert('RGB'))
            current = np.array(current_img.convert('RGB'))
            
            baseline_gray = cv2.cvtColor(baseline, cv2.COLOR_RGB2GRAY)
            current_gray = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
            
            return ssim(baseline_gray, current_gray)
        except Exception as e:
            self.logger.error(f"Similarity calculation error: {str(e)}")
            raise

    def analyze_with_gpt4(self, baseline_img: Image.Image, current_img: Image.Image) -> Dict:
        try:
            def encode_image(img):
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                return base64.b64encode(buffered.getvalue()).decode('utf-8')

            api_endpoint = "https://api.openai.com/v1/chat/completions"
            
            payload = {
                "model": "gpt-4o",  # Updated to current model name
                "temperature": 0.7,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Compare these two screenshots and describe the visual differences between them. Focus on UI changes, content modifications, and layout adjustments. Ignore minor color variations or pixel-level differences."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{encode_image(baseline_img)}"
                                }
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{encode_image(current_img)}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 500
            }

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.openai_api_key}"
            }

            self.logger.info("Making request to GPT-4 Vision API")
            response = requests.post(api_endpoint, headers=headers, json=payload, timeout=30)
            self.logger.info(f"GPT-4 API Status Code: {response.status_code}")
            
            try:
                response_data = response.json()
                self.logger.info(f"GPT-4 API Response: {response_data}")
            except json.JSONDecodeError:
                return {"error": "Invalid JSON response from API"}

            if response.status_code != 200:
                error_message = response_data.get('error', {}).get('message', 'Unknown error')
                error_details = response_data.get('error', {})
                return {
                    "error": f"API Error {response.status_code}: {error_message}",
                    "details": error_details
                }

            return response_data

        except Exception as e:
            self.logger.error(f"GPT-4 analysis error: {str(e)}")
            return {"error": str(e)}

    def __del__(self):
        if hasattr(self, 'driver'):
            self.driver.quit()

def create_streamlit_app():
    st.set_page_config(page_title="AI Visual Testing Tool", layout="wide")
    
    st.title("AI-Powered Visual Regression Testing")
    
    # Configuration
    st.sidebar.title("Configuration")
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    
    # Initialize session state
    if 'baseline_image' not in st.session_state:
        st.session_state.baseline_image = None
    if 'test_results' not in st.session_state:
        st.session_state.test_results = []
    
    # Test configuration form
    with st.form("test_config"):
        col1, col2 = st.columns(2)
        
        with col1:
            url = st.text_input("URL to Test", "https://example.com")
            css_selector = st.text_input("CSS Selector (optional)")
            
        with col2:
            threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.95)
            test_name = st.text_input("Test Name", "visual_test")
        
        submit_button = st.form_submit_button("Run Test")
        
        if submit_button and openai_api_key:
            if not openai_api_key.startswith("sk-"):
                st.error("Please enter a valid OpenAI API key starting with 'sk-'")
                return
                
            try:
                # Initialize tester
                tester = AIVisualTester(openai_api_key=openai_api_key)
                
                # Capture current screenshot
                current_image, current_path = tester.capture_screenshot(url, test_name, css_selector)
                
                # Handle baseline
                if st.session_state.baseline_image is None:
                    baseline_image, baseline_path = tester.save_baseline(current_image, test_name)
                    st.session_state.baseline_image = baseline_image
                    st.success(f"Created baseline image at: {baseline_path}")
                    current_image, current_path = tester.capture_screenshot(url, test_name, css_selector)
                
                # Calculate similarity
                similarity = tester.calculate_similarity(
                    st.session_state.baseline_image, 
                    current_image
                )
                
                # Perform GPT-4 analysis if below threshold
                gpt4_analysis = None
                if similarity < threshold:
                    with st.spinner("Analyzing changes with GPT-4..."):
                        gpt4_analysis = tester.analyze_with_gpt4(
                            st.session_state.baseline_image,
                            current_image
                        )
                        if 'error' in gpt4_analysis:
                            st.warning(f"GPT-4 Analysis Warning: {gpt4_analysis['error']}")
                
                # Store results
                result = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "test_name": test_name,
                    "similarity": similarity,
                    "passed": similarity >= threshold,
                    "baseline_image": st.session_state.baseline_image,
                    "current_image": current_image,
                    "baseline_path": baseline_path if 'baseline_path' in locals() else None,
                    "current_path": current_path,
                    "gpt4_analysis": gpt4_analysis
                }
                
                st.session_state.test_results.append(result)
                st.success(f"Test completed successfully! Images saved in test_images and baseline_images folders.")
                
            except Exception as e:
                st.error(f"Error running test: {str(e)}")
    
    # Display results
    if st.session_state.test_results:
        st.header("Test Results")
        
        for result in reversed(st.session_state.test_results):
            with st.expander(f"Test: {result['test_name']} - {result['timestamp']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(result['baseline_image'], caption="Baseline Image")
                    st.text(f"Baseline path: {result.get('baseline_path', 'Not available')}")
                
                with col2:
                    st.image(result['current_image'], caption="Current Image")
                    st.text(f"Current path: {result.get('current_path', 'Not available')}")
                
                st.metric("Similarity Score", f"{result['similarity']:.2%}")
                
                if result['passed']:
                    st.success("Test Passed")
                else:
                    st.error("Test Failed")
                
                if result.get('gpt4_analysis'):
                    st.subheader("GPT-4 Analysis")
                    if 'error' in result['gpt4_analysis']:
                        st.error(f"GPT-4 Analysis Error: {result['gpt4_analysis']['error']}")
                        st.text("Full API Response:")
                        st.code(json.dumps(result['gpt4_analysis'], indent=2))
                    else:
                        try:
                            analysis_text = result['gpt4_analysis']['choices'][0]['message']['content']
                            st.write(analysis_text)
                        except (KeyError, IndexError) as e:
                            st.error("Failed to parse GPT-4 analysis results")
                            st.text("Raw API Response:")
                            st.code(json.dumps(result['gpt4_analysis'], indent=2))

if __name__ == "__main__":
    create_streamlit_app()