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
            self.logger.info(f"Directory verified: {directory}")
        
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

    def take_screenshot(self, url: str, test_name: str) -> Tuple[Image.Image, str]:
        try:
            self.driver.get(url)
            time.sleep(2)  # Wait for page load
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{test_name}_{timestamp}.png"
            filepath = os.path.join(self.results_dir, filename)
            
            screenshot = self.driver.get_screenshot_as_png()
            image = Image.open(io.BytesIO(screenshot))
            image.save(filepath)
            
            self.logger.info(f"Screenshot saved: {filepath}")
            return image, filepath
        except Exception as e:
            self.logger.error(f"Screenshot error: {str(e)}")
            raise

    def save_baseline(self, image: Image.Image, test_name: str) -> str:
        try:
            baseline_path = os.path.join(self.baseline_dir, f"{test_name}_baseline.png")
            image.save(baseline_path)
            self.logger.info(f"Baseline saved: {baseline_path}")
            return baseline_path
        except Exception as e:
            self.logger.error(f"Error saving baseline: {str(e)}")
            raise

    def calculate_similarity(self, baseline_path: str, current_path: str) -> float:
        try:
            baseline = cv2.imread(baseline_path)
            current = cv2.imread(current_path)
            
            baseline_gray = cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY)
            current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
            
            return ssim(baseline_gray, current_gray)
        except Exception as e:
            self.logger.error(f"Similarity calculation error: {str(e)}")
            raise

    def analyze_with_gpt4(self, baseline_path: str, current_path: str) -> Dict:
        try:
            def encode_image(image_path):
                with open(image_path, "rb") as f:
                    return base64.b64encode(f.read()).decode()

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.openai_api_key}"
            }

            payload = {
                "model": "gpt-4-vision-preview",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Compare these screenshots and identify meaningful UI changes."},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(baseline_path)}"}},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(current_path)}"}}
                        ]
                    }
                ],
                "max_tokens": 500
            }

            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            )
            return response.json()
        except Exception as e:
            self.logger.error(f"GPT-4 analysis error: {str(e)}")
            raise

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
    if 'test_results' not in st.session_state:
        st.session_state.test_results = []
    
    # Test configuration form
    with st.form("test_config"):
        url = st.text_input("URL to Test", "https://play.google.com/store/apps/details?id=com.aurahealth")
        test_name = st.text_input("Test Name", "aura_playstore")
        threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.95)
        
        submitted = st.form_submit_button("Run Test")
        
        if submitted and openai_api_key:
            try:
                # Initialize tester
                tester = AIVisualTester(openai_api_key=openai_api_key)
                
                # Take current screenshot
                current_image, current_path = tester.take_screenshot(url, test_name)
                
                # Handle baseline
                baseline_path = tester.save_baseline(current_image, test_name)
                
                if not os.path.exists(baseline_path):
                    st.error(f"Failed to save baseline image at: {baseline_path}")
                    st.stop()
                
                # Take new screenshot for comparison
                current_image, current_path = tester.take_screenshot(url, test_name)
                
                # Calculate similarity
                similarity = tester.calculate_similarity(baseline_path, current_path)
                
                # Analyze with GPT-4 if below threshold
                gpt4_analysis = None
                if similarity < threshold:
                    with st.spinner("Analyzing changes with GPT-4..."):
                        gpt4_analysis = tester.analyze_with_gpt4(baseline_path, current_path)
                
                # Store results
                result = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "test_name": test_name,
                    "baseline_path": baseline_path,
                    "current_path": current_path,
                    "similarity": similarity,
                    "passed": similarity >= threshold,
                    "gpt4_analysis": gpt4_analysis
                }
                
                st.session_state.test_results.append(result)
                st.success("Test completed successfully!")
                
            except Exception as e:
                st.error(f"Error running test: {str(e)}")
                st.exception(e)
    
    # Display results
    if st.session_state.test_results:
        st.header("Test Results")
        
        for result in reversed(st.session_state.test_results):
            with st.expander(f"Test: {result['test_name']} - {result['timestamp']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(Image.open(result['baseline_path']), caption="Baseline Image")
                    st.text(f"Baseline: {result['baseline_path']}")
                
                with col2:
                    st.image(Image.open(result['current_path']), caption="Current Image")
                    st.text(f"Current: {result['current_path']}")
                
                st.metric("Similarity Score", f"{result['similarity']:.2%}")
                
                if result['passed']:
                    st.success("Test Passed")
                else:
                    st.error("Test Failed")
                
                if result.get('gpt4_analysis'):
                    st.subheader("GPT-4 Analysis")
                    st.write(result['gpt4_analysis']['choices'][0]['message']['content'])

if __name__ == "__main__":
    create_streamlit_app()