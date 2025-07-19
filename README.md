# 🔍 AI-Powered Visual Testing Tool

Automated visual regression testing tool combining screenshot comparison with GPT-4 analysis to detect UI changes in web applications. 🚀

## ✨ Features

- 📸 Automated screenshot capture
- 🤖 AI-powered visual difference analysis using GPT-4
- 📊 Similarity score calculation
- 🎯 Full-page and element-specific testing
- 📝 Detailed test reports

## 🚀 Quick Start

1. **Prerequisites**
```bash
# Required
- Python 3.8+
- Chrome browser
- OpenAI API key
```

2. **Installation**
```bash
git clone https://github.com/jaimeman84/ai-powered-visual-tester.git
cd ai-powered-visual-tester
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt

# Create directories
mkdir baseline_images test_images
```

3. **Run Application**
```bash
streamlit run app.py
```

## 🔑 Getting an OpenAI API Key

1. Go to [OpenAI Platform](https://platform.openai.com/signup)
2. Create account/Sign in
3. Visit [API Keys](https://platform.openai.com/api-keys)
4. Click "Create new secret key"
5. Set up billing in [Billing settings](https://platform.openai.com/account/billing/overview)

⚠️ **Important**: Keep your API key secure and never share it publicly

## 🧪 Testing

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run tests
pytest test_visual_tester.py -v

# Run with coverage
pytest test_visual_tester.py --cov=app --cov-report=term-missing -v
```

## 📁 Project Structure
```
ai-powered-visual-testing/
├── app.py                 # Main application
├── test_visual_tester.py  # Tests
├── pytest.ini            # Pytest config
├── requirements.txt      # Dependencies
├── baseline_images/      # Baseline screenshots
└── test_images/         # Test screenshots
```

## 🤝 Contributing
1. Fork repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add Feature'`)
4. Push (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📜 License
MIT License

## 💬 Support
Open an issue in the GitHub repository for support.