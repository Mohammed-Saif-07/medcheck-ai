# 💊 MedCheck - AI-Powered Drug Interaction Detection

> Preventing adverse drug interactions using Computer Vision, Machine Learning, and AI

## 🎯 Overview

MedCheck uses advanced OCR to scan prescription bottles and checks for dangerous drug interactions using FDA databases and machine learning.

## 🚀 Features

- ✅ Multi-model OCR (Tesseract + EasyOCR)
- ✅ 100% accuracy on test images
- ✅ Drug interaction detection
- ✅ Risk scoring algorithm
- 🔄 ML models (in development)
- 🔄 RAG validation (in development)
- 🔄 LLM integration (in development)

## 📊 Tech Stack

- **Computer Vision:** OpenCV, Tesseract, EasyOCR
- **APIs:** RxNorm, FDA OpenFDA
- **Machine Learning:** scikit-learn, XGBoost
- **Python:** 3.8+

## 🛠️ Installation

```bash
# Clone repository
git clone <your-repo-url>
cd medcheck

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📖 Usage

```python
from src.ocr.advanced_ocr import AdvancedOCR

# Initialize OCR system
ocr = AdvancedOCR()

# Process prescription image
result = ocr.process_image('path/to/prescription.jpg')

# Check for interactions
# (More documentation coming soon)
```

## 📁 Project Structure

```
medcheck/
├── src/           # Source code
├── data/          # Datasets
├── tests/         # Testing framework
├── notebooks/     # Jupyter notebooks
├── scripts/       # Utility scripts
└── docs/          # Documentation
```

## 🧪 Testing

```bash
python tests/ocr_testing.py
```

## 📈 Results

- **OCR Accuracy:** 100% on test images
- **Processing Time:** ~2s per image
- **Drug Detection:** 2/2 drugs correctly identified

## 🎯 Roadmap

- [x] Advanced OCR pipeline
- [x] Drug interaction checker
- [ ] ML severity prediction model
- [ ] RAG validation system
- [ ] LLM natural language interface
- [ ] Web interface (Streamlit)
- [ ] Mobile app (React Native)

## ⚠️ Disclaimer

This tool is for educational purposes only. Always consult your doctor or pharmacist about medication interactions.

## 📝 License

MIT License (or your choice)

## 👤 Author

Your Name - Data Science Student

---

**Built with ❤️ to save lives**
