#!/bin/bash

# MedCheck - Complete Project Setup Script
# This will set up EVERYTHING for the legendary project!

echo "🚀 Setting up MedCheck - Professional Drug Interaction System"
echo "============================================================"
echo ""

# Create complete project structure
echo "📁 Creating project structure..."

mkdir -p medcheck_pro/{data/{raw,processed,models,embeddings},src/{ocr,ml,api,analysis,utils},notebooks,frontend/{web,mobile},tests,scripts,docker,docs}

cd medcheck_pro

# Create Python virtual environment
echo "🐍 Creating Python environment..."
python3 -m venv venv
source venv/bin/activate

# Create comprehensive requirements.txt
echo "📦 Creating requirements.txt..."
cat > requirements.txt << 'EOF'
# Core Data Science
pandas==2.1.0
numpy==1.24.3
scipy==1.11.1

# Machine Learning
scikit-learn==1.3.0
xgboost==1.7.6
torch==2.0.1
torchvision==0.15.2

# Computer Vision & OCR
opencv-python==4.8.0
pytesseract==0.3.10
easyocr==1.7.0
paddleocr==2.7.0
Pillow==10.0.0

# NLP
transformers==4.31.0
spacy==3.6.0
nltk==3.8.1

# Chemical/Drug Analysis
rdkit==2023.3.2
chembl-webresource-client==0.10.8

# Network Analysis
networkx==3.1
python-louvain==0.16

# Visualization
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.16.1
kaleido==0.2.1

# Web Framework
fastapi==0.101.0
uvicorn==0.23.2
pydantic==2.1.1
python-multipart==0.0.6

# Database
psycopg2-binary==2.9.7
sqlalchemy==2.0.19
redis==4.6.0

# API Clients
requests==2.31.0
httpx==0.24.1

# Utilities
python-dotenv==1.0.0
tqdm==4.66.1
click==8.1.6
pyyaml==6.0.1
joblib==1.3.2

# Model Explainability
shap==0.42.1
lime==0.2.0.1

# Testing
pytest==7.4.0
pytest-cov==4.1.0

# Deployment
docker==6.1.3
gunicorn==21.2.0

# Jupyter
jupyter==1.0.0
ipykernel==6.25.0
jupyterlab==4.0.5
EOF

echo "📥 Installing Python packages (this will take a few minutes)..."
pip install --upgrade pip
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Create data download script
echo "📊 Creating FDA data download script..."
cat > scripts/download_fda_data.py << 'PYEOF'
"""
Download FDA FAERS data (10M+ adverse event reports)
This is FREE and publicly available!
"""

import requests
import zipfile
import os
from pathlib import Path
from tqdm import tqdm

def download_file(url, destination):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as file, tqdm(
        desc=destination.name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def download_faers_data():
    """Download latest FAERS quarterly data"""
    
    base_url = "https://fis.fda.gov/content/Exports/"
    
    # Data for recent quarters (example: 2023Q1, 2023Q2, etc.)
    quarters = ['2023Q1', '2023Q2', '2023Q3', '2023Q4', '2024Q1']
    
    data_dir = Path('data/raw/faers')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔽 Downloading FDA FAERS Data...")
    print("This contains millions of adverse event reports!")
    print()
    
    for quarter in quarters:
        print(f"Downloading {quarter}...")
        
        # FAERS data URL pattern
        url = f"{base_url}faers_ascii_{quarter}.zip"
        destination = data_dir / f"faers_{quarter}.zip"
        
        try:
            download_file(url, destination)
            
            # Extract
            print(f"Extracting {quarter}...")
            with zipfile.ZipFile(destination, 'r') as zip_ref:
                zip_ref.extractall(data_dir / quarter)
            
            print(f"✅ {quarter} complete!\n")
            
        except Exception as e:
            print(f"❌ Error downloading {quarter}: {e}\n")
            continue
    
    print("✅ FAERS data download complete!")
    print(f"Data saved to: {data_dir.absolute()}")

if __name__ == "__main__":
    download_faers_data()
PYEOF

# Create DrugBank download script
cat > scripts/download_drugbank.py << 'PYEOF'
"""
Instructions to get DrugBank data (FREE for academic use)

DrugBank contains comprehensive drug interaction data!
"""

def get_drugbank_instructions():
    instructions = """
    📚 HOW TO GET DRUGBANK DATA (FREE Academic License)
    ===================================================
    
    1. Go to: https://go.drugbank.com/releases/latest
    
    2. Click "Create Account" (FREE!)
    
    3. Fill in:
       - Use your .edu email if you have one
       - Purpose: "Academic Research / Learning"
       - Project: "Drug Interaction Analysis System"
    
    4. Download these files:
       ✅ "All Drugs" (XML format) - Full database
       ✅ "Drug Interactions" (CSV) - ~3M interactions
       ✅ "Drug Target" data
    
    5. Save files to: data/raw/drugbank/
    
    6. Run: python scripts/process_drugbank.py
    
    📊 What you'll get:
    - 13,000+ drugs
    - 3M+ drug-drug interactions
    - Chemical structures
    - Mechanisms of action
    - Side effects
    
    ⏱️  Time: 5-10 minutes for signup + download
    💾 Size: ~500MB compressed
    
    FREE FOR ACADEMIC USE! 🎓
    """
    
    print(instructions)
    
    # Create placeholder
    import os
    os.makedirs('data/raw/drugbank', exist_ok=True)
    
    with open('data/raw/drugbank/README.txt', 'w') as f:
        f.write(instructions)
    
    print("\n✅ Instructions saved to: data/raw/drugbank/README.txt")

if __name__ == "__main__":
    get_drugbank_instructions()
PYEOF

# Create RxNorm download script
cat > scripts/download_rxnorm.py << 'PYEOF'
"""
Download RxNorm database (140,000+ drug names)
FREE from NIH!
"""

import requests
import zipfile
from pathlib import Path
from tqdm import tqdm

def download_rxnorm():
    """Download RxNorm full monthly release"""
    
    print("🔽 Downloading RxNorm Database...")
    print("This contains 140,000+ drug names and codes!")
    print()
    
    # RxNorm full release URL
    url = "https://download.nlm.nih.gov/umls/kss/rxnorm/RxNorm_full_current.zip"
    
    data_dir = Path('data/raw/rxnorm')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    destination = data_dir / "RxNorm_full.zip"
    
    print("Downloading (this is ~1GB)...")
    
    try:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as file, tqdm(
            desc="RxNorm",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
        
        print("\nExtracting...")
        with zipfile.ZipFile(destination, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        print("\n✅ RxNorm download complete!")
        print(f"Data saved to: {data_dir.absolute()}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nAlternative: Access via API (no download needed)")
        print("API: https://rxnav.nlm.nih.gov/REST/")

if __name__ == "__main__":
    download_rxnorm()
PYEOF

# Create README
cat > README.md << 'MDEOF'
# 💊 MedCheck Professional - AI-Powered Drug Interaction System

## 🎯 Project Overview

A production-grade drug interaction checking system using:
- **Computer Vision** (OCR for prescription labels)
- **Machine Learning** (interaction severity prediction)
- **Big Data** (10M+ FDA adverse event records)
- **Network Analysis** (drug interaction graphs)
- **Web Application** (FastAPI + React)

## 📊 Datasets

### Included (Automatic Download):
- ✅ FDA FAERS (10M+ records) - FREE
- ✅ RxNorm (140K+ drugs) - FREE
- ✅ PubChem (chemical data) - FREE via API

### Manual Download (FREE Academic License):
- 📚 DrugBank (13K+ drugs, 3M+ interactions)
  - Get it: https://go.drugbank.com/releases/latest
  - Instructions: See `data/raw/drugbank/README.txt`

## 🚀 Quick Start

### 1. Setup
```bash
# Already done if you ran project_setup.sh!
source venv/bin/activate
```

### 2. Download Data
```bash
# FDA FAERS data (10M+ records)
python scripts/download_fda_data.py

# RxNorm (140K+ drugs)
python scripts/download_rxnorm.py

# DrugBank (manual - follow instructions)
python scripts/download_drugbank.py
```

### 3. Build Database
```bash
python scripts/build_database.py
```

### 4. Train Models
```bash
python scripts/train_models.py
```

### 5. Run Application
```bash
# API
uvicorn src.api.main:app --reload

# Frontend (separate terminal)
cd frontend/web && npm start
```

## 📁 Project Structure

```
medcheck_pro/
├── data/               # Datasets (10GB+)
├── src/                # Source code
│   ├── ocr/           # Computer vision
│   ├── ml/            # Machine learning
│   ├── api/           # FastAPI backend
│   └── analysis/      # Data analysis
├── notebooks/          # Jupyter notebooks
├── tests/             # Unit tests
└── scripts/           # Utility scripts
```

## 🔬 Features

### Completed:
- ✅ Basic API integration
- ✅ Drug lookup

### In Progress:
- 🔄 OCR pipeline
- 🔄 ML models
- 🔄 Web interface

### Planned:
- 📋 Mobile app
- 📋 Real-time monitoring
- 📋 Voice input

## 🎓 Learning Resources

- FDA Data: https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html
- RxNorm: https://www.nlm.nih.gov/research/umls/rxnorm/
- DrugBank: https://go.drugbank.com/
- OpenCV: https://docs.opencv.org/
- PyTorch: https://pytorch.org/tutorials/

## 📈 Goals

- 🎯 95%+ OCR accuracy
- 🎯 90%+ ML F1-score
- 🎯 Sub-second API response
- 🎯 Production deployment

## ⚠️ Disclaimer

For educational/research purposes only.
Not medical advice. Always consult healthcare professionals.

---

**Author:** [Your Name]
**Contact:** [Your Email]
**Portfolio:** [Your Website]
MDEOF

echo ""
echo "✅ Project structure created!"
echo ""
echo "📋 NEXT STEPS:"
echo "============================================================"
echo "1. cd medcheck_pro"
echo "2. source venv/bin/activate"
echo "3. python scripts/download_fda_data.py    # Get 10M records!"
echo "4. python scripts/download_rxnorm.py       # Get 140K drugs!"
echo "5. python scripts/download_drugbank.py     # Instructions for DrugBank"
echo ""
echo "🎯 This will give you GIGABYTES of real medical data!"
echo "============================================================"
EOF

chmod +x project_setup.sh

echo "✅ Master setup script created!"
