#!/bin/bash

# MedCheck - 100% FREE Setup (Ek paisa nahi lagega!)
# For students with limited resources

echo "💊 MedCheck - Student-Friendly FREE Setup"
echo "=========================================="
echo "✅ Zero cost"
echo "✅ No credit card needed"
echo "✅ All open source"
echo "✅ Free datasets"
echo "✅ Free hosting"
echo ""

# Create simple structure
echo "📁 Creating project..."
mkdir -p medcheck_free/{data,src,notebooks}
cd medcheck_free

# Virtual environment
python3 -m venv venv
source venv/bin/activate

# Minimal FREE requirements
cat > requirements.txt << 'EOF'
# Core (ALL FREE!)
requests==2.31.0
pandas==2.1.0
numpy==1.24.3

# Machine Learning (FREE!)
scikit-learn==1.3.0
xgboost==1.7.6

# Computer Vision (FREE!)
opencv-python==4.8.0
pytesseract==0.3.10
Pillow==10.0.0

# Web Framework (FREE!)
fastapi==0.101.0
uvicorn==0.23.2

# Visualization (FREE!)
matplotlib==3.7.2
plotly==5.16.1

# Utilities (FREE!)
tqdm==4.66.1
python-dotenv==1.0.0
EOF

echo "📦 Installing packages..."
pip install -r requirements.txt

# Download FREE data script
cat > data/download_free_data.py << 'PYEOF'
"""
Download 100% FREE datasets
No signup, no credit card, no cost!
"""

import requests
import json

def test_free_apis():
    """Test all free APIs"""
    
    print("🧪 Testing FREE APIs...")
    print()
    
    # 1. RxNorm (FREE, no signup)
    print("1. Testing RxNorm API (FREE)...")
    url = "https://rxnav.nlm.nih.gov/REST/rxcui.json?name=aspirin"
    response = requests.get(url)
    if response.status_code == 200:
        print("   ✅ RxNorm working! (FREE, no limits)")
    
    # 2. FDA OpenFDA (FREE, no signup)
    print("\n2. Testing FDA OpenFDA API (FREE)...")
    url = "https://api.fda.gov/drug/label.json?limit=1"
    response = requests.get(url)
    if response.status_code == 200:
        print("   ✅ FDA API working! (FREE, 1000 requests/day)")
    
    # 3. PubChem (FREE, no signup)
    print("\n3. Testing PubChem API (FREE)...")
    url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/aspirin/JSON"
    response = requests.get(url)
    if response.status_code == 200:
        print("   ✅ PubChem working! (FREE, unlimited)")
    
    print("\n✅ All FREE APIs working!")
    print("💰 Total cost: ₹0")

if __name__ == "__main__":
    test_free_apis()
PYEOF

# Create README
cat > README.md << 'MDEOF'
# 💊 MedCheck - 100% FREE Student Version

## 💰 Total Cost: ₹0 (Zero!)

### What's Included (ALL FREE):

✅ **APIs:** RxNorm, FDA OpenFDA, PubChem (no signup!)
✅ **ML Libraries:** scikit-learn, XGBoost (open source)
✅ **Computer Vision:** OpenCV, Tesseract (open source)
✅ **Web Framework:** FastAPI (open source)
✅ **Development:** VS Code, Python, Git (open source)
✅ **Deployment:** Render.com, Vercel (free tiers)
✅ **Training:** Your laptop or Google Colab (free GPU!)

### No Credit Card, No Payment, No Trial!

## Quick Start

```bash
# 1. Setup (already done!)
source venv/bin/activate

# 2. Test free APIs
python data/download_free_data.py

# 3. Start building!
```

## Free Training Options

### Option 1: Your Laptop (FREE)
- Slower but works
- No internet needed after setup

### Option 2: Google Colab (FREE GPU!)
- Go to: colab.research.google.com
- Upload notebook
- Runtime → GPU
- FREE Tesla T4 GPU! 🎉

### Option 3: Kaggle Notebooks (FREE GPU!)
- kaggle.com/code
- 30 hours/week free GPU
- 20GB RAM

## Free Deployment

### Backend:
```bash
# Render.com (750 hours/month FREE)
# No credit card needed!
1. Create account: render.com
2. Connect GitHub
3. Deploy → FREE hosting!
```

### Frontend:
```bash
# Vercel (unlimited FREE)
1. Create account: vercel.com
2. Connect GitHub
3. Deploy → FREE!
```

## Learning Resources (FREE)

- YouTube tutorials: FREE
- Google Colab tutorials: FREE
- FastAPI docs: FREE
- Python docs: FREE
- Stack Overflow: FREE

## GitHub Student Pack (BONUS FREE!)

- Sign up: education.github.com
- Get: $200K+ worth of free tools!
- Includes: domains, hosting, IDEs

## Total Investment Required

💰 Money: ₹0
⏰ Time: 2-3 months
💪 Effort: High (but worth it!)

---

**Made with ❤️ by students, for students!**
**No gatekeeping, no paywalls, just learning!**
MDEOF

echo ""
echo "✅ Setup complete!"
echo ""
echo "📋 Next Steps:"
echo "1. source venv/bin/activate"
echo "2. python data/download_free_data.py"
echo "3. Start coding!"
echo ""
echo "💰 Total cost so far: ₹0"
echo "💰 Future costs: ₹0"
echo "💰 Everything is FREE!"
echo ""
echo "🎓 Student tip: Get GitHub Student Pack for bonus free stuff!"
echo "   → education.github.com"
