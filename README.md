# MedCheck

**AI-Powered Drug Interaction Detection Platform**

A production-grade 7-stage AI pipeline that detects dangerous drug interactions using real FDA government data. Built to protect patients by cross-referencing multiple clinical sources before any interaction reaches the user.

**Live Demo:** [medcheck-ai-saif.streamlit.app](https://medcheck-ai-saif.streamlit.app)

---

## How It Works

MedCheck uses a multi-source validation approach — the same methodology used by hospital clinical decision support systems. No single model makes the final call. Instead, seven independent stages contribute evidence, and the system synthesizes a clinically grounded result.

```
User Input (OCR or Manual Entry)
    │
    ├── Step 1: RxNorm API ─────────── Drug normalization + interaction lookup
    ├── Step 2: FDA Drug Label API ─── FDA-approved prescribing information
    ├── Step 3: XGBoost ML Model ───── Severity prediction (99.97% accuracy)
    ├── Step 4: RAG Vector Database ── Retrieves local FDA evidence
    ├── Step 5: FDA FAERS API ──────── Real-time adverse event reports (10M+)
    ├── Step 6: Groq LLM ──────────── Natural language explanation
    └── Step 7: RAG Validation ─────── Verifies LLM output against evidence
                │
                ▼
        Evidence-Based Report
```

### Why Multi-Source Validation Matters

In real-world testing, the ML model predicted warfarin + aspirin as LOW risk (due to limited training data). However, the FDA Drug Label API returned explicit label evidence listing aspirin as a bleeding risk with warfarin, and the FDA FAERS API returned 16,020 adverse event reports. The system correctly overrode the ML prediction and reported HIGH risk.

This is the difference between a demo project and a clinical-grade system.

---

## Features

- **Prescription OCR** — Scan prescription bottles using Tesseract + EasyOCR with OpenCV preprocessing
- **7-Stage AI Pipeline** — RxNorm, FDA Labels, XGBoost, RAG, FAERS, LLM, and Validation
- **Real FDA Data** — Every interaction backed by government clinical data, not hardcoded dictionaries
- **Anti-Hallucination** — RAG validation checks LLM explanations against retrieved evidence
- **Two User Modes** — Non-technical (clean results) and Technical (full pipeline transparency)
- **Evidence Citations** — FDA label sections, FAERS report counts, PubMed references
- **Professional UI** — B2B SaaS interface with proper clinical disclaimers

---

## Data Sources

| Source | Type | Coverage |
|--------|------|----------|
| FDA FAERS | Adverse event reports | 10M+ real patient reports |
| FDA Drug Labels | Prescribing information | All FDA-approved drugs |
| RxNorm (NIH/NLM) | Drug normalization | Clinical standard used by Epic, Cerner |
| PubMed | Clinical literature | 30M+ peer-reviewed papers |

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| ML Model | XGBoost (trained on FDA FAERS data, 99.97% accuracy) |
| RAG | Sentence Transformers (all-MiniLM-L6-v2) + NumPy vector search |
| LLM | Groq API (Llama 3.1) |
| Computer Vision | OpenCV + Tesseract + EasyOCR |
| Backend | Python, pandas, scikit-learn |
| Frontend | Streamlit |
| APIs | FDA FAERS, FDA Drug Labels, RxNorm, PubMed |
| Deployment | Streamlit Cloud |

---

## Installation

```bash
git clone https://github.com/Mohammed-Saif-07/medcheck-ai.git
cd medcheck-ai

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### System Dependencies

Tesseract OCR is required for prescription scanning:

```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-eng
```

### Run Locally

```bash
streamlit run web/app.py
```

### Optional: Groq API Key

For LLM-powered explanations, get a free API key from [console.groq.com](https://console.groq.com) and enter it in the sidebar.

---

## Project Structure

```
medcheck-ai/
├── web/
│   └── app.py                          # Streamlit application
├── src/
│   ├── ocr/
│   │   └── advanced_ocr.py             # Multi-model OCR pipeline
│   ├── drug_interactions/
│   │   ├── medcheck_pipeline.py         # Core 7-stage pipeline
│   │   ├── fda_api_checker.py           # FDA FAERS API integration
│   │   └── clinical_safety_net.py       # Critical interaction fallback
│   ├── ml/
│   │   └── train_interaction_model.py   # XGBoost training
│   ├── rag/
│   │   └── rag_retriever.py            # RAG evidence retrieval
│   └── llm/
│       └── llm_explainer.py            # LLM explanation generation
├── models/
│   ├── xgboost_model.json              # Trained ML model
│   ├── drug_label_encoder.pkl          # Drug name encoder
│   ├── rag_embeddings.npy              # Vector embeddings
│   ├── rag_metadata.pkl                # RAG metadata
│   └── fda_interactions.csv            # FDA training data
├── data/                               # Datasets
├── tests/                              # Test suite
├── scripts/                            # Data processing scripts
├── packages.txt                        # System dependencies
├── requirements.txt                    # Python dependencies
└── README.md
```

---

## Model Performance

| Metric | Value |
|--------|-------|
| XGBoost Accuracy | 99.97% |
| Training Samples | 15,881 (FDA FAERS data) |
| Drug Coverage | 1,885 unique drugs |
| RAG Documents | 48 evidence vectors |
| FAERS Coverage | 10M+ adverse event reports |

---

## Example Results

**Warfarin + Aspirin:**
- ML Prediction: LOW (limited training data)
- FDA Drug Label: HIGH (explicit label warning)
- FDA FAERS: 16,020 adverse event reports
- Final Result: **HIGH** (FDA label evidence overrides ML)
- System correctly identifies the interaction using multi-source validation

**Metformin + Lisinopril:**
- All sources: No significant interaction
- Final Result: **SAFE**

---

## Training Pipeline

The ML model and RAG database were trained using a Google Colab notebook with GPU acceleration:

1. Downloaded FDA FAERS adverse event data via API (5,000+ records)
2. Processed into 15,881 drug pair training samples
3. Trained XGBoost classifier on GPU (Tesla T4)
4. Built sentence embeddings using all-MiniLM-L6-v2
5. Created NumPy-based vector search index
6. Exported models in cross-platform format (JSON + pickle)

---

## Disclaimer

This tool is for informational and educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult your doctor or pharmacist about medication interactions.

---

## Contact

**Mohammed Saif**
Master's Student, Seattle University, United States

Email: smohammed8@seattleu.edu
GitHub: [Mohammed-Saif-07](https://github.com/Mohammed-Saif-07)
