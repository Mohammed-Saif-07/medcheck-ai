"""
MedCheck - Beautiful Web Interface
Claude-inspired design + Mobile responsive

Professional, modern, and gorgeous! ✨
"""

import streamlit as st
import sys
from pathlib import Path
import json
from datetime import datetime
from PIL import Image
import io
import base64

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ocr.advanced_ocr import AdvancedOCR
from src.ml.train_interaction_model import InteractionClassifier
from src.rag.rag_retriever import RAGRetriever
from src.llm.llm_explainer import LLMExplainer

# Page config
st.set_page_config(
    page_title="MedCheck - AI Drug Safety",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Claude-inspired beautiful design
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main container */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem 1rem;
    }
    
    /* Header */
    .header {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .header h1 {
        color: #1a1a1a;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .header p {
        color: #666;
        font-size: 1.1rem;
    }
    
    /* Cards */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
        border-left: 4px solid #6366f1;
    }
    
    /* Upload area */
    .uploadedFile {
        border-radius: 12px !important;
        border: 2px dashed #6366f1 !important;
        background: #f8f9ff !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: transform 0.2s;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(102,126,234,0.4);
    }
    
    /* Status badges */
    .status-safe {
        background: #10b981;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    .status-warning {
        background: #f59e0b;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    .status-danger {
        background: #ef4444;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    /* Drug cards */
    .drug-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        font-weight: 500;
    }
    
    /* Info boxes */
    .info-box {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .danger-box {
        background: #fee2e2;
        border-left: 4px solid #ef4444;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Progress */
    .stProgress > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: white;
    }
    
    /* Mobile responsive */
    @media (max-width: 768px) {
        .header h1 {
            font-size: 1.8rem;
        }
        
        .main {
            padding: 1rem 0.5rem;
        }
        
        .card {
            padding: 1rem;
        }
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Animations */
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .card {
        animation: slideIn 0.5s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'result' not in st.session_state:
    st.session_state.result = None

# Initialize AI components (cached)
@st.cache_resource
def load_ai_components():
    """Load all AI components (cached for performance)"""
    ocr = AdvancedOCR()
    
    classifier = InteractionClassifier()
    classifier.load_model()
    
    rag = RAGRetriever()
    
    llm = LLMExplainer()
    
    return ocr, classifier, rag, llm

# Header
st.markdown("""
<div class="header">
    <h1>💊 MedCheck AI</h1>
    <p>AI-Powered Drug Interaction Detection</p>
    <p style="font-size: 0.9rem; color: #999;">Computer Vision • Machine Learning • RAG • LLM</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📋 How It Works")
    st.markdown("""
    1. **Upload** prescription image
    2. **AI analyzes** using 4 technologies
    3. **Get** instant safety report
    
    ---
    
    ### 🔬 AI Technologies
    
    ✅ **Computer Vision** (OCR)  
    ✅ **Machine Learning** (XGBoost)  
    ✅ **RAG** (FDA Database)  
    ✅ **LLM** (Natural Language)
    
    ---
    
    ### ⚠️ Disclaimer
    
    This is an AI tool for informational purposes only. Always consult your doctor or pharmacist.
    
    ---
    
    ### 💻 Tech Stack
    
    Python • OpenCV • Tesseract  
    XGBoost • scikit-learn  
    Vector DB • FDA API • LLM
    """)
    
    st.markdown("---")
    st.markdown("Built with ❤️ by Data Science Student")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📸 Upload Prescription Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear photo of your prescription bottle or label"
    )
    
    if uploaded_file:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Prescription", use_container_width=True)
        
        # Analyze button
        if st.button("🔍 Analyze Prescription", use_container_width=True):
            with st.spinner("🤖 AI is analyzing your prescription..."):
                # Save temp image
                temp_path = Path("temp_prescription.png")
                image.save(temp_path)
                
                # Load AI components
                ocr, classifier, rag, llm = load_ai_components()
                
                # Process
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Stage 1: OCR
                status_text.text("📸 Reading prescription...")
                progress_bar.progress(25)
                ocr_result = ocr.process_image(str(temp_path))
                
                if 'error' not in ocr_result:
                    verified_drugs = ocr_result.get('drugs_verified', [])
                    drug_names = [d.get('matched', d.get('original', '')) for d in verified_drugs]
                    
                    # Stage 2: ML
                    status_text.text("🤖 Checking interactions...")
                    progress_bar.progress(50)
                    
                    interactions = []
                    if len(drug_names) >= 2:
                        from itertools import combinations
                        for drug1, drug2 in combinations(drug_names, 2):
                            prediction = classifier.predict_interaction(drug1, drug2)
                            if prediction['severity'] != 'NONE':
                                interactions.append({
                                    'drug1': drug1,
                                    'drug2': drug2,
                                    'severity': prediction['severity'],
                                    'confidence': prediction['confidence']
                                })
                    
                    # Stage 3: RAG (if interactions)
                    if interactions:
                        status_text.text("📚 Finding evidence...")
                        progress_bar.progress(75)
                        for interaction in interactions:
                            evidence = rag.get_interaction_evidence(
                                interaction['drug1'], interaction['drug2']
                            )
                            interaction['evidence'] = evidence
                    
                    # Stage 4: LLM
                    status_text.text("💬 Generating explanation...")
                    progress_bar.progress(90)
                    
                    if interactions:
                        for interaction in interactions:
                            explanation = llm.explain_interaction(
                                interaction['drug1'],
                                interaction['drug2'],
                                interaction['severity'],
                                interaction.get('evidence')
                            )
                            interaction['explanation'] = explanation
                    else:
                        safe_explanation = llm.explain_safe_combination(
                            drug_names[0], drug_names[1]
                        ) if len(drug_names) >= 2 else None
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Analysis complete!")
                    
                    # Store result
                    st.session_state.result = {
                        'drugs': drug_names,
                        'interactions': interactions,
                        'safe_explanation': safe_explanation if not interactions and len(drug_names) >= 2 else None,
                        'ocr_confidence': ocr_result.get('ocr_confidence', 0)
                    }
                    st.session_state.processed = True
                    
                    # Clean up
                    temp_path.unlink()
                    
                    st.rerun()

with col2:
    if st.session_state.processed and st.session_state.result:
        result = st.session_state.result
        
        st.markdown("### 📊 Analysis Results")
        
        # Detected drugs
        st.markdown("#### 💊 Detected Medications")
        for drug in result['drugs']:
            st.markdown(f'<div class="drug-card">✓ {drug.upper()}</div>', unsafe_allow_html=True)
        
        st.markdown(f"<p style='color: #1a1a1a; font-weight: 600;'>Confidence: {result['ocr_confidence']*100:.0f}%</p>", unsafe_allow_html=True)
        
        # Interactions
        st.markdown("#### ⚠️ Interaction Analysis")
        
        if result['interactions']:
            for i, interaction in enumerate(result['interactions'], 1):
                severity = interaction['severity']
                
                if severity == 'HIGH':
                    box_class = "danger-box"
                    status_class = "status-danger"
                    icon = "🔴"
                elif severity == 'MODERATE':
                    box_class = "warning-box"
                    status_class = "status-warning"
                    icon = "🟡"
                else:
                    box_class = "info-box"
                    status_class = "status-safe"
                    icon = "🟢"
                
                st.markdown(f"""
                <div class="{box_class}">
                    <h4>{icon} Interaction #{i}</h4>
                    <p><strong>{interaction['drug1'].upper()} + {interaction['drug2'].upper()}</strong></p>
                    <span class="{status_class}">{severity} RISK</span>
                    <p style="margin-top: 1rem; line-height: 1.6;">
                        {interaction.get('explanation', 'Contact your doctor about this combination.')}
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            safe_text = result.get('safe_explanation', 'These medications appear safe to take together. Continue as prescribed and keep your doctor informed.')
            # Clean up any extra whitespace and format nicely
            safe_text = safe_text.strip()
            st.markdown(f"""
            <div class="info-box">
                <h4 style="color: #1a1a1a;">✅ No Dangerous Interactions Detected</h4>
                <div style="line-height: 1.6; white-space: pre-wrap; color: #333;">
{safe_text}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Download report button
        st.markdown("---")
        report_data = json.dumps(result, indent=2)
        st.download_button(
            label="📥 Download Full Report (JSON)",
            data=report_data,
            file_name=f"medcheck_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    else:
        # Welcome message
        st.markdown("""
        <div class="card">
            <h3>👋 Welcome to MedCheck AI!</h3>
            <p style="line-height: 1.8;">
                Upload a prescription image to get started. Our AI will:
            </p>
            <ul style="line-height: 2;">
                <li>📸 Read your prescription using Computer Vision</li>
                <li>🤖 Check for interactions with Machine Learning</li>
                <li>📚 Find evidence from FDA database</li>
                <li>💬 Explain in simple language</li>
            </ul>
            <p style="margin-top: 1rem; color: #666;">
                Results in less than 5 seconds! ⚡
            </p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>💊 <strong>MedCheck AI</strong> - Preventing Adverse Drug Interactions</p>
    <p style="font-size: 0.9rem;">Powered by Computer Vision • Machine Learning • RAG • LLM</p>
    <p style="font-size: 0.8rem; margin-top: 1rem;">
        ⚠️ For informational purposes only. Not a substitute for professional medical advice.
    </p>
</div>
""", unsafe_allow_html=True)
