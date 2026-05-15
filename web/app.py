import streamlit as st
import sys
import os
import tempfile
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

st.set_page_config(
    page_title="MedCheck",
    layout="wide"
)


@st.cache_resource
def load_pipeline(api_key=None):
    try:
        from drug_interactions.medcheck_pipeline import MedCheckPipeline
        return MedCheckPipeline(groq_api_key=api_key)
    except Exception as e:
        st.error(f"Pipeline load error: {e}")
        return None


st.markdown("""
<style>
    :root {
        --bg: #020617;
        --panel: #0f172a;
        --panel-soft: #111827;
        --border: rgba(255,255,255,0.10);
        --text: #ffffff;
        --muted: #94a3b8;
        --blue: #3b82f6;
        --red: #dc2626;
        --amber: #d97706;
        --green: #16a34a;
    }

    .stApp {
        background: var(--bg);
        color: var(--text);
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2.25rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        background: #020617;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] * {
        color: var(--text);
    }

    h1, h2, h3, h4, p, li, label, span {
        letter-spacing: 0;
    }

    .app-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        padding: 1.15rem 1.25rem;
        border: 1px solid var(--border);
        border-radius: 12px;
        background: var(--panel);
        margin-bottom: 1.5rem;
    }

    .brand-wrap {
        display: flex;
        align-items: center;
        gap: 0.85rem;
    }

    .pill-mark {
        width: 34px;
        height: 18px;
        border-radius: 999px;
        background: linear-gradient(90deg, #ffffff 0 48%, #3b82f6 48% 100%);
        border: 1px solid rgba(255,255,255,0.24);
        transform: rotate(-28deg);
        flex: 0 0 auto;
    }

    .brand-name {
        font-size: 1.45rem;
        font-weight: 700;
        color: var(--text);
        line-height: 1.05;
    }

    .brand-tagline {
        color: var(--muted);
        font-size: 0.95rem;
        margin-top: 0.22rem;
    }

    .mode-chip {
        color: var(--muted);
        border: 1px solid var(--border);
        background: rgba(255,255,255,0.04);
        border-radius: 999px;
        padding: 0.45rem 0.75rem;
        font-size: 0.86rem;
        white-space: nowrap;
    }

    .landing {
        max-width: 760px;
        margin: 4rem auto 1rem auto;
        padding: 2rem;
        border: 1px solid var(--border);
        border-radius: 12px;
        background: var(--panel);
    }

    .landing h1 {
        margin: 0.6rem 0 0.35rem 0;
        color: var(--text);
        font-size: 2.4rem;
        line-height: 1.05;
    }

    .landing p {
        color: var(--muted);
        font-size: 1rem;
        line-height: 1.6;
    }

    .section-card, .result-card, .metric-card {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.25rem;
        margin: 0.85rem 0;
    }

    .section-title {
        font-size: 1.05rem;
        font-weight: 650;
        color: var(--text);
        margin-bottom: 0.5rem;
    }

    .muted {
        color: var(--muted);
        line-height: 1.55;
    }

    .drug-list {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 0.75rem;
        margin-top: 0.75rem;
    }

    .drug-pill {
        background: rgba(255,255,255,0.04);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.7rem 0.85rem;
        color: var(--text);
        font-weight: 600;
    }

    .risk-band {
        border-left: 4px solid var(--green);
    }

    .risk-high {
        border-left-color: var(--red);
    }

    .risk-moderate {
        border-left-color: var(--amber);
    }

    .risk-low, .risk-none {
        border-left-color: var(--green);
    }

    .risk-unknown {
        border-left-color: var(--muted);
    }

    .risk-label {
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 0.28rem 0.62rem;
        font-weight: 700;
        font-size: 0.82rem;
        color: #ffffff;
        background: var(--green);
    }

    .risk-label.high {
        background: var(--red);
    }

    .risk-label.moderate {
        background: var(--amber);
    }

    .risk-label.unknown {
        background: #475569;
    }

    .pair-title {
        font-size: 1.22rem;
        font-weight: 700;
        color: var(--text);
        margin: 0.75rem 0 0.45rem 0;
    }

    .source-line {
        color: var(--muted);
        font-size: 0.9rem;
        margin-top: 0.9rem;
    }

    .stat-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
        gap: 0.85rem;
        margin: 1rem 0 1.2rem 0;
    }

    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--text);
        margin-top: 0.25rem;
    }

    .metric-card .label {
        color: var(--muted);
        font-size: 0.88rem;
    }

    .footer {
        margin-top: 2.5rem;
        padding-top: 1rem;
        border-top: 1px solid var(--border);
        color: var(--muted);
        font-size: 0.86rem;
        line-height: 1.65;
    }

    .stButton > button {
        background: var(--blue);
        color: #ffffff;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 0.7rem 1rem;
        font-weight: 650;
    }

    .stButton > button:hover {
        background: #2563eb;
        color: #ffffff;
        border-color: rgba(255,255,255,0.16);
    }

    .stTextArea textarea, .stTextInput input {
        background: #0b1220;
        color: var(--text);
        border: 1px solid var(--border);
        border-radius: 10px;
    }

    .stRadio [role="radiogroup"] {
        gap: 0.75rem;
    }

    div[data-testid="stExpander"] {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 12px;
    }

    div[data-testid="stExpander"] details {
        border: 0;
    }

    a {
        color: #93c5fd;
    }

    @media (max-width: 720px) {
        .app-header {
            align-items: flex-start;
            flex-direction: column;
        }

        .mode-chip {
            white-space: normal;
        }

        .landing {
            margin-top: 1.5rem;
            padding: 1.25rem;
        }
    }
</style>
""", unsafe_allow_html=True)


def safe_text(value, default="Not available"):
    if value is None:
        return default
    text = str(value).strip()
    return escape(text) if text else default


def severity_key(result):
    severity = str(result.get('overall_severity') or 'UNKNOWN').upper()
    if not result.get('interaction_found') and severity in {'LOW', 'NONE', 'SAFE', 'NO INTERACTION'}:
        return 'NONE'
    if severity in {'NONE', 'SAFE', 'NO INTERACTION'}:
        return 'NONE'
    if severity in {'HIGH', 'MODERATE', 'LOW'}:
        return severity
    if result.get('interaction_found'):
        return 'MODERATE'
    return 'UNKNOWN'


def severity_display(severity):
    if severity == 'NONE':
        return 'NO CLINICALLY SIGNIFICANT INTERACTION DETECTED'
    return severity


def severity_class(severity):
    if severity == 'HIGH':
        return 'high'
    if severity == 'MODERATE':
        return 'moderate'
    if severity in {'LOW', 'NONE'}:
        return 'low'
    return 'unknown'


def recommendation_for(result):
    safety_net = result.get('safety_net') or {}
    action = safety_net.get('action')
    if action:
        return str(action)

    severity = severity_key(result)
    if severity == 'HIGH':
        return "Do not combine these medicines unless a clinician specifically tells you to. Contact your doctor or pharmacist before taking them together."
    if severity == 'MODERATE':
        return "Ask a doctor or pharmacist before combining these medicines. Monitoring or dose adjustment may be needed."
    if severity == 'LOW':
        return "A serious interaction was not identified, but continue to follow the directions from your prescriber and pharmacist."
    if severity == 'NONE':
        return "No clinically significant interaction was detected by the checked sources. This is not a medical clearance; confirm with your healthcare provider."
    return "The available information is not enough to make a clear safety call. Ask a clinician or pharmacist before relying on this result."


def plain_summary(result):
    safety_net = result.get('safety_net') or {}
    if safety_net.get('found') and safety_net.get('description'):
        return str(safety_net['description'])

    explanation = str(result.get('explanation') or '').strip()
    blocked_terms = ('xgboost', 'rag', 'faiss', 'embedding', 'vector', 'ml evidence', 'faers evidence')
    if explanation and not any(term in explanation.lower() for term in blocked_terms):
        first_para = explanation.split('\n\n')[0].strip()
        if len(first_para) <= 700:
            return first_para

    severity = severity_key(result)
    drug1 = result.get('drug1', 'Drug 1')
    drug2 = result.get('drug2', 'Drug 2')
    if severity in {'HIGH', 'MODERATE'}:
        return f"A possible interaction was detected between {drug1} and {drug2}. Review this combination with a healthcare professional before using them together."
    if severity in {'LOW', 'NONE'}:
        return f"No clinically significant interaction was detected between {drug1} and {drug2} by the checked sources."
    return f"The checked sources did not provide enough clear information to classify the interaction between {drug1} and {drug2}."


def render_header(user_type):
    is_tech = user_type == "Technical / Developer"
    tagline = "AI-Powered Drug Interaction Detection Platform" if is_tech else "Drug Interaction Safety Checker"
    st.markdown(
        f"""
        <div class="app-header">
            <div class="brand-wrap">
                <div class="pill-mark" aria-hidden="true"></div>
                <div>
                    <div class="brand-name">MedCheck</div>
                    <div class="brand-tagline">{tagline}</div>
                </div>
            </div>
            <div class="mode-chip">{safe_text(user_type)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer(user_type):
    tech_line = ""
    if user_type == "Technical / Developer":
        tech_line = "<br>Built with Python, XGBoost, Sentence Transformers, NumPy, Streamlit"
    st.markdown(
        f"""
        <div class="footer">
            MedCheck | Built by Mohammed Saif | Master's Student, Seattle University, United States<br>
            Contact: smohammed8@seattleu.edu{tech_line}<br>
            For informational purposes only. Always consult your healthcare provider.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_landing():
    st.markdown(
        """
        <div class="landing">
            <div class="brand-wrap">
                <div class="pill-mark" aria-hidden="true"></div>
                <div class="brand-name">MedCheck</div>
            </div>
            <h1>Drug interaction safety, shown clearly.</h1>
            <p>Select the experience that matches how you want to read the results.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    choice = st.radio(
        "I am a...",
        ["Healthcare Professional / Patient", "Technical / Developer"],
        horizontal=True,
        key="user_type_choice",
    )

    if st.button("Continue", type="primary", use_container_width=True):
        st.session_state['user_type'] = choice
        st.rerun()


def render_sidebar(user_type):
    with st.sidebar:
        if st.button("Change user type"):
            st.session_state.pop('user_type', None)
            st.rerun()

        st.markdown("---")
        if user_type == "Technical / Developer":
            st.markdown("### AI Stack")
            st.markdown("- XGBoost severity model")
            st.markdown("- RAG evidence retrieval")
            st.markdown("- FDA FAERS real-time checks")
            st.markdown("- Groq LLM explanation")
            st.markdown("- Evidence validation")

            st.markdown("### Data Sources")
            st.markdown("- [FDA FAERS](https://www.fda.gov/drugs/questions-and-answers-fdas-adverse-event-reporting-system-faers)")
            st.markdown("- [openFDA Drug Labels](https://open.fda.gov/apis/drug/label/)")
            st.markdown("- [RxNorm](https://www.nlm.nih.gov/research/umls/rxnorm/)")

            st.markdown("### Model Metrics")
            st.markdown("- Accuracy: 99.97%")
            st.markdown("- Training samples: 15881")
            st.markdown("- RAG documents: 48")
            key_label = "Groq API Key"
        else:
            st.markdown("### How to use")
            st.markdown("1. Upload a prescription image or enter drug names.")
            st.markdown("2. Review each medicine pair.")
            st.markdown("3. Confirm important decisions with a clinician.")
            key_label = "Optional: AI Enhancement Key"

        st.markdown("---")
        groq_key = st.text_input(key_label, type="password")
        st.markdown("---")
        st.markdown("### Disclaimer")
        st.markdown("For informational purposes only. Always consult your healthcare provider.")
        st.markdown("Contact: smohammed8@seattleu.edu")
    return groq_key


def render_drug_chips(drugs):
    chips = "".join(f'<div class="drug-pill">{safe_text(drug)}</div>' for drug in drugs)
    st.markdown(f'<div class="drug-list">{chips}</div>', unsafe_allow_html=True)


def extract_meds_from_ocr_result(ocr_result):
    if not ocr_result:
        return []

    medications = ocr_result.get('medications')
    if medications:
        return [str(med).strip().title() for med in medications if str(med).strip()]

    verified_drugs = ocr_result.get('drugs_verified', [])
    return [
        drug.get('matched', drug.get('original', '')).title()
        for drug in verified_drugs
        if drug.get('matched') or drug.get('original')
    ]


def render_input_area():
    st.markdown('<div class="section-card"><div class="section-title">Check medications</div><div class="muted">Enter at least two medicines. Every unique pair will be checked.</div></div>', unsafe_allow_html=True)
    mode = st.radio(
        "Choose input method",
        ["Upload Prescription Image", "Enter Drug Names Manually"],
        horizontal=True,
    )

    drugs_to_check = []

    if mode == "Upload Prescription Image":
        uploaded_file = st.file_uploader("Upload Prescription Image", type=['png', 'jpg', 'jpeg'])

        if uploaded_file:
            col1, col2 = st.columns([1, 1])

            with col1:
                st.image(uploaded_file, caption="Uploaded prescription", use_container_width=True)

            with col2:
                st.markdown("#### Image analysis")
                if st.button("Extract Medications", type="primary"):
                    with st.spinner("Reading prescription with OCR..."):
                        try:
                            from ocr.advanced_ocr import AdvancedOCR

                            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                                tmp.write(uploaded_file.getbuffer())
                                tmp_path = tmp.name

                            ocr = AdvancedOCR()
                            ocr_result = ocr.process_image(tmp_path)
                            os.unlink(tmp_path)

                            meds = extract_meds_from_ocr_result(ocr_result)

                            if meds:
                                st.success(f"Found {len(meds)} medications.")
                                st.session_state['extracted_drugs'] = meds
                                render_drug_chips(meds)
                            else:
                                st.warning("Could not extract medications. Try manual entry.")
                        except Exception as e:
                            st.error(f"OCR Error: {e}")
                            st.info("Use manual entry mode instead.")

        if 'extracted_drugs' in st.session_state:
            drugs_to_check = st.session_state['extracted_drugs']
            render_drug_chips(drugs_to_check)

    else:
        drug_input = st.text_area(
            "Medications",
            placeholder="Warfarin\nAspirin\nMetformin\nLisinopril",
            height=150,
        )

        if drug_input:
            drugs_to_check = [d.strip() for d in drug_input.strip().split('\n') if d.strip()]
            if drugs_to_check:
                st.success(f"{len(drugs_to_check)} medications entered.")
                render_drug_chips(drugs_to_check)

    return drugs_to_check


def render_nontech_result(result):
    severity = severity_key(result)
    risk_class = severity_class(severity)
    drug1 = safe_text(result.get('drug1'))
    drug2 = safe_text(result.get('drug2'))
    fda_rt = result.get('fda_realtime') or {}
    safety_net = result.get('safety_net') or {}

    st.markdown(
        f"""
        <div class="result-card risk-band risk-{risk_class}">
            <span class="risk-label {risk_class}">{severity_display(severity)}</span>
            <div class="pair-title">{drug1} + {drug2}</div>
            <div class="muted">{safe_text(plain_summary(result))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if safety_net.get('found') and safety_net.get('description'):
        st.markdown(
            f"""
            <div class="section-card">
                <div class="section-title">Clinical safety note</div>
                <div class="muted">{safe_text(safety_net.get('description'))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="section-card">
            <div class="section-title">What to do</div>
            <div class="muted">{safe_text(recommendation_for(result))}</div>
            <div class="source-line">Source: FDA FAERS Database</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    reactions = fda_rt.get('top_reactions') or safety_net.get('reactions') or []
    if reactions:
        st.markdown("#### Reported side effects")
        for reaction in reactions[:6]:
            st.markdown(f"- {reaction}")


def render_technical_result(result, index):
    severity = severity_key(result)
    risk_class = severity_class(severity)
    drug1 = result.get('drug1', 'Drug 1')
    drug2 = result.get('drug2', 'Drug 2')

    st.markdown(
        f"""
        <div class="result-card risk-band risk-{risk_class}">
            <span class="risk-label {risk_class}">{safe_text(severity_display(severity))}</span>
            <div class="pair-title">{safe_text(drug1)} + {safe_text(drug2)}</div>
            <div class="muted">Sources used: {safe_text(", ".join(result.get('sources_used', [])) or "None reported")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    rxnorm = result.get('rxnorm_interaction') or {}
    with st.expander(f"Step 1. RxNorm real-time API", expanded=index == 1):
        if rxnorm.get('found'):
            st.write(f"Severity: {rxnorm.get('severity', 'UNKNOWN')}")
            st.write(rxnorm.get('description', 'RxNorm reports an interaction for this pair.'))
            st.write(f"Normalized: {rxnorm.get('drug1_normalized', drug1)} + {rxnorm.get('drug2_normalized', drug2)}")
            st.write(f"RxCUI: {rxnorm.get('drug1_rxcui', 'N/A')} + {rxnorm.get('drug2_rxcui', 'N/A')}")
        elif rxnorm.get('api_unavailable'):
            st.warning("RxNorm interaction data was unavailable for this check.")
        else:
            st.write("No RxNorm interaction returned.")

    fda_label = result.get('fda_label') or {}
    with st.expander("Step 2. FDA drug label evidence", expanded=index == 1):
        if fda_label.get('found') or fda_label.get('general_evidence_found'):
            st.write(f"Severity: {fda_label.get('severity', 'UNKNOWN')}")
            st.write(f"Section: {fda_label.get('section', 'N/A')}")
            st.write(f"Label drug: {fda_label.get('label_drug', 'N/A')}")
            if fda_label.get('class_based'):
                st.info(fda_label.get('note', 'FDA label evidence found through a drug-class match.'))
            st.write(fda_label.get('snippet', 'No excerpt available.'))
        else:
            st.write("No FDA label interaction text found.")

    ml = result.get('ml_prediction') or {}
    with st.expander("Step 3. ML prediction", expanded=False):
        st.write(f"Severity: {ml.get('severity', 'UNKNOWN')}")
        if ml.get('confidence') is not None:
            st.write(f"Confidence: {ml.get('confidence')}")
        probabilities = ml.get('probabilities') or {}
        if probabilities:
            st.write("Probability scores")
            st.json(probabilities)
        fda_data = ml.get('fda_data') or {}
        if fda_data:
            st.write(f"FDA reports in model features: {fda_data.get('total_reports', 0):,}")
            st.write(f"Serious reports in model features: {fda_data.get('serious_reports', 0):,}")
        st.caption("ML is displayed for transparency and should not override clinical evidence.")

    rag = result.get('rag_evidence') or []
    with st.expander(f"Step 4. RAG evidence ({len(rag)} documents)", expanded=False):
        if rag:
            for item in rag[:5]:
                metadata = item.get('metadata') or {}
                st.markdown(
                    f"""
                    <div class="section-card">
                        <div class="section-title">{safe_text(metadata.get('drug1', drug1)).title()} + {safe_text(metadata.get('drug2', drug2)).title()}</div>
                        <div class="muted">Severity: {safe_text(metadata.get('severity', 'UNKNOWN'))} | Reports: {safe_text(metadata.get('total_reports', 0))} | Similarity score: {item.get('score', 0):.2f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.write("No matching RAG evidence returned.")

    fda_rt = result.get('fda_realtime') or {}
    with st.expander("Step 5. FDA FAERS real-time API", expanded=False):
        if fda_rt.get('found'):
            st.write(f"Reports: {fda_rt.get('total_reports', 0):,}")
            st.write(f"Signal: {fda_rt.get('severity', 'UNKNOWN')}")
            if fda_rt.get('query'):
                st.code(fda_rt.get('query'), language="text")
            reactions = fda_rt.get('top_reactions') or []
            if reactions:
                st.write("Top reactions")
                for reaction in reactions[:8]:
                    st.write(f"- {reaction}")
        elif fda_rt.get('error'):
            st.warning("FDA FAERS API was unavailable.")
            if fda_rt.get('query'):
                st.code(fda_rt.get('query'), language="text")
        else:
            st.write("No real-time FDA FAERS reports found.")

    with st.expander("Step 6. LLM explanation", expanded=False):
        st.markdown(result.get('explanation') or "No explanation available.")

    validation = result.get('validation') or {}
    with st.expander("Step 7. RAG validation", expanded=False):
        if validation.get('validated'):
            st.success("Validated: explanation is grounded in retrieved evidence.")
        else:
            st.warning("Not validated against local retrieved evidence.")
        st.write(f"Validation confidence: {validation.get('confidence', 0):.2f}")


def render_results(results, user_type):
    total_pairs = len(results)
    interaction_count = len([r for r in results if r.get('interaction_found')])
    high_count = len([r for r in results if severity_key(r) == 'HIGH'])
    review_count = len([r for r in results if severity_key(r) == 'UNKNOWN'])

    st.markdown("## Results")
    st.markdown(
        f"""
        <div class="stat-grid">
            <div class="metric-card"><div class="label">Pairs checked</div><div class="value">{total_pairs}</div></div>
            <div class="metric-card"><div class="label">Interactions detected</div><div class="value">{interaction_count}</div></div>
            <div class="metric-card"><div class="label">High risk</div><div class="value">{high_count}</div></div>
            <div class="metric-card"><div class="label">Needs review</div><div class="value">{review_count}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for index, result in enumerate(results, 1):
        if user_type == "Technical / Developer":
            render_technical_result(result, index)
        else:
            render_nontech_result(result)


def render_technical_reference():
    st.markdown("## Pipeline Architecture")
    st.code(
        """User Input -> OCR/Manual Entry
    -> RxNorm API: real-time drug normalization and interaction lookup
    -> FDA Drug Label API: FDA-approved prescribing information
    -> XGBoost ML Model: supportive severity prediction
    -> RAG Vector Database: retrieves local FDA evidence
    -> FDA FAERS API: real-time adverse event co-reporting
    -> Groq LLM: explanation generation
    -> RAG Validation: evidence support check
    -> Evidence-based Report""",
        language="text",
    )

    st.markdown("## Data Sources")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**FDA FAERS**  \n[Official FDA page](https://www.fda.gov/drugs/questions-and-answers-fdas-adverse-event-reporting-system-faers)")
    c2.markdown("**openFDA Drug Labels**  \n[API documentation](https://open.fda.gov/apis/drug/label/)")
    c3.markdown("**RxNorm**  \n[NLM RxNorm](https://www.nlm.nih.gov/research/umls/rxnorm/)")


if 'user_type' not in st.session_state:
    render_landing()
    st.stop()


user_type = st.session_state['user_type']
render_header(user_type)
groq_key = render_sidebar(user_type)
drugs_to_check = render_input_area()

if drugs_to_check and len(drugs_to_check) >= 2:
    if st.button("Analyze Drug Interactions", type="primary", use_container_width=True):
        pipeline = load_pipeline(api_key=groq_key if groq_key else None)

        if not pipeline:
            st.error("Failed to load pipeline. Check model files.")
            st.stop()

        with st.spinner("Checking each medication pair..."):
            all_results = []
            total_drugs = len(drugs_to_check)
            for i in range(total_drugs):
                for j in range(i + 1, total_drugs):
                    result = pipeline.check_interaction(drugs_to_check[i], drugs_to_check[j])
                    all_results.append(result)

        render_results(all_results, user_type)

        if user_type == "Technical / Developer":
            render_technical_reference()

elif drugs_to_check and len(drugs_to_check) < 2:
    st.warning("Enter at least 2 medications to check interactions.")
else:
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Ready when you are</div>
            <div class="muted">Upload a prescription image or enter drug names manually to begin.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

render_footer(user_type)
