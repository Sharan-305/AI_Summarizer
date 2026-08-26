"""
Streamlit Web Interface for Sharp AI Text Summarizer.

Provides an executive interface for ultra-concise, information-dense summarization
powered by Transformer architectures (BART, T5, PEGASUS).
"""

import os
import time
import json
import streamlit as st
import pandas as pd

import config
from src.preprocessing import clean_text
from src.pdf_processor import extract_text_from_pdf, extract_text_from_url
from src.summarizer import (
    TransformerSummarizer, 
    ExtractiveSummarizer, 
    generate_sharp_summary
)
from src.evaluation import format_evaluation_report

# Streamlit Page Configuration
st.set_page_config(
    page_title="SHARP AI SUMMARIZER",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Modern Glassmorphic CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .header-box {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
        padding: 28px 32px;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.4);
        margin-bottom: 24px;
        text-align: center;
    }
    
    .title-banner {
        font-weight: 800;
        font-size: 2.5rem;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 40%, #c084fc 80%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .subtitle-banner {
        color: #94a3b8;
        font-size: 1.05rem;
        font-weight: 400;
        margin-top: 8px;
    }
    
    .tagline-pill {
        display: inline-block;
        background: rgba(99, 102, 241, 0.2);
        border: 1px solid rgba(99, 102, 241, 0.4);
        color: #a5b4fc;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-top: 10px;
        text-transform: uppercase;
    }

    .summary-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.9) 100%);
        border: 2px solid #6366f1;
        border-radius: 16px;
        padding: 24px;
        font-size: 1.15rem;
        line-height: 1.7;
        color: #f8fafc;
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.2);
        margin-top: 12px;
        margin-bottom: 20px;
    }
    
    .summary-card-header {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #818cf8;
        margin-bottom: 12px;
    }

    .metric-container {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px 14px;
        text-align: center;
        backdrop-filter: blur(12px);
    }
    
    .metric-value {
        font-size: 1.9rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 6px;
    }

    .stButton>button {
        background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 50%, #d946ef 100%);
        color: white;
        font-size: 1.1rem;
        font-weight: 700;
        border-radius: 12px;
        border: none;
        padding: 14px 28px;
        width: 100%;
        box-shadow: 0 6px 20px rgba(124, 58, 237, 0.35);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 28px rgba(124, 58, 237, 0.5);
    }
</style>
""", unsafe_allow_html=True)


# Cache Transformer Model loading across user sessions
@st.cache_resource(show_spinner=False)
def load_transformer_model(model_name: str):
    return TransformerSummarizer(model_name_or_path=model_name)

@st.cache_resource(show_spinner=False)
def load_extractive_model():
    return ExtractiveSummarizer()


# Header Section
st.markdown("""
<div class="header-box">
    <div class="title-banner">⚡ SHARP AI SUMMARIZER</div>
    <div class="subtitle-banner">Transformer-Based Deep Learning Engine for Ultra-Concise Executive Summaries</div>
    <div class="tagline-pill">🎯 Core Principle: Maximum Information in Minimum Words</div>
</div>
""", unsafe_allow_html=True)


# Sidebar Configuration Panel
with st.sidebar:
    st.image("https://img.icons8.com/isometric/96/brain.png", width=60)
    st.title("Summarizer Controls")
    
    # Model Selection Options
    model_options = {
        "DistilBART CNN (Fast & Crisp)": "sshleifer/distilbart-cnn-12-6",
        "T5 Small (Lightweight CPU)": "t5-small",
        "BART Base (Standard Document)": "facebook/bart-base",
        "PEGASUS XSum (Extreme Concise)": "google/pegasus-xsum",
    }
    
    # Add custom fine-tuned model option if model directory exists and contains files
    if os.path.exists(config.BEST_MODEL_DIR) and len(os.listdir(config.BEST_MODEL_DIR)) > 0:
        model_options["⭐ Fine-Tuned Custom Model"] = config.BEST_MODEL_DIR

    selected_model_label = st.selectbox(
        "Transformer Model Architecture", 
        list(model_options.keys()),
        index=0,
        key="model_select"
    )
    selected_model_path = model_options[selected_model_label]

    # Summarization Strategy
    strategy_mode = st.radio(
        "Model Strategy",
        ["Abstractive (Transformer)", "Extractive (TextRank Graph)"],
        index=0,
        key="strategy_select"
    )
    use_extractive = "Extractive" in strategy_mode

    st.markdown("---")
    st.markdown("### Executive Summary Mode")
    
    summary_mode = st.radio(
        "Select Output Mode:",
        ["Ultra Short", "Short", "Brief", "Custom Budget"],
        index=1,
        help="Ultra Short: 1-2 sents | Short: 2-4 sents | Brief: 4-6 sents",
        key="summary_mode_radio"
    )

    custom_max_w = None
    if summary_mode == "Custom Budget":
        custom_max_w = st.number_input("Maximum Word Limit", min_value=10, max_value=400, value=50, step=5, key="max_words_num")
    else:
        preset_info = config.SUMMARY_MODES[summary_mode]
        st.caption(f"**Target:** `{preset_info['description']}` (max ~{preset_info['max_words']} words)")
        
        # Optional override slider
        enable_custom_slider = st.checkbox("Override max word limit?", value=False, key="enable_custom_slider")
        if enable_custom_slider:
            custom_max_w = st.slider("Maximum Words Limit", 15, 200, preset_info['max_words'], 5, key="max_words_slider")

    st.markdown("---")
    high_precision_toggle = st.checkbox("🎯 High Precision Mode (Ground 100% on Source)", value=True, key="high_precision_check", help="Anchors facts, numbers, and proper entities directly from uploaded document.")

    st.markdown("---")
    st.markdown("### Hardware Diagnostics")

    st.info(f"**Execution Device:** `{config.DEVICE.upper()}`\n\n**PyTorch Device Active:** `{config.DEVICE}`")


# Main Input Tabs
tab_text, tab_file, tab_pdf, tab_url, tab_eval = st.tabs([
    "📝 Direct Text Input", 
    "📄 Upload TXT", 
    "📕 Upload PDF", 
    "🌐 Article URL",
    "📊 Benchmarks & Ground Truth"
])

input_text = ""
reference_target = ""

with tab_text:
    st.markdown("##### Paste or type document text below:")
    input_text = st.text_area(
        "Source Document Text", 
        height=220, 
        placeholder="Paste your article, report, or document text here...\n\nExample:\nArtificial intelligence is becoming increasingly important across different industries. Companies are using AI to automate repetitive tasks, analyze large amounts of data, improve customer experiences, and support decision-making. Healthcare organizations use AI for medical diagnosis and drug discovery, while financial institutions use it for fraud detection and risk analysis. As AI technology continues to improve, organizations are expected to adopt it more widely.",
        key="text_area_input"
    )

with tab_file:
    st.markdown("##### Upload a raw text file (.txt):")
    uploaded_txt = st.file_uploader("Choose a .txt file", type=["txt"], key="txt_file_uploader")
    if uploaded_txt is not None:
        raw_bytes = uploaded_txt.read()
        try:
            input_text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            input_text = raw_bytes.decode("latin-1")
        st.success(f"File loaded: `{uploaded_txt.name}` ({len(input_text.split())} words)")
        st.text_area("File Content Preview", input_text[:500] + "...", height=100, disabled=True)

with tab_pdf:
    st.markdown("##### Upload a PDF document:")
    uploaded_pdf = st.file_uploader("Choose a .pdf file", type=["pdf"], key="pdf_file_uploader")
    if uploaded_pdf is not None:
        with st.spinner("Extracting text from PDF pages..."):
            try:
                pdf_data = extract_text_from_pdf(uploaded_pdf.read())
                input_text = pdf_data["text"]
                st.success(f"PDF processed! Pages: {pdf_data['num_pages']} | Total Words: {pdf_data['word_count']}")
                st.text_area("Extracted PDF Preview", input_text[:600] + "...", height=120, disabled=True)
            except Exception as e:
                st.error(f"Failed to extract PDF text: {e}")

with tab_url:
    st.markdown("##### Enter a public news or blog article URL:")
    url_input = st.text_input("Article Web Address", placeholder="https://example.com/article", key="url_input_box")
    if url_input:
        if st.button("Fetch Article", key="btn_fetch_url"):
            with st.spinner("Scraping clean article text..."):
                try:
                    url_data = extract_text_from_url(url_input)
                    input_text = url_data["text"]
                    st.session_state["fetched_url_text"] = input_text
                    st.success(f"Article fetched: '{url_data['title']}' ({url_data['word_count']} words)")
                except Exception as e:
                    st.error(f"Failed to fetch article: {e}")
        elif "fetched_url_text" in st.session_state:
            input_text = st.session_state["fetched_url_text"]

with tab_eval:
    st.markdown("##### Optional Ground Truth Reference Summary")
    st.write("If evaluating against a reference summary, paste it below to compute ROUGE-1, ROUGE-2, and ROUGE-L scores:")
    reference_target = st.text_area("Reference Summary", height=100, placeholder="Paste reference target summary here...", key="ref_summary_area")


st.markdown("---")

# Primary Generate Action Button
col_btn, col_blank = st.columns([1, 1])
with col_btn:
    generate_btn = st.button("🚀 GENERATE SHARP SUMMARY", key="btn_generate_sharp")

if generate_btn:
    cleaned_input = clean_text(input_text)
    
    if not cleaned_input:
        st.warning("⚠️ Please provide non-empty source text via text input, .txt, PDF, or URL.")
    else:
        start_time = time.time()
        
        with st.spinner("Running Sharp Summarization Pipeline..."):
            try:
                # Load corresponding model with graceful fallback
                if use_extractive:
                    transformer_engine = None
                    extractive_engine = load_extractive_model()
                    active_use_extractive = True
                else:
                    try:
                        transformer_engine = load_transformer_model(selected_model_path)
                        extractive_engine = None
                        active_use_extractive = False
                    except Exception as model_err:
                        st.warning(f"⚠️ Could not load requested Transformer model (`{selected_model_path}`). Falling back to Extractive TextRank strategy. Details: {model_err}")
                        transformer_engine = None
                        extractive_engine = load_extractive_model()
                        active_use_extractive = True

                # Execute sharp pipeline
                result = generate_sharp_summary(
                    text=cleaned_input,
                    transformer_engine=transformer_engine,
                    extractive_engine=extractive_engine,
                    mode=summary_mode if summary_mode != "Custom Budget" else "Short",
                    custom_max_words=custom_max_w,
                    reference_summary=reference_target,
                    use_extractive=active_use_extractive,
                    high_precision=high_precision_toggle
                )


                
                elapsed_time = round(time.time() - start_time, 2)
                stats = result["stats"]

                # Display Executive Summary Output Box
                st.markdown("### ⚡ Sharp Executive Summary")
                st.markdown(f"""
                <div class="summary-card">
                    <div class="summary-card-header">Target Mode: {summary_mode} | Max Words: {result['max_words_budget']}</div>
                    {result['summary']}
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 📊 Metrics & Information Density Dashboard")

                # Metrics Display Grid
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                
                with c1:
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-value">{stats['original_word_count']}</div>
                        <div class="metric-label">Original Words</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with c2:
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-value">{stats['summary_word_count']}</div>
                        <div class="metric-label">Summary Words</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with c3:
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-value">{stats['compression_ratio_pct']}%</div>
                        <div class="metric-label">Compression</div>
                    </div>
                    """, unsafe_allow_html=True)

                with c4:
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-value">{stats['info_density_index']}</div>
                        <div class="metric-label">Info Density</div>
                    </div>
                    """, unsafe_allow_html=True)

                with c5:
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-value">{stats['redundancy_label']}</div>
                        <div class="metric-label">Redundancy</div>
                    </div>
                    """, unsafe_allow_html=True)

                with c6:
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-value">{stats.get('factuality_confidence_pct', 100.0)}%</div>
                        <div class="metric-label">Factuality</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.caption(f"⚡ Pipeline Latency: `{elapsed_time}s` | Model Architecture: `{selected_model_label}` | Strategy: `{strategy_mode}`")

                # Detailed Expanders for Pipeline Stages and Long Documents
                st.markdown("<br>", unsafe_allow_html=True)
                
                with st.expander("🔍 View Core Pipeline Step Breakdown"):
                    st.markdown("#### 1. Transformer Raw Output")
                    st.write(result["pipeline_stages"].get("raw_summary", ""))
                    
                    st.markdown("#### 2. Semantic Redundancy Removal Pass")
                    st.write(result["pipeline_stages"].get("non_redundant_summary", ""))
                    st.json(result["redundancy"])

                    st.markdown("#### 3. Factuality & Entity Verification")
                    st.json(result["factuality"])

                if result["num_chunks"] > 1:
                    with st.expander("📄 View Hierarchical Long-Document Chunk Summaries"):
                        st.info(f"Document was split into {result['num_chunks']} overlapping section chunks:")
                        for idx, chunk_sum in enumerate(result["intermediate_summaries"]):
                            st.write(f"**Section {idx+1} Intermediate Summary:**")
                            st.write(chunk_sum)

                with st.expander("📊 Full Evaluation Metrics & ROUGE Breakdown"):
                    report_text = format_evaluation_report(stats)
                    st.code(report_text, language="text")

                # Download Options
                st.markdown("<br>", unsafe_allow_html=True)
                d1, d2 = st.columns(2)
                
                with d1:
                    st.download_button(
                        label="📥 Download Sharp Summary (.txt)",
                        data=result["summary"],
                        file_name="sharp_summary.txt",
                        mime="text/plain",
                        key="dl_txt_summary"
                    )
                    
                with d2:
                    export_report = {
                        "model": selected_model_label,
                        "strategy": strategy_mode,
                        "mode": summary_mode,
                        "max_words_budget": result["max_words_budget"],
                        "summary": result["summary"],
                        "metrics": stats,
                        "pipeline_stages": result["pipeline_stages"],
                        "generation_time_seconds": elapsed_time
                    }
                    st.download_button(
                        label="📊 Download Full JSON Report",
                        data=json.dumps(export_report, indent=2),
                        file_name="sharp_summary_report.json",
                        mime="application/json",
                        key="dl_json_report"
                    )

            except Exception as e:
                st.error(f"Error executing summary pipeline: {e}")
                st.exception(e)
