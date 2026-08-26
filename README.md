# AI-Based Sharp & Crisp Text Summarization System

An advanced Transformer-based Deep Learning Text Summarization System engineered to generate **extremely concise, sharp, information-dense executive summaries**.

---

## 🎯 Core Principle

> **"Maximum information in minimum words."**

The goal is **NOT** simply to shorten text or produce generic multi-sentence paragraph paraphrases. The system extracts only essential facts, decisions, conclusions, numbers, and key concepts while aggressively stripping filler sentences, introductory background, and repetitive statements.

---

## 💡 Example

### Source Input Text:
> *"Artificial intelligence is becoming increasingly important across different industries. Companies are using AI to automate repetitive tasks, analyze large amounts of data, improve customer experiences, and support decision-making. Healthcare organizations use AI for medical diagnosis and drug discovery, while financial institutions use it for fraud detection and risk analysis. As AI technology continues to improve, organizations are expected to adopt it more widely."*

### Poor / Ordinary Summary:
> *"Artificial intelligence is becoming increasingly important in many industries. It is used by companies for many different purposes. AI is expected to become more widely adopted in the future."*

### Desired Sharp Summary:
> **"AI is transforming industries through automation, data analysis, diagnosis, fraud detection, and decision support, with adoption expected to expand."**

---

## 🏗️ Core Architecture & Pipeline

```text
Input Text / PDF / TXT / URL
           ↓
Text Extraction & Cleaning (preprocessing.py / pdf_processor.py)
           ↓
Hierarchical Long-Doc Chunking (chunking.py - Map-Reduce)
           ↓
Transformer Abstractive Generation (summarizer.py - BART/T5/PEGASUS)
           ↓
Post-Summarization Redundancy Removal (redundancy.py - TF-IDF Cosine Similarity)
           ↓
Length Control & Sentence Truncation (summarizer.py - Ultra Short / Short / Brief / Max Words)
           ↓
Factuality & Entity Verification (factuality.py - Entity Grounding Check)
           ↓
Information Density & ROUGE Evaluation (evaluation.py - ROUGE-1/2/L & Info Density Index)
           ↓
Executive Streamlit App & Export (app.py)
```

---

## 📂 Project Structure

```text
sharp_summarizer/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── checkpoints/
│   └── best_model/
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── pdf_processor.py
│   ├── summarizer.py
│   ├── chunking.py
│   ├── redundancy.py
│   ├── factuality.py
│   ├── evaluation.py
│   └── dataset.py
│
├── app.py
├── train.py
├── config.py
├── requirements.txt
└── README.md
```

---

## ⚡ Key Features

1. **Pretrained Transformer Models**:
   - `sshleifer/distilbart-cnn-12-6` (Default - ultra fast & sharp)
   - `t5-small` / `t5-base`
   - `facebook/bart-base`
   - `google/pegasus-xsum`
   - ⭐ Local Fine-Tuned Custom Models (`models/best_model`)

2. **Summary Length Modes**:
   - **Ultra Short**: 1–2 sentences, only central idea (~15–30 max words)
   - **Short**: 2–4 sentences, main facts and conclusion (~35–60 max words)
   - **Brief**: 4–6 sentences, important details without unnecessary explanation (~70–120 max words)
   - **Custom Max Word Limit**: User-defined word budget (e.g. `[ 50 ]` words)

3. **Hierarchical Long-Document Processing**:
   - Handles large PDFs and long documents via Map-Reduce section splitting without truncation or token overflow.

4. **Semantic Redundancy Removal**:
   - Computes pairwise cosine similarity between summary sentences and removes repetitive statements while preserving the highest information-density version.

5. **Factuality & Hallucination Guard**:
   - Extracts numbers, percentages, years, and named entities from source text and summary to verify 100% factual grounding.

6. **Information Density Index**:
   - Calculates a custom information density score (0 to 100) based on content words, named entities, and numerical facts per word.

---

## 🚀 Installation & Usage

### 1. Prerequisites & Installation

```bash
# Clone or navigate to project workspace
cd "c:/Users/shadh/OneDrive/Desktop/DL/Text summarizer"

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch Streamlit Executive Application

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 🏋️ Fine-Tuning a Custom Transformer Model

Fine-tune BART or T5 models on summarization datasets (`xsum`, `cnn_dailymail`, `samsum`, `wikihow`, `multi_news`):

```bash
# Fine-tune DistilBART on XSum dataset for 3 epochs
python train.py --model sshleifer/distilbart-cnn-12-6 --dataset xsum --epochs 3 --batch_size 4 --limit 1000

# Fine-tune T5-Small on SAMSum dialogue dataset
python train.py --model t5-small --dataset samsum --epochs 3 --batch_size 4 --limit 500
```

Fine-tuned weights automatically save to `models/best_model/` and are detected in the Streamlit UI.

---

## 📊 Evaluation & Metrics

The system measures:
- **Compression Ratio %**: \((1 - \frac{\text{Summary Words}}{\text{Original Words}}) \times 100\%\)
- **Information Density Index**: Score (0-100) representing content word concentration and entity density.
- **ROUGE-1, ROUGE-2, ROUGE-L**: Unigram, bigram, and longest common subsequence F1 scores.
- **Redundancy Score**: Measure of sentence-level semantic overlap.
- **Factuality Confidence %**: Percentage of summary numbers and entities grounded in the original source document.

> [!NOTE]
> ROUGE alone does not determine whether a summary is sharp and useful. The system prioritizes information density and conciseness.
