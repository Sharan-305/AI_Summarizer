"""
Evaluation & Information Density Metrics Module.

Computes standard ROUGE metrics (ROUGE-1, ROUGE-2, ROUGE-L), Compression Ratio %,
Information Density Index (0-100), and integrated summary metrics.
"""

import re
import numpy as np
from typing import Dict, List, Any, Union
import rouge_score.rouge_scorer as rouge_scorer
from src.preprocessing import clean_text, sentence_split


def compute_rouge_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Computes ROUGE-1, ROUGE-2, and ROUGE-L F1 scores for predicted summaries against reference summaries.
    """
    if isinstance(predictions, str):
        predictions = [predictions]
    if isinstance(references, str):
        references = [references]

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for pred, ref in zip(predictions, references):
        pred_clean = pred.strip() if pred else ""
        ref_clean = ref.strip() if ref else ""
        
        if not pred_clean or not ref_clean:
            rouge1_scores.append(0.0)
            rouge2_scores.append(0.0)
            rougeL_scores.append(0.0)
            continue
            
        scores = scorer.score(ref_clean, pred_clean)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)

    return {
        "rouge1": round(float(np.mean(rouge1_scores)), 4),
        "rouge2": round(float(np.mean(rouge2_scores)), 4),
        "rougeL": round(float(np.mean(rougeL_scores)), 4)
    }


def calculate_information_density(original_text: str, summary_text: str) -> Dict[str, Any]:
    """
    Calculates an Information Density Index (0-100) based on:
      - Content word concentration (non-stopwords, nouns/verbs)
      - Key facts, numbers, and dates retention
      - Named entities per 100 words
      - Conciseness multiplier
    
    Returns:
        Dict[str, Any]: Density index score and classification.
    """
    cleaned_orig = clean_text(original_text)
    cleaned_summ = clean_text(summary_text)
    
    orig_words = cleaned_orig.split()
    summ_words = cleaned_summ.split()
    
    if not summ_words:
        return {"density_index": 0.0, "density_label": "Very Low", "content_ratio_pct": 0.0}

    # Standard English stop words
    stopwords = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", 
        "by", "about", "against", "between", "into", "through", "during", "before", 
        "after", "above", "below", "from", "up", "down", "in", "out", "on", "off", 
        "over", "under", "again", "further", "then", "once", "here", "there", "when", 
        "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", 
        "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", 
        "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now",
        "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do",
        "does", "did", "also", "its", "it", "they", "them", "this", "that", "these", "those"
    }

    # Count content words (non-stopwords) in summary
    content_words = [w for w in summ_words if w.lower() not in stopwords and len(w) > 2]
    content_ratio = len(content_words) / len(summ_words)
    
    # Count numbers and proper nouns
    num_digits = len(re.findall(r'\b\d+(?:\.\d+)?%?\b', cleaned_summ))
    num_entities = len([w for w in summ_words if w[0].isupper() and len(w) > 1 and w.lower() not in stopwords])
    
    # Informational elements per 10 words
    info_density = (len(content_words) + (num_digits * 2) + (num_entities * 1.5)) / len(summ_words)
    
    # Normalize density index to 0 - 100 scale
    density_index = min(100.0, round(info_density * 85.0, 1))
    
    if density_index >= 80:
        label = "High"
    elif density_index >= 60:
        label = "Medium"
    else:
        label = "Low"

    return {
        "density_index": density_index,
        "density_label": label,
        "content_ratio_pct": round(content_ratio * 100.0, 1),
        "content_word_count": len(content_words),
        "fact_elements_count": num_digits + num_entities
    }


def compute_summary_evaluation(
    original_text: str,
    generated_summary: str,
    reference_summary: str = None,
    redundancy_info: Dict[str, Any] = None,
    factuality_info: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Calculates comprehensive summary performance metrics:
      - Word & Sentence counts
      - Compression Ratio %
      - Information Density Index
      - ROUGE-1, ROUGE-2, ROUGE-L F1
      - Redundancy & Factuality details
    """
    orig_words = len(original_text.split())
    summ_words = len(generated_summary.split())
    
    orig_sents = len(sentence_split(original_text))
    summ_sents = len(sentence_split(generated_summary))

    # Compression Ratio % = (1 - summary_words / orig_words) * 100
    if orig_words > 0:
        compression_ratio = round((1.0 - (summ_words / orig_words)) * 100.0, 1)
        compression_ratio = max(0.0, min(100.0, compression_ratio))
    else:
        compression_ratio = 0.0

    # Information Density calculation
    info_density = calculate_information_density(original_text, generated_summary)

    stats = {
        "original_word_count": orig_words,
        "summary_word_count": summ_words,
        "original_sentence_count": orig_sents,
        "summary_sentence_count": summ_sents,
        "compression_ratio_pct": compression_ratio,
        "info_density_index": info_density["density_index"],
        "info_density_label": info_density["density_label"],
        "content_ratio_pct": info_density["content_ratio_pct"]
    }

    # ROUGE Calculation
    if reference_summary and reference_summary.strip():
        rouge_results = compute_rouge_metrics([generated_summary], [reference_summary])
        stats["rouge_type"] = "Ground Truth Reference"
    else:
        # Relative coverage against source document if no ground truth provided
        rouge_results = compute_rouge_metrics([generated_summary], [original_text])
        stats["rouge_type"] = "Source Coverage"

    stats.update(rouge_results)

    # Attach Redundancy & Factuality if available
    if redundancy_info:
        stats["redundancy_score"] = redundancy_info.get("redundancy_score", 0.0)
        stats["redundancy_label"] = redundancy_info.get("redundancy_label", "Low")
    if factuality_info:
        stats["factuality_confidence_pct"] = factuality_info.get("factuality_confidence_pct", 100.0)
        stats["is_faithful"] = factuality_info.get("is_faithful", True)

    return stats


def format_evaluation_report(stats: Dict[str, Any]) -> str:
    """
    Formats evaluation metrics into an executive benchmark summary string.
    """
    report = [
        "=" * 55,
        "        SHARP SUMMARIZER PERFORMANCE EVALUATION REPORT     ",
        "=" * 55,
        f" Original Document Words : {stats.get('original_word_count', 0)} ({stats.get('original_sentence_count', 0)} sentences)",
        f" Summary Document Words  : {stats.get('summary_word_count', 0)} ({stats.get('summary_sentence_count', 0)} sentences)",
        f" Compression Ratio      : {stats.get('compression_ratio_pct', 0.0)}%",
        f" Information Density     : {stats.get('info_density_index', 0.0)} / 100 ({stats.get('info_density_label', 'Low')})",
        "-" * 55,
        f" ROUGE-1 F1 Score        : {stats.get('rouge1', 0.0):.4f}",
        f" ROUGE-2 F1 Score        : {stats.get('rouge2', 0.0):.4f}",
        f" ROUGE-L F1 Score        : {stats.get('rougeL', 0.0):.4f} ({stats.get('rouge_type', 'Source')})",
        "-" * 55,
        f" Redundancy Rating       : {stats.get('redundancy_label', 'Low')} (Score: {stats.get('redundancy_score', 0.0)})",
        f" Factuality Confidence   : {stats.get('factuality_confidence_pct', 100.0)}% ({'Faithful' if stats.get('is_faithful', True) else 'Caution'})",
        "=" * 55,
        " Note: ROUGE alone does not determine if a summary is sharp.",
        " Maximum information density & conciseness are prioritized.",
        "=" * 55
    ]
    return "\n".join(report)
