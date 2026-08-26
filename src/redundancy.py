"""
Semantic Redundancy Removal & Relevance Filtering Module.

Compares summary sentences for semantic similarity, removes repetitive ideas,
filters low-relevance sentences against source document topics, and preserves
the highest-density, most relevant concepts.
"""

import re
import numpy as np
from typing import List, Tuple, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocessing import clean_text, sentence_split


def compute_sentence_similarity_matrix(sentences: List[str]) -> np.ndarray:
    """
    Computes TF-IDF cosine similarity matrix across sentences.
    """
    if len(sentences) <= 1:
        return np.ones((len(sentences), len(sentences)))
        
    try:
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(sentences)
        sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        return sim_matrix
    except Exception:
        # Fallback character/n-gram jaccard similarity matrix if TF-IDF fails
        n = len(sentences)
        sim_matrix = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                set1 = set(sentences[i].lower().split())
                set2 = set(sentences[j].lower().split())
                union = set1.union(set2)
                sim = len(set1.intersection(set2)) / len(union) if union else 0.0
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim
        return sim_matrix


def compute_sentence_density(sentence: str) -> float:
    """
    Calculates an information density score for a sentence based on length,
    presence of numbers, proper nouns, and content words (non-stopwords).
    """
    words = sentence.split()
    if not words:
        return 0.0
        
    # Count numbers/digits
    num_digits = len(re.findall(r'\b\d+(?:\.\d+)?%?\b', sentence))
    # Count capitalized words (potential named entities/concepts)
    num_caps = len([w for w in words if w[0].isupper() and len(w) > 1])
    # Average word length
    avg_len = sum(len(w) for w in words) / len(words)
    
    # Score rewards structured specific words over filler
    density_score = (num_digits * 2.0) + (num_caps * 1.5) + (avg_len * 0.5) + (len(words) * 0.1)
    return density_score


def filter_low_relevance_sentences(
    summary_text: str, 
    source_text: str, 
    min_relevance_threshold: float = 0.12
) -> Tuple[str, Dict[str, Any]]:
    """
    Filters out generated summary sentences that have low topical relevance 
    compared to the main source document concepts.
    """
    cleaned_summary = clean_text(summary_text)
    cleaned_source = clean_text(source_text)
    
    sentences = sentence_split(cleaned_summary)
    if len(sentences) <= 1 or not cleaned_source:
        return cleaned_summary, {"removed_low_relevance": 0, "avg_relevance": 1.0}

    try:
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        # Fit vectorizer on source text sentences + summary sentences
        source_sentences = sentence_split(cleaned_source)
        all_corpus = source_sentences + sentences
        vectorizer.fit(all_corpus)

        source_vector = vectorizer.transform([cleaned_source])
        summary_vectors = vectorizer.transform(sentences)

        relevance_scores = cosine_similarity(summary_vectors, source_vector).flatten()
        
        kept_sentences = []
        removed_count = 0

        for idx, score in enumerate(relevance_scores):
            if score >= min_relevance_threshold or len(sentences) <= 2:
                kept_sentences.append(sentences[idx])
            else:
                removed_count += 1

        if not kept_sentences:  # Fallback to keep at least 1 highest relevance sentence
            best_idx = int(np.argmax(relevance_scores))
            kept_sentences = [sentences[best_idx]]

        filtered_summary = " ".join(kept_sentences)
        avg_rel = float(np.mean(relevance_scores)) if len(relevance_scores) > 0 else 0.0

        return filtered_summary, {
            "removed_low_relevance": removed_count,
            "avg_relevance_score": round(avg_rel, 3)
        }
    except Exception:
        return cleaned_summary, {"removed_low_relevance": 0, "avg_relevance": 1.0}


def remove_semantic_redundancy(
    text: str,
    similarity_threshold: float = 0.65
) -> Tuple[str, Dict[str, Any]]:
    """
    Identifies and removes redundant sentences from summary text.
    """
    cleaned = clean_text(text)
    sentences = sentence_split(cleaned)
    
    if len(sentences) <= 1:
        return cleaned, {
            "redundancy_score": 0.0,
            "redundancy_label": "Low",
            "removed_sentences": 0
        }

    sim_matrix = compute_sentence_similarity_matrix(sentences)
    
    n = len(sentences)
    keep_indices = []
    removed_count = 0
    max_pair_sim = 0.0

    for i in range(n):
        is_duplicate = False
        density_i = compute_sentence_density(sentences[i])
        
        for j in keep_indices:
            sim = sim_matrix[i, j]
            if sim > max_pair_sim:
                max_pair_sim = sim
                
            if sim >= similarity_threshold:
                density_j = compute_sentence_density(sentences[j])
                if density_i > density_j:
                    keep_indices.remove(j)
                    keep_indices.append(i)
                is_duplicate = True
                removed_count += 1
                break
                
        if not is_duplicate:
            keep_indices.append(i)

    keep_indices.sort()
    filtered_sentences = [sentences[idx] for idx in keep_indices]
    filtered_text = " ".join(filtered_sentences)

    if n > 1:
        off_diag = sim_matrix[np.triu_indices(n, k=1)]
        avg_sim = float(np.mean(off_diag)) if len(off_diag) > 0 else 0.0
    else:
        avg_sim = 0.0

    redundancy_score = round(max(0.0, min(1.0, avg_sim)), 3)
    
    if redundancy_score < 0.25:
        redundancy_label = "Low"
    elif redundancy_score < 0.50:
        redundancy_label = "Medium"
    else:
        redundancy_label = "High"

    return filtered_text, {
        "redundancy_score": redundancy_score,
        "redundancy_label": redundancy_label,
        "removed_sentences": removed_count,
        "original_sentence_count": n,
        "final_sentence_count": len(filtered_sentences)
    }
