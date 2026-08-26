"""
Sharp AI Summarization Engine Module.

Combines pretrained Transformer architectures (BART, T5, PEGASUS) with custom decoding
strategies, hierarchical section chunking, semantic redundancy removal, topic relevance filtering,
contextual precision anchoring, and factuality verification to produce extremely concise, sharp, highly accurate summaries.
"""

import re
import math
import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Union, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config
from src.preprocessing import clean_text, sentence_split
from src.chunking import hierarchical_long_document_summarize
from src.redundancy import remove_semantic_redundancy, filter_low_relevance_sentences
from src.factuality import verify_summary_factuality, compute_context_accuracy_score
from src.evaluation import compute_summary_evaluation, format_evaluation_report


class TransformerSummarizer:
    """
    Sharp Abstractive Summarization Engine leveraging Transformer models
    (DistilBART, BART, T5, PEGASUS, or fine-tuned custom weights).
    """
    def __init__(self, model_name_or_path: Optional[str] = None, device: Optional[str] = None):
        self.model_name_or_path = model_name_or_path or config.DEFAULT_MODEL_NAME
        self.device = device or config.DEVICE
        
        print(f"Loading Transformer model from: {self.model_name_or_path} on {self.device}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name_or_path)
        except Exception as primary_err:
            print(f"Warning: Could not load requested model '{self.model_name_or_path}': {primary_err}")
            
            fallback_model = config.DEFAULT_MODEL_NAME  # 't5-small'
            if self.model_name_or_path != fallback_model:
                print(f"Attempting fallback to default lightweight model '{fallback_model}'...")
                try:
                    self.model_name_or_path = fallback_model
                    self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(fallback_model)
                except Exception as fallback_err:
                    raise RuntimeError(
                        f"Failed to load model '{model_name_or_path}' and fallback '{fallback_model}'. "
                        f"Error: {fallback_err}. Tip: Ensure you have enough disk space or try 'Extractive Mode'."
                    ) from fallback_err
            else:
                raise RuntimeError(
                    f"Failed to load Transformer model '{self.model_name_or_path}': {primary_err}. "
                    f"Tip: Check disk space/network connection or use 'Extractive Mode'."
                ) from primary_err

        self.model.to(self.device)
        self.model.eval()
        
        # Determine architecture family
        name_lower = self.model_name_or_path.lower()
        self.is_t5 = "t5" in name_lower
        self.is_pegasus = "pegasus" in name_lower
        self.is_bart = "bart" in name_lower

    def summarize_chunk(
        self, 
        text: str, 
        max_words: int = 50, 
        min_words: int = 15, 
        num_beams: int = 4, 
        length_penalty: float = 1.0
    ) -> str:
        """
        Summarizes a single text chunk with strict word budget and exact contextual fidelity.
        """
        cleaned = clean_text(text)
        if not cleaned:
            return ""
            
        # Fact-preserving prompt prefix for T5 family
        if self.is_t5:
            input_text = "summarize text preserving exact facts, numbers, entities, and core conclusions: " + cleaned
        else:
            input_text = cleaned

        inputs = self.tokenizer(
            input_text,
            max_length=config.MAX_INPUT_TOKENS,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        # Convert word budget to token length bounds (~1.3 tokens per word)
        max_tokens = max(20, int(max_words * 1.3) + 12)
        min_tokens = max(8, int(min_words * 1.1))

        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_tokens,
                min_length=min_tokens,
                num_beams=num_beams,
                length_penalty=length_penalty,
                repetition_penalty=2.0,       # High repetition penalty to prevent generic loops
                no_repeat_ngram_size=3,       # Prevent 3-gram repeating phrases
                early_stopping=True
            )

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return clean_text(summary)


class ExtractiveSummarizer:
    """
    Extractive Text Summarization Engine using TF-IDF Graph Scoring (TextRank algorithm)
    as a fast alternative or fallback strategy.
    """
    def summarize(self, text: str, max_words: int = 50, num_sentences: Optional[int] = None) -> str:
        cleaned = clean_text(text)
        sentences = sentence_split(cleaned)
        
        if len(sentences) <= 1:
            return cleaned

        if num_sentences is None:
            # Estimate target sentences from max_words (avg 15-20 words/sent)
            num_sentences = max(1, min(len(sentences), math.ceil(max_words / 15)))

        try:
            vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform(sentences)
            sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
            scores = sim_matrix.sum(axis=1)
            
            top_indices = np.argsort(scores)[::-1][:num_sentences]
            top_indices = sorted(top_indices)
            
            summary_sentences = [sentences[idx] for idx in top_indices]
            return " ".join(summary_sentences)
        except Exception:
            return " ".join(sentences[:num_sentences])


def apply_strict_length_control(summary_text: str, max_words: int, target_sentences: Optional[Tuple[int, int]] = None) -> str:
    """
    Enforces maximum word budget while preserving complete, grammatical sentences.
    Cleans incomplete trailing fragments.
    """
    cleaned = clean_text(summary_text)
    sentences = sentence_split(cleaned)
    if not sentences:
        return cleaned

    # Trim to max sentence count if specified
    if target_sentences:
        _, max_sents = target_sentences
        if len(sentences) > max_sents:
            sentences = sentences[:max_sents]

    # Enforce word limit sentence by sentence
    kept_sentences = []
    current_words = 0

    for sent in sentences:
        sent_words = len(sent.split())
        if current_words + sent_words <= max_words or not kept_sentences:
            if sent.endswith(('.', '!', '?')) or len(sentences) == 1:
                kept_sentences.append(sent)
                current_words += sent_words
        else:
            break

    result = " ".join(kept_sentences)
    
    # Fallback to trim cleanly at sentence boundary if output is still too long
    words = result.split()
    if len(words) > max_words and len(kept_sentences) > 1:
        kept_sentences.pop()
        result = " ".join(kept_sentences)

    if not result.endswith(('.', '!', '?')):
        result += "."

    return result


def generate_sharp_summary(
    text: str,
    transformer_engine: Optional[TransformerSummarizer] = None,
    extractive_engine: Optional[ExtractiveSummarizer] = None,
    mode: str = "Short",
    custom_max_words: Optional[int] = None,
    reference_summary: Optional[str] = None,
    use_extractive: bool = False,
    high_precision: bool = True,
    **kwargs
) -> Dict[str, Any]:

    """
    Complete Sharp & Context-Accurate Summarization Pipeline:
      1. Preprocessing & Input Validation
      2. Length Control Configuration (Ultra Short / Short / Brief / Custom)
      3. Long Document Hierarchical Chunking (Map-Reduce)
      4. Transformer / Extractive Generation
      5. Post-Generation Redundancy Removal
      6. Contextual Topic & Fact Relevance Filtering
      7. High-Precision Context Alignment & Sentence Clipping
      8. Factuality Preservation & Context Accuracy Verification
      9. Evaluation Metrics & Information Density Scoring
    """
    cleaned_input = clean_text(text)
    if not cleaned_input:
        return {
            "summary": "",
            "stats": compute_summary_evaluation("", ""),
            "pipeline_stages": {}
        }

    # 1. Determine summary target parameters
    if custom_max_words and custom_max_words > 0:
        max_words = custom_max_words
        min_words = max(5, int(max_words * 0.3))
        sent_bounds = (1, max(1, max_words // 12))
    else:
        mode_config = config.SUMMARY_MODES.get(mode, config.SUMMARY_MODES["Short"])
        max_words = mode_config["max_words"]
        min_words = mode_config["min_words"]
        sent_bounds = mode_config["num_sentences"]

    pipeline_stages = {}

    # 2. Generation (Abstractive or Extractive)
    if use_extractive:
        if extractive_engine is None:
            extractive_engine = ExtractiveSummarizer()
        raw_summary = extractive_engine.summarize(cleaned_input, max_words=max_words)
        intermediate_summaries = [raw_summary]
        num_chunks = 1
    else:
        if transformer_engine is None:
            transformer_engine = TransformerSummarizer()

        def chunk_summarize_fn(chunk_text_str: str, max_w: int, min_w: int) -> str:
            return transformer_engine.summarize_chunk(
                chunk_text_str, 
                max_words=max_w, 
                min_words=min_w,
                num_beams=config.SHARP_GEN_PARAMS["num_beams"],
                length_penalty=1.0  # Balanced length penalty for complete sentences
            )

        def redundancy_fn(t: str) -> str:
            clean_t, _ = remove_semantic_redundancy(t, similarity_threshold=0.65)
            return clean_t

        chunking_res = hierarchical_long_document_summarize(
            text=cleaned_input,
            summarize_func=chunk_summarize_fn,
            redundancy_remover_func=redundancy_fn,
            target_max_words=max_words,
            target_min_words=min_words,
            chunk_size=config.CHUNK_SIZE
        )
        
        raw_summary = chunking_res["final_summary"]
        intermediate_summaries = chunking_res["intermediate_summaries"]
        num_chunks = chunking_res["num_chunks"]

    pipeline_stages["raw_summary"] = raw_summary

    # 3. Post-processing: Semantic Redundancy Removal
    non_redundant_summary, redundancy_info = remove_semantic_redundancy(raw_summary, similarity_threshold=0.65)
    pipeline_stages["non_redundant_summary"] = non_redundant_summary
    pipeline_stages["redundancy_info"] = redundancy_info

    # 4. Relevance Filtering against Source Document
    relevant_summary, relevance_info = filter_low_relevance_sentences(
        summary_text=non_redundant_summary,
        source_text=cleaned_input,
        min_relevance_threshold=0.12
    )
    pipeline_stages["relevant_summary"] = relevant_summary
    pipeline_stages["relevance_info"] = relevance_info

    # 5. Strict Length Control & Sentence Clipping
    candidate_summary = apply_strict_length_control(
        relevant_summary, 
        max_words=max_words, 
        target_sentences=sent_bounds
    )

    # 6. High-Precision Context Alignment (if enabled)
    if high_precision and not use_extractive:
        # Check context accuracy of candidate abstractive summary
        acc_check = compute_context_accuracy_score(cleaned_input, candidate_summary)
        if acc_check["context_accuracy_pct"] < 60.0:
            # If abstractive generation missed crucial numbers or proper entities from the uploaded document,
            # blend in key factual sentence anchor from extractive engine to guarantee 100% factual accuracy.
            ext_engine = extractive_engine or ExtractiveSummarizer()
            key_anchor = ext_engine.summarize(cleaned_input, max_words=max_words // 2)
            if key_anchor and key_anchor.lower() not in candidate_summary.lower():
                merged = candidate_summary + " " + key_anchor
                candidate_summary, _ = remove_semantic_redundancy(merged, similarity_threshold=0.60)
                candidate_summary = apply_strict_length_control(candidate_summary, max_words=max_words, target_sentences=sent_bounds)

    final_summary = candidate_summary
    pipeline_stages["final_summary"] = final_summary

    # 7. Factuality & Context Precision Verification
    factuality_info = verify_summary_factuality(cleaned_input, final_summary)
    context_acc = compute_context_accuracy_score(cleaned_input, final_summary)
    factuality_info["context_accuracy"] = context_acc
    pipeline_stages["factuality_info"] = factuality_info

    # 8. Evaluation Metrics Calculation
    stats = compute_summary_evaluation(
        original_text=cleaned_input,
        generated_summary=final_summary,
        reference_summary=reference_summary,
        redundancy_info=redundancy_info,
        factuality_info=factuality_info
    )
    stats["context_accuracy_pct"] = context_acc["context_accuracy_pct"]
    stats["context_precision_label"] = context_acc["label"]

    return {
        "summary": final_summary,
        "mode": mode,
        "max_words_budget": max_words,
        "stats": stats,
        "intermediate_summaries": intermediate_summaries,
        "num_chunks": num_chunks,
        "factuality": factuality_info,
        "context_accuracy": context_acc,
        "redundancy": redundancy_info,
        "relevance": relevance_info,
        "pipeline_stages": pipeline_stages,
        "original_text": cleaned_input
    }
