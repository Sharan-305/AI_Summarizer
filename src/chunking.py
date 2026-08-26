"""
Hierarchical Long Document Chunking & Summarization Module.

Implements hierarchical section splitting and map-reduce summarization
for large documents without truncating content or exceeding Transformer token limits.
"""

from typing import List, Dict, Any, Callable
from src.preprocessing import clean_text, sentence_split
import config


def split_into_sections(text: str, chunk_size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> List[str]:
    """
    Splits long text documents into overlapping sections/chunks while preserving sentence boundaries.
    
    Args:
        text (str): Input long document text.
        chunk_size (int): Max target words per section chunk.
        overlap (int): Overlap word budget between consecutive chunks.
        
    Returns:
        List[str]: List of text section chunks.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []
        
    sentences = sentence_split(cleaned)
    if not sentences:
        words = cleaned.split()
        if len(words) <= chunk_size:
            return [cleaned]
        # Word-level fallback if sentence splitting fails
        return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]

    chunks = []
    current_chunk = []
    current_word_count = 0

    for sent in sentences:
        sent_words = len(sent.split())
        
        # If adding sentence exceeds chunk limit, finalize current chunk
        if current_word_count + sent_words > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            
            # Construct overlapping prefix from trailing sentences
            overlap_chunk = []
            overlap_words = 0
            for prev_sent in reversed(current_chunk):
                pw = len(prev_sent.split())
                if overlap_words + pw <= overlap:
                    overlap_chunk.insert(0, prev_sent)
                    overlap_words += pw
                else:
                    break
            current_chunk = overlap_chunk
            current_word_count = overlap_words

        current_chunk.append(sent)
        current_word_count += sent_words

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks if chunks else [cleaned]


def hierarchical_long_document_summarize(
    text: str,
    summarize_func: Callable[[str, int, int], str],
    redundancy_remover_func: Callable[[str], str] = None,
    target_max_words: int = 50,
    target_min_words: int = 15,
    chunk_size: int = config.CHUNK_SIZE
) -> Dict[str, Any]:
    """
    Executes hierarchical Map-Reduce summarization for large text documents:
      1. Section Splitting
      2. Chunk-level Summarization
      3. Inter-chunk Redundancy Removal
      4. Intermediate Combination
      5. Final Sharp Compression Pass
    
    Args:
        text (str): Raw input text document.
        summarize_func (Callable): Function that takes (text, max_words, min_words) and returns summary str.
        redundancy_remover_func (Callable, optional): Function that removes duplicate sentences.
        target_max_words (int): Desired final maximum word limit.
        target_min_words (int): Desired final minimum word limit.
        chunk_size (int): Max word size per section.
        
    Returns:
        Dict[str, Any]: Dictionary with 'final_summary', 'intermediate_summaries', 'num_chunks'.
    """
    cleaned_input = clean_text(text)
    orig_words = len(cleaned_input.split())

    # If document fits into a single chunk, run single-pass summarization
    if orig_words <= chunk_size:
        single_summary = summarize_func(cleaned_input, target_max_words, target_min_words)
        if redundancy_remover_func:
            single_summary = redundancy_remover_func(single_summary)
        return {
            "final_summary": single_summary,
            "intermediate_summaries": [single_summary],
            "num_chunks": 1
        }

    # Map Phase: Split document into sections
    sections = split_into_sections(cleaned_input, chunk_size=chunk_size, overlap=config.CHUNK_OVERLAP)
    num_chunks = len(sections)
    
    # Calculate intermediate max words per section
    intermediate_max_words = max(30, (target_max_words * 2) // num_chunks + 35)
    
    intermediate_summaries = []
    for chunk in sections:
        chunk_summary = summarize_func(chunk, intermediate_max_words, 15)
        if chunk_summary.strip():
            intermediate_summaries.append(chunk_summary)

    # Combine intermediate summaries
    combined_intermediate = " ".join(intermediate_summaries)
    
    # Apply redundancy removal across section summaries if function provided
    if redundancy_remover_func:
        combined_intermediate = redundancy_remover_func(combined_intermediate)

    # Reduce Phase: Final sharp compression pass if combined output exceeds target limit
    combined_words = len(combined_intermediate.split())
    if combined_words > target_max_words:
        final_summary = summarize_func(combined_intermediate, target_max_words, target_min_words)
        if redundancy_remover_func:
            final_summary = redundancy_remover_func(final_summary)
    else:
        final_summary = combined_intermediate

    return {
        "final_summary": final_summary,
        "intermediate_summaries": intermediate_summaries,
        "num_chunks": num_chunks
    }
