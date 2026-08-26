"""
Factuality, Context Precision & Hallucination Verification Module.

Enforces strict factual fidelity by verifying that all numbers, dates, statistics,
and named entities in the generated summary are grounded in the original source document.
"""

import re
from typing import Dict, List, Set, Any
from src.preprocessing import clean_text


def extract_key_entities(text: str) -> Dict[str, Set[str]]:
    """
    Extracts key numerical data, dates/years, and proper noun entities from text.
    
    Returns:
        Dict containing sets of 'numbers', 'years', and 'named_entities'.
    """
    cleaned = clean_text(text)
    
    # Extract numbers (integers, decimals, percentages)
    numbers = set(re.findall(r'\b\d+(?:\.\d+)?%?\b', cleaned))
    
    # Extract 4-digit years (19xx, 20xx)
    years = set(re.findall(r'\b(?:19|20)\d{2}\b', cleaned))
    
    # Extract proper nouns (capitalized multi-character words not starting a sentence)
    words = cleaned.split()
    capitalized_words = set()
    for i, word in enumerate(words):
        clean_w = re.sub(r'[^\w\s]', '', word)
        if clean_w and clean_w[0].isupper() and len(clean_w) > 2:
            # Skip if it's the very first word of input unless it's repeated later
            if i > 0 or clean_w in text[len(word):]:
                capitalized_words.add(clean_w.lower())
                
    return {
        "numbers": numbers,
        "years": years,
        "named_entities": capitalized_words
    }


def compute_context_accuracy_score(original_text: str, summary_text: str) -> Dict[str, Any]:
    """
    Calculates a Context Precision Score (0-100%) measuring how accurately the summary
    retains key factual entities, numerical figures, and core terms from the original source.
    """
    orig_entities = extract_key_entities(original_text)
    summ_entities = extract_key_entities(summary_text)
    
    orig_numbers = orig_entities["numbers"]
    orig_ne = orig_entities["named_entities"]
    
    # Measure numerical retention ratio
    if orig_numbers:
        retained_nums = [n for n in orig_numbers if n.lower() in summary_text.lower()]
        num_precision = len(retained_nums) / len(orig_numbers)
    else:
        num_precision = 1.0

    # Measure named entity retention ratio
    if orig_ne:
        retained_ne = [e for e in orig_ne if e in summary_text.lower()]
        ne_precision = len(retained_ne) / len(orig_ne)
    else:
        ne_precision = 1.0

    # Overall Context Precision & Accuracy Score (0.0 to 100.0%)
    context_precision_pct = round(((num_precision * 0.5) + (ne_precision * 0.5)) * 100.0, 1)

    return {
        "context_accuracy_pct": context_precision_pct,
        "num_retained_ratio": f"{len(retained_nums if orig_numbers else [])}/{len(orig_numbers)}",
        "entity_retained_ratio": f"{len(retained_ne if orig_ne else [])}/{len(orig_ne)}",
        "label": "High Precision" if context_precision_pct >= 75.0 else ("Moderate Precision" if context_precision_pct >= 50.0 else "Standard")
    }


def verify_summary_factuality(original_text: str, generated_summary: str) -> Dict[str, Any]:
    """
    Compares extracted entities & numbers between original text and summary to verify factual accuracy.
    """
    orig_entities = extract_key_entities(original_text)
    summ_entities = extract_key_entities(generated_summary)
    
    orig_text_lower = original_text.lower()
    
    # Check numbers
    summ_numbers = summ_entities["numbers"]
    unsupported_numbers = []
    supported_numbers = 0
    for num in summ_numbers:
        if num.lower() in orig_text_lower:
            supported_numbers += 1
        else:
            unsupported_numbers.append(num)
            
    # Check proper nouns / named entities
    summ_ne = summ_entities["named_entities"]
    unsupported_ne = []
    supported_ne = 0
    for ne in summ_ne:
        if ne in orig_text_lower:
            supported_ne += 1
        else:
            unsupported_ne.append(ne)

    # Calculate sub-scores
    num_score = (supported_numbers / len(summ_numbers)) if summ_numbers else 1.0
    ne_score = (supported_ne / len(summ_ne)) if summ_ne else 1.0
    
    # Combined Factual Grounding Confidence Score (0.0 to 100.0%)
    weighted_confidence = (num_score * 0.6) + (ne_score * 0.4)
    confidence_pct = round(weighted_confidence * 100.0, 1)
    
    context_acc = compute_context_accuracy_score(original_text, generated_summary)
    is_faithful = confidence_pct >= 90.0 and len(unsupported_numbers) == 0

    return {
        "factuality_confidence_pct": confidence_pct,
        "context_accuracy": context_acc,
        "is_faithful": is_faithful,
        "total_summary_numbers": len(summ_numbers),
        "unsupported_numbers": unsupported_numbers,
        "total_summary_entities": len(summ_ne),
        "unsupported_entities": unsupported_ne,
        "status": "High Fidelity" if is_faithful else ("Moderate Caution" if confidence_pct >= 70 else "Potential Discrepancy")
    }
