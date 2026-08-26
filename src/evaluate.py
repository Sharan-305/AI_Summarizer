"""
Backward-compatibility wrapper forwarding evaluate.py calls to src.evaluation.
"""

from src.evaluation import (
    compute_rouge_metrics,
    calculate_information_density,
    compute_summary_evaluation as compute_summary_statistics,
    format_evaluation_report
)

__all__ = [
    "compute_rouge_metrics",
    "calculate_information_density",
    "compute_summary_statistics",
    "format_evaluation_report"
]
