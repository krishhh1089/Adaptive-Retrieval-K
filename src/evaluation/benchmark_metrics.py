"""
src/evaluation/benchmark_metrics.py

Quantitative evaluation metrics for retrieval quality, generation accuracy,
latency distributions, and relative efficiency for Load-Aware SAGE.
"""

from collections import Counter
import re
import string
from typing import Any, Dict, List, Optional, Sequence
import numpy as np

from src.retrieval.hybrid import ScoredPassage


def normalize_text(text: str) -> str:
    """
    Standard QA string normalization:
    - Lowercase
    - Remove punctuation
    - Remove articles (a, an, the)
    - Collapse extra whitespace
    """
    text = text.lower()
    # Remove punctuation
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Collapse whitespace
    return " ".join(text.split())


def compute_retrieval_hit(passages: Sequence[ScoredPassage], gold_answers: Sequence[str]) -> int:
    """
    Returns 1 if any gold answer is a substring of any retrieved passage text, else 0.
    """
    if not passages or not gold_answers:
        return 0

    clean_golds = [g.strip().lower() for g in gold_answers if g.strip()]
    if not clean_golds:
        return 0

    for p in passages:
        p_lower = p.text.lower()
        if any(gold in p_lower for gold in clean_golds):
            return 1
    return 0


def compute_first_hit_rank(passages: Sequence[ScoredPassage], gold_answers: Sequence[str]) -> Optional[int]:
    """
    Returns the 1-based rank of the first passage containing a gold answer substring, or None.
    """
    if not passages or not gold_answers:
        return None

    clean_golds = [g.strip().lower() for g in gold_answers if g.strip()]
    for rank, p in enumerate(passages, start=1):
        p_lower = p.text.lower()
        if any(gold in p_lower for gold in clean_golds):
            return rank
    return None


def compute_answer_presence(generated_answer: str, gold_answers: Sequence[str]) -> int:
    """
    Returns 1 if any gold answer substring appears in the generated answer text, else 0.
    """
    if not generated_answer or not gold_answers:
        return 0

    norm_gen = normalize_text(generated_answer)
    for gold in gold_answers:
        norm_gold = normalize_text(gold)
        if norm_gold and norm_gold in norm_gen:
            return 1
    return 0


def compute_exact_match(generated_answer: str, gold_answers: Sequence[str]) -> int:
    """
    Returns 1 if normalized generated answer exactly matches any normalized gold answer, else 0.
    """
    if not generated_answer or not gold_answers:
        return 0

    norm_gen = normalize_text(generated_answer)
    for gold in gold_answers:
        norm_gold = normalize_text(gold)
        if norm_gen == norm_gold:
            return 1
    return 0


def compute_token_f1(generated_answer: str, gold_answers: Sequence[str]) -> float:
    """
    Computes standard unigram token-level F1 score between generated answer and gold answers.
    Takes the maximum F1 score over all reference gold answers.
    """
    if not generated_answer or not gold_answers:
        return 0.0

    gen_tokens = normalize_text(generated_answer).split()
    if not gen_tokens:
        return 0.0

    max_f1 = 0.0
    for gold in gold_answers:
        gold_tokens = normalize_text(gold).split()
        if not gold_tokens:
            continue

        common = Counter(gen_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            f1 = 0.0
        else:
            precision = 1.0 * num_same / len(gen_tokens)
            recall = 1.0 * num_same / len(gold_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
        max_f1 = max(max_f1, f1)

    return float(max_f1)


def compute_faithfulness(generated_answer: str, context_passages: Sequence[ScoredPassage]) -> float:
    """
    Computes unigram faithfulness (hallucination guard):
    Measures the proportion of content tokens in the generated answer supported by retrieved context.
    If the model expresses inability to answer, faithfulness is 1.0.
    """
    if not generated_answer:
        return 0.0

    norm_gen = normalize_text(generated_answer)
    if "cannot answer" in norm_gen or "provided context" in norm_gen:
        return 1.0

    gen_tokens = [tok for tok in norm_gen.split() if len(tok) > 2]
    if not gen_tokens:
        return 1.0

    combined_context = " ".join(normalize_text(p.text) for p in context_passages)
    context_tokens = set(combined_context.split())

    supported_count = sum(1 for tok in gen_tokens if tok in context_tokens)
    return float(supported_count / len(gen_tokens))


def calculate_distribution_stats(values: Sequence[float]) -> Dict[str, float]:
    """
    Computes mean, median (P50), P90, P95, P99, min, and max across numeric observations.
    """
    if len(values) == 0:
        return {
            "mean": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def compute_relative_efficiency(
    baseline_val: float,
    target_val: float,
    higher_is_better: bool = False,
) -> float:
    """
    Calculates percentage improvement of target relative to baseline.
    For latency / token count (lower is better):
        Improvement % = ((baseline - target) / baseline) * 100
    For quality (higher is better):
        Improvement % = ((target - baseline) / baseline) * 100 (or absolute difference)
    """
    if baseline_val == 0.0:
        return 0.0

    if higher_is_better:
        return float(((target_val - baseline_val) / abs(baseline_val)) * 100.0)
    else:
        return float(((baseline_val - target_val) / abs(baseline_val)) * 100.0)

