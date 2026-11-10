"""
src/evaluation/latency_tracker.py

Fine-grained monotonic latency instrumentation for Load-Aware SAGE RAG pipeline.
Accurately records latency independently for each execution stage using time.perf_counter().
Prevents double-counting and ensures consistency across fixed and adaptive configurations.
"""

from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import Dict, Iterator, Optional
import time


@dataclass
class StageLatencies:
    """Latency breakdown across all distinct pipeline stages (in milliseconds)."""
    feature_extraction_ms: float = 0.0
    policy_inference_ms: float = 0.0
    dense_retrieval_ms: float = 0.0
    sparse_retrieval_ms: float = 0.0
    hybrid_fusion_ms: float = 0.0
    reranking_ms: float = 0.0  # Marked 0.0 / N/A as reranking is not part of this baseline pipeline
    context_preparation_ms: float = 0.0
    llm_generation_ms: float = 0.0
    total_end_to_end_ms: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    def validate(self) -> None:
        """Ensures all recorded latencies are non-negative."""
        for field_name, value in self.to_dict().items():
            if value < 0:
                raise ValueError(f"Latency for stage '{field_name}' cannot be negative: {value} ms")


class StageLatencyTracker:
    """
    Monotonic clock latency tracker for RAG pipeline evaluation.
    Provides context managers to time individual stages without double counting.
    """

    def __init__(self):
        self.latencies = StageLatencies()
        self._global_start: Optional[float] = None

    def start_request(self) -> float:
        """Marks the start of the end-to-end request."""
        self._global_start = time.perf_counter()
        return self._global_start

    def stop_request(self) -> float:
        """Finalizes total end-to-end request latency."""
        if self._global_start is None:
            raise RuntimeError("Request timer was not started.")
        total_ms = (time.perf_counter() - self._global_start) * 1000.0
        self.latencies.total_end_to_end_ms = max(0.0, total_ms)
        self.latencies.validate()
        return self.latencies.total_end_to_end_ms

    @contextmanager
    def time_stage(self, stage_name: str) -> Iterator[None]:
        """Context manager to measure the latency of an individual pipeline stage."""
        if not hasattr(self.latencies, stage_name):
            raise AttributeError(f"Unknown latency stage: {stage_name}")

        t_start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            setattr(self.latencies, stage_name, max(0.0, elapsed_ms))

    def record_stage(self, stage_name: str, latency_ms: float) -> None:
        """Explicitly sets a measured stage latency."""
        if not hasattr(self.latencies, stage_name):
            raise AttributeError(f"Unknown latency stage: {stage_name}")
        setattr(self.latencies, stage_name, max(0.0, float(latency_ms)))

