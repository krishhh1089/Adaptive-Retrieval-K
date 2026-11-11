"""
src/generation/local_generator.py

LocalLLMGenerator: Local Ollama-based LLM generation backend for Load-Aware SAGE.

Replaces GroqGenerator as the Oracle generation engine, eliminating:
  - Groq TPM rate-limit pacing and artificial sleeps.
  - External API dependency.
  - Network dependency for inference.

Interface is designed to be compatible with GroqGenerator.generate() so that
oracle.py and generate_oracle_dataset.py require only minimal changes.

Timing Boundaries (all in milliseconds):
  - prompt_time_ms        : Ollama-reported prompt evaluation time (from response stats).
  - generation_time_ms    : Ollama-reported token generation time (from response stats).
  - llm_latency_ms        : prompt_time_ms + generation_time_ms (pure inference cost).
  - client_request_latency_ms : Total round-trip measured by time.perf_counter().

Ollama duration values are reported in nanoseconds; this module converts to ms.

Backend identity is embedded in GenerationResult.model_id as:
  "ollama/<model_name>"

No artificial pacing is introduced between requests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import time

from src.config import (
    CANDIDATE_K_VALUES,
    GENERATION_MAX_TOKENS,
    GENERATION_TEMPERATURE,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT_SECONDS,
    RETRIEVAL_CANDIDATE_POOL,
    SLO_P95_LATENCY_SECONDS,
)
from src.generation.generator import (
    SYSTEM_PROMPT,
    GenerationResult,
    SAGEGenerationResult,
)
from src.retrieval.features import FeatureExtractor, QueryRetrievalFeatures
from src.retrieval.hybrid import HybridRetriever, ScoredPassage
from src.utils.logger import setup_logger

logger = setup_logger("local_generator")

# Nanosecond-to-millisecond conversion factor
_NS_TO_MS = 1e-6


@dataclass
class LocalGenerationTelemetry:
    """
    Extended telemetry captured from Ollama inference response.
    All duration fields are in milliseconds.
    """
    prompt_tokens: int
    completion_tokens: int
    prompt_time_ms: float          # Ollama eval_duration converted from ns
    generation_time_ms: float      # Ollama eval_duration for token generation in ms
    llm_latency_ms: float          # prompt_time_ms + generation_time_ms
    client_request_latency_ms: float  # Measured by perf_counter() (includes serialization)
    tokens_per_second: float
    model_name: str                # e.g., "ollama/qwen2.5:7b"
    generation_success: bool
    finish_reason: str             # "stop", "length", "unknown"
    empty_generation: bool


class OllamaConnectionError(RuntimeError):
    """Raised when the Ollama server is not reachable."""


class OllamaModelNotFoundError(RuntimeError):
    """Raised when the requested model is not installed in Ollama."""


class OllamaTimeoutError(RuntimeError):
    """Raised when an Ollama request exceeds the configured timeout."""


class OllamaGenerationError(RuntimeError):
    """Raised for unexpected generation failures."""


def _ns_to_ms(ns: Optional[int]) -> float:
    """Safely converts nanoseconds (int) to milliseconds (float)."""
    if ns is None or ns <= 0:
        return 0.0
    return float(ns) * _NS_TO_MS


class LocalLLMGenerator:
    """
    Local LLM generation client using Ollama as the inference runtime.

    Provides the same .generate() and .generate_with_sage() interface as
    GroqGenerator so oracle.py requires only a type annotation change.

    No rate-limit pacing or artificial sleeps are introduced.
    The .rate_limit_wait_ms attribute is always 0.0 for compatibility with
    oracle.py's e2e latency exclusion logic.

    Backend identity in GenerationResult.model_id:
        "ollama/<model_name>"

    Example:
        generator = LocalLLMGenerator()
        result = generator.generate(query, passages)
        print(result.generated_answer, result.latency_ms)
    """

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL,
        temperature: float = GENERATION_TEMPERATURE,
        max_tokens: int = GENERATION_MAX_TOKENS,
        timeout: float = OLLAMA_TIMEOUT_SECONDS,
    ):
        """
        Args:
            base_url: Ollama server URL (default: http://localhost:11434).
            model: Ollama model tag, e.g. "qwen2.5:7b" (configurable via OLLAMA_MODEL env).
            temperature: Generation temperature. 0.0 = greedy/deterministic decoding.
            max_tokens: Maximum completion tokens (num_predict in Ollama API).
            timeout: Request timeout in seconds for a single inference call.
        """
        try:
            import ollama as _ollama_module
            self._ollama = _ollama_module
        except ImportError as exc:
            raise ImportError(
                "The 'ollama' package is required for LocalLLMGenerator. "
                "Install it with: pip install ollama"
            ) from exc

        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        # GroqGenerator compatibility: oracle.py reads this to exclude wait time from L(q,K)
        self.rate_limit_wait_ms: float = 0.0

        # Build Ollama client pointed at configured base URL
        self._client = self._ollama.Client(host=self.base_url)

        logger.info(
            f"LocalLLMGenerator initialised: model={self.model!r} "
            f"base_url={self.base_url!r} temperature={self.temperature} "
            f"max_tokens={self.max_tokens} timeout={self.timeout}s"
        )

    @property
    def model_id(self) -> str:
        """Returns a namespaced model identifier for experiment metadata."""
        return f"ollama/{self.model}"

    @staticmethod
    def format_context(
        passages: List[ScoredPassage],
        max_tokens_budget: Optional[int] = None,
    ) -> str:
        """
        Formats retrieved passages into the same structured context string
        as GroqGenerator.format_context() for prompt consistency.
        """
        context_blocks = []
        for i, p in enumerate(passages, start=1):
            block = f"[Passage {i}] (Title: {p.title})\n{p.text.strip()}"
            context_blocks.append(block)

        full_context = "\n\n".join(context_blocks)

        # Approximate character truncation if a budget is enforced (~4 chars per token)
        if max_tokens_budget:
            char_limit = max_tokens_budget * 4
            if len(full_context) > char_limit:
                full_context = full_context[:char_limit] + "\n[Context truncated by budget]"

        return full_context

    def _build_messages(self, query: str, context: str) -> List[Dict[str, str]]:
        """Builds the chat message list in the same format as GroqGenerator."""
        user_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    def check_server(self) -> bool:
        """
        Pings the Ollama server.

        Returns:
            True if the server is reachable, False otherwise.
        """
        try:
            self._client.list()
            return True
        except Exception:
            return False

    def check_model_available(self) -> bool:
        """
        Checks whether self.model is installed in the local Ollama runtime.

        Returns:
            True if the model tag is found in the local model list.
        """
        try:
            response = self._client.list()
            installed = [m.model for m in response.models]
            # Normalize: strip digest suffix if present
            installed_names = {m.split(":")[0] for m in installed} | set(installed)
            target = self.model
            return target in installed_names or target.split(":")[0] in installed_names
        except Exception:
            return False

    def generate(
        self,
        query: str,
        passages: List[ScoredPassage],
        context_token_budget: Optional[int] = None,
    ) -> GenerationResult:
        """
        Executes zero-shot generation over retrieved passages using Ollama.

        Timing:
          - client_request_latency_ms: wall-clock measured by perf_counter.
          - prompt_time_ms / generation_time_ms: Ollama-reported durations (ns→ms).
          - llm_latency_ms = prompt_time_ms + generation_time_ms.

        Returns:
            GenerationResult with GroqGenerator-compatible fields.
            rate_limit_wait_ms is always 0.0.

        Raises:
            OllamaConnectionError: Ollama server not reachable.
            OllamaModelNotFoundError: Model not installed.
            OllamaTimeoutError: Request exceeded self.timeout.
            OllamaGenerationError: Other generation failure.
        """
        context = self.format_context(passages, max_tokens_budget=context_token_budget)
        messages = self._build_messages(query, context)

        t0 = time.perf_counter()
        try:
            response = self._client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            )
        except Exception as exc:
            client_latency_ms = (time.perf_counter() - t0) * 1000.0
            exc_str = str(exc).lower()
            if "connection" in exc_str or "refused" in exc_str or "unreachable" in exc_str:
                raise OllamaConnectionError(
                    f"Ollama server at {self.base_url!r} is not reachable. "
                    "Start it with: ollama serve"
                ) from exc
            if "not found" in exc_str or "model" in exc_str and ("pull" in exc_str or "404" in exc_str):
                raise OllamaModelNotFoundError(
                    f"Model {self.model!r} is not installed. "
                    f"Install it with: ollama pull {self.model}"
                ) from exc
            if "timeout" in exc_str or "timed out" in exc_str:
                raise OllamaTimeoutError(
                    f"Ollama request timed out after {self.timeout}s for model {self.model!r}."
                ) from exc
            raise OllamaGenerationError(
                f"Ollama generation failed after {client_latency_ms:.0f}ms: {exc}"
            ) from exc

        client_latency_ms = (time.perf_counter() - t0) * 1000.0

        # Extract text
        generated_answer = ""
        if response.message and response.message.content:
            generated_answer = response.message.content.strip()

        # Token usage from Ollama response
        prompt_tokens = getattr(response, "prompt_eval_count", 0) or 0
        completion_tokens = getattr(response, "eval_count", 0) or 0
        total_tokens = prompt_tokens + completion_tokens

        # Timing from Ollama (in nanoseconds → convert to ms)
        # prompt_eval_duration: time spent evaluating the prompt
        # eval_duration: time spent generating completion tokens
        prompt_time_ms = _ns_to_ms(getattr(response, "prompt_eval_duration", None))
        generation_time_ms = _ns_to_ms(getattr(response, "eval_duration", None))
        llm_latency_ms = prompt_time_ms + generation_time_ms

        # tokens_per_second based on generation phase only
        if generation_time_ms > 0 and completion_tokens > 0:
            tokens_per_second = completion_tokens / (generation_time_ms / 1000.0)
        else:
            tokens_per_second = 0.0

        finish_reason = "stop"
        done_reason = getattr(response, "done_reason", None)
        if done_reason:
            finish_reason = str(done_reason)
        elif getattr(response, "done", True) is False:
            finish_reason = "unknown"

        # Detect truncated/empty generation
        empty_generation = len(generated_answer) == 0

        # Log telemetry
        telemetry = LocalGenerationTelemetry(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_time_ms=prompt_time_ms,
            generation_time_ms=generation_time_ms,
            llm_latency_ms=llm_latency_ms,
            client_request_latency_ms=client_latency_ms,
            tokens_per_second=tokens_per_second,
            model_name=self.model_id,
            generation_success=not empty_generation,
            finish_reason=finish_reason,
            empty_generation=empty_generation,
        )
        if empty_generation:
            logger.warning(
                f"Empty generation returned by {self.model!r} "
                f"(finish_reason={finish_reason!r}, completion_tokens={completion_tokens}). "
                "Consider increasing GENERATION_MAX_TOKENS."
            )
        else:
            logger.debug(
                f"Generated {completion_tokens} tokens in {generation_time_ms:.1f}ms "
                f"({tokens_per_second:.1f} tok/s) | finish={finish_reason!r}"
            )

        return GenerationResult(
            query=query,
            context_used=context,
            num_passages=len(passages),
            generated_answer=generated_answer,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            # latency_ms exposes llm_latency_ms so oracle.py's e2e calculation is consistent.
            # client_request_latency_ms is separately stored in telemetry.
            latency_ms=llm_latency_ms if llm_latency_ms > 0 else client_latency_ms,
            model_id=self.model_id,
            rate_limit_wait_ms=0.0,  # Always 0 — no pacing for local inference
        )

    def generate_with_sage(
        self,
        query: str,
        retriever: HybridRetriever,
        policy: Optional[Any] = None,
        default_k: int = 5,
        candidate_pool_depth: int = RETRIEVAL_CANDIDATE_POOL,
        context_token_budget: Optional[int] = None,
        slo_budget_ms: Optional[float] = None,
        request_start_time: Optional[float] = None,
    ) -> SAGEGenerationResult:
        """
        Executes end-to-end SAGE adaptive generation using the local Ollama backend.
        Identical logic to GroqGenerator.generate_with_sage().
        """
        req_start = request_start_time if request_start_time is not None else time.perf_counter()

        # 1. Candidate retrieval
        t_probe_start = time.perf_counter()
        pool_depth = max(candidate_pool_depth, 2)
        dense_candidates = retriever.dense_retriever.search(query, top_k=pool_depth)
        sparse_candidates = retriever.sparse_retriever.search(query, top_k=pool_depth)
        probe_latency_ms = (time.perf_counter() - t_probe_start) * 1000.0

        chunks = retriever.chunks
        dense_ids = [
            chunks[idx].chunk_id if chunks and idx < len(chunks) else f"chunk_{idx}"
            for idx, _ in dense_candidates
        ]
        sparse_ids = [
            chunks[idx].chunk_id if chunks and idx < len(chunks) else f"chunk_{idx}"
            for idx, _ in sparse_candidates
        ]
        dense_scores = [score for _, score in dense_candidates]
        sparse_scores = [score for _, score in sparse_candidates]

        # 2. Extract SAGE probe features
        features = FeatureExtractor.extract_features(
            query=query,
            dense_scores=dense_scores,
            dense_passage_ids=dense_ids,
            sparse_scores=sparse_scores,
            sparse_passage_ids=sparse_ids,
            request_start_time=req_start,
            slo_budget_ms=slo_budget_ms,
        )

        # 3. Predict adaptive budget K
        if policy is not None:
            feature_vector = features.to_array().reshape(1, -1)
            pred = policy.predict(feature_vector)
            if hasattr(pred, "__len__"):
                pred = pred[0]
            raw_k = int(round(float(pred)))
            selected_k = min(CANDIDATE_K_VALUES, key=lambda x: abs(x - raw_k))
        else:
            selected_k = default_k

        # 4. RRF fusion
        fused_passages = retriever.fuse_candidates(
            dense_results=dense_candidates,
            sparse_results=sparse_candidates,
            top_k=selected_k,
        )

        # 5. Local generation
        gen_result = self.generate(
            query=query,
            passages=fused_passages,
            context_token_budget=context_token_budget,
        )

        total_request_latency_ms = (time.perf_counter() - req_start) * 1000.0

        return SAGEGenerationResult(
            query=query,
            selected_k=selected_k,
            features=features,
            generation_result=gen_result,
            passages=fused_passages,
            probe_latency_ms=probe_latency_ms,
            total_request_latency_ms=total_request_latency_ms,
        )
