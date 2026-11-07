"""
src/generation/generator.py

LLM Generation client using Groq API and openai/gpt-oss-120b.
Enforces deterministic greedy decoding (temperature=0.0) and records token telemetry.
Integrates SAGE adaptive retrieval budget allocation using lightweight probe features.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import os
import time

from groq import Groq, RateLimitError, APIError

from src.config import (
    CANDIDATE_K_VALUES,
    GENERATION_MAX_TOKENS,
    GENERATION_TEMPERATURE,
    GROQ_API_KEY,
    GROQ_MODEL,
    MAX_RETRIES,
    REQUEST_TIMEOUT_SECONDS,
    RETRIEVAL_CANDIDATE_POOL,
    SLO_P95_LATENCY_SECONDS,
)
from src.retrieval.features import FeatureExtractor, QueryRetrievalFeatures
from src.retrieval.hybrid import HybridRetriever, ScoredPassage
from src.utils.logger import setup_logger

logger = setup_logger("generator")

# Standardized research prompt template
SYSTEM_PROMPT = (
    "You are a factual, concise question-answering assistant. "
    "Answer the question using ONLY the provided context passages. "
    "If the answer cannot be determined from the context, state 'I cannot answer based on the provided context.' "
    "Keep your answer brief, direct, and factual."
)


@dataclass
class GenerationResult:
    query: str
    context_used: str
    num_passages: int
    generated_answer: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    model_id: str


@dataclass
class SAGEGenerationResult:
    query: str
    selected_k: int
    features: QueryRetrievalFeatures
    generation_result: GenerationResult
    passages: List[ScoredPassage]
    probe_latency_ms: float
    total_request_latency_ms: float


class GroqGenerator:
    """
    Client for Groq LLM generation with retry logic, telemetry instrumentation,
    and adaptive SAGE retrieval budget execution.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = GROQ_MODEL,
        temperature: float = GENERATION_TEMPERATURE,
        max_tokens: int = GENERATION_MAX_TOKENS,
    ):
        self.api_key = api_key or GROQ_API_KEY
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Please ensure it is defined in your local .env file."
            )

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = Groq(api_key=self.api_key, timeout=REQUEST_TIMEOUT_SECONDS)

    @staticmethod
    def format_context(passages: List[ScoredPassage], max_tokens_budget: Optional[int] = None) -> str:
        """
        Formats retrieved passages into a structured context string.
        Optionally truncates at a context token budget b.
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

    def generate(
        self,
        query: str,
        passages: List[ScoredPassage],
        context_token_budget: Optional[int] = None,
    ) -> GenerationResult:
        """
        Executes zero-shot generation over retrieved passages with greedy decoding.
        """
        context = self.format_context(passages, max_tokens_budget=context_token_budget)
        user_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        retries = 0
        last_error = None

        while retries <= MAX_RETRIES:
            try:
                t0 = time.perf_counter()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                latency_ms = (time.perf_counter() - t0) * 1000.0

                generated_answer = response.choices[0].message.content.strip()
                usage = response.usage

                return GenerationResult(
                    query=query,
                    context_used=context,
                    num_passages=len(passages),
                    generated_answer=generated_answer,
                    prompt_tokens=usage.prompt_tokens if usage else 0,
                    completion_tokens=usage.completion_tokens if usage else 0,
                    total_tokens=usage.total_tokens if usage else 0,
                    latency_ms=latency_ms,
                    model_id=self.model,
                )

            except RateLimitError as e:
                retries += 1
                wait_time = 2 ** retries
                logger.warning(f"Groq RateLimit encountered. Retrying in {wait_time}s... ({retries}/{MAX_RETRIES})")
                time.sleep(wait_time)
                last_error = e

            except APIError as e:
                retries += 1
                logger.error(f"Groq APIError: {e}. Retrying... ({retries}/{MAX_RETRIES})")
                time.sleep(1)
                last_error = e

        raise RuntimeError(f"Failed to generate answer after {MAX_RETRIES} retries. Last error: {last_error}")

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
        Executes end-to-end SAGE adaptive generation:
        1. Records monotonic request start time.
        2. Retrieves candidate pool (depth >= 2) from dense and BM25 channels.
        3. Captures dense and BM25 scores and canonical passage IDs before RRF fusion.
        4. Extracts lightweight k=2 probe features & latency budget.
        5. Predicts adaptive budget K using the Random Forest policy.
        6. Fuses candidates using RRF up to predicted K (no redundant retrieval passes).
        7. Sends selected passages to LLM with greedy decoding.
        """
        req_start = request_start_time if request_start_time is not None else time.perf_counter()

        # 1. Candidate retrieval (serving both the k=2 probe and the subsequent budget K)
        t_probe_start = time.perf_counter()
        pool_depth = max(candidate_pool_depth, 2)
        dense_candidates = retriever.dense_retriever.search(query, top_k=pool_depth)
        sparse_candidates = retriever.sparse_retriever.search(query, top_k=pool_depth)
        probe_latency_ms = (time.perf_counter() - t_probe_start) * 1000.0

        # Map to canonical chunk IDs and scores
        chunks = retriever.chunks
        dense_ids = [chunks[idx].chunk_id if chunks and idx < len(chunks) else f"chunk_{idx}" for idx, _ in dense_candidates]
        sparse_ids = [chunks[idx].chunk_id if chunks and idx < len(chunks) else f"chunk_{idx}" for idx, _ in sparse_candidates]
        dense_scores = [score for _, score in dense_candidates]
        sparse_scores = [score for _, score in sparse_candidates]

        # 2. Extract research-aligned features
        features = FeatureExtractor.extract_features(
            query=query,
            dense_scores=dense_scores,
            dense_passage_ids=dense_ids,
            sparse_scores=sparse_scores,
            sparse_passage_ids=sparse_ids,
            request_start_time=req_start,
            slo_budget_ms=slo_budget_ms,
        )

        # 3. Predict adaptive budget K using trained Random Forest policy
        if policy is not None:
            feature_vector = features.to_array().reshape(1, -1)
            pred = policy.predict(feature_vector)
            if hasattr(pred, "__len__"):
                pred = pred[0]
            raw_k = int(round(float(pred)))
            # Snap to valid candidate budgets: [2, 3, 5, 7, 10, 15, 20, 25, 30]
            selected_k = min(CANDIDATE_K_VALUES, key=lambda x: abs(x - raw_k))
        else:
            selected_k = default_k

        # 4. Perform RRF fusion over the retrieved candidate pool up to selected_k
        fused_passages = retriever.fuse_candidates(
            dense_results=dense_candidates,
            sparse_results=sparse_candidates,
            top_k=selected_k,
        )

        # 5. Generate answer using selected passages
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
