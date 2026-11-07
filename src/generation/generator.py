"""
src/generation/generator.py

LLM Generation client using Groq API and openai/gpt-oss-120b.
Enforces deterministic greedy decoding (temperature=0.0) and records token telemetry.
"""

from dataclasses import dataclass
from typing import List, Optional
import os
import time

from groq import Groq, RateLimitError, APIError

from src.config import (
    GENERATION_MAX_TOKENS,
    GENERATION_TEMPERATURE,
    GROQ_API_KEY,
    GROQ_MODEL,
    MAX_RETRIES,
    REQUEST_TIMEOUT_SECONDS,
)
from src.retrieval.hybrid import ScoredPassage
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


class GroqGenerator:
    """
    Client for Groq LLM generation with retry logic and telemetry instrumentation.
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
        Executes zero-shot generation over retrieved passages.
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

