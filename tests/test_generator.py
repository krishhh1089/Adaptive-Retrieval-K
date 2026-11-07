"""
tests/test_generator.py

Verifies the GroqGenerator client with openai/gpt-oss-120b over hybrid retrieved context.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.chunker import DocumentChunker
from src.retrieval.hybrid import HybridRetriever
from src.generation.generator import GroqGenerator
from src import config


def main():
    print("=== Step 7.1: Testing Groq LLM Generation Engine ===")

    # 1. Load chunks and retrieve passages for test query
    chunker = DocumentChunker()
    chunks = chunker.load_chunks()
    retriever = HybridRetriever(rrf_k=config.RRF_K_CONSTANT, candidate_pool=config.RETRIEVAL_CANDIDATE_POOL)
    retriever.initialize(chunks)

    test_query = "when is the last episode of season 8 of the walking dead"
    top_k = 3
    retrieval_result = retriever.search(test_query, top_k=top_k)

    print(f"Query: '{test_query}'")
    print(f"Retrieved {len(retrieval_result.passages)} passages via Hybrid RRF.")

    # 2. Initialize generator
    generator = GroqGenerator()
    print(f"Connected to Groq API. Model: {generator.model} | Temperature: {generator.temperature}")

    # 3. Generate answer
    print("\nSending prompt to Groq API...")
    result = generator.generate(test_query, retrieval_result.passages)

    print("\n=== Generation Results ===")
    print(f"Model:              {result.model_id}")
    print(f"Generated Answer:   {result.generated_answer}")
    print(f"Prompt Tokens:      {result.prompt_tokens}")
    print(f"Completion Tokens:  {result.completion_tokens}")
    print(f"Total Tokens:       {result.total_tokens}")
    print(f"API Latency:        {result.latency_ms:.2f} ms")

    assert len(result.generated_answer) > 0, "Generated answer is empty!"
    print("\nStep 7.1 Verification Complete!")


if __name__ == "__main__":
    main()

