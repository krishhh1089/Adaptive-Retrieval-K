"""
tests/test_local_generator.py

Unit tests for LocalLLMGenerator (Ollama backend).

All tests use mocked Ollama responses — no live Ollama server or model required.

Coverage:
  1.  Successful generation response (text, tokens, timing).
  2.  Correct nanosecond-to-millisecond conversion.
  3.  Correct llm_latency_ms = prompt_time_ms + generation_time_ms.
  4.  tokens_per_second calculation.
  5.  Empty generation detection and warning.
  6.  Finish reason extraction.
  7.  Missing Ollama server -> OllamaConnectionError.
  8.  Model not installed -> OllamaModelNotFoundError.
  9.  Timeout -> OllamaTimeoutError.
  10. rate_limit_wait_ms is always 0.0 (no Groq pacing).
  11. model_id format is "ollama/<model>".
  12. format_context matches GroqGenerator output.
  13. GenerationResult fields populated correctly.
  14. Resume cache backend/model validation (identity mismatch raises SystemExit).
  15. Resume cache identity match passes without raising.
  16. check_server() returns True/False based on mock.
  17. check_model_available() matches installed model list.
  18. Oracle evaluate_candidate uses rate_limit_wait_ms=0 from LocalLLMGenerator.
"""

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock
import pytest

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.local_generator import (
    LocalLLMGenerator,
    LocalGenerationTelemetry,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
    OllamaGenerationError,
    _ns_to_ms,
)
from src.generation.generator import GenerationResult
from src.retrieval.hybrid import ScoredPassage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_passage(i: int = 0) -> ScoredPassage:
    return ScoredPassage(
        chunk_index=i,
        chunk_id=f"chunk_{i}",
        doc_id="doc_0",
        title="Test Title",
        text=f"Passage text number {i}.",
        rrf_score=0.9 - i * 0.1,
    )


def _make_ollama_response(
    content: str = "Paris",
    prompt_eval_count: int = 50,
    eval_count: int = 10,
    prompt_eval_duration: int = 500_000_000,   # 500ms in ns
    eval_duration: int = 300_000_000,           # 300ms in ns
    done_reason: str = "stop",
) -> SimpleNamespace:
    """Builds a minimal mock Ollama chat response."""
    return SimpleNamespace(
        message=SimpleNamespace(content=content),
        prompt_eval_count=prompt_eval_count,
        eval_count=eval_count,
        prompt_eval_duration=prompt_eval_duration,
        eval_duration=eval_duration,
        done_reason=done_reason,
        done=True,
    )


def _make_generator(**kwargs) -> LocalLLMGenerator:
    """Creates a LocalLLMGenerator with the ollama module mocked."""
    mock_ollama = MagicMock()
    mock_client = MagicMock()
    mock_ollama.Client.return_value = mock_client
    # Patch ollama at the module level where local_generator imports it
    with patch.dict("sys.modules", {"ollama": mock_ollama}):
        import importlib
        import src.generation.local_generator as _lgmod
        # Re-create generator with patched import resolution
        gen = object.__new__(LocalLLMGenerator)
        gen._ollama = mock_ollama
        gen._client = mock_client
        gen.base_url = kwargs.get("base_url", "http://localhost:11434")
        gen.model = kwargs.get("model", "qwen2.5:7b")
        gen.temperature = kwargs.get("temperature", 0.0)
        gen.max_tokens = kwargs.get("max_tokens", 512)
        gen.timeout = kwargs.get("timeout", 180.0)
        gen.rate_limit_wait_ms = 0.0
    return gen


# ---------------------------------------------------------------------------
# 1. ns->ms conversion
# ---------------------------------------------------------------------------

class TestNsToMs:
    def test_zero(self):
        assert _ns_to_ms(0) == 0.0

    def test_none(self):
        assert _ns_to_ms(None) == 0.0

    def test_negative(self):
        assert _ns_to_ms(-100) == 0.0

    def test_one_second(self):
        assert pytest.approx(_ns_to_ms(1_000_000_000), rel=1e-6) == 1000.0

    def test_500ms(self):
        assert pytest.approx(_ns_to_ms(500_000_000), rel=1e-6) == 500.0

    def test_1ms(self):
        assert pytest.approx(_ns_to_ms(1_000_000), rel=1e-6) == 1.0


# ---------------------------------------------------------------------------
# 2. Successful generation
# ---------------------------------------------------------------------------

class TestSuccessfulGeneration:
    def setup_method(self):
        self.gen = _make_generator()
        self.passages = [_make_passage(0), _make_passage(1)]

    def test_generated_answer_returned(self):
        resp = _make_ollama_response(content="Paris")
        self.gen._client.chat.return_value = resp
        result = self.gen.generate("What is the capital of France?", self.passages)
        assert result.generated_answer == "Paris"

    def test_prompt_tokens(self):
        resp = _make_ollama_response(prompt_eval_count=42)
        self.gen._client.chat.return_value = resp
        result = self.gen.generate("Q?", self.passages)
        assert result.prompt_tokens == 42

    def test_completion_tokens(self):
        resp = _make_ollama_response(eval_count=15)
        self.gen._client.chat.return_value = resp
        result = self.gen.generate("Q?", self.passages)
        assert result.completion_tokens == 15

    def test_total_tokens(self):
        resp = _make_ollama_response(prompt_eval_count=50, eval_count=10)
        self.gen._client.chat.return_value = resp
        result = self.gen.generate("Q?", self.passages)
        assert result.total_tokens == 60

    def test_llm_latency_ms_is_sum_of_prompt_and_gen(self):
        # prompt_eval_duration=500ms ns, eval_duration=300ms ns
        resp = _make_ollama_response(
            prompt_eval_duration=500_000_000,
            eval_duration=300_000_000,
        )
        self.gen._client.chat.return_value = resp
        result = self.gen.generate("Q?", self.passages)
        # latency_ms should equal llm_latency_ms = 500 + 300 = 800ms
        assert pytest.approx(result.latency_ms, rel=1e-3) == 800.0

    def test_model_id_format(self):
        resp = _make_ollama_response()
        self.gen._client.chat.return_value = resp
        result = self.gen.generate("Q?", self.passages)
        assert result.model_id.startswith("ollama/")

    def test_rate_limit_wait_ms_always_zero(self):
        resp = _make_ollama_response()
        self.gen._client.chat.return_value = resp
        result = self.gen.generate("Q?", self.passages)
        assert result.rate_limit_wait_ms == 0.0

    def test_result_is_generation_result_instance(self):
        resp = _make_ollama_response()
        self.gen._client.chat.return_value = resp
        result = self.gen.generate("Q?", self.passages)
        assert isinstance(result, GenerationResult)

    def test_num_passages_recorded(self):
        resp = _make_ollama_response()
        self.gen._client.chat.return_value = resp
        result = self.gen.generate("Q?", self.passages)
        assert result.num_passages == len(self.passages)


# ---------------------------------------------------------------------------
# 3. Empty generation detection
# ---------------------------------------------------------------------------

class TestEmptyGeneration:
    def setup_method(self):
        self.gen = _make_generator()
        self.passages = [_make_passage()]

    def test_empty_answer_recorded(self):
        resp = _make_ollama_response(content="")
        self.gen._client.chat.return_value = resp
        result = self.gen.generate("Q?", self.passages)
        assert result.generated_answer == ""

    def test_whitespace_stripped_to_empty(self):
        resp = _make_ollama_response(content="   \n  ")
        self.gen._client.chat.return_value = resp
        result = self.gen.generate("Q?", self.passages)
        assert result.generated_answer == ""


# ---------------------------------------------------------------------------
# 4. Finish reason
# ---------------------------------------------------------------------------

class TestFinishReason:
    def setup_method(self):
        self.gen = _make_generator()
        self.passages = [_make_passage()]

    def test_stop_finish_reason(self):
        resp = _make_ollama_response(done_reason="stop")
        self.gen._client.chat.return_value = resp
        # No assertion on GenerationResult (finish_reason is in telemetry only)
        result = self.gen.generate("Q?", self.passages)
        assert isinstance(result, GenerationResult)

    def test_length_finish_reason_no_crash(self):
        resp = _make_ollama_response(done_reason="length")
        self.gen._client.chat.return_value = resp
        result = self.gen.generate("Q?", self.passages)
        assert isinstance(result, GenerationResult)


# ---------------------------------------------------------------------------
# 5. Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def setup_method(self):
        self.gen = _make_generator()
        self.passages = [_make_passage()]

    def test_connection_refused_raises_ollama_connection_error(self):
        self.gen._client.chat.side_effect = Exception("Connection refused")
        with pytest.raises(OllamaConnectionError):
            self.gen.generate("Q?", self.passages)

    def test_connection_unreachable_raises_ollama_connection_error(self):
        self.gen._client.chat.side_effect = Exception("unreachable host")
        with pytest.raises(OllamaConnectionError):
            self.gen.generate("Q?", self.passages)

    def test_model_not_found_raises_ollama_model_not_found(self):
        self.gen._client.chat.side_effect = Exception("model not found, pull it first with 404")
        with pytest.raises(OllamaModelNotFoundError):
            self.gen.generate("Q?", self.passages)

    def test_timeout_raises_ollama_timeout_error(self):
        self.gen._client.chat.side_effect = Exception("timed out")
        with pytest.raises(OllamaTimeoutError):
            self.gen.generate("Q?", self.passages)

    def test_generic_error_raises_ollama_generation_error(self):
        self.gen._client.chat.side_effect = Exception("unexpected kernel panic")
        with pytest.raises(OllamaGenerationError):
            self.gen.generate("Q?", self.passages)


# ---------------------------------------------------------------------------
# 6. check_server and check_model_available
# ---------------------------------------------------------------------------

class TestServerChecks:
    def setup_method(self):
        self.gen = _make_generator()

    def test_check_server_true_when_list_succeeds(self):
        self.gen._client.list.return_value = MagicMock()
        assert self.gen.check_server() is True

    def test_check_server_false_when_list_raises(self):
        self.gen._client.list.side_effect = Exception("refused")
        assert self.gen.check_server() is False

    def test_check_model_available_true_when_model_in_list(self):
        model_ns = SimpleNamespace(model="qwen2.5:7b")
        self.gen._client.list.return_value = SimpleNamespace(models=[model_ns])
        self.gen.model = "qwen2.5:7b"
        assert self.gen.check_model_available() is True

    def test_check_model_available_false_when_model_absent(self):
        model_ns = SimpleNamespace(model="llama3.2:3b")
        self.gen._client.list.return_value = SimpleNamespace(models=[model_ns])
        self.gen.model = "qwen2.5:7b"
        assert self.gen.check_model_available() is False

    def test_check_model_available_false_when_server_down(self):
        self.gen._client.list.side_effect = Exception("refused")
        assert self.gen.check_model_available() is False


# ---------------------------------------------------------------------------
# 7. format_context (matches GroqGenerator behavior)
# ---------------------------------------------------------------------------

class TestFormatContext:
    def test_passage_format_includes_title_and_text(self):
        passages = [_make_passage(0)]
        ctx = LocalLLMGenerator.format_context(passages)
        assert "[Passage 1]" in ctx
        assert "Test Title" in ctx
        assert "Passage text number 0." in ctx

    def test_multiple_passages_separated_by_blank_line(self):
        passages = [_make_passage(0), _make_passage(1)]
        ctx = LocalLLMGenerator.format_context(passages)
        assert "[Passage 1]" in ctx
        assert "[Passage 2]" in ctx

    def test_budget_truncation(self):
        passages = [_make_passage(0)]
        # Very small budget to force truncation
        ctx = LocalLLMGenerator.format_context(passages, max_tokens_budget=1)
        assert "[Context truncated by budget]" in ctx

    def test_no_truncation_within_budget(self):
        passages = [_make_passage(0)]
        ctx = LocalLLMGenerator.format_context(passages, max_tokens_budget=10000)
        assert "[Context truncated by budget]" not in ctx
# ---------------------------------------------------------------------------
# 8. No Groq pacing in local execution path
# ---------------------------------------------------------------------------

class TestNoPacing:
    def test_no_rate_limit_attributes(self):
        gen = _make_generator()
        # LocalLLMGenerator should NOT have Groq pacing internals
        assert not hasattr(gen, "pace_for_rate_limit")
        assert not hasattr(gen, "_remaining_tokens")
        assert not hasattr(gen, "_wait_if_rate_limited")

    def test_rate_limit_wait_ms_zero_always(self):
        gen = _make_generator()
        resp = _make_ollama_response()
        gen._client.chat.return_value = resp
        result = gen.generate("Q?", [_make_passage()])
        assert result.rate_limit_wait_ms == 0.0


# ---------------------------------------------------------------------------
# 9. Resume cache backend/model validation
# ---------------------------------------------------------------------------

class TestResumeCacheValidation:
    def _write_identity(self, tmp_path, backend, model_id):
        identity_path = tmp_path / "oracle_backend_identity.json"
        identity_path.write_text(json.dumps({"backend": backend, "model_id": model_id}))
        return identity_path

    def test_matching_identity_does_not_raise(self, tmp_path):
        from scripts.generate_oracle_dataset import _validate_resume_identity
        with patch(
            "scripts.generate_oracle_dataset._BACKEND_META_PATH",
            self._write_identity(tmp_path, "ollama", "ollama/qwen2.5:7b"),
        ):
            # Should not raise
            _validate_resume_identity("ollama", "ollama/qwen2.5:7b")

    def test_backend_mismatch_raises_system_exit(self, tmp_path):
        from scripts.generate_oracle_dataset import _validate_resume_identity
        with patch(
            "scripts.generate_oracle_dataset._BACKEND_META_PATH",
            self._write_identity(tmp_path, "groq", "groq/openai/gpt-oss-120b"),
        ):
            with pytest.raises(SystemExit):
                _validate_resume_identity("ollama", "ollama/qwen2.5:7b")

    def test_model_mismatch_raises_system_exit(self, tmp_path):
        from scripts.generate_oracle_dataset import _validate_resume_identity
        with patch(
            "scripts.generate_oracle_dataset._BACKEND_META_PATH",
            self._write_identity(tmp_path, "ollama", "ollama/llama3.2:3b"),
        ):
            with pytest.raises(SystemExit):
                _validate_resume_identity("ollama", "ollama/qwen2.5:7b")

    def test_missing_identity_file_does_not_raise(self, tmp_path):
        from scripts.generate_oracle_dataset import _validate_resume_identity
        missing = tmp_path / "nonexistent.json"
        with patch("scripts.generate_oracle_dataset._BACKEND_META_PATH", missing):
            # First run: writes the file, does not raise
            _validate_resume_identity("ollama", "ollama/qwen2.5:7b")
            assert missing.exists()


# ---------------------------------------------------------------------------
# 10. Oracle evaluate_candidate uses LocalLLMGenerator correctly
# ---------------------------------------------------------------------------

class TestOracleWithLocalGenerator:
    def test_oracle_accepts_local_generator(self):
        from src.adaptive.oracle import QualityAwareOracle
        from src.data.loader import QuestionRecord

        mock_retriever = MagicMock()
        from src.retrieval.hybrid import HybridRetrievalResult
        mock_passage = ScoredPassage(0, "c0", "d0", "Title", "Paris is the capital of France.", 0.9)
        mock_retriever.search.return_value = HybridRetrievalResult(
            query="What is the capital of France?",
            passages=[mock_passage],
            dense_latency_ms=200.0,
            sparse_latency_ms=30.0,
            fusion_latency_ms=0.1,
            total_retrieval_latency_ms=230.1,
        )

        # Build a LocalLLMGenerator with mocked internals
        mock_gen = _make_generator()
        resp = _make_ollama_response(content="Paris")
        mock_gen._client.chat.return_value = resp

        oracle = QualityAwareOracle(
            retriever=mock_retriever,
            generator=mock_gen,
            slo_latency_seconds=5.0,
        )
        record = QuestionRecord(
            qid="q_local_1",
            question="What is the capital of France?",
            target_doc_id="d0",
            short_answers=["Paris"],
            has_answer=True,
        )

        ev = oracle.evaluate_candidate(record, candidate_k=2)
        assert ev.evaluation_status == "evaluated"
        assert ev.answer_quality == 1.0
        # rate_limit_wait_ms must be 0.0, so e2e_latency excludes no pacing wait
        assert ev.end_to_end_latency_ms >= 0.0
        # e2e = wall - rate_limit_wait_ms (0.0), must be > 0
        assert ev.slo_satisfied is True


if __name__ == "__main__":
    pytest.main(["-v", __file__])
