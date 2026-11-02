import os
import unittest
from unittest.mock import Mock, patch

from src.gitcontext.llm import GitContextLLM


class LLMProviderTests(unittest.TestCase):
    def test_generation_uses_gemini_after_groq_failure(self):
        llm = GitContextLLM.__new__(GitContextLLM)
        llm.groq_client = object()
        llm.gemini_client = object()
        llm.groq_api_key = "groq-test-key"
        llm.gemini_api_key = "gemini-test-key"
        llm.build_prompt = Mock(return_value="evidence prompt")
        llm.generate_with_groq = Mock(side_effect=RuntimeError("unavailable"))
        llm.generate_with_gemini = Mock(
            return_value="[Source: app.py, Lines: 1-2]"
        )
        llm.normalize_citations = Mock(side_effect=lambda answer: answer)
        llm.validate_citations = Mock(return_value=True)

        answer = llm.generate_answer(
            "What does this do?",
            [{"content": "def run(): pass", "metadata": {}}],
        )

        self.assertEqual(answer, "[Source: app.py, Lines: 1-2]")
        llm.generate_with_groq.assert_called_once_with("evidence prompt")
        llm.generate_with_gemini.assert_called_once_with("evidence prompt")

    def test_streaming_resets_partial_groq_output_before_gemini_fallback(self):
        llm = GitContextLLM.__new__(GitContextLLM)
        llm.groq_client = object()
        llm.gemini_client = object()
        llm.groq_api_key = None
        llm.gemini_api_key = None

        def partial_then_fail(prompt):
            yield "partial"
            raise RuntimeError("primary failed")

        def backup(prompt):
            yield "backup"

        with (
            patch.object(
                GitContextLLM,
                "generate_groq_stream",
                side_effect=lambda prompt: partial_then_fail(prompt),
            ),
            patch.object(
                GitContextLLM,
                "generate_gemini_stream",
                side_effect=lambda prompt: backup(prompt),
            ),
        ):
            events = list(llm.generate_answer_stream(
                "question",
                [{
                    "content": "evidence",
                    "metadata": {
                        "file_path": "app.py",
                        "chunk_name": "run",
                        "start_line": 1,
                        "end_line": 2,
                    },
                }],
            ))

        self.assertEqual(
            events,
            [
                {"type": "token", "text": "partial"},
                {"type": "reset"},
                {"type": "token", "text": "backup"},
            ],
        )

    def test_query_rewriting_is_disabled_unless_enabled(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(GitContextLLM.query_rewriting_enabled())
        with patch.dict(os.environ, {"GITCONTEXT_QUERY_REWRITE": "true"}):
            self.assertTrue(GitContextLLM.query_rewriting_enabled())


if __name__ == "__main__":
    unittest.main()
