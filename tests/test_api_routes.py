import asyncio
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

from starlette.datastructures import UploadFile

import api


def make_repository_zip():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("repo/app.py", "def answer():\n    return 42\n")
    return buffer.getvalue()


class ApiRouteTests(unittest.TestCase):
    def test_api_starts_when_saved_zip_source_is_missing(self):
        database = Mock()
        database.get_active_repository.return_value = {
            "source_type": "zip",
            "source_location": "C:/missing/repository.zip",
        }

        async def run_lifespan():
            with (
                patch.object(api, "components", {}),
                patch.object(api, "repository_components", {}),
                patch.object(api, "startup_error", None),
                patch.object(api, "get_database", return_value=database),
            ):
                async with api.lifespan(api.app):
                    self.assertFalse(api.health()["repository_loaded"])
                    self.assertIn("could not be restored", api.health()["message"])

        asyncio.run(run_lifespan())

    def test_chat_skips_rewriting_when_disabled(self):
        llm = Mock()
        llm.query_rewriting_enabled.return_value = False
        retriever = Mock()
        retriever.search_multiple.return_value = []
        backend = {
            "llm": llm,
            "embedding_model": Mock(),
            "hybrid_retriever": retriever,
            "reranker": Mock(),
            "repository_id": "test-repository",
        }

        with patch.object(api, "components", backend):
            response = api.chat(
                api.ChatRequest(question="Where is the answer handler?")
            )

        llm.generate_search_queries.assert_not_called()
        self.assertIn("couldn't find relevant evidence", response.answer)
        self.assertGreaterEqual(response.response_time_ms, 0)

    def test_zip_upload_passes_loaded_repository_to_indexing(self):
        repository_files = [object()]
        indexed = {
            "repository_id": "zip:test",
            "repository_files": repository_files,
            "indexing_result": {"successful": 1},
        }
        upload = UploadFile(
            filename="repository.zip",
            file=io.BytesIO(make_repository_zip()),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(api, "PROJECT_ROOT", Path(temp_dir)),
                patch.object(api, "components", {}),
                patch.object(api, "initialize_backend", return_value=indexed) as init,
            ):
                result = api.upload_repository(upload)

        self.assertEqual(result["repository_id"], "zip:test")
        self.assertEqual(result["total_files"], 1)
        uploaded_files = init.call_args.kwargs["uploaded_repository_files"]
        self.assertEqual(len(uploaded_files), 1)
        self.assertEqual(uploaded_files[0].file_path, "repo/app.py")

    def test_chat_stream_emits_answer_sources_and_persists_history(self):
        chunk = {
            "metadata": {
                "file_path": "app.py",
                "chunk_name": "answer",
                "chunk_type": "function",
                "start_line": 1,
                "end_line": 2,
            },
            "content": "def answer(): return 42",
            "rerank_score": 0.9,
        }
        llm = Mock()
        llm.query_rewriting_enabled.return_value = False
        llm.generate_answer_stream.return_value = iter([
            {
                "type": "token",
                "text": "The answer is 42. [Source: app.py, Lines: 1-2]",
            }
        ])
        llm.normalize_citations.side_effect = lambda answer: answer
        llm.validate_citations.return_value = True
        database = Mock()
        backend = {
            "llm": llm,
            "repository_id": "test-repository",
            "embedding_model": Mock(),
            "hybrid_retriever": Mock(),
            "reranker": Mock(),
        }
        backend["hybrid_retriever"].search_multiple.return_value = [chunk]
        backend["reranker"].rerank.return_value = [chunk]

        with (
            patch.object(api, "components", backend),
            patch.object(api, "get_database", return_value=database),
            patch.object(api, "load_chat_exchanges", return_value=[]),
        ):
            response = api.chat_stream(
                api.ChatRequest(
                    question="What is the answer?",
                    session_id="test-session",
                )
            )

            async def collect():
                parts = [
                    part async for part in response.body_iterator
                ]
                return "".join(
                    part.decode("utf-8") if isinstance(part, bytes) else part
                    for part in parts
                )

            body = asyncio.run(collect())

        events = [
            json.loads(line[6:])
            for line in body.splitlines()
            if line.startswith("data: ")
        ]
        self.assertEqual(events[0]["type"], "token")
        self.assertEqual(events[-1]["type"], "final")
        self.assertEqual(events[-1]["sources"][0]["file_path"], "app.py")
        self.assertEqual(database.save_chat_message.call_count, 2)


if __name__ == "__main__":
    unittest.main()
