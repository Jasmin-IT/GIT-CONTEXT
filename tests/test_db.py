import tempfile
import unittest
from pathlib import Path

from src.gitcontext.db import IndexDatabase


class RepositoryAndChatDatabaseTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database = IndexDatabase(Path(self.temp_dir.name) / "app.db")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_repository_catalog_persists_and_switches_active_repository(self):
        self.database.register_repository(
            "local:first",
            "First",
            "local",
            "C:/repos/first",
            activate=True,
        )
        self.database.register_repository(
            "github:second@main",
            "Second",
            "github",
            "https://github.com/example/second",
            branch="main",
            activate=False,
        )

        self.database.set_active_repository("github:second@main")

        self.assertEqual(
            self.database.get_active_repository()["repository_id"],
            "github:second@main",
        )
        self.assertEqual(len(self.database.list_repositories()), 2)

    def test_chat_history_is_durable_and_isolated_by_repository(self):
        self.database.register_repository(
            "local:first", "First", "local", "C:/repos/first"
        )
        self.database.save_chat_message(
            "local:first",
            "session-a",
            "user",
            "Where is the entry point?",
        )
        self.database.save_chat_message(
            "local:first",
            "session-a",
            "assistant",
            "In app.py.",
            sources=[{"file_path": "app.py"}],
            response_time_ms=12.5,
        )

        history = self.database.get_chat_history(
            "local:first", "session-a"
        )
        other_repository_history = self.database.get_chat_history(
            "local:second", "session-a"
        )

        self.assertEqual([item["role"] for item in history], ["user", "assistant"])
        self.assertEqual(history[1]["sources"], [{"file_path": "app.py"}])
        self.assertEqual(history[1]["response_time_ms"], 12.5)
        self.assertEqual(other_repository_history, [])


if __name__ == "__main__":
    unittest.main()
