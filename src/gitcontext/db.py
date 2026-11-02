import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path


class IndexDatabase:

    def __init__(self, db_path=None):

        if db_path is None:
            db_path = Path(__file__).resolve().parents[2] / "data" / "gitcontext.db"

        Path(db_path).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.db_path = db_path

        self.initialize()

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.db_path)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self):

        with self.connect() as connection:

            connection.execute("""
                CREATE TABLE IF NOT EXISTS indexed_files (

                    repository_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                    PRIMARY KEY (repository_id, file_path)
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS repositories (
                    repository_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_location TEXT NOT NULL,
                    branch TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    is_active INTEGER NOT NULL DEFAULT 0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    repository_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    sources_json TEXT NOT NULL DEFAULT '[]',
                    response_time_ms REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_chat_messages_session
                ON chat_messages(repository_id, session_id, id)
            """)

    def register_repository(
        self,
        repository_id,
        display_name,
        source_type,
        source_location,
        branch="",
        metadata=None,
        activate=True,
    ):
        with self.connect() as connection:
            if activate:
                connection.execute("UPDATE repositories SET is_active = 0")
            connection.execute(
                """
                INSERT INTO repositories (
                    repository_id, display_name, source_type,
                    source_location, branch, metadata_json, is_active
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(repository_id) DO UPDATE SET
                    display_name = excluded.display_name,
                    source_type = excluded.source_type,
                    source_location = excluded.source_location,
                    branch = excluded.branch,
                    metadata_json = excluded.metadata_json,
                    is_active = excluded.is_active,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    repository_id,
                    display_name,
                    source_type,
                    source_location,
                    branch,
                    json.dumps(metadata or {}),
                    int(activate),
                ),
            )

    def list_repositories(self):
        with self.connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT repository_id, display_name, source_type,
                       source_location, branch, metadata_json,
                       is_active, updated_at
                FROM repositories
                ORDER BY is_active DESC, display_name COLLATE NOCASE
                """
            ).fetchall()
        return [
            {
                **dict(row),
                "metadata": json.loads(row["metadata_json"]),
                "is_active": bool(row["is_active"]),
            }
            for row in rows
        ]

    def get_repository(self, repository_id):
        return next(
            (
                repository
                for repository in self.list_repositories()
                if repository["repository_id"] == repository_id
            ),
            None,
        )

    def get_active_repository(self):
        return next(
            (
                repository
                for repository in self.list_repositories()
                if repository["is_active"]
            ),
            None,
        )

    def set_active_repository(self, repository_id):
        with self.connect() as connection:
            cursor = connection.execute(
                "UPDATE repositories SET is_active = 0"
            )
            cursor = connection.execute(
                """
                UPDATE repositories
                SET is_active = 1, updated_at = CURRENT_TIMESTAMP
                WHERE repository_id = ?
                """,
                (repository_id,),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Unknown repository: {repository_id}")

    def delete_repository(self, repository_id):
        with self.connect() as connection:
            connection.execute(
                "DELETE FROM indexed_files WHERE repository_id = ?",
                (repository_id,),
            )
            connection.execute(
                "DELETE FROM chat_messages WHERE repository_id = ?",
                (repository_id,),
            )
            connection.execute(
                "DELETE FROM repositories WHERE repository_id = ?",
                (repository_id,),
            )

    def save_chat_message(
        self,
        repository_id,
        session_id,
        role,
        content,
        sources=None,
        response_time_ms=None,
    ):
        if role not in {"user", "assistant"}:
            raise ValueError(f"Unsupported chat role: {role}")
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO chat_messages (
                    repository_id, session_id, role, content,
                    sources_json, response_time_ms
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    repository_id,
                    session_id,
                    role,
                    content,
                    json.dumps(sources or []),
                    response_time_ms,
                ),
            )

    def get_chat_history(self, repository_id, session_id, limit=100):
        with self.connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT role, content, sources_json, response_time_ms, created_at
                FROM chat_messages
                WHERE repository_id = ? AND session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (repository_id, session_id, limit),
            ).fetchall()
        return [
            {
                **dict(row),
                "sources": json.loads(row["sources_json"]),
            }
            for row in reversed(rows)
        ]

    def clear_chat_history(self, repository_id, session_id):
        with self.connect() as connection:
            connection.execute(
                """
                DELETE FROM chat_messages
                WHERE repository_id = ? AND session_id = ?
                """,
                (repository_id, session_id),
            )

    def get_files(self, repository_id):

        with self.connect() as connection:

            rows = connection.execute(
                """
                SELECT file_path, content_hash
                FROM indexed_files
                WHERE repository_id = ?
                """,
                (repository_id,)
            ).fetchall()

        return {
            file_path: content_hash
            for file_path, content_hash in rows
        }

    def upsert_file(
        self,
        repository_id,
        file_path,
        content_hash
    ):

        with self.connect() as connection:

            connection.execute(
                """
                INSERT INTO indexed_files (
                    repository_id,
                    file_path,
                    content_hash
                )
                VALUES (?, ?, ?)

                ON CONFLICT(repository_id, file_path)
                DO UPDATE SET
                    content_hash = excluded.content_hash,
                    indexed_at = CURRENT_TIMESTAMP
                """,
                (
                    repository_id,
                    file_path,
                    content_hash
                )
            )

    def delete_file(self, repository_id, file_path):

        with self.connect() as connection:

            connection.execute(
                """
                DELETE FROM indexed_files
                WHERE repository_id = ?
                AND file_path = ?
                """,
                (repository_id, file_path)
            )