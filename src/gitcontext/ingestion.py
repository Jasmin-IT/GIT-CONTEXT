from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib


SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".ts", ".java",
    ".cpp", ".c", ".html", ".css",
    ".json", ".md", ".txt", ".yaml",
    ".yml", ".toml", ".sql"
}


IGNORED_DIRECTORIES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".idea",
    ".vscode",
    ".dart_tool",
    ".flutter_tmp",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".next",
    ".cache",
    "dataset",
    "datasets",
    "dataset_3class",
    "labels",
    "runs",
    "outputs",
    "coverage",
}


MAX_FILE_SIZE = 1 * 1024 * 1024


LANGUAGE_MAP = {
    ".py": "Python",
    ".js": "JavaScript",
    ".ts": "TypeScript",
    ".java": "Java",
    ".cpp": "C++",
    ".c": "C",
    ".html": "HTML",
    ".css": "CSS",
    ".json": "JSON",
    ".md": "Markdown",
    ".txt": "Text",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".toml": "TOML",
    ".sql": "SQL"
}


@dataclass
class RepositoryFile:
    file_path: str
    file_name: str
    extension: str
    language: str
    content: str
    size: int
    last_modified: str
    content_hash: str
    metadata: dict = field(default_factory=dict)


def should_ignore(file_path: Path):

    if any(
        part.lower() in IGNORED_DIRECTORIES
        for part in file_path.parts
    ):
        return True

    name = file_path.name.lower()

    if name.startswith(".env"):
        return True

    if name.startswith(("id_rsa", "id_ed25519")):
        return True

    if file_path.suffix.lower() in {
        ".pem", ".key", ".p12", ".pfx"
    }:
        return True

    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return True

    return False


def calculate_hash(content: bytes):

    return hashlib.sha256(content).hexdigest()


def create_repository_file(
    relative_path,
    raw_content,
    last_modified,
    metadata=None,
):

    path = Path(relative_path)

    if should_ignore(path):
        return None

    if len(raw_content) > MAX_FILE_SIZE:
        return None

    if b"\x00" in raw_content:
        return None

    content = raw_content.decode(
        "utf-8",
        errors="replace"
    )

    return RepositoryFile(
        file_path=path.as_posix(),
        file_name=path.name,
        extension=path.suffix.lower(),
        language=LANGUAGE_MAP.get(
            path.suffix.lower(),
            "Unknown"
        ),
        content=content,
        size=len(raw_content),
        last_modified=last_modified,
        content_hash=calculate_hash(raw_content),
        metadata=metadata or {},
    )


def read_repository(repo_path: str):

    root = Path(repo_path).resolve()

    if not root.exists():
        raise FileNotFoundError(
            f"Repository does not exist: {root}"
        )

    if not root.is_dir():
        raise NotADirectoryError(
            f"Expected a directory: {root}"
        )

    repository_files = []

    for file_path in root.rglob("*"):

        if not file_path.is_file():
            continue

        if should_ignore(file_path.relative_to(root)):
            continue

        try:

            resolved_path = file_path.resolve()

            if not resolved_path.is_relative_to(root):
                continue

            stat = resolved_path.stat()

            if stat.st_size > MAX_FILE_SIZE:
                continue

            raw_content = resolved_path.read_bytes()

            repository_file = create_repository_file(
                relative_path=resolved_path.relative_to(root),
                raw_content=raw_content,
                last_modified=datetime.fromtimestamp(
                    stat.st_mtime,
                    tz=timezone.utc
                ).isoformat()
            )

            if repository_file:
                repository_files.append(repository_file)

        except OSError as error:

            print(f"Could not read {file_path}: {error}")

    return repository_files