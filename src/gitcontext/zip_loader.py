import hashlib
import stat
import zipfile
from io import BytesIO
from pathlib import PurePosixPath

from src.gitcontext.ingestion import MAX_FILE_SIZE, create_repository_file


MAX_ARCHIVE_SIZE = 25 * 1024 * 1024
MAX_UNCOMPRESSED_SIZE = 100 * 1024 * 1024
MAX_ARCHIVE_FILES = 10_000


def load_repository_zip(archive_bytes, filename="repository.zip"):
    if len(archive_bytes) > MAX_ARCHIVE_SIZE:
        raise ValueError("ZIP upload exceeds the 25 MB size limit.")

    files = []
    total_size = 0

    try:
        archive = zipfile.ZipFile(BytesIO(archive_bytes))
    except zipfile.BadZipFile as error:
        raise ValueError("Uploaded file is not a valid ZIP archive.") from error

    with archive:
        members = archive.infolist()
        if len(members) > MAX_ARCHIVE_FILES:
            raise ValueError("ZIP contains too many files.")

        for member in members:
            path = PurePosixPath(member.filename.replace("\\", "/"))
            if (
                path.is_absolute()
                or ".." in path.parts
                or (path.parts and ":" in path.parts[0])
            ):
                raise ValueError("ZIP contains an unsafe file path.")

            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError("ZIP contains a symbolic link.")

            total_size += member.file_size
            if total_size > MAX_UNCOMPRESSED_SIZE:
                raise ValueError(
                    "ZIP expands beyond the 100 MB uncompressed size limit."
                )

        for member in members:
            if member.is_dir():
                continue

            if member.file_size > MAX_FILE_SIZE:
                continue

            path = PurePosixPath(member.filename.replace("\\", "/"))
            repository_file = create_repository_file(
                relative_path=path,
                raw_content=archive.read(member),
                last_modified="",
            )
            if repository_file:
                files.append(repository_file)

    if not files:
        raise ValueError("ZIP contains no supported repository files.")

    archive_id = hashlib.sha256(archive_bytes).hexdigest()
    repository_id = f"zip:{archive_id}"
    print(
        f"[ZIP] Loaded {len(files)} supported files from {filename}"
    )
    return repository_id, files
