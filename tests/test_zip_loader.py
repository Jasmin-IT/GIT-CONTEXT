import io
import unittest
import zipfile

from src.gitcontext.zip_loader import load_repository_zip


def make_zip(entries):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for path, content in entries:
            archive.writestr(path, content)
    return buffer.getvalue()


class ZipLoaderTests(unittest.TestCase):
    def test_loads_supported_files_and_assigns_content_id(self):
        archive_bytes = make_zip([
            ("project/src/app.py", "def answer():\n    return 42\n"),
            ("project/image.png", b"\x89PNG"),
        ])

        repository_id, files = load_repository_zip(archive_bytes)

        self.assertTrue(repository_id.startswith("zip:"))
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].file_path, "project/src/app.py")

    def test_rejects_path_traversal(self):
        archive_bytes = make_zip([("../outside.py", "unsafe")])

        with self.assertRaisesRegex(ValueError, "unsafe file path"):
            load_repository_zip(archive_bytes)

    def test_rejects_non_zip_input(self):
        with self.assertRaisesRegex(ValueError, "not a valid ZIP"):
            load_repository_zip(b"not a zip")

    def test_rejects_archives_without_supported_files(self):
        archive_bytes = make_zip([("project/image.png", b"\x89PNG")])

        with self.assertRaisesRegex(ValueError, "no supported"):
            load_repository_zip(archive_bytes)


if __name__ == "__main__":
    unittest.main()
