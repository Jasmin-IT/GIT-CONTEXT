import unittest
from pathlib import Path

from src.gitcontext.ingestion import create_repository_file, should_ignore


class RepositoryIngestionFilterTests(unittest.TestCase):
    def test_ignores_dataset_and_generated_directories(self):
        paths = [
            "dataset_3class/labels/train/item.txt",
            "datasets/annotations/labels.json",
            "mobile/.dart_tool/package_config.json",
            "mobile/.flutter_tmp/generated.dart",
            "app/.pytest_cache/lastfailed",
            "frontend/.next/server/manifest.json",
            "app/outputs/predictions.json",
        ]

        for path in paths:
            with self.subTest(path=path):
                self.assertTrue(should_ignore(Path(path)))

    def test_keeps_application_source_outside_ignored_directories(self):
        self.assertFalse(should_ignore(Path("app/analyzer.py")))
        self.assertIsNotNone(
            create_repository_file("app/analyzer.py", b"def analyze(): pass", "")
        )

    def test_zip_ingestion_uses_same_directory_filters(self):
        self.assertIsNone(
            create_repository_file(
                "dataset_3class/labels/train/item.txt",
                b"0 0.1 0.2",
                "",
            )
        )


if __name__ == "__main__":
    unittest.main()
