import unittest
from pathlib import Path

from src.gitcontext.architecture import analyze_repository, dependencies_for_file
from src.gitcontext.ingestion import create_repository_file


class RepositoryArchitectureTests(unittest.TestCase):
    def setUp(self):
        self.files = [
            create_repository_file(
                Path("app/main.py"),
                b"from app.services import run\n\nclass App:\n    def start(self):\n        run()\n",
                "",
            ),
            create_repository_file(
                Path("app/services.py"),
                b"def run():\n    return True\n",
                "",
            ),
            create_repository_file(
                Path("README.md"),
                b"# App\n",
                "",
            ),
        ]

    def test_architecture_counts_files_symbols_and_local_dependencies(self):
        result = analyze_repository(self.files)

        self.assertEqual(result["total_files"], 3)
        self.assertEqual(result["python_functions"], 2)
        self.assertEqual(result["python_classes"], 1)
        self.assertEqual(result["dependency_edges"], 1)
        self.assertEqual(result["dependencies"][0]["to"], "app/services.py")

    def test_dependencies_are_reported_in_both_directions(self):
        result = dependencies_for_file(self.files, "app/services.py")

        self.assertEqual(result["imports"], [])
        self.assertEqual(result["imported_by"][0]["from"], "app/main.py")


if __name__ == "__main__":
    unittest.main()
