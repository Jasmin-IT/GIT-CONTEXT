import unittest
from unittest.mock import patch

from evaluate import evaluate_mode, unknown_gold_paths


class EvaluationMetricTests(unittest.TestCase):
    def test_recall_counts_each_gold_target(self):
        retrieved = [
            {
                "metadata": {
                    "file_path": "src/first.py",
                    "start_line": 1,
                    "end_line": 5,
                }
            },
            {
                "metadata": {
                    "file_path": "src/unrelated.py",
                    "start_line": 1,
                    "end_line": 2,
                }
            },
        ]
        dataset = [{
            "question": "Find both modules",
            "expected": [
                {"file_path": "src/first.py"},
                {"file_path": "src/second.py"},
            ],
        }]

        with patch("evaluate.retrieve", return_value=(retrieved, 0.01)):
            result = evaluate_mode({}, dataset, "hybrid", rewrite=False)

        self.assertEqual(result["recall@1"], 0.5)
        self.assertEqual(result["recall@3"], 0.5)
        self.assertEqual(result["hit_rate@1"], 1.0)
        self.assertEqual(result["precision@1"], 1.0)
        self.assertEqual(result["question_results"][0]["top_results"][0]["relevant"], True)

    def test_gold_labels_are_checked_against_selected_repository(self):
        dataset = [{
            "question": "Find a module",
            "expected": [{"file_path": "src/missing.py"}],
        }]

        self.assertEqual(unknown_gold_paths(dataset, []), ["src/missing.py"])

    def test_gold_paths_match_repository_relative_paths(self):
        dataset = [{
            "question": "Find a module",
            "expected": [{"file_path": "src/module.py"}],
        }]
        repository_files = [
            type("RepositoryFile", (), {"file_path": "src/module.py"})()
        ]

        self.assertEqual(unknown_gold_paths(dataset, repository_files), [])


if __name__ == "__main__":
    unittest.main()
