import unittest
from unittest.mock import Mock

from src.gitcontext.github_loader import GitHubLoader


class GitHubLoaderTests(unittest.TestCase):
    def test_loads_selected_branch_and_preserves_source_metadata(self):
        loader = GitHubLoader()
        loader.session = Mock()

        def response(payload=None, content=None):
            result = Mock()
            result.json.return_value = payload
            result.content = content
            return result

        def get(url, **kwargs):
            self.assertEqual(kwargs["timeout"], 30)
            if url.endswith("/repos/example/project"):
                return response({
                    "private": False,
                    "default_branch": "main",
                })
            if "/branches/" in url:
                return response({
                    "commit": {
                        "sha": "commit-sha",
                        "commit": {"tree": {"sha": "tree-sha"}},
                    }
                })
            if "/git/trees/" in url:
                return response({
                    "truncated": False,
                    "tree": [{
                        "type": "blob",
                        "path": "src/app.py",
                        "size": 20,
                    }],
                })
            if url.startswith("https://raw.githubusercontent.com/"):
                return response(content=b"def run():\n    return 1\n")
            self.fail(f"Unexpected GitHub request: {url}")

        loader.session.get.side_effect = get

        repository_id, files = loader.load_repository(
            "https://github.com/example/project",
            branch="feature/cleanup",
            include_context=False,
        )

        self.assertEqual(
            repository_id,
            "github:example/project@feature/cleanup",
        )
        self.assertEqual(loader.last_branch, "feature/cleanup")
        self.assertEqual(files[0].file_path, "src/app.py")
        self.assertEqual(files[0].metadata["branch"], "feature/cleanup")
        self.assertEqual(files[0].metadata["commit_sha"], "commit-sha")

    def test_rejects_private_repositories(self):
        loader = GitHubLoader()
        loader.session = Mock()
        result = Mock()
        result.json.return_value = {"private": True}
        loader.session.get.return_value = result

        with self.assertRaisesRegex(ValueError, "Only public"):
            loader.list_branches("https://github.com/example/private")


if __name__ == "__main__":
    unittest.main()
