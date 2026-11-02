import unittest
from unittest.mock import Mock

import numpy as np

from src.gitcontext.vector_store import VectorStore


class VectorStoreTests(unittest.TestCase):
    def test_filtered_search_avoids_fetching_all_repository_ids(self):
        store = VectorStore.__new__(VectorStore)
        store.collection = Mock()
        store.collection.count.return_value = 10
        store.collection.query.return_value = {
            "documents": [["code"]],
            "metadatas": [[{"file_path": "app.py"}]],
            "distances": [[0.1]],
            "ids": [["chunk-id"]],
        }

        result = store.search(
            np.asarray([0.2, 0.3]),
            top_k=3,
            repository_id="repo:one",
        )

        store.collection.get.assert_not_called()
        store.collection.query.assert_called_once_with(
            query_embeddings=[[0.2, 0.3]],
            n_results=3,
            where={"repository_id": "repo:one"},
        )
        self.assertEqual(result["documents"], [["code"]])

    def test_empty_collection_returns_empty_results(self):
        store = VectorStore.__new__(VectorStore)
        store.collection = Mock()
        store.collection.count.return_value = 0

        result = store.search(np.asarray([0.2, 0.3]))

        store.collection.query.assert_not_called()
        self.assertEqual(result["documents"], [[]])


if __name__ == "__main__":
    unittest.main()
