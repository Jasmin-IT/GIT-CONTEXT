import unittest
from unittest.mock import Mock

from src.gitcontext.hybrid_search import HybridRetriever


class HybridSearchTests(unittest.TestCase):
    def test_keyword_only_does_not_query_vector_store(self):
        vector_store = Mock()
        bm25 = Mock()
        bm25.search.return_value = []
        retriever = HybridRetriever(vector_store, bm25)

        retriever.search(
            query="find function",
            query_embedding=[0.1],
            repository_id="repo",
            vector_weight=0,
            bm25_weight=1,
        )

        vector_store.search.assert_not_called()
        bm25.search.assert_called_once()

    def test_vector_only_does_not_query_bm25(self):
        vector_store = Mock()
        vector_store.search.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        bm25 = Mock()
        retriever = HybridRetriever(vector_store, bm25)

        retriever.search(
            query="find function",
            query_embedding=[0.1],
            repository_id="repo",
            vector_weight=1,
            bm25_weight=0,
        )

        vector_store.search.assert_called_once()
        bm25.search.assert_not_called()


if __name__ == "__main__":
    unittest.main()
