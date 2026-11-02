import re
from collections import defaultdict

from rank_bm25 import BM25Okapi


def tokenize(text):
    """
    Tokenizer optimized for source code.

    Handles:
    - snake_case
    - camelCase
    - PascalCase
    - file paths
    - normal English words
    """

    text = str(text).lower()

    # Split camelCase and PascalCase.
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

    # Extract identifiers and words.
    parts = re.findall(
        r"[a-z][a-z0-9_]*",
        text
    )

    tokens = []

    for part in parts:

        tokens.append(part)

        # Split snake_case.
        tokens.extend(
            token for token in part.split("_") if token
        )

    return tokens


class BM25Retriever:

    def __init__(self, chunks):

        self.chunks = chunks

        self.tokenized_documents = []

        # Separate BM25 indexes by repository.
        self.repository_indexes = {}

        self.repository_documents = defaultdict(list)

        for chunk in chunks:

            text = (
                f"{chunk.chunk_name} "
                f"{chunk.file_path} "
                f"{chunk.chunk_type} "
                f"{chunk.content}"
            )

            tokens = tokenize(text)

            self.tokenized_documents.append(tokens)

            repository_id = getattr(
                chunk,
                "repository_id",
                ""
            )

            self.repository_documents[repository_id].append(
                (chunk, tokens)
            )

        # Build one BM25 index per repository.
        for repository_id, documents in (
            self.repository_documents.items()
        ):

            tokenized = [
                tokens for _, tokens in documents
            ]

            if tokenized and any(tokenized):

                self.repository_indexes[repository_id] = (
                    BM25Okapi(tokenized)
                )

        # Global index for searches without repository filtering.
        self.bm25 = (
            BM25Okapi(self.tokenized_documents)
            if self.tokenized_documents
            and any(self.tokenized_documents)
            else None
        )

    def search(
        self,
        query,
        top_k=5,
        repository_id=None
    ):

        if top_k <= 0:
            return []

        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        # Select repository-specific index.
        if repository_id is not None:

            bm25 = self.repository_indexes.get(repository_id)

            documents = self.repository_documents.get(
                repository_id,
                []
            )

        else:

            bm25 = self.bm25

            documents = list(
                zip(self.chunks, self.tokenized_documents)
            )

        if bm25 is None or not documents:
            return []

        scores = bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True
        )

        results = []

        for index in ranked_indices:

            score = float(scores[index])

            if score <= 0:
                continue

            chunk = documents[index][0]

            results.append({
                "chunk": chunk,
                "score": score
            })

            if len(results) >= top_k:
                break

        return results