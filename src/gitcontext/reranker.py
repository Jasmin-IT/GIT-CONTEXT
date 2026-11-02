from sentence_transformers import CrossEncoder


class CodeReranker:

    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size=16
    ):

        self.model = CrossEncoder(model_name)

        self.batch_size = batch_size

    def rerank(
        self,
        query,
        retrieved_chunks,
        top_k=5
    ):

        if not retrieved_chunks or top_k <= 0:
            return []

        pairs = []

        for result in retrieved_chunks:

            metadata = result.get("metadata", {})

            # Include code context in reranking.
            context = (
                f"File: {metadata.get('file_path', '')}\n"
                f"Type: {metadata.get('chunk_type', '')}\n"
                f"Name: {metadata.get('chunk_name', '')}\n"
                f"Code:\n{result.get('content', '')}"
            )

            pairs.append([
                query,
                context
            ])

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False
        )

        reranked_results = []

        for index, result in enumerate(retrieved_chunks):

            reranked_results.append({

                **result,

                "rerank_score": float(scores[index])

            })

        reranked_results.sort(
            key=lambda item: item["rerank_score"],
            reverse=True
        )

        return reranked_results[:top_k]