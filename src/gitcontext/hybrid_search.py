class HybridRetriever:

    def __init__(
        self,
        vector_store,
        bm25_retriever
    ):

        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever

    @staticmethod
    def get_chunk_key(metadata):

        return (
            metadata.get("repository_id", ""),
            metadata.get("file_path", ""),
            metadata.get("chunk_type", ""),
            metadata.get("chunk_name", ""),
            metadata.get("start_line", 0),
            metadata.get("end_line", 0)
        )

    def search(
        self,
        query,
        query_embedding,
        top_k=5,
        rrf_k=60,
        repository_id=None,
        vector_weight=1.0,
        bm25_weight=1.0,
        candidate_multiplier=3
    ):

        if top_k <= 0:
            return []

        candidate_k = max(
            top_k,
            top_k * candidate_multiplier
        )

        combined_scores = {}
        combined_chunks = {}

        # ---------------------------------
        # 1. Semantic search
        # ---------------------------------

        vector_results = (
            self.vector_store.search(
                query_embedding,
                top_k=candidate_k,
                max_distance=1.0,
                repository_id=repository_id
            )
            if vector_weight
            else {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        )

        vector_documents = (
            vector_results.get("documents", [[]])[0]
        )

        vector_metadatas = (
            vector_results.get("metadatas", [[]])[0]
        )

        vector_distances = (
            vector_results.get("distances", [[]])[0]
        )

        for rank, metadata in enumerate(
            vector_metadatas,
            start=1
        ):

            key = self.get_chunk_key(metadata)

            score = vector_weight / (rrf_k + rank)

            combined_scores[key] = (
                combined_scores.get(key, 0.0) + score
            )

            combined_chunks[key] = {
                "metadata": metadata,
                "content": vector_documents[rank - 1],
                "vector_distance": (
                    float(vector_distances[rank - 1])
                    if rank - 1 < len(vector_distances)
                    else None
                )
            }

        # ---------------------------------
        # 2. Keyword search
        # ---------------------------------

        bm25_results = (
            self.bm25_retriever.search(
                query,
                top_k=candidate_k,
                repository_id=repository_id
            )
            if bm25_weight
            else []
        )

        for rank, result in enumerate(
            bm25_results,
            start=1
        ):

            chunk = result["chunk"]

            metadata = {
                "repository_id": getattr(
                    chunk, "repository_id", ""
                ),
                "file_path": chunk.file_path,
                "chunk_type": chunk.chunk_type,
                "chunk_name": chunk.chunk_name,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "language": chunk.language
            }
            metadata.update({
                key: value
                for key, value in chunk.metadata.items()
                if isinstance(value, (str, int, float, bool))
            })

            key = self.get_chunk_key(metadata)

            score = bm25_weight / (rrf_k + rank)

            combined_scores[key] = (
                combined_scores.get(key, 0.0) + score
            )

            if key not in combined_chunks:

                combined_chunks[key] = {
                    "metadata": metadata,
                    "content": chunk.content,
                    "vector_distance": None
                }

        # ---------------------------------
        # 3. Rank using RRF
        # ---------------------------------

        ranked_results = sorted(
            combined_scores.items(),
            key=lambda item: item[1],
            reverse=True
        )

        final_results = []

        for key, score in ranked_results[:top_k]:

            result = combined_chunks[key]

            final_results.append({
                "metadata": result["metadata"],
                "content": result["content"],
                "rrf_score": float(score),
                "vector_distance": result["vector_distance"]
            })

        return final_results

    # =====================================
    # Multi-query retrieval
    # =====================================

    def search_multiple(
        self,
        queries,
        query_embeddings,
        top_k=10,
        rrf_k=60,
        repository_id=None,
        vector_weight=1.0,
        bm25_weight=1.0
    ):

        if len(queries) != len(query_embeddings):

            raise ValueError(
                "Number of queries and embeddings must match."
            )

        combined_results = {}

        for query, query_embedding in zip(
            queries,
            query_embeddings
        ):

            results = self.search(
                query=query,
                query_embedding=query_embedding,
                top_k=top_k,
                rrf_k=rrf_k,
                repository_id=repository_id,
                vector_weight=vector_weight,
                bm25_weight=bm25_weight
            )

            for rank, result in enumerate(
                results,
                start=1
            ):

                metadata = result["metadata"]

                chunk_id = self.get_chunk_key(metadata)

                if chunk_id not in combined_results:

                    combined_results[chunk_id] = {
                        "metadata": metadata,
                        "content": result["content"],
                        "rrf_score": 0.0,
                        "vector_distance": result.get(
                            "vector_distance"
                        )
                    }

                # Fuse rankings from multiple queries.
                combined_results[chunk_id]["rrf_score"] += (
                    1 / (rrf_k + rank)
                )

        ranked_results = sorted(
            combined_results.values(),
            key=lambda item: item["rrf_score"],
            reverse=True
        )

        return ranked_results[:top_k]