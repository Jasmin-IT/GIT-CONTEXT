from sentence_transformers import SentenceTransformer


class EmbeddingModel:

    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=32
    ):

        self.model = SentenceTransformer(model_name)

        self.batch_size = batch_size

        self.dimension = self.model.get_sentence_embedding_dimension()

    # =====================================
    # Document embeddings
    # =====================================

    def embed_documents(self, documents):

        if not documents:
            return []

        embeddings = self.model.encode(
            documents,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        return embeddings

    # =====================================
    # Query embedding
    # =====================================

    def embed_query(self, query):

        if not query or not query.strip():

            raise ValueError(
                "Query cannot be empty."
            )

        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        return embedding