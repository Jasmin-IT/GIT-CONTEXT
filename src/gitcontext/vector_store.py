import hashlib
from pathlib import Path
import chromadb


class VectorStore:

    def __init__(self, collection_name="gitcontext"):

        self.client = chromadb.PersistentClient(
            path=str(Path(__file__).resolve().parents[2] / "data" / "chroma_db")
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(
        self,
        chunks,
        embeddings,
        repository_id
    ):

        if not chunks:
            return

        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:

            identity = (
                f"{repository_id}:"
                f"{chunk.file_path}:"
                f"{chunk.chunk_type}:"
                f"{chunk.chunk_name}:"
                f"{chunk.start_line}:"
                f"{chunk.end_line}"
            )

            chunk_id = hashlib.sha256(
                identity.encode("utf-8")
            ).hexdigest()

            ids.append(chunk_id)

            documents.append(chunk.content)

            metadata = {
                "repository_id": repository_id,
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
            metadatas.append(metadata)

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )

    def delete_file(self, repository_id, file_path):

        self.collection.delete(
            where={
                "$and": [
                    {"repository_id": repository_id},
                    {"file_path": file_path}
                ]
            }
        )

    def delete_repository(self, repository_id):

        self.collection.delete(
            where={
                "repository_id": repository_id
            }
        )

    def search(
        self,
        query_embedding,
        top_k=5,
        max_distance=0.50,
        repository_id=None
    ):

        where = None

        if repository_id:
            where = {
                "repository_id": repository_id
            }

        if self.collection.count() == 0:
            return {
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
                "ids": [[]]
            }

        results = self.collection.query(
            query_embeddings=[
                query_embedding.tolist()
            ],
            n_results=top_k,
            where=where
        )

        filtered_results = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
            "ids": [[]]
        }

        for index, distance in enumerate(
            results["distances"][0]
        ):

            if distance <= max_distance:

                for key in filtered_results:

                    filtered_results[key][0].append(
                        results[key][0][index]
                    )

        return filtered_results