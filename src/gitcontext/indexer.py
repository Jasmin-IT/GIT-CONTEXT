class IncrementalIndexer:

    def __init__(
        self,
        database,
        vector_store,
        embedding_model,
        chunker
    ):

        self.database = database
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.chunker = chunker

    def index_repository(
        self,
        repository_id,
        repository_files
    ):

        print("\n========== INCREMENTAL INDEXING ==========")

        previous_files = self.database.get_files(
            repository_id
        )

        current_files = {
            file.file_path: file
            for file in repository_files
        }

        previous_paths = set(previous_files)
        current_paths = set(current_files)

        # Detect new files.
        new_paths = current_paths - previous_paths

        # Detect modified files.
        modified_paths = {
            path
            for path in current_paths & previous_paths
            if current_files[path].content_hash
            != previous_files[path]
        }

        # Detect deleted files.
        deleted_paths = previous_paths - current_paths

        print(f"[NEW] {len(new_paths)}")
        print(f"[MODIFIED] {len(modified_paths)}")
        print(f"[DELETED] {len(deleted_paths)}")

        # Process new and modified files.
        changed_paths = new_paths | modified_paths

        successful_paths = set()

        for path in sorted(changed_paths):

            file = current_files[path]

            try:

                chunks = self.chunker(
                    file,
                    repository_id=repository_id
                    )

                if chunks:

                    embeddings = (
                        self.embedding_model.embed_documents(
                            [chunk.content for chunk in chunks]
                        )
                    )

                    # Replace old chunks only after
                    # successful chunking and embedding.
                    self.vector_store.delete_file(
                        repository_id,
                        path
                    )

                    self.vector_store.add_documents(
                        chunks=chunks,
                        embeddings=embeddings,
                        repository_id=repository_id
                    )

                else:

                    # A valid file can produce zero chunks.
                    # Remove any previously indexed chunks.
                    self.vector_store.delete_file(
                        repository_id,
                        path
                    )

                successful_paths.add(path)

                print(f"[INDEXED] {path}")

            except Exception as error:

                print(
                    f"[ERROR] Could not index {path}: {error}"
                )

        # Only update SQLite after successful processing.
        for path in successful_paths:

            self.database.upsert_file(
                repository_id=repository_id,
                file_path=path,
                content_hash=current_files[path].content_hash
            )

        # Remove deleted files from ChromaDB and SQLite.
        for path in deleted_paths:

            try:

                self.vector_store.delete_file(
                    repository_id,
                    path
                )

                self.database.delete_file(
                    repository_id,
                    path
                )

                print(f"[DELETED] {path}")

            except Exception as error:

                print(
                    f"[ERROR] Could not remove {path}: {error}"
                )

        print("\n========== INDEXING SUMMARY ==========")

        print(f"New files: {len(new_paths)}")
        print(f"Modified files: {len(modified_paths)}")
        print(f"Deleted files: {len(deleted_paths)}")
        print(f"Successfully indexed: {len(successful_paths)}")

        return {
            "new": len(new_paths),
            "modified": len(modified_paths),
            "deleted": len(deleted_paths),
            "successful": len(successful_paths)
        }