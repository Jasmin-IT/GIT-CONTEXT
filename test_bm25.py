from src.gitcontext.ingestion import read_repository
from src.gitcontext.chunking import chunk_repository_file
from src.gitcontext.retrieval import BM25Retriever


REPOSITORY_PATH = "data/sample_repo"


# 1. Read repository
files = read_repository(REPOSITORY_PATH)


# 2. Create chunks
all_chunks = []

for file in files:
    chunks = chunk_repository_file(file)
    all_chunks.extend(chunks)


# 3. Build BM25 retriever
retriever = BM25Retriever(all_chunks)


# 4. Search
query = "verify_password"

results = retriever.search(query, top_k=3)


# 5. Display results
print("\n========== BM25 RESULTS ==========")

for index, result in enumerate(results):

    chunk = result["chunk"]

    print(f"\nResult {index + 1}")
    print("File:", chunk.file_path)
    print("Function:", chunk.chunk_name)

    print(
        f"Lines: {chunk.start_line}-{chunk.end_line}"
    )

    print("BM25 Score:", result["score"])