import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables before initializing GitHubLoader.
load_dotenv()

from src.gitcontext.chunking import chunk_repository_file
from src.gitcontext.db import IndexDatabase
from src.gitcontext.embeddings import EmbeddingModel
from src.gitcontext.github_loader import GitHubLoader
from src.gitcontext.hybrid_search import HybridRetriever
from src.gitcontext.indexer import IncrementalIndexer
from src.gitcontext.ingestion import read_repository
from src.gitcontext.llm import GitContextLLM
from src.gitcontext.reranker import CodeReranker
from src.gitcontext.retrieval import BM25Retriever
from src.gitcontext.vector_store import VectorStore

# ============================================
# CONFIGURATION
# ============================================

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_LOCAL_REPOSITORY = PROJECT_ROOT / "data" / "sample_repo"

TOP_K_RETRIEVAL = 10
TOP_K_RERANK = 5
RRF_K = 60
MAX_CHAT_HISTORY = 10

# ============================================
# REPOSITORY SELECTION
# ============================================


def select_repository():
    print("\n" + "=" * 55)
    print("          SELECT REPOSITORY")
    print("=" * 55)

    print("\n1. Local repository")
    print("2. GitHub repository")

    while True:
        choice = input("\nEnter choice (1/2): ").strip()

        if choice == "1":
            default_path = str(DEFAULT_LOCAL_REPOSITORY)

            path_input = input(
                f"Enter repository path [{default_path}]: "
            ).strip()

            repository_path = (
                Path(path_input if path_input else default_path)
                .expanduser()
                .resolve()
            )

            if not repository_path.is_dir():
                print("[ERROR] Repository directory not found.")
                continue

            files = read_repository(str(repository_path))

            repository_id = f"local:{repository_path}"

            if not files:
                raise ValueError("No supported files found.")

            print(f"\n[OK] Loaded {len(files)} local files.")

            return repository_id, files

        elif choice == "2":
            repository_url = input("\nEnter GitHub repository URL: ").strip()

            try:
                github_loader = GitHubLoader()

                repository_id, files = github_loader.load_repository(
                    repository_url
                )

                if not files:
                    raise ValueError(
                        "No supported files found in repository."
                    )

                print(f"\n[OK] Loaded {len(files)} GitHub files.")

                return repository_id, files

            except Exception as error:
                print(f"\n[ERROR] GitHub loading failed: {error}")
                print("Please try again.")

        else:
            print("Invalid choice. Enter 1 or 2.")


# ============================================
# INITIALIZE COMPONENTS
# ============================================


def initialize_components():
    print("\n" + "=" * 55)
    print("          GITCONTEXT INITIALIZATION")
    print("=" * 55)

    # 1. Select repository
    print("\n[1/7] Selecting repository...")

    repository_id, repository_files = select_repository()

    print(f"[OK] Repository ID: {repository_id}")

    # 2. Embedding model
    print("\n[2/7] Loading embedding model...")

    embedding_model = EmbeddingModel()

    print(f"[OK] Embedding dimension: {embedding_model.dimension}")

    # 3. Vector database
    print("\n[3/7] Connecting to ChromaDB...")

    vector_store = VectorStore()

    print("[OK] ChromaDB connected.")

    # 4. SQLite database
    print("\n[4/7] Connecting to SQLite...")

    database = IndexDatabase()

    print("[OK] SQLite connected.")

    # 5. Incremental indexing
    print("\n[5/7] Indexing repository...")

    indexer = IncrementalIndexer(
        database=database,
        vector_store=vector_store,
        embedding_model=embedding_model,
        chunker=chunk_repository_file,
    )

    indexing_stats = indexer.index_repository(
        repository_id=repository_id, repository_files=repository_files
    )

    print("\nINDEXING SUMMARY")

    print(f"New files        : {indexing_stats['new']}")
    print(f"Modified files   : {indexing_stats['modified']}")
    print(f"Deleted files    : {indexing_stats['deleted']}")
    print(f"Successful files : {indexing_stats['successful']}")

    # 6. Build BM25 and hybrid retriever
    print("\n[6/7] Building retrieval system...")

    chunks = []

    for file in repository_files:
        file_chunks = chunk_repository_file(file, repository_id=repository_id)

        chunks.extend(file_chunks)

    if not chunks:
        raise ValueError("No chunks generated from repository.")

    bm25_retriever = BM25Retriever(chunks)

    hybrid_retriever = HybridRetriever(
        vector_store=vector_store, bm25_retriever=bm25_retriever
    )

    print(f"[OK] Generated {len(chunks)} chunks.")
    print("[OK] BM25 and hybrid retrieval initialized.")

    # 7. Reranker and LLM
    print("\n[7/7] Loading AI components...")

    reranker = CodeReranker()

    llm = GitContextLLM()

    print("[OK] Reranker loaded.")
    print("[OK] LLM initialized.")

    print("\n" + "=" * 55)
    print("          GITCONTEXT READY")
    print("=" * 55)

    return {
        "repository_id": repository_id,
        "embedding_model": embedding_model,
        "vector_store": vector_store,
        "bm25_retriever": bm25_retriever,
        "hybrid_retriever": hybrid_retriever,
        "reranker": reranker,
        "llm": llm,
        "chunks": chunks,
    }


# ============================================
# QUESTION ANSWERING
# ============================================


def answer_question(query, components, chat_history):
    embedding_model = components["embedding_model"]
    hybrid_retriever = components["hybrid_retriever"]
    reranker = components["reranker"]
    llm = components["llm"]

    repository_id = components["repository_id"]

    if llm.query_rewriting_enabled():
        print("\n[1/5] Generating search queries...")
        search_queries = llm.generate_search_queries(query, chat_history)
    else:
        search_queries = [query]

    if not search_queries:
        search_queries = [query]

    print("\n[2/5] Generating query embeddings...")

    query_embeddings = [
        embedding_model.embed_query(search_query)
        for search_query in search_queries
    ]

    print("\n[3/5] Performing hybrid retrieval...")

    retrieved_chunks = hybrid_retriever.search_multiple(
        queries=search_queries,
        query_embeddings=query_embeddings,
        top_k=TOP_K_RETRIEVAL,
        rrf_k=RRF_K,
        repository_id=repository_id,
    )

    print(f"[OK] Retrieved {len(retrieved_chunks)} candidates.")

    if not retrieved_chunks:
        return (
            "I couldn't find sufficient evidence "
            "in the repository to answer this question."
        )

    print("\n[4/5] Reranking candidates...")

    reranked_chunks = reranker.rerank(
        query=query, retrieved_chunks=retrieved_chunks, top_k=TOP_K_RERANK
    )

    if not reranked_chunks:
        return (
            "I couldn't find sufficient evidence "
            "in the repository to answer this question."
        )

    print("\nSELECTED EVIDENCE")

    for index, result in enumerate(reranked_chunks, start=1):
        metadata = result["metadata"]

        print(f"\n{index}. {metadata['file_path']}")

        print(
            f"   {metadata['chunk_name']} "
            f"(Lines {metadata['start_line']}-"
            f"{metadata['end_line']})"
        )

        print(f"   Rerank score: " f"{result['rerank_score']:.4f}")

    print("\n[5/5] Generating answer...")

    answer = llm.generate_answer(
        query=query, retrieved_chunks=reranked_chunks, chat_history=chat_history
    )

    return answer


# ============================================
# CHAT INTERFACE
# ============================================


def start_chat(components):
    chat_history = []

    print("\n" + "=" * 55)
    print("              GITCONTEXT CHAT")
    print("=" * 55)

    print(f"\nActive repository: {components['repository_id']}")

    print("\nCommands:")
    print("  exit  - Exit GitContext")
    print("  clear - Clear conversation history")

    while True:
        try:
            query = input("\nYou: ").strip()

            if not query:
                continue

            if query.lower() in ("exit", "quit"):
                print("\nGitContext shutting down.")
                break

            if query.lower() == "clear":
                chat_history.clear()

                print("\nConversation history cleared.")
                continue

            answer = answer_question(
                query=query, components=components, chat_history=chat_history
            )

            print("\n" + "=" * 55)
            print("GITCONTEXT ANSWER")
            print("=" * 55)

            print("\n" + answer)

            print("\n" + "=" * 55)

            chat_history.append({"question": query, "answer": answer})

            if len(chat_history) > MAX_CHAT_HISTORY:
                chat_history = chat_history[-MAX_CHAT_HISTORY:]

        except KeyboardInterrupt:
            print("\nGitContext interrupted.")
            break

        except Exception as error:
            print(f"\n[ERROR] {error}")


# ============================================
# MAIN
# ============================================


def main():
    try:
        components = initialize_components()

        start_chat(components)

    except Exception as error:
        print("\n[FATAL ERROR]")
        print(error)

        raise


if __name__ == "__main__":
    main()