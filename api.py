import json
import os
import time
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
import traceback

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

load_dotenv()

from src.gitcontext.chunking import chunk_repository_file
from src.gitcontext.architecture import analyze_repository, dependencies_for_file
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
from src.gitcontext.zip_loader import MAX_ARCHIVE_SIZE, load_repository_zip

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_LOCAL_REPOSITORY = PROJECT_ROOT / "data" / "sample_repo"

components = {}
chat_sessions = {}
repository_components = {}
startup_error = None


def get_database():
    database = IndexDatabase()
    database.initialize()
    return database


class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"
    rewrite_queries: bool | None = None


class RepositoryRequest(BaseModel):
    repository_url: str
    branch: str | None = None


class RepositorySelectRequest(BaseModel):
    repository_id: str


class ChatResponse(BaseModel):
    answer: str
    sources: list = Field(default_factory=list)
    response_time_ms: float = 0.0


def load_chat_exchanges(database, repository_id, session_id):
    stored_messages = database.get_chat_history(
        repository_id, session_id, limit=20
    )
    exchanges = []
    pending_question = None
    for message in stored_messages:
        if message["role"] == "user":
            pending_question = message["content"]
        elif message["role"] == "assistant" and pending_question is not None:
            exchanges.append({
                "question": pending_question,
                "answer": message["content"],
            })
            pending_question = None
    return exchanges


def initialize_backend(
    repository_url=None,
    repository_path=None,
    branch=None,
    reuse_components=None,
    uploaded_repository_files=None,
    uploaded_repository_id=None,
    uploaded_archive_path=None,
):
    print("Initializing GitContext backend...")

    if repository_url is None and repository_path is None:
        repository_url = os.getenv("GITHUB_REPOSITORY_URL")

    # Load repository
    if uploaded_repository_files is not None:
        repository_files = uploaded_repository_files
        repository_id = uploaded_repository_id
        if not repository_id:
            raise ValueError("An uploaded repository ID is required.")
        source_type = "zip"
        source_location = str(uploaded_archive_path or "")
        display_name = Path(source_location).stem or repository_id
    elif repository_url:
        loader = GitHubLoader()

        repository_id, repository_files = loader.load_repository(
            repository_url,
            branch=branch,
        )
        source_type = "github"
        source_location = repository_url
        display_name = repository_id
        branch = loader.last_branch

    else:
        repository_path = Path(
            repository_path
            or os.getenv("REPOSITORY_PATH", str(DEFAULT_LOCAL_REPOSITORY))
        ).expanduser()
        if not repository_path.is_absolute():
            repository_path = (PROJECT_ROOT / repository_path).resolve()
        else:
            repository_path = repository_path.resolve()

        if not repository_path.exists():
            raise FileNotFoundError(
                f"Repository path does not exist: {repository_path}"
            )

        repository_files = read_repository(str(repository_path))

        repository_id = f"local:{repository_path}"
        source_type = "local"
        source_location = str(repository_path)
        display_name = repository_path.name or str(repository_path)
        branch = ""

    if not repository_files:
        raise ValueError("No supported repository files found.")

    print(f"Repository: {repository_id}")
    print(f"Files discovered: {len(repository_files)}")

    # Reuse expensive models and stores when switching repositories.
    embedding_model = (
        reuse_components["embedding_model"]
        if reuse_components
        else EmbeddingModel()
    )
    vector_store = (
        reuse_components["vector_store"]
        if reuse_components
        else VectorStore()
    )

    # Initialize database
    database = get_database()

    # Incremental indexing
    indexer = IncrementalIndexer(
        database=database,
        vector_store=vector_store,
        embedding_model=embedding_model,
        chunker=chunk_repository_file,
    )

    indexing_result = indexer.index_repository(repository_id, repository_files)
    database.register_repository(
        repository_id=repository_id,
        display_name=display_name,
        source_type=source_type,
        source_location=source_location,
        branch=branch or "",
        metadata={"total_files": len(repository_files)},
    )

    print("Indexing completed:", indexing_result)

    # Build BM25 index from current repository snapshot
    chunks = []

    for file in repository_files:
        file_chunks = chunk_repository_file(file, repository_id=repository_id)
        chunks.extend(file_chunks)

    bm25_retriever = BM25Retriever(chunks)

    # Hybrid retrieval
    hybrid_retriever = HybridRetriever(
        vector_store=vector_store, bm25_retriever=bm25_retriever
    )

    # Reranker
    reranker = (
        reuse_components["reranker"]
        if reuse_components
        else CodeReranker()
    )

    # LLM
    llm = reuse_components["llm"] if reuse_components else GitContextLLM()

    return {
        "repository_id": repository_id,
        "branch": branch or "",
        "repository_files": repository_files,
        "embedding_model": embedding_model,
        "vector_store": vector_store,
        "bm25_retriever": bm25_retriever,
        "hybrid_retriever": hybrid_retriever,
        "reranker": reranker,
        "llm": llm,
        "indexing_result": indexing_result,
        "database": database,
    }


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global components, startup_error

    try:
        database = get_database()
        active_repository = database.get_active_repository()
        if active_repository:
            source_type = active_repository["source_type"]
            if source_type == "github":
                components = initialize_backend(
                    repository_url=active_repository["source_location"],
                    branch=active_repository["branch"] or None,
                )
            elif source_type == "local":
                components = initialize_backend(
                    repository_path=active_repository["source_location"],
                )
            elif source_type == "zip":
                archive_path = Path(active_repository["source_location"])
                if not archive_path.is_file():
                    raise FileNotFoundError(
                        f"Saved repository ZIP is missing: {archive_path}"
                    )
                repository_id, repository_files = load_repository_zip(
                    archive_path.read_bytes(),
                    filename=archive_path.name,
                )
                components = initialize_backend(
                    uploaded_repository_files=repository_files,
                    uploaded_repository_id=repository_id,
                    uploaded_archive_path=str(archive_path),
                )
            else:
                raise ValueError(f"Unknown repository source: {source_type}")
        else:
            components = initialize_backend()

        repository_components.clear()
        repository_components[components["repository_id"]] = components
        startup_error = None
    except Exception as error:
        components = {}
        startup_error = (
            "The saved repository could not be restored. "
            "Load or select a repository from the sidebar."
        )
        print(
            "[startup] Repository restoration failed: "
            f"{type(error).__name__}"
        )

    yield

    components.clear()
    chat_sessions.clear()
    repository_components.clear()


app = FastAPI(
    title="GitContext API",
    description="RAG-powered codebase intelligence API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def home():
    return {"message": "GitContext API is running", "status": "active"}


@app.get("/health")
def health():
    return {
        "status": "healthy" if components else "not_initialized",
        "repository_loaded": bool(components),
        "message": startup_error,
    }


@app.get("/repository")
def get_repository():
    if not components:
        raise HTTPException(
            status_code=503, detail="Backend is not initialized"
        )

    return {
        "repository_id": components["repository_id"],
        "total_files": len(components["repository_files"]),
        "indexing": components["indexing_result"],
        "branch": components.get("branch", ""),
    }


@app.get("/repository/architecture")
def get_repository_architecture():
    if not components:
        raise HTTPException(
            status_code=503, detail="Backend is not initialized"
        )
    return analyze_repository(components["repository_files"])


@app.get("/repository/dependencies")
def get_repository_dependencies(file_path: str = Query(min_length=1)):
    if not components:
        raise HTTPException(
            status_code=503, detail="Backend is not initialized"
        )
    result = dependencies_for_file(
        components["repository_files"], file_path
    )
    known_paths = {
        file.file_path.replace("\\", "/").lstrip("./")
        for file in components["repository_files"]
    }
    if result["file_path"].replace("\\", "/").lstrip("./") not in known_paths:
        raise HTTPException(status_code=404, detail="File not found in repository.")
    return result


@app.get("/chat/history")
def get_chat_history(session_id: str = Query(min_length=1, max_length=128)):
    if not components:
        raise HTTPException(
            status_code=503, detail="Backend is not initialized"
        )
    return get_database().get_chat_history(
        components["repository_id"], session_id
    )


@app.delete("/chat/history")
def delete_chat_history(session_id: str = Query(min_length=1, max_length=128)):
    if not components:
        raise HTTPException(
            status_code=503, detail="Backend is not initialized"
        )
    get_database().clear_chat_history(
        components["repository_id"], session_id
    )
    chat_sessions.pop(session_id, None)
    return {"status": "ok"}


@app.get("/repositories")
def list_repositories():
    return [
        {
            "repository_id": repository["repository_id"],
            "display_name": repository["display_name"],
            "source_type": repository["source_type"],
            "branch": repository["branch"],
            "is_active": repository["is_active"],
            "updated_at": repository["updated_at"],
        }
        for repository in get_database().list_repositories()
    ]


@app.delete("/repositories/{repository_id:path}")
def delete_repository(repository_id: str):
    database = get_database()
    repository = database.get_repository(repository_id)
    if repository is None:
        raise HTTPException(status_code=404, detail="Repository not found.")
    if repository["is_active"]:
        raise HTTPException(
            status_code=409,
            detail="Select another repository before deleting the active repository.",
        )

    vector_store = (
        components.get("vector_store")
        if components
        else VectorStore()
    )
    vector_store.delete_repository(repository_id)
    database.delete_repository(repository_id)
    repository_components.pop(repository_id, None)
    if repository["source_type"] == "zip" and repository["source_location"]:
        Path(repository["source_location"]).unlink(missing_ok=True)
    return {"status": "ok", "repository_id": repository_id}


@app.get("/repository/branches")
def list_repository_branches(repository_url: str = Query(min_length=1)):
    try:
        return {
            "branches": GitHubLoader().list_branches(repository_url.strip())
        }
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Could not list public repository branches: {error}",
        ) from error


@app.post("/repository/select")
def select_registered_repository(request: RepositorySelectRequest):
    global components

    database = get_database()
    repository = database.get_repository(request.repository_id)
    if repository is None:
        raise HTTPException(status_code=404, detail="Repository not found.")

    cached = repository_components.get(request.repository_id)
    if cached:
        database.set_active_repository(request.repository_id)
        components = cached
    elif repository["source_type"] == "github":
        components = initialize_backend(
            repository_url=repository["source_location"],
            branch=repository["branch"] or None,
            reuse_components=components or None,
        )
    elif repository["source_type"] == "local":
        components = initialize_backend(
            repository_path=repository["source_location"],
            reuse_components=components or None,
        )
    elif repository["source_type"] == "zip":
        archive_path = Path(repository["source_location"])
        if not archive_path.is_file():
            raise HTTPException(
                status_code=410,
                detail="Saved ZIP source is missing; upload the repository again.",
            )
        repository_id, repository_files = load_repository_zip(
            archive_path.read_bytes(),
            filename=archive_path.name,
        )
        components = initialize_backend(
            reuse_components=components or None,
            uploaded_repository_files=repository_files,
            uploaded_repository_id=repository_id,
            uploaded_archive_path=str(archive_path),
        )
    else:
        raise HTTPException(status_code=400, detail="Unsupported repository source.")

    repository_components.clear()
    repository_components[components["repository_id"]] = components
    return {
        "repository_id": components["repository_id"],
        "total_files": len(components["repository_files"]),
        "indexing": components["indexing_result"],
    }


@app.post("/repository")
def select_repository(request: RepositoryRequest):
    global components

    repository_url = request.repository_url.strip()
    if not repository_url:
        raise HTTPException(
            status_code=400, detail="Repository URL cannot be empty"
        )

    try:
        updated_components = initialize_backend(
            repository_url=repository_url,
            branch=request.branch,
            reuse_components=components or None,
        )
    except Exception as error:
        traceback.print_exc()
        raise HTTPException(
            status_code=400,
            detail=f"Could not load repository: {error}",
        ) from error

    components = updated_components
    repository_components.clear()
    repository_components[components["repository_id"]] = components

    return {
        "repository_id": components["repository_id"],
        "total_files": len(components["repository_files"]),
        "indexing": components["indexing_result"],
    }


@app.post("/repository/upload")
def upload_repository(file: UploadFile = File(...)):
    global components

    if not file.filename or Path(file.filename).suffix.lower() != ".zip":
        raise HTTPException(
            status_code=400, detail="Upload a repository ZIP file."
        )

    archive_bytes = file.file.read(MAX_ARCHIVE_SIZE + 1)
    if len(archive_bytes) > MAX_ARCHIVE_SIZE:
        raise HTTPException(
            status_code=413, detail="ZIP upload exceeds the 25 MB size limit."
        )

    try:
        repository_id, repository_files = load_repository_zip(
            archive_bytes, filename=Path(file.filename).name
        )
        archive_path = (
            PROJECT_ROOT
            / "data"
            / "repository_archives"
            / f"{repository_id.removeprefix('zip:')}.zip"
        )
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        archive_path.write_bytes(archive_bytes)
        updated_components = initialize_backend(
            reuse_components=components or None,
            uploaded_repository_files=repository_files,
            uploaded_repository_id=repository_id,
            uploaded_archive_path=str(archive_path),
        )
    except (ValueError, zipfile.BadZipFile) as error:
        raise HTTPException(
            status_code=400, detail=f"Could not load repository ZIP: {error}"
        ) from error
    except Exception as error:
        traceback.print_exc()
        raise HTTPException(
            status_code=400,
            detail=f"Could not index repository ZIP: {error}",
        ) from error

    components = updated_components
    repository_components.clear()
    repository_components[components["repository_id"]] = components

    return {
        "repository_id": components["repository_id"],
        "total_files": len(components["repository_files"]),
        "indexing": components["indexing_result"],
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    request_started = time.perf_counter()
    backend = components
    if not backend:
        raise HTTPException(
            status_code=503, detail="Backend is not initialized"
        )

    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        llm = backend["llm"]

        embedding_model = backend["embedding_model"]

        hybrid_retriever = backend["hybrid_retriever"]

        reranker = backend["reranker"]

        repository_id = backend["repository_id"]

        database = get_database()
        history = load_chat_exchanges(
            database, repository_id, request.session_id
        )

        # Skip the extra LLM request unless query rewriting is enabled.
        rewrite_queries = request.rewrite_queries
        if rewrite_queries is None:
            rewrite_queries = llm.query_rewriting_enabled()

        if rewrite_queries:
            print("STEP 1: Starting query rewriting")
            search_queries = llm.generate_search_queries(question, history)
            print("STEP 1 COMPLETED")
        else:
            search_queries = [question]

        # Generate embeddings
        print("STEP 2: Generating embeddings")
        query_embeddings = [
            embedding_model.embed_query(query) for query in search_queries
        ]
        print("STEP 2 COMPLETED")

        # Hybrid retrieval + RRF
        print("STEP 3: Starting hybrid retrieval")
        retrieved_chunks = hybrid_retriever.search_multiple(
            queries=search_queries,
            query_embeddings=query_embeddings,
            top_k=10,
            rrf_k=60,
            repository_id=repository_id,
        )

        if not retrieved_chunks:
            response_time_ms = (time.perf_counter() - request_started) * 1000
            database.save_chat_message(
                repository_id, request.session_id, "user", question
            )
            database.save_chat_message(
                repository_id,
                request.session_id,
                "assistant",
                "I couldn't find relevant evidence in the indexed repository.",
                response_time_ms=response_time_ms,
            )
            return ChatResponse(
                answer=(
                    "I couldn't find relevant evidence "
                    "in the indexed repository."
                ),
                sources=[],
                response_time_ms=response_time_ms,
            )
        print("STEP 3 COMPLETED")

        # Cross-encoder reranking
        print("STEP 4: Starting reranking")
        reranked_chunks = reranker.rerank(
            query=question, retrieved_chunks=retrieved_chunks, top_k=5
        )
        print("STEP 4 COMPLETED")

        # LLM generation
        print("STEP 5: Generating answer")
        answer = llm.generate_answer(
            query=question,
            retrieved_chunks=reranked_chunks,
            chat_history=history,
        )
        print("STEP 5 COMPLETED")

        # Save conversation history
        if components is backend:
            history.append({"question": question, "answer": answer})
            chat_sessions[request.session_id] = history[-10:]

        # Extract source metadata
        sources = []

        for chunk in reranked_chunks:
            metadata = chunk["metadata"]

            sources.append(
                {
                    "file_path": metadata.get("file_path"),
                    "chunk_name": metadata.get("chunk_name"),
                    "chunk_type": metadata.get("chunk_type"),
                    "start_line": metadata.get("start_line"),
                    "end_line": metadata.get("end_line"),
                    "score": chunk.get("rerank_score"),
                    "source_type": metadata.get("source_type", "source"),
                    "commit_sha": metadata.get("commit_sha"),
                    "author": metadata.get("author"),
                    "branch": metadata.get("branch"),
                    "pull_request_number": metadata.get("pull_request_number"),
                    "issue_number": metadata.get("issue_number"),
                    "last_modified": metadata.get("last_modified"),
                }
            )

        response_time_ms = (time.perf_counter() - request_started) * 1000
        database.save_chat_message(
            repository_id, request.session_id, "user", question
        )
        database.save_chat_message(
            repository_id,
            request.session_id,
            "assistant",
            answer,
            sources=sources,
            response_time_ms=response_time_ms,
        )
        return ChatResponse(
            answer=answer,
            sources=sources,
            response_time_ms=response_time_ms,
        )

    except Exception as error:
        print("\n========== GITCONTEXT ERROR ==========")
        traceback.print_exc()
        print("======================================\n")

        raise HTTPException(status_code=500, detail=str(error))


@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    def event_stream():
        request_started = time.perf_counter()
        backend = components

        def send(event):
            return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        if not backend:
            yield send({"type": "error", "detail": "Backend is not initialized"})
            return

        question = request.question.strip()
        if not question:
            yield send({"type": "error", "detail": "Question cannot be empty"})
            return

        try:
            llm = backend["llm"]
            repository_id = backend["repository_id"]
            database = get_database()
            history = load_chat_exchanges(
                database, repository_id, request.session_id
            )
            rewrite_queries = request.rewrite_queries
            if rewrite_queries is None:
                rewrite_queries = llm.query_rewriting_enabled()
            queries = (
                llm.generate_search_queries(question, history)
                if rewrite_queries
                else [question]
            )
            embeddings = [
                backend["embedding_model"].embed_query(query)
                for query in queries
            ]
            retrieved = backend["hybrid_retriever"].search_multiple(
                queries=queries,
                query_embeddings=embeddings,
                top_k=10,
                rrf_k=60,
                repository_id=repository_id,
            )
            if not retrieved:
                answer = "I couldn't find relevant evidence in the indexed repository."
                sources = []
                yield send({"type": "token", "text": answer})
            else:
                reranked = backend["reranker"].rerank(
                    query=question, retrieved_chunks=retrieved, top_k=5
                )
                answer_parts = []
                for event in llm.generate_answer_stream(
                    query=question,
                    retrieved_chunks=reranked,
                    chat_history=history,
                ):
                    if event["type"] == "reset":
                        answer_parts.clear()
                        yield send({"type": "reset"})
                    else:
                        answer_parts.append(event["text"])
                        yield send(event)

                answer = llm.normalize_citations("".join(answer_parts))
                if not llm.validate_citations(answer, reranked):
                    answer = (
                        "I couldn't produce an answer with verifiable source "
                        "citations. Please try rephrasing your question."
                    )
                    yield send({"type": "reset"})
                    yield send({"type": "token", "text": answer})
                else:
                    normalized = llm.normalize_citations("".join(answer_parts))
                    if normalized != "".join(answer_parts):
                        answer_parts = [normalized]
                        yield send({"type": "reset"})
                        yield send({"type": "token", "text": normalized})
                sources = [
                    {
                        "file_path": chunk["metadata"].get("file_path"),
                        "chunk_name": chunk["metadata"].get("chunk_name"),
                        "chunk_type": chunk["metadata"].get("chunk_type"),
                        "start_line": chunk["metadata"].get("start_line"),
                        "end_line": chunk["metadata"].get("end_line"),
                        "score": chunk.get("rerank_score"),
                        "source_type": chunk["metadata"].get("source_type", "source"),
                        "commit_sha": chunk["metadata"].get("commit_sha"),
                        "author": chunk["metadata"].get("author"),
                        "branch": chunk["metadata"].get("branch"),
                        "pull_request_number": chunk["metadata"].get("pull_request_number"),
                        "issue_number": chunk["metadata"].get("issue_number"),
                        "last_modified": chunk["metadata"].get("last_modified"),
                    }
                    for chunk in reranked
                ]

            elapsed_ms = (time.perf_counter() - request_started) * 1000
            database.save_chat_message(
                repository_id, request.session_id, "user", question
            )
            database.save_chat_message(
                repository_id,
                request.session_id,
                "assistant",
                answer,
                sources=sources,
                response_time_ms=elapsed_ms,
            )
            yield send({
                "type": "final",
                "sources": sources,
                "response_time_ms": elapsed_ms,
            })
        except Exception as error:
            traceback.print_exc()
            yield send({"type": "error", "detail": str(error)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )