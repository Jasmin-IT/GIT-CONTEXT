# GitContext

GitContext is a local codebase intelligence app. It indexes a repository and
answers questions using hybrid vector/BM25 retrieval, reciprocal-rank fusion,
cross-encoder reranking, and evidence-checked citations.

## Features

- Load a local repository, a **public** GitHub repository and branch, or a ZIP.
- Index recent GitHub commits, pull-request descriptions/reviews, and
  bug/incident-style issues alongside repository files.
- Incrementally index changed and deleted files into persistent local Chroma.
- Search per repository using vector similarity and BM25, then rerank results.
- Generate answers with Groq as the primary LLM and Gemini as fallback.
- Stream answers in the Streamlit chat UI, persist repository-specific chat
  history, and show response timing and source metadata.
- Inspect a repository architecture summary and local Python/JS/TS dependencies.
- Compare BM25, vector, hybrid, and reranked retrieval against manually labeled
  repository-specific evaluation data.

The app runs locally. SQLite and Chroma data are stored under `data/`; API keys
are read from environment variables or a local `.env` file.

## Setup (Windows PowerShell)

Use Python 3.10 or newer:

```powershell
cd folder_name
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Set at least one LLM key in `.env`. Groq is tried first; Gemini is only used
when Groq is unavailable or fails:

```dotenv
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
```

Both keys are optional for indexing and retrieval, but an LLM key is needed to
generate answers. Never commit `.env` or put API keys in source code.

## Run the application

Start the API in one terminal:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn api:app --reload
```

Start Streamlit in another terminal:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run frontend\app.py
```
or 

```powershell
cd frontend
streamlit run app.py
```

Open the Streamlit URL printed by the command (normally
`http://localhost:8501`). The API is at `http://127.0.0.1:8000`; its interactive
OpenAPI page is `/docs`. Set `API_BASE_URL` if the frontend should use a
different API address.

On first start, the embedding and reranking models may need to download. If a
saved repository cannot be restored, the API remains available so another
repository can be loaded from the UI.

## Repository input and indexing

In the sidebar, paste a public GitHub repository URL and optionally select a
branch, or upload a ZIP archive. Private GitHub repositories are not supported.
ZIP files are limited to 25 MB compressed and 100 MB uncompressed; archives are
validated for unsafe paths and symbolic links and are not extracted to disk.
Files larger than 1 MB and unsupported extensions are skipped. Common virtual
environments, dependency/build caches, generated output, and dataset folders
are excluded.
GitHub code files are loaded from the selected branch. Git history and issue
context are intentionally bounded to the five most recent records per category
and at most ten comments per record to limit public API usage; this is not a
complete archive of all repository discussions.

Repository catalog, chat history, ZIP archives, and Chroma vectors persist
locally under `data/`. The repository selector switches the active index;
each query is filtered to the selected repository.

Query rewriting is disabled by default to avoid an extra LLM call. Enable
**Use AI query rewriting** in the UI when useful, or set
`GITCONTEXT_QUERY_REWRITE=true` for CLI use.

## Evaluation

Evaluation requires repository-specific gold labels; retrieval metrics are not
meaningful without manually verified expected files/ranges. First inspect the
selected repository's chunk paths and line ranges:

```powershell
python evaluate.py --repository-path "C:\work\my-project" --list-chunks
```

Then create a JSON dataset and run:

```powershell
python evaluate.py --repository-path "C:\work\my-project" --eval-file "eval_sets\my-project.json"
python evaluate.py --repository-url "https://github.com/owner/project" --eval-file "eval_sets\project.json"
python evaluate.py --repository-path "C:\work\my-project" --eval-file "eval_sets\my-project.json" --rewrite
python evaluate.py --repository-path "C:\work\my-project" --eval-file "eval_sets\my-project.json" --answers
```


The evaluator checks BM25, vector, hybrid, and hybrid-plus-reranking retrieval
with Recall@K, Precision@K, hit rate, MRR, and latency; it records per-question
results and misses. `--answers` additionally measures citation validity and
overlap with the gold labels, plus retrieval and end-to-end response latency.
Results are written to `eval_results.json`. A representative dataset is
necessary to draw conclusions about retrieval or answer quality.
Citation checks are not a substitute for human review of answer correctness or
hallucination resistance.

## Tests

```powershell
python -m unittest discover -s tests -v
```
