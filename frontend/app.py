import os
import json
import uuid
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

CHAT_STREAM_ENDPOINT = f"{API_BASE_URL}/chat/stream"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"
REPOSITORY_ENDPOINT = f"{API_BASE_URL}/repository"
REPOSITORY_UPLOAD_ENDPOINT = f"{API_BASE_URL}/repository/upload"
REPOSITORIES_ENDPOINT = f"{API_BASE_URL}/repositories"
REPOSITORY_SELECT_ENDPOINT = f"{API_BASE_URL}/repository/select"
ARCHITECTURE_ENDPOINT = f"{API_BASE_URL}/repository/architecture"
DEPENDENCIES_ENDPOINT = f"{API_BASE_URL}/repository/dependencies"
CHAT_HISTORY_ENDPOINT = f"{API_BASE_URL}/chat/history"

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="GitContext",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown(
    """

<style>

.main {
    background-color: #0e1117;
}

.block-container {
    padding-top: 2rem;
    max-width: 1200px;
}

[data-testid="stSidebar"] {
    background-color: #161b22;
}

.header-title {
    font-size: 36px;
    font-weight: 700;
    color: #58a6ff;
}

.header-subtitle {
    font-size: 16px;
    color: #8b949e;
}

.source-card {
    background-color: #161b22;
    padding: 12px;
    border-radius: 8px;
    border: 1px solid #30363d;
    margin-bottom: 10px;
}

.status-online {
    color: #3fb950;
    font-weight: bold;
}

.status-offline {
    color: #f85149;
    font-weight: bold;
}

</style>

""",
    unsafe_allow_html=True,
)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = st.query_params.get("session_id", str(uuid.uuid4()))
    st.query_params["session_id"] = st.session_state.session_id

# --------------------------------------------------
# API FUNCTIONS
# --------------------------------------------------


def check_backend():
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)

        response.raise_for_status()

        return response.json()

    except requests.RequestException:
        return None


def get_repository_info():
    try:
        response = requests.get(REPOSITORY_ENDPOINT, timeout=10)

        response.raise_for_status()

        return response.json()

    except requests.RequestException:
        return None


def get_repositories():
    try:
        response = requests.get(REPOSITORIES_ENDPOINT, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return []


def get_repository_branches(repository_url):
    try:
        response = requests.get(
            f"{API_BASE_URL}/repository/branches",
            params={"repository_url": repository_url},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    except requests.HTTPError:
        try:
            return {"error": response.json().get("detail", "Backend returned an error.")}
        except ValueError:
            return {"error": "Backend returned an invalid response."}
    except requests.RequestException:
        return {"error": "Cannot connect to GitContext backend."}


def select_registered_repository(repository_id):
    try:
        response = requests.post(
            REPOSITORY_SELECT_ENDPOINT,
            json={"repository_id": repository_id},
            timeout=600,
        )
        response.raise_for_status()
        return response.json()
    except requests.Timeout:
        return {"error": "Repository indexing timed out. Try again."}
    except requests.HTTPError:
        try:
            return {"error": response.json().get("detail", "Backend returned an error.")}
        except ValueError:
            return {"error": "Backend returned an invalid response."}
    except requests.RequestException:
        return {"error": "Cannot connect to GitContext backend."}


def select_repository(repository_url, branch=None):
    try:
        response = requests.post(
            REPOSITORY_ENDPOINT,
            json={
                "repository_url": repository_url,
                "branch": branch or None,
            },
            timeout=600,
        )
        response.raise_for_status()
        return response.json()

    except requests.Timeout:
        return {"error": "Repository indexing timed out. Try again."}

    except requests.HTTPError:
        try:
            detail = response.json().get("detail", "Backend returned an error.")
        except ValueError:
            detail = "Backend returned an invalid response."

        return {"error": detail}

    except requests.RequestException:
        return {"error": "Cannot connect to GitContext backend."}


def upload_repository(uploaded_file):
    try:
        response = requests.post(
            REPOSITORY_UPLOAD_ENDPOINT,
            files={
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    "application/zip",
                )
            },
            timeout=600,
        )
        response.raise_for_status()
        return response.json()

    except requests.Timeout:
        return {"error": "Repository indexing timed out. Try again."}

    except requests.HTTPError:
        try:
            detail = response.json().get("detail", "Backend returned an error.")
        except ValueError:
            detail = "Backend returned an invalid response."
        return {"error": detail}

    except requests.RequestException:
        return {"error": "Cannot connect to GitContext backend."}


def ask_question(question, answer_placeholder=None):
    payload = {
        "question": question,
        "session_id": st.session_state.session_id,
        "rewrite_queries": st.session_state.get("rewrite_queries", False),
    }

    try:
        response = requests.post(
            CHAT_STREAM_ENDPOINT,
            json=payload,
            timeout=(10, 300),
            stream=True,
        )
        response.raise_for_status()
        answer_parts = []
        result = {"sources": [], "response_time_ms": None}
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            if isinstance(line, bytes):
                line = line.decode("utf-8")
            if not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            if event["type"] == "reset":
                answer_parts.clear()
            elif event["type"] == "token":
                answer_parts.append(event["text"])
                if answer_placeholder is not None:
                    answer_placeholder.markdown("".join(answer_parts))
            elif event["type"] == "final":
                result.update(event)
            elif event["type"] == "error":
                return {"error": event["detail"]}
        result["answer"] = "".join(answer_parts)
        return result

    except requests.Timeout:
        return {"error": "The request timed out. Try again."}

    except requests.HTTPError:
        try:
            detail = response.json().get("detail", "Backend returned an error.")
        except ValueError:
            detail = "Backend returned an invalid response."

        return {"error": detail}

    except requests.RequestException:
        return {"error": "Cannot connect to GitContext backend."}


def get_saved_history():
    try:
        response = requests.get(
            CHAT_HISTORY_ENDPOINT,
            params={"session_id": st.session_state.session_id},
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def clear_saved_history():
    try:
        response = requests.delete(
            CHAT_HISTORY_ENDPOINT,
            params={"session_id": st.session_state.session_id},
            timeout=10,
        )
        response.raise_for_status()
    except requests.RequestException:
        st.error("Could not clear saved conversation history.")


def get_architecture():
    try:
        response = requests.get(ARCHITECTURE_ENDPOINT, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def get_dependencies(file_path):
    try:
        response = requests.get(
            DEPENDENCIES_ENDPOINT,
            params={"file_path": file_path},
            timeout=15,
        )
        response.raise_for_status()
        return response.json()
    except requests.HTTPError:
        try:
            return {"error": response.json().get("detail", "Backend returned an error.")}
        except ValueError:
            return {"error": "Backend returned an invalid response."}
    except requests.RequestException:
        return {"error": "Cannot connect to GitContext backend."}


if not st.session_state.get("history_loaded"):
    saved_history = get_saved_history()
    if saved_history is not None:
        st.session_state.messages = [
            {
                "role": message["role"],
                "content": message["content"],
                "sources": message.get("sources", []),
                "response_time_ms": message.get("response_time_ms"),
            }
            for message in saved_history
        ]
        st.session_state.history_loaded = True


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown("## 💻 GitContext")

    st.caption("AI-powered Codebase Intelligence")

    st.divider()

    st.markdown("### Backend Status")

    backend_status = check_backend()

    if backend_status:
        st.markdown(
            '<p class="status-online">● Backend Online</p>',
            unsafe_allow_html=True,
        )
        if not backend_status.get("repository_loaded"):
            st.info(
                backend_status.get("message")
                or "Load or select a repository to start chatting."
            )

    else:
        st.markdown(
            '<p class="status-offline">● Backend Offline</p>',
            unsafe_allow_html=True,
        )

        st.warning("Start the FastAPI server before chatting.")

    st.divider()

    st.markdown("### Repository")

    repository_info = get_repository_info()

    if repository_info:
        st.success("Repository Indexed")

        st.write("**Repository ID:**", repository_info["repository_id"])

        st.metric("Total Files", repository_info["total_files"])

        indexing = repository_info.get("indexing", {})

        col1, col2 = st.columns(2)

        col1.metric("New", indexing.get("new", 0))

        col2.metric("Modified", indexing.get("modified", 0))

        col1, col2 = st.columns(2)

        col1.metric("Deleted", indexing.get("deleted", 0))

        col2.metric("Successful", indexing.get("successful", 0))

        architecture = get_architecture()
        if architecture:
            with st.expander("Repository architecture"):
                st.caption(
                    f"{architecture['total_files']} files · "
                    f"{architecture['python_functions']} Python functions · "
                    f"{architecture['python_classes']} classes · "
                    f"{architecture['dependency_edges']} local dependencies"
                )
                st.write("**Top-level directories**")
                st.json(architecture["top_level_directories"])
                dependency_file = st.text_input(
                    "Inspect file dependencies",
                    key="dependency_file_path",
                    placeholder="src/module.py",
                )
                if st.button("Show dependencies", key="show_file_dependencies"):
                    if not dependency_file.strip():
                        st.warning("Enter a repository-relative file path.")
                    else:
                        dependency_result = get_dependencies(
                            dependency_file.strip()
                        )
                        if "error" in dependency_result:
                            st.error(dependency_result["error"])
                        else:
                            st.write("**Imports from this repository**")
                            st.json(dependency_result["imports"])
                            st.write("**Imported by**")
                            st.json(dependency_result["imported_by"])

    else:
        st.warning("Repository information unavailable.")

    registered_repositories = get_repositories()
    if registered_repositories:
        active_id = next(
            (
                item["repository_id"]
                for item in registered_repositories
                if item["is_active"]
            ),
            registered_repositories[0]["repository_id"],
        )
        selected_id = st.selectbox(
            "Indexed repositories",
            [item["repository_id"] for item in registered_repositories],
            index=[
                item["repository_id"] for item in registered_repositories
            ].index(active_id),
            format_func=lambda repository_id: next(
                item["display_name"]
                for item in registered_repositories
                if item["repository_id"] == repository_id
            ),
        )
        if st.button("Switch repository", use_container_width=True):
            with st.spinner("Switching repository..."):
                result = select_registered_repository(selected_id)
            if "error" in result:
                st.error(result["error"])
            else:
                st.session_state.messages = []
                st.session_state.history_loaded = False
                st.rerun()

    repository_source = st.radio(
        "Repository source",
        ("GitHub URL", "ZIP upload"),
        horizontal=True,
    )

    repository_url = ""
    uploaded_file = None
    branch = None
    if repository_source == "GitHub URL":
        repository_url = st.text_input(
            "GitHub repository URL",
            key="repository_url",
            placeholder="https://github.com/owner/repository",
        )
        if st.button("List branches", use_container_width=True):
            if not repository_url.strip():
                st.warning("Enter a GitHub repository URL first.")
            else:
                branch_result = get_repository_branches(repository_url.strip())
                if "error" in branch_result:
                    st.error(branch_result["error"])
                else:
                    st.session_state.github_branches = branch_result["branches"]
                    st.session_state.github_branches_url = repository_url.strip()

        if (
            repository_url.strip()
            and st.session_state.get("github_branches_url")
            == repository_url.strip()
            and st.session_state.get("github_branches")
        ):
            branch = st.selectbox(
                "Branch",
                st.session_state.github_branches,
                key="selected_github_branch",
            )
        else:
            branch = st.text_input(
                "Branch (optional)",
                key="repository_branch",
                placeholder="Default branch",
            )
    else:
        uploaded_file = st.file_uploader(
            "Repository ZIP (up to 25 MB)",
            type=["zip"],
        )

    submitted = st.button("Load repository", use_container_width=True)

    if submitted:
        if repository_source == "GitHub URL" and not repository_url.strip():
            st.error("Enter a GitHub repository URL.")
        elif repository_source == "ZIP upload" and uploaded_file is None:
            st.error("Choose a repository ZIP file.")
        else:
            with st.spinner("Loading and indexing repository..."):
                if repository_source == "GitHub URL":
                    result = select_repository(
                        repository_url.strip(),
                        branch.strip() if branch else None,
                    )
                else:
                    result = upload_repository(uploaded_file)

            if "error" in result:
                st.error(result["error"])
            else:
                st.session_state.messages = []
                st.session_state.history_loaded = False
                st.rerun()

    st.divider()

    st.checkbox(
        "Use AI query rewriting",
        key="rewrite_queries",
        help=(
            "Uses an extra LLM request to expand the question. "
            "Leave off for faster retrieval."
        ),
    )

    st.divider()

    st.markdown("### Actions")

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        clear_saved_history()
        st.session_state.messages = []

        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.history_loaded = False
        st.query_params["session_id"] = st.session_state.session_id

        st.rerun()

    st.divider()

    st.caption("Built with Python • FastAPI • RAG • ChromaDB")

# --------------------------------------------------
# MAIN HEADER
# --------------------------------------------------

st.markdown(
    '<div class="header-title">GitContext</div>', unsafe_allow_html=True
)

st.markdown(
    '<div class="header-subtitle">'
    "Understand your codebase using AI-powered retrieval."
    "</div>",
    unsafe_allow_html=True,
)

st.divider()

# --------------------------------------------------
# WELCOME SCREEN
# --------------------------------------------------

if not st.session_state.messages:
    st.markdown("### 👋 Welcome to GitContext")

    st.write(
        "Ask questions about your repository. "
        "GitContext retrieves relevant code and generates "
        "evidence-grounded answers."
    )

    st.markdown("### Try asking")

    examples = [
        "Explain the overall architecture of this repository.",
        "Where is the database connection initialized?",
        "Find the function responsible for processing user input.",
        "Which files depend on the database module?",
        "Explain how authentication works.",
    ]

    cols = st.columns(2)

    for index, example in enumerate(examples):
        with cols[index % 2]:
            if st.button(
                example, key=f"example_{index}", use_container_width=True
            ):
                st.session_state.pending_question = example

                st.rerun()

# --------------------------------------------------
# DISPLAY CHAT HISTORY
# --------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            response_time_ms = message.get("response_time_ms")
            if response_time_ms is not None:
                st.caption(f"Response time: {response_time_ms:.0f} ms")

            sources = message.get("sources", [])

            if sources:
                with st.expander(f"📁 View Sources ({len(sources)})"):
                    for source in sources:
                        file_path = source.get("file_path", "Unknown file")

                        start_line = source.get("start_line", "?")

                        end_line = source.get("end_line", "?")

                        chunk_name = source.get("chunk_name", "Unknown chunk")

                        score = source.get("score")

                        st.markdown(f"**{file_path}**")

                        st.caption(
                            f"{chunk_name} | "
                            f"Lines {start_line}-{end_line}"
                        )

                        if score is not None:
                            st.caption(f"Rerank score: {score:.4f}")
                        source_type = source.get("source_type", "source")
                        details = [
                            f"Source: {source_type}",
                            source.get("branch"),
                            source.get("author"),
                        ]
                        if source.get("pull_request_number") is not None:
                            details.append(
                                f"PR #{source['pull_request_number']}"
                            )
                        if source.get("issue_number") is not None:
                            details.append(f"Issue #{source['issue_number']}")
                        st.caption(" · ".join(item for item in details if item))

                        st.divider()

# --------------------------------------------------
# QUESTION PROCESSING
# --------------------------------------------------

question = st.chat_input("Ask anything about your codebase...")

if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")

if question:
    question = question.strip()

    if question:
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            answer_placeholder = st.empty()
            with st.spinner("Searching codebase and generating answer..."):
                result = ask_question(question, answer_placeholder)

            if "error" in result:
                answer = result["error"]

                sources = []
                response_time_ms = None

                answer_placeholder.error(answer)

            else:
                answer = result.get("answer", "No answer returned.")

                sources = result.get("sources", [])
                response_time_ms = result.get("response_time_ms")

                answer_placeholder.markdown(answer)
                if response_time_ms is not None:
                    st.caption(f"Response time: {response_time_ms:.0f} ms")

                if sources:
                    with st.expander(f"📁 View Sources ({len(sources)})"):
                        for source in sources:
                            st.markdown(f"**{source.get('file_path')}**")

                            st.caption(
                                f"Lines "
                                f"{source.get('start_line')}-"
                                f"{source.get('end_line')}"
                            )
                            details = [
                                f"Source: {source.get('source_type', 'source')}",
                                source.get("branch"),
                                source.get("author"),
                            ]
                            if source.get("pull_request_number") is not None:
                                details.append(
                                    f"PR #{source['pull_request_number']}"
                                )
                            if source.get("issue_number") is not None:
                                details.append(f"Issue #{source['issue_number']}")
                            st.caption(
                                " · ".join(item for item in details if item)
                            )

                            st.divider()

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "response_time_ms": response_time_ms,
            }
        )