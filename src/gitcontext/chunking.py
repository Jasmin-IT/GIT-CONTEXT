import ast
from dataclasses import dataclass, field

from src.gitcontext.ingestion import RepositoryFile


@dataclass
class CodeChunk:
    file_path: str
    chunk_type: str
    chunk_name: str
    start_line: int
    end_line: int
    content: str
    language: str
    repository_id: str = ""
    metadata: dict = field(default_factory=dict)


# ==========================================
# 1. CREATE CHUNK
# ==========================================

def create_chunk(
    file,
    chunk_type,
    chunk_name,
    start_line,
    end_line,
    lines,
    repository_id=""
):

    content = "\n".join(
        lines[start_line - 1:end_line]
    )

    return CodeChunk(
        file_path=file.file_path,
        chunk_type=chunk_type,
        chunk_name=chunk_name,
        start_line=start_line,
        end_line=end_line,
        content=content,
        language=file.language,
        repository_id=repository_id,
        metadata=file.metadata,
    )


# ==========================================
# 2. GET START LINE INCLUDING DECORATORS
# ==========================================

def get_start_line(node):

    decorators = getattr(node, "decorator_list", [])

    if decorators:
        return min(
            decorator.lineno
            for decorator in decorators
        )

    return node.lineno


# ==========================================
# 3. EXTRACT CLASS METHODS
# ==========================================

def extract_class_methods(
    file,
    class_node,
    lines,
    repository_id=""
):

    chunks = []

    for node in class_node.body:

        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef)
        ):

            method_name = (
                f"{class_node.name}.{node.name}"
            )

            chunk = create_chunk(
                file=file,
                chunk_type="method",
                chunk_name=method_name,
                start_line=get_start_line(node),
                end_line=node.end_lineno,
                lines=lines,
                repository_id=repository_id
            )

            chunks.append(chunk)

    return chunks


# ==========================================
# 4. CHUNK CLASS
# ==========================================

def chunk_class(
    file,
    node,
    lines,
    repository_id=""
):

    methods = extract_class_methods(
        file,
        node,
        lines,
        repository_id
    )

    class_start = get_start_line(node)

    # Find the first method.
    method_nodes = [
        child
        for child in node.body
        if isinstance(
            child,
            (ast.FunctionDef, ast.AsyncFunctionDef)
        )
    ]

    if method_nodes:

        first_method_line = min(
            get_start_line(method)
            for method in method_nodes
        )

        class_end = first_method_line - 1

    else:
        class_end = node.end_lineno

    chunks = []

    # Keep class declaration, docstring and initial
    # class-level attributes as context.
    if class_end >= class_start:

        class_context = create_chunk(
            file=file,
            chunk_type="class_context",
            chunk_name=node.name,
            start_line=class_start,
            end_line=class_end,
            lines=lines,
            repository_id=repository_id
        )

        chunks.append(class_context)

    # Add individual methods.
    chunks.extend(methods)

    # Classes without methods need their complete content.
    if not method_nodes and not chunks:

        chunks.append(
            create_chunk(
                file=file,
                chunk_type="class",
                chunk_name=node.name,
                start_line=class_start,
                end_line=node.end_lineno,
                lines=lines,
                repository_id=repository_id
            )
        )

    return chunks


# ==========================================
# 5. CHUNK PYTHON FILE
# ==========================================

def chunk_python_file(
    file,
    repository_id=""
):

    chunks = []

    try:
        tree = ast.parse(file.content)

    except SyntaxError:
        print(f"Could not parse: {file.file_path}")
        return []

    lines = file.content.splitlines()

    module_nodes = []

    def flush_module_nodes():

        if not module_nodes:
            return

        start_line = min(
            node.lineno for node in module_nodes
        )

        end_line = max(
            node.end_lineno for node in module_nodes
        )

        chunks.append(
            create_chunk(
                file=file,
                chunk_type="module",
                chunk_name="module_context",
                start_line=start_line,
                end_line=end_line,
                lines=lines,
                repository_id=repository_id
            )
        )

        module_nodes.clear()

    for node in tree.body:

        # Top-level functions
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef)
        ):

            flush_module_nodes()

            chunks.append(
                create_chunk(
                    file=file,
                    chunk_type="function",
                    chunk_name=node.name,
                    start_line=get_start_line(node),
                    end_line=node.end_lineno,
                    lines=lines,
                    repository_id=repository_id
                )
            )

        # Classes
        elif isinstance(node, ast.ClassDef):

            flush_module_nodes()

            chunks.extend(
                chunk_class(
                    file,
                    node,
                    lines,
                    repository_id
                )
            )

        # Imports, constants and other module-level statements
        else:
            module_nodes.append(node)

    flush_module_nodes()

    return chunks


# ==========================================
# 6. CHUNK ANY REPOSITORY FILE
# ==========================================

def chunk_repository_file(
    file,
    repository_id=""
):

    if file.language == "Python":

        return chunk_python_file(
            file,
            repository_id
        )

    lines = file.content.splitlines()

    return [
        CodeChunk(
            file_path=file.file_path,
            chunk_type="file",
            chunk_name=file.file_name,
            start_line=1,
            end_line=len(lines),
            content=file.content,
            language=file.language,
            repository_id=repository_id,
            metadata=file.metadata,
        )
    ]