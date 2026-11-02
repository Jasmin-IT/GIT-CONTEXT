import ast
import re
from collections import Counter


JS_IMPORT = re.compile(
    r"""(?:import\s+(?:[^"']*?\s+from\s+)?|require\()\s*["']([^"']+)["']"""
)


def _python_imports(file_path, content):
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []

    module = file_path.rsplit(".", 1)[0].replace("/", ".")
    if module.endswith(".__init__"):
        module = module[: -len(".__init__")]

    dependencies = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            dependencies.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                package = module.split(".")[:-1]
                if node.level > 1:
                    package = package[: -(node.level - 1)]
                base = ".".join(package)
                if node.module:
                    base = ".".join(part for part in (base, node.module) if part)
            else:
                base = node.module or ""
            if base:
                dependencies.add(base)
    return sorted(dependencies)


def analyze_repository(repository_files):
    file_paths = {file.file_path.replace("\\", "/") for file in repository_files}
    python_modules = {
        path.rsplit(".", 1)[0].replace("/", "."): path
        for path in file_paths
        if path.endswith(".py")
    }
    python_modules.update({
        path[: -len(".__init__")]: source_path
        for path, source_path in list(python_modules.items())
        if path.endswith(".__init__")
    })

    edges = []
    functions = 0
    classes = 0
    for file in repository_files:
        path = file.file_path.replace("\\", "/")
        if file.language == "Python":
            try:
                tree = ast.parse(file.content)
            except SyntaxError:
                imports = []
            else:
                functions += sum(
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    for node in ast.walk(tree)
                )
                classes += sum(
                    isinstance(node, ast.ClassDef) for node in ast.walk(tree)
                )
                imports = _python_imports(path, file.content)
            for imported in imports:
                target = python_modules.get(imported)
                if target and target != path:
                    edges.append({"from": path, "to": target, "import": imported})
        elif file.language in {"JavaScript", "TypeScript"}:
            for imported in JS_IMPORT.findall(file.content):
                if not imported.startswith("."):
                    continue
                base = path.rsplit("/", 1)[0] if "/" in path else ""
                target_base = "/".join(
                    part for part in (base, imported) if part
                )
                candidates = {
                    target_base,
                    *(f"{target_base}{suffix}" for suffix in (".js", ".ts", ".tsx", ".jsx")),
                    *(f"{target_base}/index{suffix}" for suffix in (".js", ".ts", ".tsx", ".jsx")),
                }
                target = next(
                    (candidate for candidate in candidates if candidate in file_paths),
                    None,
                )
                if target:
                    edges.append({"from": path, "to": target, "import": imported})

    extensions = Counter(file.extension for file in repository_files)
    top_level_directories = Counter(
        file.file_path.replace("\\", "/").split("/", 1)[0]
        if "/" in file.file_path.replace("\\", "/")
        else "."
        for file in repository_files
    )
    return {
        "total_files": len(repository_files),
        "files_by_extension": dict(sorted(extensions.items())),
        "top_level_directories": dict(sorted(top_level_directories.items())),
        "python_functions": functions,
        "python_classes": classes,
        "dependency_edges": len(edges),
        "dependencies": edges,
    }


def dependencies_for_file(repository_files, file_path):
    normalized = file_path.replace("\\", "/").lstrip("./")
    repository = analyze_repository(repository_files)
    dependencies = repository["dependencies"]
    return {
        "file_path": file_path,
        "imports": [
            edge for edge in dependencies if edge["from"] == normalized
        ],
        "imported_by": [
            edge for edge in dependencies if edge["to"] == normalized
        ],
    }
