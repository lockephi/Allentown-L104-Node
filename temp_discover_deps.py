# temp_discover_deps.py
import ast
from pathlib import Path

def get_imports(path):
    with open(path, "r", encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        yield alias.name.split('.')[0]
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        yield node.module.split('.')[0]
        except Exception:
            pass # Ignore files that can't be parsed

def main():
    project_root = Path(__file__).parent
    local_modules = {p.name for p in project_root.iterdir() if p.is_dir()}
    local_modules.add(project_root.name)

    all_imports = set()
    for py_file in project_root.rglob("*.py"):
        if "site-packages" in str(py_file): # Exclude venv
            continue
        for imp in get_imports(py_file):
            all_imports.add(imp)

    # A best-effort filter for standard library modules
    # This is not perfect but good enough for this purpose.
    # A more robust solution would use a package like `stdlibs`.
    std_libs = {
        "os", "sys", "math", "json", "time", "datetime", "pathlib", "ast", "uuid", "textwrap", "sqlite3",
        "concurrent", "collections", "typing", "functools", "abc", "asyncio", "logging", "re",
        "copy", "hashlib", "importlib", "enum", "threading"
    }

    third_party_imports = sorted([
        imp for imp in all_imports 
        if imp not in local_modules and imp not in std_libs and not imp.startswith("_")
    ])

    print("\n".join(third_party_imports))

if __name__ == "__main__":
    main()
