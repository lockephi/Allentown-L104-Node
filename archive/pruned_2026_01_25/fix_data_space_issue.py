# L104_GOD_CODE_ALIGNED: 527.5184818492537

import os
import shutil
import subprocess
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════


def get_dir_size(path):
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    except (PermissionError, FileNotFoundError):
        pass
    return total

def format_size(size):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.2f}{unit}"
        size /= 1024
    return f"{size:.2f}TB"

def run_fix():
    print("🚀 Starting Data Space Fix (File-based Implementation)")
    workspace = Path("/workspaces/Allentown-L104-Node")

    # 1. Analyze
    print("\n📊 Analyzing large directories...")
    targets = [
        workspace / "node_modules",
        workspace / ".venv",
        workspace / ".venv-1",
        workspace / "__pycache__",
        workspace / ".git",
        workspace / "data",
        workspace / "archive",
        workspace / ".l104_backups"
    ]

    for target in targets:
        if target.exists():
            size = get_dir_size(target)
            print(f"   {target.name}: {format_size(size)}")

    # 2. Cleanup __pycache__
    print("\n🧹 Cleaning up __pycache__ files...")
    count = 0
    for p in workspace.rglob("__pycache__"):
        if p.is_dir():
            shutil.rmtree(p)
            count += 1
    print(f"   Removed {count} __pycache__ directories.")

    # 3. Aggressive Git GC (using os.system as backup)
    print("\n⚙️ Running Git Garbage Collection...")
    os.system(f"cd {workspace} && git gc --aggressive --prune=now")

    # 4. Cleanup .npm and pip cache if possible (approximated)
    print("\n🧹 Cleaning up external caches...")
    npm_cache = Path.home() / ".npm"
    if npm_cache.exists():
        print(f"   Removing {npm_cache}...")
        os.system(f"rm -rf {npm_cache}")

    pip_cache = Path.home() / ".cache/pip"
    if pip_cache.exists():
        print(f"   Removing {pip_cache}...")
        os.system(f"rm -rf {pip_cache}")

    print("\n✅ Data space optimization complete.")

if __name__ == "__main__":
    run_fix()
