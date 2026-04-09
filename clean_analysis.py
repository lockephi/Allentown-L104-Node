#!/usr/bin/env python3
import os
import json
from collections import defaultdict
from pathlib import Path

def clean_analysis():
    root = Path(".")
    exclude = {
        '.venv', '__pycache__', '.git', '.pytest_cache', 'node_modules', '.mcp',
        '.l104_backups', '.l104_circuits', '.l104_mailbox', '.l104_save_states',
        '.soul_backups', '.soul_state', '.quantum_storage', '.unified_evolution',
        '.vscode', '.devcontainer', '.kernel_build', '.claude'
    }
    
    # Also exclude paths containing "venv" or "site-packages"
    def should_exclude(path):
        parts = path.parts
        for part in parts:
            if part in exclude:
                return True
            if 'venv' in part.lower() or 'site-packages' in part.lower():
                return True
            if part.startswith('.') and part not in {'.', '..'}:
                return True
        return False
    
    # Find all Python files
    python_files = []
    for path in root.rglob("*.py"):
        if should_exclude(path):
            continue
        python_files.append(path)
    
    print(f"Found {len(python_files)} Python files (excluding virtual envs and hidden dirs)")
    
    # Group by directory
    dir_stats = defaultdict(lambda: {'files': 0, 'lines': 0, 'size': 0})
    
    for filepath in python_files:
        try:
            # Count lines
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # Get directory
            rel_path = filepath.relative_to(root)
            dir_name = str(rel_path.parent)
            if dir_name == '.':
                dir_name = 'root'
            
            # Update stats
            dir_stats[dir_name]['files'] += 1
            dir_stats[dir_name]['lines'] += len(lines)
            dir_stats[dir_name]['size'] += filepath.stat().st_size
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
    
    # Find packages (directories with __init__.py)
    packages = []
    for path in root.rglob("__init__.py"):
        if should_exclude(path):
            continue
        rel_path = path.parent.relative_to(root)
        if str(rel_path) != '.':
            packages.append(str(rel_path))
    
    print(f"\nFound {len(packages)} Python packages")
    
    # Print summary
    print("\n" + "=" * 80)
    print("L104 CODEBASE ANALYSIS - CLEANED")
    print("=" * 80)
    
    total_lines = sum(stats['lines'] for stats in dir_stats.values())
    total_files = len(python_files)
    
    print(f"\nTotal: {total_files:,} files, {total_lines:,} lines")
    
    # Top directories
    print("\nTop 40 directories by file count:")
    print("-" * 80)
    sorted_dirs = sorted(dir_stats.items(), key=lambda x: x[1]['files'], reverse=True)
    
    for i, (dir_name, stats) in enumerate(sorted_dirs[:40], 1):
        avg_lines = stats['lines'] / stats['files'] if stats['files'] > 0 else 0
        print(f"{i:2}. {dir_name:40} {stats['files']:4} files  {stats['lines']:8,} lines  {avg_lines:6.1f} avg")
    
    # L104 packages only
    l104_packages = [p for p in packages if p.startswith('l104_') or p == 'routers' or p == 'tests']
    
    print("\n" + "=" * 80)
    print("L104 PROJECT PACKAGES:")
    print("=" * 80)
    
    for i, package in enumerate(sorted(l104_packages), 1):
        # Count files in package
        package_path = root / package
        package_files = 0
        package_lines = 0
        if package_path.exists():
            for py_file in package_path.rglob("*.py"):
                if not should_exclude(py_file):
                    package_files += 1
                    try:
                        with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                            package_lines += len(f.readlines())
                    except:
                        pass
        
        print(f"{i:2}. {package:45} {package_files:3} files  {package_lines:6,} lines")
    
    # Check for missing __init__.py in l104 directories
    print("\n" + "=" * 80)
    print("CHECKING FOR MISSING __init__.py IN L104 DIRECTORIES:")
    print("=" * 80)
    
    l104_dirs = []
    for path in root.rglob("*"):
        if path.is_dir() and path.name.startswith('l104_') and not should_exclude(path):
            rel_path = path.relative_to(root)
            l104_dirs.append(str(rel_path))
    
    missing_init = []
    for l104_dir in sorted(l104_dirs):
        init_file = root / l104_dir / "__init__.py"
        if not init_file.exists():
            missing_init.append(l104_dir)
    
    if missing_init:
        print(f"\nFound {len(missing_init)} L104 directories missing __init__.py:")
        for dir_name in missing_init[:20]:
            print(f"  {dir_name}")
        if len(missing_init) > 20:
            print(f"  ... and {len(missing_init) - 20} more")
    else:
        print("\nAll L104 directories have __init__.py files!")
    
    # Save results
    results = {
        'total_files': total_files,
        'total_lines': total_lines,
        'l104_packages': sorted(l104_packages),
        'all_packages': sorted(packages),
        'directory_stats': {k: v for k, v in sorted_dirs},
        'missing_init_files': missing_init
    }
    
    with open('l104_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed analysis saved to: l104_analysis.json")
    
    return results

if __name__ == '__main__':
    clean_analysis()