#!/usr/bin/env python3
import os
import json
from collections import defaultdict
from pathlib import Path

def analyze_structure():
    root = Path(".")
    exclude = {'.venv', '__pycache__', '.git', '.pytest_cache', 'node_modules', '.mcp'}
    
    # Find all Python files
    python_files = []
    for path in root.rglob("*.py"):
        # Skip excluded directories
        if any(part in exclude for part in path.parts):
            continue
        # Skip hidden directories
        if any(part.startswith('.') and part not in {'.', '..'} for part in path.parts):
            continue
        python_files.append(path)
    
    print(f"Found {len(python_files)} Python files")
    
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
        if any(part in exclude for part in path.parts):
            continue
        rel_path = path.parent.relative_to(root)
        if str(rel_path) != '.':
            packages.append(str(rel_path))
    
    print(f"\nFound {len(packages)} Python packages")
    
    # Print summary
    print("\n" + "=" * 80)
    print("CODEBASE STRUCTURE ANALYSIS")
    print("=" * 80)
    
    total_lines = sum(stats['lines'] for stats in dir_stats.values())
    total_files = len(python_files)
    
    print(f"\nTotal: {total_files:,} files, {total_lines:,} lines")
    
    # Top directories
    print("\nTop 30 directories by file count:")
    print("-" * 80)
    sorted_dirs = sorted(dir_stats.items(), key=lambda x: x[1]['files'], reverse=True)
    
    for i, (dir_name, stats) in enumerate(sorted_dirs[:30], 1):
        print(f"{i:2}. {dir_name:40} {stats['files']:4} files  {stats['lines']:8,} lines  {stats['size']/1024/1024:6.1f} MB")
    
    # Packages
    print("\n" + "=" * 80)
    print("PYTHON PACKAGES:")
    print("=" * 80)
    
    for i, package in enumerate(sorted(packages)[:50], 1):
        print(f"{i:2}. {package}")
    
    if len(packages) > 50:
        print(f"\n... and {len(packages) - 50} more packages")
    
    # Save results
    results = {
        'total_files': total_files,
        'total_lines': total_lines,
        'packages': sorted(packages),
        'directory_stats': {k: v for k, v in sorted_dirs}
    }
    
    with open('structure_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed analysis saved to: structure_analysis.json")
    
    return results

if __name__ == '__main__':
    analyze_structure()