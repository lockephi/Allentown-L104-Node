#!/usr/bin/env python3
import os
import sys
import ast
from pathlib import Path
from collections import defaultdict
import json

def analyze_codebase():
    print("L104 Codebase Analysis")
    print("=" * 80)
    
    # Configuration
    root_dir = "."
    exclude_dirs = {'.venv', '__pycache__', '.git', '.pytest_cache', 'node_modules', '.mcp', '.l104_backups'}
    
    # Find Python files
    python_files = []
    for root, dirs, files in os.walk(root_dir):
        # Filter directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith('.l104_')]
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(Path(root) / file)
    
    print(f"Total Python files found: {len(python_files)}")
    
    # Find packages (directories with __init__.py)
    packages = []
    for root, dirs, files in os.walk(root_dir):
        if '__init__.py' in files:
            rel_path = os.path.relpath(root, root_dir)
            if rel_path != '.':
                packages.append(rel_path)
    
    print(f"Python packages found: {len(packages)}")
    
    # Analyze files
    stats = {
        'total_files': len(python_files),
        'total_lines': 0,
        'total_functions': 0,
        'total_classes': 0,
        'syntax_errors': [],
        'import_issues': defaultdict(list),
        'by_directory': defaultdict(lambda: {'files': 0, 'lines': 0, 'functions': 0, 'classes': 0}),
        'packages': packages
    }
    
    print("\nAnalyzing files...")
    for i, filepath in enumerate(python_files):
        if i % 500 == 0 and i > 0:
            print(f"  Processed {i}/{len(python_files)} files...")
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Count lines
            lines = content.split('\n')
            stats['total_lines'] += len(lines)
            
            # Parse AST
            try:
                tree = ast.parse(content)
                
                # Count functions and classes
                functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
                classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
                
                stats['total_functions'] += len(functions)
                stats['total_classes'] += len(classes)
                
                # Check imports
                imports = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        module = node.module or ''
                        for alias in node.names:
                            imports.append(f"{module}.{alias.name}" if module else alias.name)
                
                # Simple import check - look for missing __init__.py in local packages
                for imp in imports:
                    if '.' in imp:
                        first_part = imp.split('.')[0]
                        package_dir = Path(root_dir) / first_part
                        if package_dir.exists() and package_dir.is_dir():
                            init_file = package_dir / '__init__.py'
                            if not init_file.exists():
                                stats['import_issues'][str(filepath)].append(f"Package '{first_part}' missing __init__.py")
                
            except SyntaxError as e:
                stats['syntax_errors'].append({
                    'file': str(filepath),
                    'error': str(e),
                    'line': e.lineno if hasattr(e, 'lineno') else None
                })
            
            # Update directory stats
            rel_path = os.path.relpath(filepath, root_dir)
            dir_name = os.path.dirname(rel_path)
            if dir_name == '':
                dir_name = '.'
            
            stats['by_directory'][dir_name]['files'] += 1
            stats['by_directory'][dir_name]['lines'] += len(lines)
            stats['by_directory'][dir_name]['functions'] += len(functions) if 'functions' in locals() else 0
            stats['by_directory'][dir_name]['classes'] += len(classes) if 'classes' in locals() else 0
            
        except Exception as e:
            print(f"Error analyzing {filepath}: {e}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"\nTotal Python files: {stats['total_files']:,}")
    print(f"Total lines of code: {stats['total_lines']:,}")
    print(f"Total functions: {stats['total_functions']:,}")
    print(f"Total classes: {stats['total_classes']:,}")
    print(f"Python packages: {len(packages)}")
    
    if stats['syntax_errors']:
        print(f"\nSyntax errors found: {len(stats['syntax_errors'])}")
        for error in stats['syntax_errors'][:5]:
            print(f"  {error['file']}: {error['error']}")
    
    if stats['import_issues']:
        total_issues = sum(len(issues) for issues in stats['import_issues'].values())
        print(f"\nImport issues found: {total_issues}")
        for file, issues in list(stats['import_issues'].items())[:5]:
            print(f"  {file}: {len(issues)} issues")
    
    # Top directories
    print("\n" + "=" * 80)
    print("TOP 20 DIRECTORIES BY FILE COUNT")
    print("=" * 80)
    sorted_dirs = sorted(stats['by_directory'].items(), key=lambda x: x[1]['files'], reverse=True)
    for dir_name, dir_stats in sorted_dirs[:20]:
        print(f"{dir_name:40} {dir_stats['files']:4} files  {dir_stats['lines']:8,} lines  "
              f"{dir_stats['functions']:6} funcs  {dir_stats['classes']:4} classes")
    
    # Packages list
    print("\n" + "=" * 80)
    print("PYTHON PACKAGES (first 50)")
    print("=" * 80)
    for package in sorted(packages)[:50]:
        print(f"  {package}")
    
    # Save detailed report
    with open('codebase_stats.json', 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    print(f"\nDetailed statistics saved to: codebase_stats.json")
    
    return stats

if __name__ == '__main__':
    analyze_codebase()