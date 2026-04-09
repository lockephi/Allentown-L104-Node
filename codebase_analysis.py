#!/usr/bin/env python3
"""
Comprehensive L104 codebase analysis script.
Scans the codebase to identify packages, modules, lines of code, and import errors.
"""

import os
import sys
import ast
import subprocess
from pathlib import Path
from collections import defaultdict, Counter
import importlib.util
import traceback

def find_python_files(root_dir, exclude_dirs=None):
    """Find all Python files in the codebase, excluding specified directories."""
    if exclude_dirs is None:
        exclude_dirs = {'.venv', '__pycache__', '.git', '.pytest_cache', 'node_modules'}
    
    python_files = []
    for root, dirs, files in os.walk(root_dir):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(Path(root) / file)
    
    return python_files

def analyze_file(filepath):
    """Analyze a single Python file."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Count lines
        lines = content.split('\n')
        total_lines = len(lines)
        non_empty_lines = len([l for l in lines if l.strip()])
        
        # Parse AST to get imports and structure
        try:
            tree = ast.parse(content)
            
            # Count imports
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        imports.append(f"{module}.{alias.name}" if module else alias.name)
            
            # Count functions and classes
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            
            return {
                'path': str(filepath),
                'total_lines': total_lines,
                'non_empty_lines': non_empty_lines,
                'imports': imports,
                'function_count': len(functions),
                'class_count': len(classes),
                'parse_success': True
            }
        except SyntaxError as e:
            return {
                'path': str(filepath),
                'total_lines': total_lines,
                'non_empty_lines': non_empty_lines,
                'imports': [],
                'function_count': 0,
                'class_count': 0,
                'parse_success': False,
                'syntax_error': str(e)
            }
            
    except Exception as e:
        return {
            'path': str(filepath),
            'error': str(e),
            'parse_success': False
        }

def check_imports(file_analysis, root_dir):
    """Check if imports in a file can be resolved."""
    import_errors = []
    
    for imp in file_analysis['imports']:
        # Skip standard library imports (simple heuristic)
        if '.' not in imp and len(imp.split('.')) == 1:
            # Could be standard library or local module
            continue
        
        # Try to find the module
        module_parts = imp.split('.')
        module_name = module_parts[0]
        
        # Check if it's a local package
        local_package_path = Path(root_dir) / module_name
        if local_package_path.exists() and local_package_path.is_dir():
            # Check for __init__.py
            init_file = local_package_path / '__init__.py'
            if not init_file.exists():
                import_errors.append(f"Package '{module_name}' missing __init__.py")
    
    return import_errors

def analyze_codebase(root_dir='.'):
    """Main analysis function."""
    print(f"Analyzing codebase at: {os.path.abspath(root_dir)}")
    print("=" * 80)
    
    # Find all Python files
    print("Scanning for Python files...")
    python_files = find_python_files(root_dir)
    print(f"Found {len(python_files)} Python files")
    
    # Analyze files
    print("\nAnalyzing files...")
    file_analyses = []
    import_errors_by_file = defaultdict(list)
    syntax_errors = []
    
    for i, filepath in enumerate(python_files):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(python_files)} files...")
        
        analysis = analyze_file(filepath)
        file_analyses.append(analysis)
        
        if not analysis.get('parse_success', False):
            if 'syntax_error' in analysis:
                syntax_errors.append((analysis['path'], analysis['syntax_error']))
        
        # Check imports
        if analysis.get('parse_success', False):
            errors = check_imports(analysis, root_dir)
            if errors:
                import_errors_by_file[analysis['path']].extend(errors)
    
    # Calculate statistics
    total_lines = sum(a.get('total_lines', 0) for a in file_analyses)
    total_non_empty = sum(a.get('non_empty_lines', 0) for a in file_analyses)
    total_functions = sum(a.get('function_count', 0) for a in file_analyses)
    total_classes = sum(a.get('class_count', 0) for a in file_analyses)
    
    # Find packages (directories with __init__.py)
    packages = []
    for root, dirs, files in os.walk(root_dir):
        if '__init__.py' in files:
            rel_path = os.path.relpath(root, root_dir)
            if rel_path != '.':
                packages.append(rel_path)
    
    # Group by directory
    dir_stats = defaultdict(lambda: {'files': 0, 'lines': 0, 'functions': 0, 'classes': 0})
    for analysis in file_analyses:
        if 'path' in analysis:
            rel_path = os.path.relpath(analysis['path'], root_dir)
            dir_name = os.path.dirname(rel_path)
            if dir_name == '':
                dir_name = '.'
            
            dir_stats[dir_name]['files'] += 1
            dir_stats[dir_name]['lines'] += analysis.get('total_lines', 0)
            dir_stats[dir_name]['functions'] += analysis.get('function_count', 0)
            dir_stats[dir_name]['classes'] += analysis.get('class_count', 0)
    
    # Print summary
    print("\n" + "=" * 80)
    print("CODEBASE ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"\nTotal Python files: {len(python_files)}")
    print(f"Total lines of code: {total_lines:,}")
    print(f"Total non-empty lines: {total_non_empty:,}")
    print(f"Total functions: {total_functions:,}")
    print(f"Total classes: {total_classes:,}")
    print(f"Python packages found: {len(packages)}")
    
    if syntax_errors:
        print(f"\nSyntax errors found: {len(syntax_errors)}")
        for file, error in syntax_errors[:10]:  # Show first 10
            print(f"  {file}: {error}")
        if len(syntax_errors) > 10:
            print(f"  ... and {len(syntax_errors) - 10} more")
    
    if import_errors_by_file:
        print(f"\nImport issues found: {sum(len(errors) for errors in import_errors_by_file.values())}")
        for file, errors in list(import_errors_by_file.items())[:10]:  # Show first 10 files
            print(f"  {file}:")
            for error in errors[:5]:  # Show first 5 errors per file
                print(f"    - {error}")
            if len(errors) > 5:
                print(f"    ... and {len(errors) - 5} more")
    
    # Print top directories by file count
    print("\n" + "=" * 80)
    print("TOP DIRECTORIES BY FILE COUNT")
    print("=" * 80)
    sorted_dirs = sorted(dir_stats.items(), key=lambda x: x[1]['files'], reverse=True)
    for dir_name, stats in sorted_dirs[:20]:
        print(f"{dir_name}: {stats['files']} files, {stats['lines']:,} lines, "
              f"{stats['functions']} functions, {stats['classes']} classes")
    
    # Print packages
    print("\n" + "=" * 80)
    print("PYTHON PACKAGES FOUND")
    print("=" * 80)
    for package in sorted(packages)[:50]:  # Show first 50 packages
        print(f"  {package}")
    if len(packages) > 50:
        print(f"  ... and {len(packages) - 50} more")
    
    # Save detailed report
    with open('codebase_analysis_report.txt', 'w') as f:
        f.write("L104 Codebase Analysis Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total Python files: {len(python_files)}\n")
        f.write(f"Total lines of code: {total_lines:,}\n")
        f.write(f"Total functions: {total_functions:,}\n")
        f.write(f"Total classes: {total_classes:,}\n")
        f.write(f"Python packages: {len(packages)}\n\n")
        
        f.write("Packages:\n")
        for package in sorted(packages):
            f.write(f"  {package}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("Directory Statistics:\n")
        for dir_name, stats in sorted(dir_stats.items(), key=lambda x: x[1]['files'], reverse=True):
            f.write(f"\n{dir_name}:\n")
            f.write(f"  Files: {stats['files']}\n")
            f.write(f"  Lines: {stats['lines']:,}\n")
            f.write(f"  Functions: {stats['functions']}\n")
            f.write(f"  Classes: {stats['classes']}\n")
        
        if syntax_errors:
            f.write("\n" + "=" * 80 + "\n")
            f.write("Syntax Errors:\n")
            for file, error in syntax_errors:
                f.write(f"\n{file}:\n  {error}\n")
        
        if import_errors_by_file:
            f.write("\n" + "=" * 80 + "\n")
            f.write("Import Issues:\n")
            for file, errors in import_errors_by_file.items():
                f.write(f"\n{file}:\n")
                for error in errors:
                    f.write(f"  - {error}\n")
    
    print(f"\nDetailed report saved to: codebase_analysis_report.txt")
    
    return {
        'total_files': len(python_files),
        'total_lines': total_lines,
        'total_functions': total_functions,
        'total_classes': total_classes,
        'packages': packages,
        'dir_stats': dict(dir_stats),
        'syntax_errors': syntax_errors,
        'import_errors': dict(import_errors_by_file)
    }

if __name__ == '__main__':
    root_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    analyze_codebase(root_dir)