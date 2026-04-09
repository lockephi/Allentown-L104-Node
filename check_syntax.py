#!/usr/bin/env python3
import os
import sys
import ast
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

def check_file_syntax(filepath):
    """Check if a Python file has syntax errors."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Try to parse the AST
        ast.parse(content)
        return filepath, None
    except SyntaxError as e:
        return filepath, {
            'error': str(e),
            'lineno': e.lineno,
            'offset': e.offset,
            'text': e.text
        }
    except Exception as e:
        return filepath, {
            'error': f"Unexpected error: {str(e)}",
            'type': type(e).__name__
        }

def main():
    root_dir = "."
    
    # Find Python files
    python_files = []
    exclude_dirs = {'.venv', '__pycache__', '.git', '.pytest_cache', 'node_modules'}
    
    for root, dirs, files in os.walk(root_dir):
        # Filter directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith('.')]
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    print(f"Found {len(python_files)} Python files")
    print("Checking syntax...")
    
    syntax_errors = []
    
    # Use ThreadPoolExecutor for parallel checking
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_file = {executor.submit(check_file_syntax, f): f for f in python_files}
        
        for i, future in enumerate(as_completed(future_to_file)):
            filepath = future_to_file[future]
            try:
                filepath, error = future.result()
                if error:
                    syntax_errors.append((filepath, error))
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
            
            if i % 100 == 0:
                print(f"  Checked {i}/{len(python_files)} files...")
    
    print(f"\nSyntax check complete. Found {len(syntax_errors)} files with syntax errors.")
    
    if syntax_errors:
        print("\nFiles with syntax errors:")
        for filepath, error in syntax_errors[:20]:  # Show first 20
            print(f"\n{filepath}:")
            print(f"  Error: {error['error']}")
            if 'lineno' in error:
                print(f"  Line: {error['lineno']}, Offset: {error['offset']}")
            if 'text' in error:
                print(f"  Context: {error['text']}")
        
        if len(syntax_errors) > 20:
            print(f"\n... and {len(syntax_errors) - 20} more files with syntax errors")
        
        # Save to file
        with open('syntax_errors.txt', 'w') as f:
            f.write(f"Found {len(syntax_errors)} files with syntax errors:\n\n")
            for filepath, error in syntax_errors:
                f.write(f"{filepath}:\n")
                f.write(f"  Error: {error['error']}\n")
                if 'lineno' in error:
                    f.write(f"  Line: {error['lineno']}, Offset: {error['offset']}\n")
                f.write("\n")
        
        print(f"\nFull list saved to: syntax_errors.txt")
    
    return len(syntax_errors)

if __name__ == '__main__':
    sys.exit(main())