#!/usr/bin/env python3
"""
Script to add missing __init__.py files to L104 directories.
"""

import os
import json
from pathlib import Path

def fix_missing_init_files():
    """Add __init__.py files to directories that are missing them."""
    
    # Load the analysis results
    with open('l104_analysis.json', 'r') as f:
        data = json.load(f)
    
    missing_init = data.get('missing_init_files', [])
    
    if not missing_init:
        print("No missing __init__.py files found!")
        return
    
    print(f"Found {len(missing_init)} directories missing __init__.py files")
    print("\nCreating __init__.py files...")
    
    created = 0
    skipped = 0
    
    for dir_path in missing_init:
        init_file = Path(dir_path) / "__init__.py"
        
        if init_file.exists():
            print(f"  ✓ {dir_path}/__init__.py already exists")
            skipped += 1
            continue
        
        # Create directory if it doesn't exist
        init_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py with basic content
        content = f'''"""
{l104_package_name(dir_path)} - L104 Module

Auto-generated __init__.py file for package structure.
"""

__version__ = "1.0.0"
__author__ = "L104 Autonomous System"

# Package metadata
PACKAGE_NAME = "{dir_path}"
MODULE_TYPE = "l104_core" if "core" in dir_path else "l104_module"

# Import key components
try:
    # Add main exports here
    pass
except ImportError:
    # Graceful degradation
    pass

print(f"  + Created {init_file}")
created += 1
    
    print(f"\nSummary:")
    print(f"  Created: {created} __init__.py files")
    print(f"  Skipped: {skipped} (already exist)")
    print(f"  Total: {len(missing_init)} directories processed")

def l104_package_name(dir_path):
    """Generate a friendly package name from directory path."""
    name = dir_path.replace('l104_', '').replace('_', ' ').title()
    if '/' in name:
        # Handle subpackages
        parts = name.split('/')
        name = f"{parts[0]} - {parts[1]}"
    return f"L104 {name}"

if __name__ == '__main__':
    fix_missing_init_files()