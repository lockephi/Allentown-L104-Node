#!/usr/bin/env python3
"""
Verify that all L104 directories now have __init__.py files.
"""

import os
from pathlib import Path

def verify_init_files():
    """Check that all L104 directories have __init__.py files."""
    
    root = Path(".")
    exclude = {
        '.venv', '__pycache__', '.git', '.pytest_cache', 'node_modules', '.mcp',
        '.l104_backups', '.l104_circuits', '.l104_mailbox', '.l104_save_states',
        '.soul_backups', '.soul_state', '.quantum_storage', '.unified_evolution',
        '.vscode', '.devcontainer', '.kernel_build', '.claude'
    }
    
    # Find all L104 directories
    l104_dirs = []
    for path in root.rglob("*"):
        if path.is_dir() and path.name.startswith('l104_') and not any(part in exclude for part in path.parts):
            rel_path = path.relative_to(root)
            l104_dirs.append(str(rel_path))
    
    print(f"Found {len(l104_dirs)} L104 directories")
    
    # Check for __init__.py
    missing_init = []
    has_init = []
    
    for dir_path in sorted(l104_dirs):
        init_file = root / dir_path / "__init__.py"
        if init_file.exists():
            has_init.append(dir_path)
        else:
            missing_init.append(dir_path)
    
    print("\n" + "=" * 80)
    print("VERIFICATION RESULTS")
    print("=" * 80)
    
    print(f"\n✅ Directories WITH __init__.py: {len(has_init)}")
    if len(has_init) <= 20:
        for dir_name in sorted(has_init):
            print(f"  ✓ {dir_name}")
    else:
        for dir_name in sorted(has_init)[:20]:
            print(f"  ✓ {dir_name}")
        print(f"  ... and {len(has_init) - 20} more")
    
    print(f"\n❌ Directories MISSING __init__.py: {len(missing_init)}")
    if missing_init:
        for dir_name in sorted(missing_init):
            print(f"  ✗ {dir_name}")
        
        print(f"\n⚠️  Recommendation: Create __init__.py files for these directories.")
    else:
        print("\n🎉 SUCCESS! All L104 directories have __init__.py files!")
        print("   The package structure is now complete.")
    
    # Quick import test for key packages
    print("\n" + "=" * 80)
    print("QUICK IMPORT TEST")
    print("=" * 80)
    
    key_packages = [
        'l104_asi',
        'l104_quantum_engine',
        'l104_code_engine',
        'l104_intellect',
        'l104_server',
        'routers'
    ]
    
    import sys
    import importlib.util
    
    sys.path.insert(0, str(root))
    
    for pkg in key_packages:
        pkg_path = root / pkg
        if pkg_path.exists():
            init_file = pkg_path / "__init__.py"
            if init_file.exists():
                try:
                    spec = importlib.util.spec_from_file_location(pkg, init_file)
                    if spec:
                        print(f"✓ {pkg}: Can be imported (spec found)")
                    else:
                        print(f"⚠️  {pkg}: No import spec (but __init__.py exists)")
                except Exception as e:
                    print(f"✗ {pkg}: Import error: {e}")
            else:
                print(f"✗ {pkg}: Missing __init__.py")
        else:
            print(f"⚠️  {pkg}: Directory not found")
    
    return len(missing_init) == 0

if __name__ == '__main__':
    success = verify_init_files()
    exit(0 if success else 1)