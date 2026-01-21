#!/usr/bin/env python3
"""
Quick access script for stable kernel and GitHub operations.

Usage:
    python3 kernel_ops.py verify        # Verify kernel integrity
    python3 kernel_ops.py sync          # Sync to GitHub
    python3 kernel_ops.py pull          # Pull from GitHub
    python3 kernel_ops.py status        # Show status
    python3 kernel_ops.py const NAME    # Get constant value
    python3 kernel_ops.py algo NAME     # Get algorithm info
"""

import sys
from l104_stable_kernel import stable_kernel
from l104_github_kernel_bridge import github_bridge, auto_sync


def verify():
    """Verify kernel integrity."""
    print("\n🔬 VERIFYING KERNEL INTEGRITY...")
    results = stable_kernel.constants.verify_all()
    
    for name, passed in results.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {name}")
    
    all_passed = all(results.values())
    print(f"\n{'✓' if all_passed else '✗'} Overall: {'PASSED' if all_passed else 'FAILED'}")
    return 0 if all_passed else 1


def sync():
    """Sync kernel to GitHub."""
    result = github_bridge.sync_kernel_to_github()
    return 0 if result['success'] else 1


def pull():
    """Pull from GitHub."""
    result = github_bridge.pull_from_github()
    return 0 if result['success'] else 1


def status():
    """Show kernel and GitHub status."""
    github_bridge.print_status()
    return 0


def get_constant(name):
    """Get constant value."""
    value = stable_kernel.get_constant(name)
    if value is not None:
        print(f"{name}: {value}")
        return 0
    else:
        print(f"✗ Constant '{name}' not found")
        return 1


def get_algorithm(name):
    """Get algorithm info."""
    algo = stable_kernel.get_algorithm(name)
    if algo:
        print(f"\nAlgorithm: {algo.name}")
        print(f"Formula: {algo.formula}")
        print(f"Description: {algo.description}")
        print(f"Complexity: {algo.complexity}")
        print(f"Resonance: {algo.resonance}")
        print(f"Entropy: {algo.entropy}")
        return 0
    else:
        print(f"✗ Algorithm '{name}' not found")
        print(f"Available: {', '.join(stable_kernel.algorithms.list_algorithms())}")
        return 1


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    
    command = sys.argv[1].lower()
    
    if command == 'verify':
        return verify()
    elif command == 'sync':
        return sync()
    elif command == 'pull':
        return pull()
    elif command == 'status':
        return status()
    elif command == 'const':
        if len(sys.argv) < 3:
            print("Usage: kernel_ops.py const NAME")
            return 1
        return get_constant(sys.argv[2])
    elif command == 'algo':
        if len(sys.argv) < 3:
            print("Usage: kernel_ops.py algo NAME")
            return 1
        return get_algorithm(sys.argv[2])
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        return 1


if __name__ == "__main__":
    sys.exit(main())
