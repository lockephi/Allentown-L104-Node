#!/usr/bin/env python3
"""
L104 MainView Splitter - Reduces sourcekit-lsp CPU by splitting large files.

Usage:
    python Scripts/split_mainview.py --analyze   # Show what would be split
    python Scripts/split_mainview.py --split      # Create extension files
"""

from pathlib import Path
import re
import sys

# Configuration
MAX_FILE_LINES = 2000  # Target max lines per file
SOURCE_FILE = "Sources/L104v2/TheHeart/H11_MainView.swift"
OUTPUT_DIR = "Sources/L104v2/TheHeart/MainViewExtensions"

# Groups of functions to extract into separate files
EXTENSION_GROUPS = {
    "Chat": {
        "prefixes": ["createChatView", "appendChat", "updateChat", "streamResponse", "sendHelp", "loadHistory"],
        "description": "Chat UI components"
    },
    "ASI": {
        "prefixes": ["createASIView", "createASIDashboard", "updateASI"],
        "description": "ASI dashboard components"
    },
    "Memory": {
        "prefixes": ["createMemoryView", "updateMemory"],
        "description": "Memory view components"
    },
    "Network": {
        "prefixes": ["createNetworkView", "updateNetwork", "createNetworkControl"],
        "description": "Network view components"
    },
    "Debug": {
        "prefixes": ["createDebugConsole", "updateDebugConsole", "appendDebug"],
        "description": "Debug console components"
    },
    "Science": {
        "prefixes": ["createScienceView", "updateScience", "appendScienceLog", "createScienceMetric"],
        "description": "Science engine UI"
    },
    "UnifiedField": {
        "prefixes": ["createUnifiedField", "updateUF", "appendUFLog"],
        "description": "Unified field view"
    },
    "Hardware": {
        "prefixes": ["createHardwareView", "updateHardware"],
        "description": "Hardware view components"
    },
    "Upgrades": {
        "prefixes": ["createUpgradesView", "updateUpgrades"],
        "description": "Upgrades view"
    },
    "GateEnvironment": {
        "prefixes": ["createGateEnvironment", "updateGate"],
        "description": "Gate environment view"
    },
    "System": {
        "prefixes": ["createSystemView", "updateSystem"],
        "description": "System view"
    },
    "QuickBar": {
        "prefixes": ["createQuickBar", "navigateToTab"],
        "description": "Quick bar navigation"
    },
    "Helpers": {
        "prefixes": ["createPanel", "addLabel", "btn(", "loadWelcome"],
        "description": "UI helper functions"
    }
}


def analyze_file(content: str) -> dict:
    """Analyze file and return function groups."""
    lines = content.split('\n')
    functions = []
    current_func = None
    brace_depth = 0
    func_start = 0

    for i, line in enumerate(lines):
        # Detect function start
        match = re.match(r'^\s*(?:public |private |internal |fileprivate |open )?(?:override |static )?func\s+(\w+)', line)
        if match:
            if current_func:
                functions.append({
                    'name': current_func['name'],
                    'start': current_func['start'],
                    'end': i - 1,
                    'lines': i - current_func['start']
                })
            current_func = {'name': match.group(1), 'start': i}
            brace_depth = 0

        # Track braces
        brace_depth += line.count('{') - line.count('}')

    # Last function
    if current_func:
        functions.append({
            'name': current_func['name'],
            'start': current_func['start'],
            'end': len(lines) - 1,
            'lines': len(lines) - current_func['start']
        })

    return {
        'total_lines': len(lines),
        'function_count': len(functions),
        'functions': functions,
        'groups': categorize_functions(functions)
    }


def categorize_functions(functions: list) -> dict:
    """Categorize functions into extension groups."""
    groups = {name: [] for name in EXTENSION_GROUPS}

    for func in functions:
        for group_name, group_info in EXTENSION_GROUPS.items():
            for prefix in group_info['prefixes']:
                if func['name'].startswith(prefix):
                    groups[group_name].append(func)
                    break

    return groups


def print_analysis(analysis: dict):
    """Print analysis results."""
    print(f"=== H11_MainView.swift Analysis ===")
    print(f"Total lines: {analysis['total_lines']}")
    print(f"Function count: {analysis['function_count']}")
    print()
    print("=== Extension Groups ===")

    for group_name, group_info in EXTENSION_GROUPS.items():
        funcs = analysis['groups'].get(group_name, [])
        if funcs:
            total_lines = sum(f['lines'] for f in funcs)
            print(f"\n{group_name} ({group_info['description']}): {len(funcs)} functions, ~{total_lines} lines")
            for f in funcs[:5]:  # Show first 5
                print(f"  - {f['name']} ({f['lines']} lines)")
            if len(funcs) > 5:
                print(f"  ... and {len(funcs) - 5} more")


def split_file(content: str, output_dir: Path):
    """Split file into extensions."""
    lines = content.split('\n')
    analysis = analyze_file(content)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract class declaration and properties
    class_start = 0
    class_end = 0
    for i, line in enumerate(lines):
        if 'class L104MainView' in line:
            class_start = i
        if class_start and '// ───' in line and i > class_start + 50:
            class_end = i
            break

    # Create extension files
    for group_name, group_info in EXTENSION_GROUPS.items():
        funcs = analysis['groups'].get(group_name, [])
        if not funcs:
            continue

        # Build extension content
        extension_lines = [
            f"// L104MainView+{group_name}.swift",
            f"// Extension for {group_info['description']}",
            f"// Auto-split from H11_MainView.swift",
            f"//",
            f"// NOTE: This file is an extension of L104MainView.",
            f"// SourceKit optimization: Reduced from 8912 lines to <{MAX_FILE_LINES} per file.",
            f"",
            f"import AppKit",
            f"import Accelerate",
            f"import Metal",
            f"",
            f"extension L104MainView {{",
        ]

        # Add function implementations
        for func in funcs:
            func_lines = lines[func['start']:func['end']+1]
            extension_lines.extend(func_lines)
            extension_lines.append("")

        extension_lines.append("}")

        # Write file
        extension_path = output_dir / f"H11_MainView+{group_name}.swift"
        extension_path.write_text('\n'.join(extension_lines))
        print(f"Created: {extension_path.name} ({len(funcs)} functions)")

    print(f"\nDone! Created extension files in {output_dir}")
    print("\nTo complete the split:")
    print("1. Review extension files")
    print("2. Add to Package.swift")
    print("3. Remove functions from H11_MainView.swift")


def main():
    """TODO: Document main."""
    script_dir = Path(__file__).parent.parent  # L104SwiftApp directory
    source_path = script_dir / SOURCE_FILE
    output_dir = script_dir / OUTPUT_DIR

    if not source_path.exists():
        print(f"Error: Source file not found: {source_path}")
        sys.exit(1)

    content = source_path.read_text()

    if '--analyze' in sys.argv:
        analysis = analyze_file(content)
        print_analysis(analysis)
    elif '--split' in sys.argv:
        split_file(content, output_dir)
    else:
        print(__doc__)
        print("\nOptions:")
        print("  --analyze   Analyze file and show extension groups")
        print("  --split    Create extension files")


if __name__ == '__main__':
    main()