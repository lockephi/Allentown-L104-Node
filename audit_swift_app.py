#!/usr/bin/env python3
"""
L104v2 Swift App Audit - Logic Incompletions & Weak Connectivity Analysis
Uses code_engine to analyze Swift codebase
"""

import os
import sys
import re
import json
from pathlib import Path
from collections import defaultdict

# Add l104_code_engine to path
sys.path.insert(0, '/Users/carolalvarez/Applications/Allentown-L104-Node')

try:
    from l104_code_engine import code_engine
    print("✓ Code engine loaded")
except ImportError as e:
    print(f"✗ Code engine import failed: {e}")
    sys.exit(1)

# Configuration
SWIFT_SOURCES = Path('/Users/carolalvarez/Applications/Allentown-L104-Node/L104SwiftApp/Sources/L104v2')
OUTPUT_FILE = Path('/Users/carolalvarez/Applications/Allentown-L104-Node/swift_audit_report.json')

class SwiftAuditor:
    def __init__(self):
        self.issues = []
        self.connectivity_issues = []
        self.logic_gaps = []
        self.incomplete_implementations = []
        self.ui_weaknesses = []

    def audit_file(self, file_path: Path) -> dict:
        """Audit a single Swift file"""
        content = file_path.read_text(encoding='utf-8', errors='ignore')
        lines = content.split('\n')

        file_issues = {
            'path': str(file_path),
            'lines': len(lines),
            'issues': [],
            'connectivity_issues': [],
            'logic_gaps': [],
            'ui_issues': []
        }

        # Pattern analysis
        self._check_stub_implementations(content, file_path, file_issues)
        self._check_unconnected_buttons(content, file_path, file_issues)
        self._check_weak_references(content, file_path, file_issues)
        self._check_missing_error_handling(content, file_path, file_issues)
        self._check_async_gaps(content, file_path, file_issues)
        self._check_state_management(content, file_path, file_issues)

        return file_issues

    def _check_stub_implementations(self, content: str, file_path: Path, issues: dict):
        """Find stub/imcomplete implementations"""
        # Patterns for incomplete code
        stub_patterns = [
            (r'func \w+\([^)]*\)\s*->\s*\w+\s*\{\s*return\s+[^}]*\}', 'Returns placeholder value'),
            (r'func \w+\([^)]*\)\s*\{\s*//\s*TODO', 'TODO implementation'),
            (r'func \w+\([^)]*\)\s*\{\s*//\s*FIXME', 'FIXME implementation'),
            (r'var \w+:\s*\w+\s*\{\s*return\s+[^}]*\}', 'Computed property stub'),
            (r'fatalError\s*\(\s*"[^"]*not implemented', 'fatalError stub'),
            (r'\{\s*//\s*TODO:[^}]*\}', 'TODO block'),
            (r'pass\s*//\s*Placeholder', 'Python-style placeholder'),
            (r'body:\s*\{\s*[^}]*\}\s*//\s*Empty', 'Empty closure body'),
        ]

        for pattern, description in stub_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                snippet = content[max(0, match.start()-30):min(len(content), match.end()+30)]
                issues['logic_gaps'].append({
                    'line': line_num,
                    'type': 'stub_implementation',
                    'description': description,
                    'snippet': snippet[:100]
                })

    def _check_unconnected_buttons(self, content: str, file_path: Path, issues: dict):
        """Find buttons with weak or missing actions"""
        # SwiftUI Button patterns
        button_patterns = [
            # Button with empty action
            (r'Button\s*\(\s*action:\s*\{\s*\}\s*\)', 'Button with empty action closure'),
            # Button with just print
            (r'Button\s*\(\s*action:\s*\{\s*print\s*\([^}]*\}\s*\)', 'Button only prints, no real action'),
            # NavigationLink without destination
            (r'NavigationLink\s*\([^)]*\)\s*\{\s*\}', 'NavigationLink with empty content'),
            # Tab without proper tagging
            (r'\.tabItem\s*\{[^}]*\}\s*[^.]*\}(?!\s*\.tag)', 'TabItem without tag'),
        ]

        for pattern, description in button_patterns:
            matches = re.finditer(pattern, content, re.DOTALL)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                issues['connectivity_issues'].append({
                    'line': line_num,
                    'type': 'weak_button_connectivity',
                    'description': description,
                    'snippet': match.group()[:100]
                })

    def _check_weak_references(self, content: str, file_path: Path, issues: dict):
        """Find potential retain cycles and weak reference issues"""
        patterns = [
            # Self capture in async without weak
            (r'(Task|async|await)[^{]*\{[^}]*\bself\.', 'Strong self capture in async context'),
            # ObservableObject without weak in closure
            (r'\{[^}]*\[\s*self\s*\][^}]*observable', 'Strong self in closure'),
            # Completion handlers without weak
            (r'\{[^}]*completion[^}]*\[\s*self\s*\]', 'Strong self in completion handler'),
        ]

        for pattern, description in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                issues['logic_gaps'].append({
                    'line': line_num,
                    'type': 'memory_management',
                    'description': description,
                    'snippet': match.group()[:100]
                })

    def _check_missing_error_handling(self, content: str, file_path: Path, issues: dict):
        """Find throws/try without proper error handling"""
        # Find try! and try? without catch
        try_bang = re.finditer(r'try!', content)
        for match in try_bang:
            line_num = content[:match.start()].count('\n') + 1
            line = content.split('\n')[line_num - 1].strip()
            if 'guard' not in line and 'if' not in line:
                issues['logic_gaps'].append({
                    'line': line_num,
                    'type': 'force_try',
                    'description': 'Force try (!) without error handling',
                    'snippet': line[:100]
                })

    def _check_async_gaps(self, content: str, file_path: Path, issues: dict):
        """Find async/await gaps"""
        # Async functions without await
        async_funcs = re.finditer(r'func\s+\w+\s*\([^)]*\)\s*async', content)
        for match in async_funcs:
            func_start = match.start()
            func_end = content.find('}', func_start)
            func_body = content[func_start:func_end]

            if 'await' not in func_body and 'Task' not in func_body:
                line_num = content[:func_start].count('\n') + 1
                issues['logic_gaps'].append({
                    'line': line_num,
                    'type': 'async_without_await',
                    'description': 'Async function without await keyword',
                    'snippet': match.group()[:50]
                })

    def _check_state_management(self, content: str, file_path: Path, issues: dict):
        """Find StateObject/ObservedObject issues"""
        # Check for state published without proper binding
        state_patterns = [
            (r'@State\s+private\s+var\s+\w+:\s*\w+[^=]*$', 'State without initial value'),
            (r'@Published\s+var[^}]*\{[^}]*didSet[^}]*\}', 'Published with didSet (anti-pattern)'),
        ]

        for pattern, description in state_patterns:
            matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                issues['ui_issues'].append({
                    'line': line_num,
                    'type': 'state_management',
                    'description': description,
                    'snippet': match.group()[:100]
                })

    def run_audit(self):
        """Run full audit on all Swift files"""
        print(f"\n🔍 Auditing Swift files in {SWIFT_SOURCES}")

        swift_files = list(SWIFT_SOURCES.rglob('*.swift'))
        print(f"Found {len(swift_files)} Swift files")

        results = []
        for i, file_path in enumerate(swift_files, 1):
            if i % 20 == 0:
                print(f"  Progress: {i}/{len(swift_files)} files...")
            result = self.audit_file(file_path)
            results.append(result)

        return results

    def generate_report(self, results: list) -> dict:
        """Generate comprehensive report"""
        total_issues = sum(len(r['issues']) for r in results)
        total_connectivity = sum(len(r['connectivity_issues']) for r in results)
        total_logic = sum(len(r['logic_gaps']) for r in results)
        total_ui = sum(len(r['ui_issues']) for r in results)

        report = {
            'summary': {
                'files_analyzed': len(results),
                'total_lines': sum(r['lines'] for r in results),
                'total_issues': total_issues,
                'connectivity_issues': total_connectivity,
                'logic_gaps': total_logic,
                'ui_issues': total_ui
            },
            'files_with_issues': [
                r for r in results
                if r['issues'] or r['connectivity_issues'] or r['logic_gaps'] or r['ui_issues']
            ],
            'high_priority': [],
            'medium_priority': [],
            'low_priority': []
        }

        # Categorize by priority
        for r in results:
            for issue in r['logic_gaps']:
                if issue['type'] in ['stub_implementation', 'force_try']:
                    report['high_priority'].append({
                        'file': r['path'],
                        **issue
                    })
                else:
                    report['medium_priority'].append({
                        'file': r['path'],
                        **issue
                    })

            for issue in r['connectivity_issues']:
                report['high_priority'].append({
                    'file': r['path'],
                    **issue
                })

            for issue in r['ui_issues']:
                report['medium_priority'].append({
                    'file': r['path'],
                    **issue
                })

        return report

def main():
    print("=" * 70)
    print("L104v2 Swift App - Logic & Connectivity Audit")
    print("=" * 70)

    auditor = SwiftAuditor()
    results = auditor.run_audit()
    report = auditor.generate_report(results)

    # Save report
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n{'=' * 70}")
    print("AUDIT SUMMARY")
    print(f"{'=' * 70}")
    print(f"Files analyzed: {report['summary']['files_analyzed']}")
    print(f"Total lines: {report['summary']['total_lines']:,}")
    print(f"Total issues: {report['summary']['total_issues']}")
    print(f"  - Connectivity issues: {report['summary']['connectivity_issues']}")
    print(f"  - Logic gaps: {report['summary']['logic_gaps']}")
    print(f"  - UI issues: {report['summary']['ui_issues']}")
    print(f"\nPriority breakdown:")
    print(f"  - High priority: {len(report['high_priority'])}")
    print(f"  - Medium priority: {len(report['medium_priority'])}")
    print(f"  - Low priority: {len(report['low_priority'])}")

    print(f"\nReport saved to: {OUTPUT_FILE}")

    # Show top 10 high priority issues
    if report['high_priority']:
        print(f"\n{'=' * 70}")
        print("TOP 10 HIGH PRIORITY ISSUES")
        print(f"{'=' * 70}")
        for issue in report['high_priority'][:10]:
            file = issue['file'].split('/')[-1]
            print(f"\n{file}:{issue.get('line', '?')} - {issue['type']}")
            print(f"  {issue['description']}")
            if 'snippet' in issue:
                snippet = issue['snippet'][:60].replace('\n', ' ')
                print(f"  Code: {snippet}...")

if __name__ == '__main__':
    main()
