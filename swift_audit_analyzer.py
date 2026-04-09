#!/usr/bin/env python3
"""
L104v2 Swift Audit Analyzer
Analyzes Swift codebase for logic incompletions and weak connectivity
"""

import os
import re
import json
from pathlib import Path
from collections import defaultdict

class SwiftAuditAnalyzer:
    def __init__(self, source_dir):
        self.source_dir = Path(source_dir)
        self.issues = defaultdict(list)
        self.stats = {
            'files_analyzed': 0,
            'lines_analyzed': 0,
            'empty_functions': 0,
            'todo_fixme': 0,
            'fatal_errors': 0,
            'weak_references': 0,
            'incomplete_switches': 0,
            'stub_implementations': 0,
            'button_connectivity': 0
        }

    def analyze(self):
        """Run full audit on all Swift files"""
        swift_files = list(self.source_dir.rglob('*.swift'))

        for swift_file in swift_files:
            self._analyze_file(swift_file)

        return self._generate_report()

    def _analyze_file(self, filepath):
        """Analyze a single Swift file"""
        try:
            content = filepath.read_text()
            lines = content.split('\n')
            self.stats['files_analyzed'] += 1
            self.stats['lines_analyzed'] += len(lines)

            # Check for empty function bodies
            self._find_empty_functions(filepath, content)

            # Check for TODO/FIXME comments
            self._find_todos(filepath, content)

            # Check for fatalError stubs
            self._find_fatal_errors(filepath, content)

            # Check for weak references (potential connectivity issues)
            self._find_weak_references(filepath, content)

            # Check for incomplete switch statements
            self._find_incomplete_switches(filepath, content)

            # Check for stub implementations
            self._find_stubs(filepath, content)

            # Check for button/tab connectivity
            self._find_button_connectivity(filepath, content)

            # Check for unimplemented protocol methods
            self._find_unimplemented_protocols(filepath, content)

        except Exception as e:
            self.issues['errors'].append(f"Error reading {filepath}: {e}")

    def _find_empty_functions(self, filepath, content):
        """Find functions with empty bodies"""
        # Pattern: func name() { } or func name() -> Type { }
        pattern = r'(func\s+\w+\([^)]*\)(?:\s*->\s*\w+)?)\s*\{\s*\}'
        matches = re.finditer(pattern, content, re.MULTILINE)
        for match in matches:
            self.issues['empty_functions'].append({
                'file': str(filepath),
                'signature': match.group(1).strip(),
                'line': content[:match.start()].count('\n') + 1
            })
            self.stats['empty_functions'] += 1

    def _find_todos(self, filepath, content):
        """Find TODO and FIXME comments"""
        pattern = r'(TODO|FIXME|XXX|HACK):?\s*(.+)'
        for match in re.finditer(pattern, content, re.IGNORECASE):
            line_num = content[:match.start()].count('\n') + 1
            self.issues['todos'].append({
                'file': str(filepath),
                'type': match.group(1).upper(),
                'description': match.group(2).strip()[:100],
                'line': line_num
            })
            self.stats['todo_fixme'] += 1

    def _find_fatal_errors(self, filepath, content):
        """Find fatalError stubs"""
        pattern = r'fatalError\s*\(\s*["\']([^"\']*)["\']\s*\)'
        for match in re.finditer(pattern, content):
            line_num = content[:match.start()].count('\n') + 1
            self.issues['fatal_errors'].append({
                'file': str(filepath),
                'message': match.group(1)[:80],
                'line': line_num
            })
            self.stats['fatal_errors'] += 1

    def _find_weak_references(self, filepath, content):
        """Find weak references that might need attention"""
        # Pattern: weak var, weak self
        patterns = [
            (r'weak\s+var\s+(\w+)', 'weak_var'),
            (r'\[\s*weak\s+\w+\s*\]', 'weak_capture'),
            (r'\[\s*unowned\s+\w+\s*\]', 'unowned_capture')
        ]
        for pattern, issue_type in patterns:
            for match in re.finditer(pattern, content):
                line_num = content[:match.start()].count('\n') + 1
                self.issues['weak_references'].append({
                    'file': str(filepath),
                    'type': issue_type,
                    'match': match.group(0),
                    'line': line_num
                })
                self.stats['weak_references'] += 1

    def _find_incomplete_switches(self, filepath, content):
        """Find switch statements that might be incomplete"""
        # Look for switches without default cases
        pattern = r'switch\s+\w+\s*\{[^}]*\}'
        for match in re.finditer(pattern, content, re.DOTALL):
            switch_block = match.group(0)
            if 'default' not in switch_block and '@unknown' not in switch_block:
                # Check if it's an enum switch (common to need default)
                line_num = content[:match.start()].count('\n') + 1
                self.issues['incomplete_switches'].append({
                    'file': str(filepath),
                    'line': line_num,
                    'snippet': switch_block[:80].replace('\n', ' ')
                })
                self.stats['incomplete_switches'] += 1

    def _find_stubs(self, filepath, content):
        """Find stub implementations"""
        stub_patterns = [
            (r'return\s+\{\s*\}', 'empty_closure_return'),
            (r'return\s+""', 'empty_string_return'),
            (r'return\s+0(?:\.0)?', 'zero_return'),
            (r'return\s+nil', 'nil_return'),
            (r'return\s+false', 'false_return'),
            (r'return\s+true', 'true_return'),
        ]

        for pattern, return_type in stub_patterns:
            for match in re.finditer(pattern, content):
                line_num = content[:match.start()].count('\n') + 1
                # Only flag if it's in a function body
                context = content[max(0, match.start()-100):match.start()]
                if 'func ' in context or 'var ' in context:
                    self.issues['stub_implementations'].append({
                        'file': str(filepath),
                        'type': return_type,
                        'line': line_num
                    })
                    self.stats['stub_implementations'] += 1

    def _find_button_connectivity(self, filepath, content):
        """Find button actions and tab connectivity issues"""
        patterns = [
            (r'Button\s*\([^)]*\)\s*\{[^}]*\}', 'button_closure'),
            (r'action:\s*\{\s*\}', 'empty_button_action'),
            (r'@IBAction\s+func\s+\w+\s*\([^)]*\)', 'ibaction'),
            (r'TabView|TabBar|tabItem', 'tab_component'),
            (r'onChange\s*\(of:\s*\w+\)', 'on_change_handler'),
            (r'\.onAppear\s*\{[^}]*\}', 'on_appear'),
            (r'\.onDisappear\s*\{[^}]*\}', 'on_disappear'),
        ]

        for pattern, btn_type in patterns:
            for match in re.finditer(pattern, content):
                line_num = content[:match.start()].count('\n') + 1
                self.issues['button_connectivity'].append({
                    'file': str(filepath),
                    'type': btn_type,
                    'match': match.group(0)[:60],
                    'line': line_num
                })
                self.stats['button_connectivity'] += 1

    def _find_unimplemented_protocols(self, filepath, content):
        """Find protocol conformances that might be incomplete"""
        # Look for protocol conformance declarations
        pattern = r'(struct|class|enum)\s+(\w+).*:\s*([\w,\s]+)\{'
        for match in re.finditer(pattern, content):
            conformances = match.group(3)
            if 'ObservableObject' in conformances or 'View' in conformances:
                line_num = content[:match.start()].count('\n') + 1
                self.issues['protocol_conformances'].append({
                    'file': str(filepath),
                    'type': match.group(1),
                    'name': match.group(2),
                    'conformances': conformances.strip(),
                    'line': line_num
                })

    def _generate_report(self):
        """Generate comprehensive audit report"""
        report = {
            'summary': self.stats,
            'issues': dict(self.issues),
            'recommendations': self._generate_recommendations()
        }
        return report

    def _generate_recommendations(self):
        """Generate recommendations based on findings"""
        recommendations = []

        if self.stats['empty_functions'] > 0:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Logic Incompletions',
                'description': f"Found {self.stats['empty_functions']} empty function bodies that need implementation",
                'action': 'Review and implement missing logic'
            })

        if self.stats['fatal_errors'] > 0:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Stub Implementations',
                'description': f"Found {self.stats['fatal_errors']} fatalError stubs that crash the app",
                'action': 'Replace fatalError with actual implementations'
            })

        if self.stats['stub_implementations'] > 50:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Stub Code',
                'description': f"Found {self.stats['stub_implementations']} stub return statements",
                'action': 'Review stub implementations for correctness'
            })

        if self.stats['weak_references'] > 0:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Memory Management',
                'description': f"Found {self.stats['weak_references']} weak/unowned references",
                'action': 'Verify weak references are properly handled to avoid premature deallocation'
            })

        if self.stats['button_connectivity'] > 0:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'UI Connectivity',
                'description': f"Found {self.stats['button_connectivity']} button/tab components",
                'action': 'Verify all button actions are properly connected and handlers are not empty'
            })

        if self.stats['todo_fixme'] > 0:
            recommendations.append({
                'priority': 'LOW',
                'category': 'Technical Debt',
                'description': f"Found {self.stats['todo_fixme']} TODO/FIXME comments",
                'action': 'Address or schedule TODO items'
            })

        return recommendations

if __name__ == '__main__':
    source_dir = '/Users/carolalvarez/Applications/Allentown-L104-Node/L104SwiftApp/Sources'

    print("🔍 L104v2 Swift Audit Analyzer")
    print("=" * 60)
    print(f"Analyzing: {source_dir}")
    print()

    analyzer = SwiftAuditAnalyzer(source_dir)
    report = analyzer.analyze()

    # Print summary
    print("📊 AUDIT SUMMARY")
    print("-" * 60)
    for key, value in report['summary'].items():
        print(f"  {key}: {value}")

    print()
    print("🚨 TOP ISSUES BY CATEGORY")
    print("-" * 60)

    # Show top issues
    for category, items in sorted(report['issues'].items(), key=lambda x: len(x[1]), reverse=True)[:5]:
        if items:
            print(f"\n{category.upper()}: {len(items)} items")
            for item in items[:3]:
                file = str(item.get('file', 'N/A')).replace(source_dir, '')
                line = item.get('line', '?')
                print(f"    {file}:{line}")

    print()
    print("💡 RECOMMENDATIONS")
    print("-" * 60)
    for rec in report['recommendations']:
        print(f"\n[{rec['priority']}] {rec['category']}")
        print(f"  {rec['description']}")
        print(f"  → Action: {rec['action']}")

    # Save full report
    report_file = '/Users/carolalvarez/Applications/Allentown-L104-Node/L104SwiftApp/audit_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print()
    print(f"✅ Full report saved to: {report_file}")
