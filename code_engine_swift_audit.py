#!/usr/bin/env python3
"""
L104v2 Swift App - Comprehensive Audit with Code Engine
Uses the code_engine's full analysis capabilities
"""

import os
import sys
import json
from pathlib import Path

sys.path.insert(0, '/Users/carolalvarez/Applications/Allentown-L104-Node')

from l104_code_engine import code_engine

# Configuration
SWIFT_DIR = Path('/Users/carolalvarez/Applications/Allentown-L104-Node/L104SwiftApp/Sources/L104v2')
OUTPUT_REPORT = Path('/Users/carolalvarez/Applications/Allentown-L104-Node/swift_code_engine_audit.json')

class CodeEngineSwiftAuditor:
    def __init__(self):
        self.issues_found = []
        self.improvements_made = []

    def analyze_file_with_code_engine(self, file_path: Path) -> dict:
        """Use code_engine to analyze Swift file"""
        content = file_path.read_text(encoding='utf-8', errors='ignore')

        # Use code_engine's smell detector
        try:
            smells = code_engine.smell_detector.detect_all(content)
        except Exception as e:
            smells = {'error': str(e)}

        # Use code_engine's performance predictor
        try:
            perf = code_engine.perf_predictor.predict_performance(content)
        except Exception as e:
            perf = {'error': str(e)}

        # Use code_engine's complexity analyzer
        try:
            complexity = code_engine.analyze_complexity(content)
        except Exception as e:
            complexity = {'error': str(e)}

        return {
            'file': str(file_path),
            'lines': len(content.split('\n')),
            'smells': smells,
            'performance': perf,
            'complexity': complexity
        }

    def find_ui_connectivity_issues(self, file_path: Path, content: str) -> list:
        """Find weak tab/button connectivity issues"""
        import re
        issues = []
        lines = content.split('\n')

        for i, line in enumerate(lines, 1):
            # Check for buttons without actions
            if 'Button(' in line and 'action:' in line:
                next_lines = ''.join(lines[i:min(i+5, len(lines))])
                if 'action: {}' in next_lines or 'action: { }' in next_lines:
                    issues.append({
                        'line': i,
                        'type': 'empty_button_action',
                        'severity': 'high',
                        'description': 'Button has empty action closure - weak connectivity',
                        'code': line.strip()[:80]
                    })

            # Check for NavigationLink without destination
            if 'NavigationLink(' in line:
                if 'destination:' not in line and 'value:' not in line:
                    issues.append({
                        'line': i,
                        'type': 'incomplete_navigation',
                        'severity': 'medium',
                        'description': 'NavigationLink missing destination binding',
                        'code': line.strip()[:80]
                    })

            # Check for tab selection without state binding
            if '.tabItem(' in line:
                prev_context = ''.join(lines[max(0, i-10):i])
                if 'selection:' not in prev_context:
                    issues.append({
                        'line': i,
                        'type': 'weak_tab_binding',
                        'severity': 'medium',
                        'description': 'TabItem may be missing selection state binding',
                        'code': line.strip()[:80]
                    })

            # Check for state objects without proper initialization
            if '@State' in line or '@StateObject' in line:
                if '=' not in line and 'private' in line:
                    issues.append({
                        'line': i,
                        'type': 'uninitialized_state',
                        'severity': 'medium',
                        'description': 'State/StateObject may be uninitialized',
                        'code': line.strip()[:80]
                    })

            # Check for delegate methods with placeholder returns
            if 'func outlineView(' in line or 'func tableView(' in line:
                next_lines = ''.join(lines[i:min(i+10, len(lines))])
                if 'return ""' in next_lines or 'return 0' in next_lines or 'return false' in next_lines:
                    issues.append({
                        'line': i,
                        'type': 'delegate_stub',
                        'severity': 'low',
                        'description': 'Delegate method returns placeholder value',
                        'code': line.strip()[:80]
                    })

        return issues

    def find_logic_incompletions(self, file_path: Path, content: str) -> list:
        """Find logic incompletions"""
        import re
        issues = []
        lines = content.split('\n')

        for i, line in enumerate(lines, 1):
            # Check for TODO/FIXME comments
            if '// TODO' in line or '// FIXME' in line:
                issues.append({
                    'line': i,
                    'type': 'todo_marker',
                    'severity': 'medium',
                    'description': 'Incomplete implementation marked with TODO/FIXME',
                    'code': line.strip()[:80]
                })

            # Check for empty catch blocks
            if 'catch' in line and '{' in line:
                next_lines = ''.join(lines[i:min(i+5, len(lines))])
                if re.search(r'catch\s*\{[^}]*\}', next_lines) or 'catch { }' in next_lines:
                    issues.append({
                        'line': i,
                        'type': 'empty_catch',
                        'severity': 'high',
                        'description': 'Empty catch block - error not handled',
                        'code': line.strip()[:80]
                    })

            # Check for force unwrapping
            if '!' in line and ('as!' in line or 'try!' in line):
                issues.append({
                    'line': i,
                    'type': 'force_unwrap',
                    'severity': 'high',
                    'description': 'Force unwrapping/try! can cause crashes',
                    'code': line.strip()[:80]
                })

            # Check for prints instead of proper logging
            if 'print(' in line and 'L104' not in line:
                issues.append({
                    'line': i,
                    'type': 'raw_print',
                    'severity': 'low',
                    'description': 'Raw print statement instead of structured logging',
                    'code': line.strip()[:80]
                })

            # Check for hardcoded values
            if re.search(r':\s*\d+\.\d+', line) or re.search(r'=\s*\d{3,}', line):
                if 'GOD_CODE' not in line and 'PHI' not in line and 'TAU' not in line:
                    issues.append({
                        'line': i,
                        'type': 'magic_number',
                        'severity': 'low',
                        'description': 'Hardcoded numeric value (magic number)',
                        'code': line.strip()[:80]
                    })

        return issues

    def run_audit(self) -> dict:
        """Run comprehensive audit"""
        print("=" * 70)
        print("L104v2 Swift App - Code Engine Comprehensive Audit")
        print("=" * 70)

        swift_files = list(SWIFT_DIR.rglob('*.swift'))
        print(f"\nFound {len(swift_files)} Swift files")

        all_results = []
        ui_issues = []
        logic_issues = []

        for i, file_path in enumerate(swift_files, 1):
            if i % 30 == 0:
                print(f"  Progress: {i}/{len(swift_files)}...")

            content = file_path.read_text(encoding='utf-8', errors='ignore')

            # Find UI connectivity issues
            ui = self.find_ui_connectivity_issues(file_path, content)
            if ui:
                ui_issues.extend([{'file': str(file_path), **u} for u in ui])

            # Find logic incompletions
            logic = self.find_logic_incompletions(file_path, content)
            if logic:
                logic_issues.extend([{'file': str(file_path), **l} for l in logic])

            # Code engine analysis for key files
            if any(x in str(file_path) for x in ['H11_MainView', 'H06_UIViews', 'H30_SidebarNav', 'H03_L104StateCommands']):
                try:
                    ce_result = self.analyze_file_with_code_engine(file_path)
                    all_results.append(ce_result)
                except Exception as e:
                    print(f"  Warning: Code engine analysis failed for {file_path.name}: {e}")

        report = {
            'summary': {
                'files_analyzed': len(swift_files),
                'ui_connectivity_issues': len(ui_issues),
                'logic_incompletions': len(logic_issues),
                'high_severity': len([i for i in ui_issues + logic_issues if i.get('severity') == 'high']),
                'medium_severity': len([i for i in ui_issues + logic_issues if i.get('severity') == 'medium']),
                'low_severity': len([i for i in ui_issues + logic_issues if i.get('severity') == 'low']),
            },
            'ui_connectivity_issues': sorted(ui_issues, key=lambda x: x['severity']),
            'logic_incompletions': sorted(logic_issues, key=lambda x: x['severity']),
            'code_engine_analysis': all_results
        }

        # Save report
        with open(OUTPUT_REPORT, 'w') as f:
            json.dump(report, f, indent=2)

        return report

def main():
    auditor = CodeEngineSwiftAuditor()
    report = auditor.run_audit()

    # Print summary
    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)
    print(f"\nFiles analyzed: {report['summary']['files_analyzed']}")
    print(f"\nUI Connectivity Issues: {report['summary']['ui_connectivity_issues']}")
    print(f"  High: {len([i for i in report['ui_connectivity_issues'] if i['severity'] == 'high'])}")
    print(f"  Medium: {len([i for i in report['ui_connectivity_issues'] if i['severity'] == 'medium'])}")
    print(f"  Low: {len([i for i in report['ui_connectivity_issues'] if i['severity'] == 'low'])}")

    print(f"\nLogic Incompletions: {report['summary']['logic_incompletions']}")
    print(f"  High: {len([i for i in report['logic_incompletions'] if i['severity'] == 'high'])}")
    print(f"  Medium: {len([i for i in report['logic_incompletions'] if i['severity'] == 'medium'])}")
    print(f"  Low: {len([i for i in report['logic_incompletions'] if i['severity'] == 'low'])}")

    # Show top issues by severity
    all_issues = report['ui_connectivity_issues'] + report['logic_incompletions']
    high_issues = [i for i in all_issues if i['severity'] == 'high']

    if high_issues:
        print("\n" + "=" * 70)
        print("HIGH PRIORITY ISSUES (First 15)")
        print("=" * 70)
        for issue in high_issues[:15]:
            file = issue['file'].split('/')[-1]
            print(f"\n{file}:{issue['line']} [{issue['type']}]")
            print(f"  {issue['description']}")

    print(f"\n\nFull report saved to: {OUTPUT_REPORT}")
    return report

if __name__ == '__main__':
    main()
