#!/usr/bin/env python3
"""
L104 Missing Features Detector
Identifies missing features and optimization opportunities in the L104 system
"""

import os
import sys
import ast
import inspect
import importlib
from typing import Dict, List, Any, Set, Tuple
from pathlib import Path

class L104FeatureDetector:
    """Detects missing features and optimization opportunities"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.missing_features = []
        self.optimization_opportunities = []
        self.security_issues = []
        self.performance_issues = []
        
    def scan_project(self):
        """Scan the entire project for issues"""
        print("🔍 Scanning L104 project for missing features and optimizations...")
        
        # Scan Python files
        python_files = list(self.project_root.rglob("*.py"))
        print(f"Found {len(python_files)} Python files")
        
        for py_file in python_files:
            self.analyze_python_file(py_file)
        
        # Check for missing directories
        self.check_missing_directories()
        
        # Check for missing dependencies
        self.check_missing_dependencies()
        
        # Check for configuration issues
        self.check_configuration_issues()
        
        # Check for API completeness
        self.check_api_completeness()
        
        # Check for monitoring and observability
        self.check_monitoring_features()
        
        # Check for security features
        self.check_security_features()
        
        return self.generate_report()
    
    def analyze_python_file(self, file_path: Path):
        """Analyze a Python file for issues"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Skip if file is too small or likely auto-generated
            if len(content) < 100:
                return
            
            # Check for common issues
            lines = content.split('\n')
            
            # Check for TODO/FIXME comments
            for i, line in enumerate(lines, 1):
                line_lower = line.lower()
                if 'todo' in line_lower or 'fixme' in line_lower or 'xxx' in line_lower:
                    self.missing_features.append({
                        'file': str(file_path.relative_to(self.project_root)),
                        'line': i,
                        'issue': 'TODO/FIXME found',
                        'code': line.strip()[:100]
                    })
            
            # Check for broad exception handling
            for i, line in enumerate(lines, 1):
                if 'except:' in line or 'except Exception:' in line:
                    self.performance_issues.append({
                        'file': str(file_path.relative_to(self.project_root)),
                        'line': i,
                        'issue': 'Broad exception handling',
                        'recommendation': 'Use specific exception types'
                    })
            
            # Check for potential performance issues
            for i, line in enumerate(lines, 1):
                if any(pattern in line for pattern in [
                    '.append() in loop',
                    'string concatenation in loop',
                    'deepcopy',
                    'eval(',
                    'exec('
                ]):
                    self.performance_issues.append({
                        'file': str(file_path.relative_to(self.project_root)),
                        'line': i,
                        'issue': 'Potential performance issue',
                        'code': line.strip()[:100]
                    })
            
            # Check for security issues
            for i, line in enumerate(lines, 1):
                if any(pattern in line for pattern in [
                    'pickle.load',
                    'yaml.load',
                    'eval(',
                    'exec(',
                    'subprocess.call(',
                    'os.system('
                ]):
                    self.security_issues.append({
                        'file': str(file_path.relative_to(self.project_root)),
                        'line': i,
                        'issue': 'Potential security issue',
                        'recommendation': 'Use safe alternatives'
                    })
            
            # Check for missing error handling
            try:
                tree = ast.parse(content)
                self.analyze_ast(tree, file_path)
            except SyntaxError:
                pass  # Skip files with syntax errors
            
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
    
    def analyze_ast(self, tree: ast.AST, file_path: Path):
        """Analyze AST for deeper issues"""
        for node in ast.walk(tree):
            # Check for functions without docstrings
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not ast.get_docstring(node):
                    self.missing_features.append({
                        'file': str(file_path.relative_to(self.project_root)),
                        'line': node.lineno,
                        'issue': f'Missing docstring for {node.name}',
                        'recommendation': 'Add docstring'
                    })
            
            # Check for missing type hints
            if isinstance(node, ast.FunctionDef):
                if not node.returns and node.name not in ['__init__', '__str__', '__repr__']:
                    # Check if any args have type hints
                    args_with_hints = sum(1 for arg in node.args.args if arg.annotation)
                    if args_with_hints == 0:
                        self.missing_features.append({
                            'file': str(file_path.relative_to(self.project_root)),
                            'line': node.lineno,
                            'issue': f'Missing type hints for function {node.name}',
                            'recommendation': 'Add type hints'
                        })
    
    def check_missing_directories(self):
        """Check for missing but expected directories"""
        expected_dirs = [
            'tests',
            'docs',
            'scripts',
            'config',
            'models',
            'utils',
            'middleware',
            'schemas',
            'migrations'
        ]
        
        for dir_name in expected_dirs:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                self.missing_features.append({
                    'file': dir_name,
                    'issue': f'Missing directory: {dir_name}',
                    'recommendation': f'Create {dir_name}/ directory'
                })
    
    def check_missing_dependencies(self):
        """Check for missing dependencies"""
        # Common dependencies that might be missing
        common_deps = [
            'pytest',  # Testing
            'mypy',    # Type checking
            'black',   # Code formatting
            'flake8',  # Linting
            'pre-commit',  # Git hooks
            'coverage',  # Test coverage
            'orjson',  # Faster JSON
            'uvicorn[standard]',  # ASGI server
            'httpx',   # Async HTTP client
            'redis',   # Caching
            'celery',  # Task queue
            'prometheus-client',  # Metrics
            'sentry-sdk',  # Error tracking
            'structlog',  # Structured logging
        ]
        
        # Check requirements.txt or pyproject.toml
        req_files = [
            self.project_root / 'requirements.txt',
            self.project_root / 'pyproject.toml',
            self.project_root / 'setup.py'
        ]
        
        existing_deps = set()
        for req_file in req_files:
            if req_file.exists():
                try:
                    with open(req_file, 'r') as f:
                        content = f.read()
                        for dep in common_deps:
                            if dep.split('[')[0] in content:
                                existing_deps.add(dep)
                except:
                    pass
        
        missing_deps = [d for d in common_deps if d not in existing_deps]
        for dep in missing_deps:
            self.optimization_opportunities.append({
                'issue': f'Missing dependency: {dep}',
                'recommendation': f'Consider adding {dep} for improved functionality'
            })
    
    def check_configuration_issues(self):
        """Check for configuration issues"""
        config_files = [
            '.env',
            '.env.example',
            'config.yaml',
            'config.json',
            'settings.py'
        ]
        
        for config_file in config_files:
            config_path = self.project_root / config_file
            if not config_path.exists():
                self.missing_features.append({
                    'file': config_file,
                    'issue': f'Missing configuration file: {config_file}',
                    'recommendation': f'Create {config_file} with appropriate settings'
                })
        
        # Check for hardcoded secrets
        secret_patterns = [
            'api_key',
            'secret',
            'password',
            'token',
            'private_key'
        ]
        
        for py_file in self.project_root.rglob("*.py"):
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    for pattern in secret_patterns:
                        if pattern in content.lower():
                            # Check if it's in a string literal
                            lines = content.split('\n')
                            for i, line in enumerate(lines, 1):
                                if pattern in line.lower() and ('"' in line or "'" in line):
                                    self.security_issues.append({
                                        'file': str(py_file.relative_to(self.project_root)),
                                        'line': i,
                                        'issue': f'Potential hardcoded secret: {pattern}',
                                        'recommendation': 'Move to environment variables'
                                    })
            except:
                pass
    
    def check_api_completeness(self):
        """Check for API completeness"""
        # Check for common API endpoints that might be missing
        expected_endpoints = [
            '/api/health',  # Health check
            '/api/metrics',  # Prometheus metrics
            '/api/docs',  # API documentation
            '/api/redoc',  # Alternative docs
            '/api/version',  # Version info
            '/api/config',  # Configuration
            '/api/logs',  # Log access
            '/api/debug',  # Debug endpoints
        ]
        
        # This would require parsing the FastAPI app to check routes
        # For now, we'll just note the expectation
        self.missing_features.append({
            'issue': 'API completeness check needed',
            'recommendation': 'Verify all expected API endpoints are implemented'
        })
    
    def check_monitoring_features(self):
        """Check for monitoring and observability features"""
        monitoring_features = [
            'Metrics collection (Prometheus)',
            'Distributed tracing (OpenTelemetry)',
            'Structured logging',
            'Error tracking (Sentry)',
            'Performance monitoring',
            'Alerting system',
            'Dashboard (Grafana)'
        ]
        
        for feature in monitoring_features:
            self.optimization_opportunities.append({
                'issue': f'Monitoring feature missing: {feature}',
                'recommendation': f'Implement {feature} for better observability'
            })
    
    def check_security_features(self):
        """Check for security features"""
        security_features = [
            'Rate limiting',
            'Authentication',
            'Authorization',
            'Input validation',
            'Output encoding',
            'CORS configuration',
            'CSRF protection',
            'Security headers',
            'API key validation',
            'Request signing'
        ]
        
        for feature in security_features:
            self.security_issues.append({
                'issue': f'Security feature missing: {feature}',
                'recommendation': f'Implement {feature} for better security'
            })
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive report"""
        report = {
            'summary': {
                'missing_features': len(self.missing_features),
                'optimization_opportunities': len(self.optimization_opportunities),
                'security_issues': len(self.security_issues),
                'performance_issues': len(self.performance_issues),
                'total_issues': len(self.missing_features) + len(self.optimization_opportunities) + 
                              len(self.security_issues) + len(self.performance_issues)
            },
            'missing_features': self.missing_features[:50],  # Limit for readability
            'optimization_opportunities': self.optimization_opportunities[:50],
            'security_issues': self.security_issues[:50],
            'performance_issues': self.performance_issues[:50],
            'recommendations': self.generate_recommendations()
        }
        
        return report
    
    def generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Prioritize security issues
        if self.security_issues:
            recommendations.append("🔒 Address security issues first (hardcoded secrets, input validation)")
        
        # Address critical missing features
        critical_missing = [f for f in self.missing_features if 'TODO' in f.get('issue', '')]
        if critical_missing:
            recommendations.append("🚨 Fix critical TODO/FIXME comments in code")
        
        # Add monitoring
        if not any('monitoring' in str(r).lower() for r in recommendations):
            recommendations.append("📊 Implement comprehensive monitoring and observability")
        
        # Add testing
        if not (self.project_root / 'tests').exists():
            recommendations.append("🧪 Create comprehensive test suite")
        
        # Add documentation
        if not (self.project_root / 'docs').exists():
            recommendations.append("📚 Create documentation (API docs, user guide)")
        
        # Add CI/CD
        ci_files = ['.github/workflows', '.gitlab-ci.yml', '.circleci/config.yml']
        if not any((self.project_root / ci).exists() for ci in ci_files):
            recommendations.append("⚙️ Set up CI/CD pipeline")
        
        # Add performance optimization
        if self.performance_issues:
            recommendations.append("⚡ Address performance issues (broad exceptions, inefficient loops)")
        
        # Add type checking
        if not any('type hints' in str(f.get('issue', '')).lower() for f in self.missing_features):
            recommendations.append("🎯 Implement comprehensive type hints for better code quality")
        
        return recommendations
    
    def print_report(self, report: Dict[str, Any]):
        """Print the report in a readable format"""
        print("\n" + "="*80)
        print("L104 SYSTEM ANALYSIS REPORT")
        print("="*80)
        
        summary = report['summary']
        print(f"\n📊 SUMMARY:")
        print(f"  Missing Features: {summary['missing_features']}")
        print(f"  Optimization Opportunities: {summary['optimization_opportunities']}")
        print(f"  Security Issues: {summary['security_issues']}")
        print(f"  Performance Issues: {summary['performance_issues']}")
        print(f"  Total Issues: {summary['total_issues']}")
        
        if report['missing_features']:
            print(f"\n❌ MISSING FEATURES ({len(report['missing_features'])}):")
            for i, feature in enumerate(report['missing_features'][:10], 1):
                print(f"  {i}. {feature.get('file', 'N/A')}:{feature.get('line', 'N/A')} - {feature.get('issue', 'N/A')}")
                if 'recommendation' in feature:
                    print(f"     💡 {feature['recommendation']}")
        
        if report['optimization_opportunities']:
            print(f"\n⚡ OPTIMIZATION OPPORTUNITIES ({len(report['optimization_opportunities'])}):")
            for i, opp in enumerate(report['optimization_opportunities'][:10], 1):
                print(f"  {i}. {opp.get('issue', 'N/A')}")
                if 'recommendation' in opp:
                    print(f"     💡 {opp['recommendation']}")
        
        if report['security_issues']:
            print(f"\n🔒 SECURITY ISSUES ({len(report['security_issues'])}):")
            for i, issue in enumerate(report['security_issues'][:10], 1):
                print(f"  {i}. {issue.get('file', 'N/A')}:{issue.get('line', 'N/A')} - {issue.get('issue', 'N/A')}")
                if 'recommendation' in issue:
                    print(f"     💡 {issue['recommendation']}")
        
        if report['performance_issues']:
            print(f"\n🐌 PERFORMANCE ISSUES ({len(report['performance_issues'])}):")
            for i, issue in enumerate(report['performance_issues'][:10], 1):
                print(f"  {i}. {issue.get('file', 'N/A')}:{issue.get('line', 'N/A')} - {issue.get('issue', 'N/A')}")
                if 'recommendation' in issue:
                    print(f"     💡 {issue['recommendation']}")
        
        if report['recommendations']:
            print(f"\n🎯 TOP RECOMMENDATIONS:")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        print("\n" + "="*80)
        print("Next steps:")
        print("1. Address security issues first")
        print("2. Fix critical TODO/FIXME comments")
        print("3. Implement monitoring and observability")
        print("4. Run the optimization engine: python l104_optimization_engine.py --optimize")
        print("="*80)

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='L104 Missing Features Detector')
    parser.add_argument('--path', type=str, default='.', help='Project path')
    parser.add_argument('--output', type=str, help='Output JSON report file')
    
    args = parser.parse_args()
    
    detector = L104FeatureDetector(args.path)
    report = detector.scan_project()
    
    detector.print_report(report)
    
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to {args.output}")

if __name__ == '__main__':
    main()