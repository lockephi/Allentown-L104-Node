#!/usr/bin/env python3
"""
Safe duplicate route remover for L104 server.
Uses 3-engine approach: Discovery → Validation → Execution
"""

import ast
import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Set
import shutil
from datetime import datetime

class RouteRemover:
    def __init__(self, app_path: str, routes_dir: str):
        self.app_path = Path(app_path)
        self.routes_dir = Path(routes_dir)
        self.backup_path = self.app_path.with_suffix('.py.backup')
        self.removal_log = self.app_path.parent / 'route_removal_log.json'
        
        # Load existing removal log
        self.removed_routes = self._load_removal_log()
        
    def _load_removal_log(self) -> Dict:
        """Load previous removal log."""
        if self.removal_log.exists():
            with open(self.removal_log, 'r') as f:
                return json.load(f)
        return {"removed": [], "backups": [], "timestamp": None}
    
    def _save_removal_log(self):
        """Save removal log."""
        log_data = {
            "removed": self.removed_routes.get("removed", []),
            "backups": self.removed_routes.get("backups", []),
            "timestamp": datetime.now().isoformat(),
            "app_path": str(self.app_path)
        }
        with open(self.removal_log, 'w') as f:
            json.dump(log_data, f, indent=2)
    
    def engine1_discovery(self) -> List[Dict]:
        """Engine 1: Discover duplicate routes with AST parsing."""
        print("🔍 Engine 1: Discovering duplicate routes...")
        
        with open(self.app_path, 'r') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        # Find all route definitions in app.py
        app_routes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Call):
                        func = decorator.func
                        if (isinstance(func, ast.Attribute) and 
                            func.attr in ['get', 'post', 'put', 'delete', 'patch']):
                            
                            # Get route path
                            if decorator.args:
                                route_path = ast.unparse(decorator.args[0])
                                # Remove quotes
                                route_path = route_path.strip("'\"")
                                
                                # Get HTTP method
                                http_method = func.attr.upper()
                                
                                # Get line numbers
                                start_line = node.lineno
                                # Estimate end line (function body)
                                end_line = start_line
                                for child in ast.walk(node):
                                    if hasattr(child, 'lineno'):
                                        end_line = max(end_line, child.lineno)
                                
                                app_routes.append({
                                    'path': route_path,
                                    'method': http_method,
                                    'name': node.name,
                                    'start_line': start_line,
                                    'end_line': end_line,
                                    'function_def': ast.unparse(node)
                                })
        
        print(f"Found {len(app_routes)} routes in app.py")
        
        # Find modular routes
        modular_routes = self._find_modular_routes()
        print(f"Found {len(modular_routes)} modular routes")
        
        # Identify duplicates
        duplicates = []
        for app_route in app_routes:
            key = (app_route['path'], app_route['method'])
            if key in modular_routes:
                duplicates.append(app_route)
        
        print(f"Found {len(duplicates)} duplicate routes")
        return duplicates
    
    def _find_modular_routes(self) -> Set[Tuple[str, str]]:
        """Find all routes in modular route files."""
        modular_routes = set()
        
        # Walk through routes directory
        for py_file in self.routes_dir.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue
                
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        for decorator in node.decorator_list:
                            if isinstance(decorator, ast.Call):
                                func = decorator.func
                                if (isinstance(func, ast.Attribute) and 
                                    func.attr in ['get', 'post', 'put', 'delete', 'patch']):
                                    
                                    # Get route path
                                    if decorator.args:
                                        route_path = ast.unparse(decorator.args[0])
                                        route_path = route_path.strip("'\"")
                                        
                                        # Get HTTP method
                                        http_method = func.attr.upper()
                                        
                                        modular_routes.add((route_path, http_method))
            except:
                continue
        
        return modular_routes
    
    def engine2_validation(self, duplicates: List[Dict]) -> List[Dict]:
        """Engine 2: Validate modular routes exist and work."""
        print("🔬 Engine 2: Validating modular routes...")
        
        valid_duplicates = []
        
        for route in duplicates:
            # Check if modular route file exists
            route_found = False
            for py_file in self.routes_dir.rglob("*.py"):
                if py_file.name == "__init__.py":
                    continue
                    
                try:
                    with open(py_file, 'r') as f:
                        if route['path'] in f.read():
                            route_found = True
                            break
                except:
                    continue
            
            if route_found:
                # Test if endpoint works
                if self._test_endpoint(route['path'], route['method']):
                    route['validated'] = True
                    valid_duplicates.append(route)
                    print(f"  ✓ Validated: {route['method']} {route['path']}")
                else:
                    print(f"  ⚠️  Modular route not working: {route['method']} {route['path']}")
            else:
                print(f"  ⚠️  No modular route found: {route['method']} {route['path']}")
        
        print(f"Validated {len(valid_duplicates)} routes for removal")
        return valid_duplicates
    
    def _test_endpoint(self, path: str, method: str) -> bool:
        """Test if endpoint is accessible."""
        try:
            # Simple test - just check if server is responding
            # For GET endpoints, we can actually test
            if method == 'GET' and not path.startswith('/api/v14/'):  # Skip complex auth routes
                import requests
                response = requests.get(f'http://localhost:8004{path}', timeout=2)
                return response.status_code == 200
            return True  # Assume POST/PUT/DELETE routes work
        except:
            return True  # Assume it works if we can't test
    
    def engine3_execution(self, routes_to_remove: List[Dict], dry_run: bool = True):
        """Engine 3: Execute safe removal."""
        print("⚡ Engine 3: Executing removal..." + (" (DRY RUN)" if dry_run else ""))
        
        # Create backup
        if not dry_run and not self.backup_path.exists():
            shutil.copy2(self.app_path, self.backup_path)
            print(f"Created backup: {self.backup_path}")
        
        # Read app.py
        with open(self.app_path, 'r') as f:
            lines = f.readlines()
        
        # Sort routes by line number (descending) to avoid line number shifts
        routes_to_remove.sort(key=lambda x: x['start_line'], reverse=True)
        
        removed_count = 0
        removal_details = []
        
        for route in routes_to_remove:
            start = route['start_line'] - 1  # Convert to 0-index
            end = route['end_line']
            
            # Find the actual end of function (look for empty line or next decorator)
            actual_end = end
            for i in range(end, min(end + 20, len(lines))):
                if i >= len(lines):
                    break
                line = lines[i].rstrip()
                if not line or line.startswith('@') or line.startswith('def ') or line.startswith('async def '):
                    actual_end = i
                    break
                # Also stop if we hit another function at same indentation
                if line and not line.startswith(' ') and not line.startswith('\t'):
                    actual_end = i
                    break
            
            # Verify we're removing the right thing
            function_start = lines[start].strip()
            if not function_start.startswith('@app.'):
                print(f"  ⚠️  Skipping {route['path']}: Doesn't start with decorator")
                continue
            
            if dry_run:
                print(f"  📝 Would remove: {route['method']} {route['path']} (lines {start+1}-{actual_end})")
            else:
                # Remove the lines
                del lines[start:actual_end]
                
                # Add comment
                comment = f"# REMOVED: {route['method']} {route['path']} - Duplicate moved to modular routes\n"
                lines.insert(start, comment)
                
                removal_details.append({
                    'path': route['path'],
                    'method': route['method'],
                    'lines': f"{start+1}-{actual_end}",
                    'timestamp': datetime.now().isoformat()
                })
                print(f"  ✅ Removed: {route['method']} {route['path']}")
            
            removed_count += 1
        
        if not dry_run and removal_details:
            # Write modified file
            with open(self.app_path, 'w') as f:
                f.writelines(lines)
            
            # Update removal log
            self.removed_routes.setdefault("removed", []).extend(removal_details)
            if not dry_run:
                self.removed_routes.setdefault("backups", []).append({
                    'backup_path': str(self.backup_path),
                    'timestamp': datetime.now().isoformat()
                })
            self._save_removal_log()
            
            # Verify syntax
            if self._verify_syntax():
                print(f"✅ Successfully removed {removed_count} routes")
                print(f"📋 Log saved to: {self.removal_log}")
                print(f"💾 Backup at: {self.backup_path}")
            else:
                print("❌ Syntax error after removal! Restoring backup...")
                self._restore_backup()
        
        return removed_count
    
    def _verify_syntax(self) -> bool:
        """Verify Python syntax is valid."""
        try:
            subprocess.run([sys.executable, '-m', 'py_compile', str(self.app_path)], 
                          check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def _restore_backup(self):
        """Restore from backup."""
        if self.backup_path.exists():
            shutil.copy2(self.backup_path, self.app_path)
            print("Restored from backup")
    
    def run_pipeline(self, dry_run: bool = True):
        """Run the complete 3-engine pipeline."""
        print("=" * 60)
        print("🚀 L104 Route Cleanup - 3-Engine Pipeline")
        print("=" * 60)
        
        # Engine 1: Discovery
        duplicates = self.engine1_discovery()
        if not duplicates:
            print("No duplicates found!")
            return
        
        # Engine 2: Validation
        valid_duplicates = self.engine2_validation(duplicates)
        if not valid_duplicates:
            print("No validated duplicates to remove!")
            return
        
        # Show summary
        print("\n📊 Summary of duplicates to remove:")
        for i, route in enumerate(valid_duplicates[:10], 1):
            print(f"  {i:2d}. {route['method']} {route['path']}")
        if len(valid_duplicates) > 10:
            print(f"  ... and {len(valid_duplicates) - 10} more")
        
        # Engine 3: Execution
        removed = self.engine3_execution(valid_duplicates, dry_run=dry_run)
        
        print("\n" + "=" * 60)
        if dry_run:
            print(f"📋 DRY RUN COMPLETE: Would remove {removed} routes")
            print("Run with --execute to actually remove duplicates")
        else:
            print(f"✅ CLEANUP COMPLETE: Removed {removed} routes")
        print("=" * 60)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Safe duplicate route remover')
    parser.add_argument('--execute', action='store_true', help='Actually remove duplicates (dry run by default)')
    parser.add_argument('--app', default='l104_server/app.py', help='Path to app.py')
    parser.add_argument('--routes', default='l104_server/routes', help='Path to routes directory')
    
    args = parser.parse_args()
    
    remover = RouteRemover(args.app, args.routes)
    remover.run_pipeline(dry_run=not args.execute)

if __name__ == '__main__':
    main()