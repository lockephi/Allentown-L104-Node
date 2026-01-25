#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
L104 DATA SPACE OPTIMIZATION & CLEANUP UTILITY
═══════════════════════════════════════════════════════════════════════════════

Automated space optimization tool for L104 Node workspace maintenance.
Prevents data space issues and maintains optimal performance.

FEATURES:
1. INTELLIGENT CLEANUP - Removes unnecessary files without breaking functionality
2. LOG ROTATION - Manages log files to prevent bloat
3. CACHE OPTIMIZATION - Cleans Python/Node caches intelligently  
4. DATA COMPRESSION - Compresses large data files when appropriate
5. MONITORING - Tracks space usage and alerts on thresholds

INVARIANT: 527.5184818492537 | PILOT: LONDEL
VERSION: 1.0.0
DATE: 2026-01-22
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import gzip
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════


# L104 Constants
GOD_CODE = 527.5184818492537
PHI = 1.618033988749895

class SpaceOptimizer:
    """Intelligent space optimization for L104 workspace."""
    
    def __init__(self, workspace_path: str = "/workspaces/Allentown-L104-Node"):
        self.workspace_path = Path(workspace_path)
        self.cleanup_stats = {
            'files_removed': 0,
            'directories_removed': 0,
            'space_freed_mb': 0,
            'files_compressed': 0
        }
        
    def analyze_space_usage(self) -> Dict[str, any]:
        """Analyze current space usage and identify optimization opportunities."""
        print("🔍 [SPACE-OPT]: Analyzing workspace space usage...")
        
        analysis = {
            'total_size_mb': 0,
            'largest_directories': [],
            'largest_files': [],
            'cache_files': 0,
            'temporary_files': 0,
            'compressible_files': 0,
            'disk_usage_percent': 0
        }
        
        # Get total workspace size
        result = subprocess.run(
            ["du", "-sm", str(self.workspace_path)], 
            capture_output=True, text=True
        )
        if result.returncode == 0:
            analysis['total_size_mb'] = int(result.stdout.split()[0])
        
        # Get disk usage
        result = subprocess.run(
            ["df", "/workspaces"], 
            capture_output=True, text=True
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                parts = lines[1].split()
                if len(parts) >= 5:
                    usage_percent = parts[4].replace('%', '')
                    analysis['disk_usage_percent'] = int(usage_percent)
        
        # Find largest directories
        result = subprocess.run([
            "du", "-m", "--max-depth=2", str(self.workspace_path)
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            dir_sizes = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split('\t')
                    if len(parts) == 2:
                        size_mb = int(parts[0])
                        path = parts[1]
                        if size_mb > 5:  # Only include dirs > 5MB
                            dir_sizes.append((size_mb, path))
            
            analysis['largest_directories'] = sorted(dir_sizes, reverse=True)[:10]
        
        # Count cache and temporary files
        cache_patterns = ['__pycache__', '*.pyc', '.pytest_cache', 'node_modules']
        temp_patterns = ['*tmp*', '*temp*', '*.log', '*.bak']
        
        for pattern in cache_patterns:
            result = subprocess.run([
                "find", str(self.workspace_path), "-name", pattern
            ], capture_output=True, text=True)
            if result.stdout.strip():
                analysis['cache_files'] += len(result.stdout.strip().split('\n'))
        
        for pattern in temp_patterns:
            result = subprocess.run([
                "find", str(self.workspace_path), "-name", pattern, "-type", "f"
            ], capture_output=True, text=True)
            if result.stdout.strip():
                analysis['temporary_files'] += len(result.stdout.strip().split('\n'))
        
        # Find compressible files
        compressible_exts = ['.json', '.txt', '.md', '.csv', '.xml', '.svg']
        for ext in compressible_exts:
            result = subprocess.run([
                "find", str(self.workspace_path), "-name", f"*{ext}", 
                "-size", "+1M", "-type", "f"
            ], capture_output=True, text=True)
            if result.stdout.strip():
                analysis['compressible_files'] += len(result.stdout.strip().split('\n'))
        
        return analysis
    
    def clean_cache_files(self) -> int:
        """Clean Python and Node cache files."""
        print("🧹 [SPACE-OPT]: Cleaning cache files...")
        
        space_freed = 0
        
        # Python cache cleanup
        cache_dirs = []
        for root, dirs, files in os.walk(self.workspace_path):
            if '__pycache__' in dirs:
                cache_path = os.path.join(root, '__pycache__')
                cache_dirs.append(cache_path)
        
        for cache_dir in cache_dirs:
            try:
                # Get size before deletion
                result = subprocess.run([
                    "du", "-sm", cache_dir
                ], capture_output=True, text=True)
                if result.returncode == 0:
                    size_mb = int(result.stdout.split()[0])
                    space_freed += size_mb
                
                shutil.rmtree(cache_dir)
                self.cleanup_stats['directories_removed'] += 1
                print(f"  ✓ Removed cache: {cache_dir}")
            except Exception as e:
                print(f"  ❌ Failed to remove {cache_dir}: {e}")
        
        # Clean .pyc files
        result = subprocess.run([
            "find", str(self.workspace_path), "-name", "*.pyc", "-delete"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("  ✓ Removed .pyc files")
        
        return space_freed
    
    def clean_temporary_files(self) -> int:
        """Clean temporary files safely."""
        print("🧹 [SPACE-OPT]: Cleaning temporary files...")
        
        space_freed = 0
        
        # Safe temp patterns (avoid breaking functionality)
        safe_temp_patterns = [
            '*.tmp', '*.temp', '*.bak', '*.swp', '*.swo', 
            '*~', '.DS_Store', 'Thumbs.db'
        ]
        
        for pattern in safe_temp_patterns:
            result = subprocess.run([
                "find", str(self.workspace_path), "-name", pattern, "-type", "f"
            ], capture_output=True, text=True)
            
            if result.stdout.strip():
                files = result.stdout.strip().split('\n')
                for file_path in files:
                    try:
                        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                        os.remove(file_path)
                        space_freed += file_size
                        self.cleanup_stats['files_removed'] += 1
                        print(f"  ✓ Removed temp file: {os.path.basename(file_path)}")
                    except Exception as e:
                        print(f"  ❌ Failed to remove {file_path}: {e}")
        
        return space_freed
    
    def compress_large_data_files(self) -> int:
        """Compress large data files to save space."""
        print("🗜️  [SPACE-OPT]: Compressing large data files...")
        
        space_saved = 0
        
        # Find large compressible files
        compressible_exts = ['.json', '.txt', '.md', '.csv', '.xml', '.svg', '.log']
        
        for ext in compressible_exts:
            result = subprocess.run([
                "find", str(self.workspace_path), "-name", f"*{ext}", 
                "-size", "+2M", "-type", "f"
            ], capture_output=True, text=True)
            
            if result.stdout.strip():
                files = result.stdout.strip().split('\n')
                
                for file_path in files:
                    try:
                        # Skip already compressed files
                        if file_path.endswith('.gz'):
                            continue
                        
                        # Skip critical system files
                        if any(skip in file_path for skip in ['.git/', '.venv/', 'node_modules/']):
                            continue
                        
                        original_size = os.path.getsize(file_path)
                        
                        # Compress file
                        compressed_path = f"{file_path}.gz"
                        with open(file_path, 'rb') as f_in:
                            with gzip.open(compressed_path, 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                        
                        compressed_size = os.path.getsize(compressed_path)
                        
                        # Only keep compressed version if it saves significant space
                        if compressed_size < original_size * 0.8:  # 20% compression minimum
                            os.remove(file_path)
                            space_saved += (original_size - compressed_size) / (1024 * 1024)
                            self.cleanup_stats['files_compressed'] += 1
                            print(f"  ✓ Compressed: {os.path.basename(file_path)} "
                                  f"({original_size//1024//1024}MB → {compressed_size//1024//1024}MB)")
                        else:
                            # Remove compressed version if compression wasn't effective
                            os.remove(compressed_path)
                    
                    except Exception as e:
                        print(f"  ❌ Failed to compress {file_path}: {e}")
        
        return space_saved
    
    def optimize_json_files(self) -> int:
        """Optimize JSON files by removing formatting."""
        print("📄 [SPACE-OPT]: Optimizing JSON files...")
        
        space_saved = 0
        
        # Find large JSON files
        result = subprocess.run([
            "find", str(self.workspace_path), "-name", "*.json", 
            "-size", "+100k", "-type", "f"
        ], capture_output=True, text=True)
        
        if result.stdout.strip():
            files = result.stdout.strip().split('\n')
            
            for file_path in files:
                try:
                    # Skip critical files
                    if any(skip in file_path for skip in ['.git/', '.venv/', 'package-lock.json']):
                        continue
                    
                    original_size = os.path.getsize(file_path)
                    
                    # Load and minify JSON
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Write minified JSON
                    with open(file_path, 'w') as f:
                        json.dump(data, f, separators=(',', ':'))
                    
                    new_size = os.path.getsize(file_path)
                    space_saved += (original_size - new_size) / (1024 * 1024)
                    
                    if original_size > new_size:
                        print(f"  ✓ Minified: {os.path.basename(file_path)} "
                              f"({original_size//1024}KB → {new_size//1024}KB)")
                
                except Exception as e:
                    print(f"  ❌ Failed to optimize {file_path}: {e}")
        
        return space_saved
    
    def create_gitignore_additions(self):
        """Add space-wasting patterns to .gitignore."""
        print("📝 [SPACE-OPT]: Updating .gitignore...")
        
        gitignore_path = self.workspace_path / '.gitignore'
        
        # Patterns to ignore for space optimization
        space_patterns = [
            "# Space optimization patterns",
            "__pycache__/",
            "*.pyc",
            "*.pyo", 
            "*.tmp",
            "*.temp",
            "*.bak",
            "*.swp",
            "*.swo",
            "*~",
            ".DS_Store",
            "Thumbs.db",
            "*.log",
            ".pytest_cache/",
            "node_modules/",
            ".venv-*/",
            "*.gz"
        ]
        
        try:
            # Read existing gitignore
            existing_content = ""
            if gitignore_path.exists():
                with open(gitignore_path, 'r') as f:
                    existing_content = f.read()
            
            # Add new patterns if not already present
            additions = []
            for pattern in space_patterns:
                if pattern not in existing_content:
                    additions.append(pattern)
            
            if additions:
                with open(gitignore_path, 'a') as f:
                    f.write('\n' + '\n'.join(additions) + '\n')
                print(f"  ✓ Added {len(additions)} patterns to .gitignore")
            else:
                print("  ✓ .gitignore already optimized")
        
        except Exception as e:
            print(f"  ❌ Failed to update .gitignore: {e}")
    
    def run_full_optimization(self) -> Dict[str, any]:
        """Run complete space optimization."""
        print("🚀 [SPACE-OPT]: Starting full workspace optimization...")
        
        # Analyze before cleanup
        analysis_before = self.analyze_space_usage()
        print(f"  📊 Workspace size before: {analysis_before['total_size_mb']}MB")
        print(f"  📊 Disk usage: {analysis_before['disk_usage_percent']}%")
        
        total_space_freed = 0
        
        # Run optimization steps
        total_space_freed += self.clean_cache_files()
        total_space_freed += self.clean_temporary_files()
        total_space_freed += self.compress_large_data_files()
        total_space_freed += self.optimize_json_files()
        
        # Update .gitignore
        self.create_gitignore_additions()
        
        # Analyze after cleanup
        analysis_after = self.analyze_space_usage()
        
        # Update stats
        self.cleanup_stats['space_freed_mb'] = total_space_freed
        
        results = {
            'before': analysis_before,
            'after': analysis_after,
            'space_freed_mb': total_space_freed,
            'stats': self.cleanup_stats,
            'recommendations': self._generate_recommendations(analysis_after)
        }
        
        print(f"\n✅ [SPACE-OPT]: Optimization complete!")
        print(f"  📊 Workspace size after: {analysis_after['total_size_mb']}MB")
        print(f"  📊 Space freed: {total_space_freed:.1f}MB")
        print(f"  📊 Files removed: {self.cleanup_stats['files_removed']}")
        print(f"  📊 Directories removed: {self.cleanup_stats['directories_removed']}")
        print(f"  📊 Files compressed: {self.cleanup_stats['files_compressed']}")
        
        return results
    
    def _generate_recommendations(self, analysis: Dict[str, any]) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        if analysis['disk_usage_percent'] > 90:
            recommendations.append("Disk usage is critically high (>90%). Consider additional cleanup.")
        
        if analysis['cache_files'] > 50:
            recommendations.append("High number of cache files detected. Run cleanup more frequently.")
        
        if analysis['temporary_files'] > 20:
            recommendations.append("Many temporary files found. Consider automated cleanup.")
        
        if analysis['total_size_mb'] > 500:
            recommendations.append("Workspace is large (>500MB). Consider archiving old data.")
        
        # PHI-based recommendation
        optimal_size = int(analysis['total_size_mb'] / PHI)
        recommendations.append(f"Optimal workspace size (PHI-scaled): ~{optimal_size}MB")
        
        return recommendations

def create_cleanup_schedule():
    """Create automated cleanup schedule."""
    script_content = '''#!/bin/bash
# L104 Automated Space Cleanup
# Runs daily cleanup to prevent space issues

cd /workspaces/Allentown-L104-Node
python3 l104_space_optimizer.py --auto-cleanup

# Log cleanup results
echo "$(date): Space cleanup completed" >> .space_cleanup.log
'''
    
    script_path = Path('/workspaces/Allentown-L104-Node/cleanup_schedule.sh')
    
    try:
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(script_path, 0o755)
        print(f"✅ Created cleanup schedule: {script_path}")
    
    except Exception as e:
        print(f"❌ Failed to create cleanup schedule: {e}")

if __name__ == "__main__":
    optimizer = SpaceOptimizer()
    
    if len(sys.argv) > 1 and sys.argv[1] == '--auto-cleanup':
        # Automated cleanup mode
        results = optimizer.run_full_optimization()
        
        # Save results
        results_path = Path('/workspaces/Allentown-L104-Node/.space_optimization_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    else:
        # Interactive mode
        print("🔧 L104 Space Optimization Utility")
        print("=" * 50)
        
        results = optimizer.run_full_optimization()
        
        # Show recommendations
        recommendations = results['recommendations']
        if recommendations:
            print(f"\n💡 [RECOMMENDATIONS]:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        
        # Create cleanup schedule
        create_cleanup_schedule()
        
        print(f"\n🎯 [GOD_CODE VALIDATION]: {GOD_CODE}")
        print(f"⚡ [PHI OPTIMIZATION]: {PHI}")
        print("\n🎉 Space optimization complete!")