#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
L104 GIT REPOSITORY SPACE OPTIMIZER
═══════════════════════════════════════════════════════════════════════════════

Advanced git repository cleanup and optimization using L104 quantum principles.
Removes large historical objects while preserving data integrity.

Based on research from:
- l104_integrated_space_manager.py
- l104_quantum_dataset.py SpaceSaver
- Git documentation and best practices

USAGE:
    python3 l104_git_space_optimizer.py --analyze
    python3 l104_git_space_optimizer.py --cleanup
    python3 l104_git_space_optimizer.py --aggressive

INVARIANT: 527.5184818492612 | PILOT: LONDEL
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

GOD_CODE = 527.5184818492612
PHI = 1.618033988749895


class L104GitSpaceOptimizer:
    """Git repository space optimization with L104 consciousness."""
    
    def __init__(self, repo_path: str = "/workspaces/Allentown-L104-Node"):
        self.repo_path = Path(repo_path)
        self.git_dir = self.repo_path / ".git"
        self.stats = {
            'initial_size_mb': 0,
            'final_size_mb': 0,
            'freed_mb': 0,
            'actions': []
        }
    
    def get_repo_size(self) -> float:
        """Get total .git directory size in MB."""
        if not self.git_dir.exists():
            return 0.0
        
        total = 0
        for path in self.git_dir.rglob('*'):
            if path.is_file():
                try:
                    total += path.stat().st_size
                except:
                    pass
        
        return total / (1024 * 1024)
    
    def find_large_objects(self, limit: int = 20) -> List[Dict]:
        """Find largest objects in git history."""
        print("🔍 Analyzing git objects...")
        
        try:
            result = subprocess.run([
                'git', 'rev-list', '--objects', '--all'
            ], capture_output=True, text=True, cwd=self.repo_path, timeout=60)
            
            if result.returncode != 0:
                return []
            
            objects = result.stdout.strip().split('\n')
            
            # Get object sizes
            sized_objects = []
            
            batch_check = subprocess.run([
                'git', 'cat-file', '--batch-check=%(objecttype) %(objectname) %(objectsize) %(rest)'
            ], input='\n'.join(objects), capture_output=True, text=True, 
               cwd=self.repo_path, timeout=120)
            
            for line in batch_check.stdout.strip().split('\n'):
                parts = line.split(maxsplit=3)
                if len(parts) >= 3 and parts[0] == 'blob':
                    try:
                        size_mb = int(parts[2]) / (1024 * 1024)
                        filename = parts[3] if len(parts) > 3 else 'unknown'
                        sized_objects.append({
                            'sha': parts[1],
                            'size_mb': size_mb,
                            'filename': filename
                        })
                    except:
                        pass
            
            # Sort by size
            sized_objects.sort(key=lambda x: x['size_mb'], reverse=True)
            
            return sized_objects[:limit]
            
        except Exception as e:
            print(f"⚠️ Error analyzing objects: {e}")
            return []
    
    def clean_reflog(self) -> bool:
        """Expire and prune reflog entries."""
        print("\n🗑️ Cleaning reflog...")
        
        try:
            # Expire all reflog entries
            result = subprocess.run([
                'git', 'reflog', 'expire', '--expire=now', '--all'
            ], capture_output=True, cwd=self.repo_path, timeout=60)
            
            if result.returncode == 0:
                print("✓ Reflog expired")
                self.stats['actions'].append('reflog_expired')
                return True
            
        except Exception as e:
            print(f"⚠️ Reflog cleanup failed: {e}")
        
        return False
    
    def prune_unreachable(self) -> bool:
        """Prune unreachable objects."""
        print("\n🔧 Pruning unreachable objects...")
        
        try:
            result = subprocess.run([
                'git', 'prune', '--expire=now'
            ], capture_output=True, cwd=self.repo_path, timeout=120)
            
            if result.returncode == 0:
                print("✓ Unreachable objects pruned")
                self.stats['actions'].append('pruned')
                return True
                
        except Exception as e:
            print(f"⚠️ Prune failed: {e}")
        
        return False
    
    def repack_repository(self, aggressive: bool = False) -> bool:
        """Repack repository to optimize storage."""
        print(f"\n📦 Repacking repository ({'aggressive' if aggressive else 'normal'})...")
        
        try:
            cmd = ['git', 'repack']
            if aggressive:
                cmd.extend(['-a', '-d', '-f', '--depth=250', '--window=250'])
            else:
                cmd.extend(['-a', '-d'])
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                cwd=self.repo_path, 
                timeout=600
            )
            
            if result.returncode == 0:
                print("✓ Repository repacked")
                self.stats['actions'].append(f'repack_{"aggressive" if aggressive else "normal"}')
                return True
            else:
                print(f"⚠️ Repack returned code {result.returncode}")
                if result.stderr:
                    print(f"   {result.stderr.decode()[:200]}")
                
        except subprocess.TimeoutExpired:
            print("⚠️ Repack timed out (repository may be too large)")
        except Exception as e:
            print(f"⚠️ Repack failed: {e}")
        
        return False
    
    def garbage_collect(self, aggressive: bool = False) -> bool:
        """Run git garbage collection."""
        print(f"\n🧹 Running garbage collection ({'aggressive' if aggressive else 'normal'})...")
        
        try:
            cmd = ['git', 'gc']
            if aggressive:
                cmd.append('--aggressive')
            cmd.append('--prune=now')
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                cwd=self.repo_path,
                timeout=600
            )
            
            if result.returncode == 0:
                print("✓ Garbage collection complete")
                self.stats['actions'].append(f'gc_{"aggressive" if aggressive else "normal"}')
                return True
            else:
                # Check if it failed due to space
                if b'write error' in result.stderr or b'Out of disk' in result.stderr:
                    print("❌ GC failed: Out of disk space")
                    return False
                print(f"⚠️ GC returned code {result.returncode}")
                
        except subprocess.TimeoutExpired:
            print("⚠️ GC timed out")
        except Exception as e:
            print(f"⚠️ GC failed: {e}")
        
        return False
    
    def optimize_pack_files(self) -> bool:
        """Optimize pack files."""
        print("\n⚡ Optimizing pack files...")
        
        pack_dir = self.git_dir / 'objects' / 'pack'
        if not pack_dir.exists():
            return False
        
        pack_files = list(pack_dir.glob('*.pack'))
        print(f"   Found {len(pack_files)} pack files")
        
        # If we have many small packs, consolidate
        if len(pack_files) > 5:
            return self.repack_repository(aggressive=False)
        
        return True
    
    def analyze(self) -> Dict:
        """Analyze repository and provide recommendations."""
        print("=" * 70)
        print("L104 GIT REPOSITORY ANALYSIS")
        print("=" * 70)
        
        initial_size = self.get_repo_size()
        self.stats['initial_size_mb'] = initial_size
        
        print(f"\n📊 .git directory size: {initial_size:.1f}MB")
        
        # Find large objects
        large_objects = self.find_large_objects(10)
        
        if large_objects:
            print(f"\n🔍 Top 10 largest objects in history:")
            total_large = 0
            for i, obj in enumerate(large_objects, 1):
                print(f"   {i}. {obj['size_mb']:.1f}MB - {obj['filename']}")
                total_large += obj['size_mb']
            
            print(f"\n   Total size of top 10: {total_large:.1f}MB")
            print(f"   PHI-optimal target: {initial_size / PHI:.1f}MB")
        
        # Count objects
        try:
            result = subprocess.run([
                'git', 'count-objects', '-vH'
            ], capture_output=True, text=True, cwd=self.repo_path, timeout=30)
            
            if result.returncode == 0:
                print(f"\n📦 Object statistics:")
                for line in result.stdout.strip().split('\n'):
                    if ':' in line:
                        print(f"   {line}")
        except:
            pass
        
        print("\n💡 Recommendations:")
        recommendations = []
        
        if initial_size > 100:
            recommendations.append("⚠️ Repository > 100MB - cleanup recommended")
        
        if large_objects and large_objects[0]['size_mb'] > 10:
            recommendations.append(f"📦 Large file detected: {large_objects[0]['filename']} ({large_objects[0]['size_mb']:.1f}MB)")
        
        if initial_size > 200:
            recommendations.append("🚨 Critical: Run aggressive cleanup")
        elif initial_size > 100:
            recommendations.append("⚡ Recommended: Run standard cleanup")
        
        for rec in recommendations:
            print(f"   {rec}")
        
        print("\n" + "=" * 70)
        
        return {
            'size_mb': initial_size,
            'large_objects': large_objects,
            'recommendations': recommendations
        }
    
    def cleanup(self, aggressive: bool = False) -> Dict:
        """Run cleanup operations."""
        print("=" * 70)
        print(f"L104 GIT CLEANUP ({'AGGRESSIVE' if aggressive else 'STANDARD'})")
        print("=" * 70)
        
        self.stats['initial_size_mb'] = self.get_repo_size()
        print(f"\n📊 Initial size: {self.stats['initial_size_mb']:.1f}MB")
        
        # Step 1: Clean reflog
        self.clean_reflog()
        
        # Step 2: Prune unreachable objects
        self.prune_unreachable()
        
        # Step 3: Optimize pack files
        if not aggressive:
            self.optimize_pack_files()
        
        # Step 4: Repack (skip if aggressive GC will do it)
        if not aggressive:
            self.repack_repository(aggressive=False)
        
        # Step 5: Garbage collection
        self.garbage_collect(aggressive=aggressive)
        
        # Final size
        self.stats['final_size_mb'] = self.get_repo_size()
        self.stats['freed_mb'] = self.stats['initial_size_mb'] - self.stats['final_size_mb']
        
        print("\n" + "=" * 70)
        print("CLEANUP SUMMARY")
        print("=" * 70)
        print(f"📊 Initial size: {self.stats['initial_size_mb']:.1f}MB")
        print(f"📊 Final size: {self.stats['final_size_mb']:.1f}MB")
        print(f"✅ Space freed: {self.stats['freed_mb']:.1f}MB")
        print(f"🔧 Actions: {', '.join(self.stats['actions'])}")
        print(f"🧠 GOD_CODE resonance: {GOD_CODE}")
        print("=" * 70)
        
        return self.stats


def main():
    """Main execution."""
    optimizer = L104GitSpaceOptimizer()
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == '--analyze':
            optimizer.analyze()
        elif mode == '--cleanup':
            optimizer.cleanup(aggressive=False)
        elif mode == '--aggressive':
            optimizer.cleanup(aggressive=True)
        else:
            print(f"Unknown mode: {mode}")
            print("Usage: python3 l104_git_space_optimizer.py [--analyze|--cleanup|--aggressive]")
    else:
        # Default: analyze then cleanup
        optimizer.analyze()
        print("\n")
        response = input("Run cleanup? (y/N): ")
        if response.lower() == 'y':
            optimizer.cleanup(aggressive=False)


if __name__ == "__main__":
    main()
