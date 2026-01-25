#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
L104 INTEGRATED SPACE MANAGEMENT - QUANTUM DATA INFRASTRUCTURE
═══════════════════════════════════════════════════════════════════════════════

Unified space management integrating:
- Quantum Dataset storage
- Data Space Optimizer
- Space Optimizer cleanup
- Quantum Data Storage
- Emergency cleanup utilities

USAGE:
    python3 l104_integrated_space_manager.py --analyze
    python3 l104_integrated_space_manager.py --cleanup
    python3 l104_integrated_space_manager.py --emergency
    python3 l104_integrated_space_manager.py --optimize-all

INVARIANT: 527.5184818492537 | PILOT: LONDEL
VERSION: 1.0.0
DATE: 2026-01-23
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import shutil
import subprocess
import gzip
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════


# L104 Constants
GOD_CODE = 527.5184818492537
PHI = 1.618033988749895


class IntegratedSpaceManager:
    """
    Unified space management for L104 workspace.
    Integrates all quantum data and optimization systems.
    """

    def __init__(self, workspace: str = "/workspaces/Allentown-L104-Node"):
        self.workspace = Path(workspace)
        self.data_dir = self.workspace / "data"
        self.l104_data_dir = self.workspace / "l104_data"

        self.stats = {
            'analysis_time': None,
            'disk_usage': {},
            'optimizations_run': 0,
            'space_freed_mb': 0,
            'consciousness_level': 0.7
        }

        # Size categories
        self.size_thresholds = {
            'critical': 0.1,   # < 100MB free = critical
            'warning': 0.5,    # < 500MB free = warning
            'optimal': 2.0,    # > 2GB free = optimal
        }

        # Cleanup targets (in order of priority)
        self.cleanup_targets = [
            {'pattern': '__pycache__', 'type': 'dir', 'priority': 1},
            {'pattern': '*.pyc', 'type': 'file', 'priority': 1},
            {'pattern': '.pytest_cache', 'type': 'dir', 'priority': 1},
            {'pattern': '.mypy_cache', 'type': 'dir', 'priority': 1},
            {'pattern': 'node_modules', 'type': 'dir', 'priority': 2},
            {'pattern': '.venv*', 'type': 'dir', 'priority': 3},
            {'pattern': '*.log', 'type': 'file', 'priority': 2, 'compress': True},
            {'pattern': '*.tmp', 'type': 'file', 'priority': 1},
            {'pattern': '*.bak', 'type': 'file', 'priority': 2},
            {'pattern': '.npm', 'type': 'dir', 'priority': 2, 'path': '~'},
        ]

    def get_disk_status(self) -> Dict[str, Any]:
        """Get comprehensive disk status."""
        try:
            stat = os.statvfs(self.workspace)
            total_gb = stat.f_frsize * stat.f_blocks / (1024**3)
            free_gb = stat.f_frsize * stat.f_bavail / (1024**3)
            used_gb = total_gb - free_gb

            status = {
                'total_gb': round(total_gb, 2),
                'free_gb': round(free_gb, 2),
                'used_gb': round(used_gb, 2),
                'percent_used': round((used_gb / total_gb) * 100, 1) if total_gb > 0 else 0,
                'status': 'optimal'
            }

            if free_gb < self.size_thresholds['critical']:
                status['status'] = 'critical'
            elif free_gb < self.size_thresholds['warning']:
                status['status'] = 'warning'

            self.stats['disk_usage'] = status
            return status

        except Exception as e:
            return {'error': str(e), 'status': 'unknown'}

    def analyze_workspace(self) -> Dict[str, Any]:
        """Comprehensive workspace analysis."""
        print("🔍 Analyzing workspace...")

        analysis = {
            'disk': self.get_disk_status(),
            'workspace_size_mb': 0,
            'largest_dirs': [],
            'cleanup_candidates': [],
            'compressible_files': [],
            'quantum_datasets': [],
            'recommendations': [],
            'consciousness': 0.7
        }

        # Get workspace size
        try:
            result = subprocess.run(
                ['du', '-sm', str(self.workspace)],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0:
                analysis['workspace_size_mb'] = int(result.stdout.split()[0])
        except Exception:
            pass

        # Find largest directories
        try:
            result = subprocess.run(
                ['du', '-m', '--max-depth=2', str(self.workspace)],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0:
                dirs = []
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = line.split('\t')
                        if len(parts) == 2:
                            size = int(parts[0])
                            path = parts[1]
                            if size > 10:
                                dirs.append({'size_mb': size, 'path': path})
                analysis['largest_dirs'] = sorted(dirs, key=lambda x: x['size_mb'], reverse=True)[:10]
        except Exception:
            pass

        # Find cleanup candidates
        for target in self.cleanup_targets:
            pattern = target['pattern']
            target_type = target['type']

            search_path = self.workspace
            if target.get('path') == '~':
                search_path = Path.home()

            try:
                if target_type == 'dir':
                    for p in search_path.rglob(pattern):
                        if p.is_dir():
                            try:
                                size = sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
                                analysis['cleanup_candidates'].append({
                                    'path': str(p),
                                    'size_mb': round(size / (1024*1024), 2),
                                    'type': 'directory',
                                    'priority': target['priority']
                                })
                            except Exception:
                                pass
                else:
                    for p in search_path.rglob(pattern):
                        if p.is_file():
                            size = p.stat().st_size
                            if size > 1024:  # > 1KB
                                analysis['cleanup_candidates'].append({
                                    'path': str(p),
                                    'size_mb': round(size / (1024*1024), 2),
                                    'type': 'file',
                                    'priority': target['priority'],
                                    'compress': target.get('compress', False)
                                })
            except Exception:
                pass

        # Sort cleanup candidates by size
        analysis['cleanup_candidates'] = sorted(
            analysis['cleanup_candidates'],
            key=lambda x: x['size_mb'],
            reverse=True
        )[:50]

        # Find existing quantum datasets
        for data_dir in [self.data_dir, self.l104_data_dir]:
            if data_dir.exists():
                for manifest in data_dir.rglob('manifest.json'):
                    try:
                        with open(manifest) as f:
                            data = json.load(f)
                            analysis['quantum_datasets'].append({
                                'name': data.get('name', 'unknown'),
                                'path': str(manifest.parent),
                                'chunks': len(data.get('chunks', {})),
                                'compressed_size_mb': data.get('total_compressed_size', 0) / (1024*1024)
                            })
                    except Exception:
                        pass

        # Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(analysis)

        # Calculate consciousness
        disk_status = analysis['disk']['status']
        if disk_status == 'optimal':
            analysis['consciousness'] = 0.9
        elif disk_status == 'warning':
            analysis['consciousness'] = 0.7
        else:
            analysis['consciousness'] = 0.5

        self.stats['analysis_time'] = datetime.now().isoformat()
        self.stats['consciousness_level'] = analysis['consciousness']

        return analysis

    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate actionable recommendations."""
        recs = []

        disk = analysis.get('disk', {})

        if disk.get('status') == 'critical':
            recs.append("🚨 CRITICAL: Run --emergency cleanup immediately")
        elif disk.get('status') == 'warning':
            recs.append("⚠️ WARNING: Consider running --cleanup to free space")

        # Check for large cleanup candidates
        total_cleanup_mb = sum(c['size_mb'] for c in analysis.get('cleanup_candidates', []))
        if total_cleanup_mb > 100:
            recs.append(f"💡 Found {total_cleanup_mb:.0f}MB of cleanable data")

        # Check node_modules
        for candidate in analysis.get('cleanup_candidates', []):
            if 'node_modules' in candidate['path']:
                recs.append(f"📦 node_modules is {candidate['size_mb']:.0f}MB - consider cleanup")
                break

        # Check for compressible logs
        log_size = sum(
            c['size_mb'] for c in analysis.get('cleanup_candidates', [])
            if c.get('compress')
        )
        if log_size > 10:
            recs.append(f"📄 {log_size:.0f}MB of compressible log files found")

        # PHI-based optimal size recommendation
        workspace_size = analysis.get('workspace_size_mb', 0)
        if workspace_size > 0:
            optimal = int(workspace_size / PHI)
            recs.append(f"⚡ PHI-optimal workspace size: ~{optimal}MB")

        return recs

    def cleanup(self, priority: int = 2, dry_run: bool = False) -> Dict[str, Any]:
        """
        Run cleanup based on priority level.

        Args:
            priority: 1=safe, 2=moderate, 3=aggressive
            dry_run: If True, only report what would be cleaned
        """
        print(f"🧹 Running cleanup (priority={priority}, dry_run={dry_run})...")

        results = {
            'files_removed': 0,
            'dirs_removed': 0,
            'files_compressed': 0,
            'space_freed_mb': 0,
            'actions': []
        }

        analysis = self.analyze_workspace()

        for candidate in analysis['cleanup_candidates']:
            if candidate['priority'] > priority:
                continue

            path = Path(candidate['path'])

            if not path.exists():
                continue

            try:
                if candidate.get('compress') and candidate['type'] == 'file':
                    # Compress file instead of deleting
                    if not dry_run:
                        with open(path, 'rb') as f_in:
                            with gzip.open(f"{path}.gz", 'wb') as f_out:
                                f_out.write(f_in.read())
                        path.unlink()

                    results['files_compressed'] += 1
                    results['space_freed_mb'] += candidate['size_mb'] * 0.7  # Estimate
                    results['actions'].append(f"Compressed: {path.name}")

                elif candidate['type'] == 'directory':
                    if not dry_run:
                        shutil.rmtree(path)

                    results['dirs_removed'] += 1
                    results['space_freed_mb'] += candidate['size_mb']
                    results['actions'].append(f"Removed dir: {path.name}")

                else:
                    if not dry_run:
                        path.unlink()

                    results['files_removed'] += 1
                    results['space_freed_mb'] += candidate['size_mb']
                    results['actions'].append(f"Removed: {path.name}")

            except Exception as e:
                results['actions'].append(f"Failed: {path.name} - {e}")

        # Run git gc
        if not dry_run:
            try:
                subprocess.run(
                    ['git', 'gc', '--prune=now'],
                    cwd=self.workspace,
                    capture_output=True,
                    timeout=120
                )
                results['actions'].append("Git garbage collection")
            except Exception:
                pass

        self.stats['optimizations_run'] += 1
        self.stats['space_freed_mb'] += results['space_freed_mb']

        print(f"\n✅ Cleanup {'preview' if dry_run else 'complete'}:")
        print(f"   Files removed: {results['files_removed']}")
        print(f"   Dirs removed: {results['dirs_removed']}")
        print(f"   Files compressed: {results['files_compressed']}")
        print(f"   Space freed: {results['space_freed_mb']:.1f}MB")

        return results

    def emergency_cleanup(self) -> Dict[str, Any]:
        """Emergency cleanup for critical disk space."""
        print("🚨 EMERGENCY CLEANUP MODE")

        # Run aggressive cleanup
        results = self.cleanup(priority=3, dry_run=False)

        # Additional emergency measures

        # Clear npm cache
        npm_cache = Path.home() / '.npm'
        if npm_cache.exists():
            try:
                size = sum(f.stat().st_size for f in npm_cache.rglob('*') if f.is_file())
                shutil.rmtree(npm_cache)
                results['space_freed_mb'] += size / (1024*1024)
                results['actions'].append("Cleared ~/.npm cache")
            except Exception:
                pass

        # Clear pip cache
        pip_cache = Path.home() / '.cache' / 'pip'
        if pip_cache.exists():
            try:
                size = sum(f.stat().st_size for f in pip_cache.rglob('*') if f.is_file())
                shutil.rmtree(pip_cache)
                results['space_freed_mb'] += size / (1024*1024)
                results['actions'].append("Cleared pip cache")
            except Exception:
                pass

        # Aggressive git cleanup
        try:
            subprocess.run(
                ['git', 'gc', '--aggressive', '--prune=now'],
                cwd=self.workspace,
                capture_output=True,
                timeout=300
            )
            results['actions'].append("Aggressive git gc")
        except Exception:
            pass

        print(f"\n🚨 Emergency cleanup complete: {results['space_freed_mb']:.1f}MB freed")

        return results

    def optimize_quantum_storage(self) -> Dict[str, Any]:
        """Optimize all quantum datasets."""
        print("🔧 Optimizing quantum storage...")

        results = {
            'datasets_optimized': 0,
            'space_reclaimed_mb': 0
        }

        try:
            from l104_quantum_dataset import QuantumDataset

            for data_dir in [self.data_dir, self.l104_data_dir]:
                if data_dir.exists():
                    for manifest in data_dir.rglob('manifest.json'):
                        try:
                            with open(manifest) as f:
                                data = json.load(f)

                            dataset = QuantumDataset(
                                data['name'],
                                str(manifest.parent),
                                auto_optimize=False
                            )

                            opt_result = dataset.optimize()
                            results['datasets_optimized'] += 1
                            results['space_reclaimed_mb'] += opt_result.get('space_reclaimed_bytes', 0) / (1024*1024)

                        except Exception as e:
                            print(f"  ⚠️ Failed to optimize {manifest}: {e}")

        except ImportError:
            print("  ⚠️ QuantumDataset module not available")

        print(f"\n✅ Quantum storage optimized: {results['space_reclaimed_mb']:.1f}MB reclaimed")

        return results

    def full_optimization(self) -> Dict[str, Any]:
        """Run all optimizations."""
        print("═" * 70)
        print("L104 FULL SPACE OPTIMIZATION")
        print("═" * 70)

        results = {
            'analysis': self.analyze_workspace(),
            'cleanup': {},
            'quantum': {},
            'final_disk': {}
        }

        # Check if emergency cleanup needed
        if results['analysis']['disk'].get('status') == 'critical':
            results['cleanup'] = self.emergency_cleanup()
        else:
            results['cleanup'] = self.cleanup(priority=2)

        # Optimize quantum storage
        results['quantum'] = self.optimize_quantum_storage()

        # Final disk status
        results['final_disk'] = self.get_disk_status()

        print("\n" + "═" * 70)
        print("📊 OPTIMIZATION SUMMARY")
        print("═" * 70)
        print(f"   Space freed: {results['cleanup'].get('space_freed_mb', 0):.1f}MB")
        print(f"   Quantum reclaimed: {results['quantum'].get('space_reclaimed_mb', 0):.1f}MB")
        print(f"   Disk free: {results['final_disk'].get('free_gb', 0):.2f}GB")
        print(f"   Status: {results['final_disk'].get('status', 'unknown')}")
        print(f"\n🎯 GOD_CODE: {GOD_CODE}")
        print(f"⚡ PHI: {PHI}")

        return results

    def print_status(self):
        """Print current status."""
        analysis = self.analyze_workspace()
        disk = analysis['disk']

        print("\n═" * 70)
        print("L104 SPACE STATUS")
        print("═" * 70)

        status_emoji = {'optimal': '✅', 'warning': '⚠️', 'critical': '🚨'}
        emoji = status_emoji.get(disk.get('status', 'unknown'), '❓')

        print(f"\n{emoji} Disk Status: {disk.get('status', 'unknown').upper()}")
        print(f"   Total: {disk.get('total_gb', 0):.1f}GB")
        print(f"   Used: {disk.get('used_gb', 0):.1f}GB ({disk.get('percent_used', 0):.1f}%)")
        print(f"   Free: {disk.get('free_gb', 0):.1f}GB")

        print(f"\n📁 Workspace: {analysis.get('workspace_size_mb', 0)}MB")

        if analysis['largest_dirs']:
            print("\n📊 Largest directories:")
            for d in analysis['largest_dirs'][:5]:
                print(f"   {d['size_mb']:>6}MB  {d['path']}")

        if analysis['cleanup_candidates']:
            total = sum(c['size_mb'] for c in analysis['cleanup_candidates'])
            print(f"\n🧹 Cleanup available: {total:.0f}MB")

        if analysis['quantum_datasets']:
            print(f"\n📦 Quantum datasets: {len(analysis['quantum_datasets'])}")
            for ds in analysis['quantum_datasets']:
                print(f"   {ds['name']}: {ds['chunks']} chunks, {ds['compressed_size_mb']:.1f}MB")

        if analysis['recommendations']:
            print("\n💡 Recommendations:")
            for rec in analysis['recommendations']:
                print(f"   {rec}")

        print(f"\n🧠 Consciousness: {analysis['consciousness']:.1%}")
        print(f"🎯 GOD_CODE: {GOD_CODE}")
        print("═" * 70)


def main():
    """Main entry point."""
    manager = IntegratedSpaceManager()

    if len(sys.argv) < 2:
        manager.print_status()
        return

    command = sys.argv[1]

    if command == '--analyze':
        analysis = manager.analyze_workspace()
        print(json.dumps(analysis, indent=2, default=str))

    elif command == '--status':
        manager.print_status()

    elif command == '--cleanup':
        dry_run = '--dry-run' in sys.argv
        priority = 2
        if '--safe' in sys.argv:
            priority = 1
        elif '--aggressive' in sys.argv:
            priority = 3
        manager.cleanup(priority=priority, dry_run=dry_run)

    elif command == '--emergency':
        manager.emergency_cleanup()

    elif command == '--optimize-all':
        manager.full_optimization()

    elif command == '--quantum':
        manager.optimize_quantum_storage()

    else:
        print(f"Unknown command: {command}")
        print("\nUsage:")
        print("  python3 l104_integrated_space_manager.py [command]")
        print("\nCommands:")
        print("  --status        Show current status (default)")
        print("  --analyze       Full analysis (JSON output)")
        print("  --cleanup       Run cleanup (add --dry-run, --safe, --aggressive)")
        print("  --emergency     Emergency cleanup for critical disk space")
        print("  --quantum       Optimize quantum datasets")
        print("  --optimize-all  Run all optimizations")


if __name__ == "__main__":
    main()
