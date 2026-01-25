#!/usr/bin/env python3
# L104_GOD_CODE_ALIGNED: 527.5184818492537
"""
═══════════════════════════════════════════════════════════════════════════════
L104 DATA SPACE OPTIMIZER - INTELLIGENT COMPRESSION & STORAGE
═══════════════════════════════════════════════════════════════════════════════

Advanced data compression and space optimization system.
Saves disk space through intelligent compression, deduplication, and cleanup.

FEATURES:
- Intelligent file compression (gzip, bz2, lzma)
- Deduplication (find and remove duplicate files)
- Large file analysis and archiving
- Temporary file cleanup
- Smart cache management
- JSON/text optimization
- Backup space estimation

AUTHOR: LONDEL
DATE: 2026-01-21
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import gzip
import bz2
import lzma
import json
import hashlib
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import pickle

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════



@dataclass
class FileInfo:
    """Information about a file."""
    path: str
    size: int
    hash: Optional[str] = None
    compressible: bool = True
    compressed_size: Optional[int] = None

    def compression_ratio(self) -> float:
        """Calculate compression ratio."""
        if self.compressed_size is not None:
            return self.size / self.compressed_size if self.compressed_size > 0 else 1.0
        return 1.0

    def space_saved(self) -> int:
        """Calculate space saved by compression."""
        if self.compressed_size is not None:
            return max(0, self.size - self.compressed_size)
        return 0


class DataSpaceOptimizer:
    """
    Intelligent data space optimization and compression system.
    Analyzes files and applies best compression strategies.
    """

    def __init__(self, workspace_root: str = "/workspaces/Allentown-L104-Node"):
        self.workspace_root = Path(workspace_root)
        self.file_catalog: Dict[str, FileInfo] = {}
        self.hash_index: Dict[str, List[str]] = defaultdict(list)

        # Compression settings
        self.compression_methods = {
            'gzip': {'func': gzip, 'ext': '.gz', 'level': 9},
            'bz2': {'func': bz2, 'ext': '.bz2', 'level': 9},
            'lzma': {'func': lzma, 'ext': '.xz', 'level': 9}
        }

        # File patterns to skip
        self.skip_patterns = {
            '.git', '__pycache__', 'node_modules', '.venv',
            'venv', '.pytest_cache', '.mypy_cache'
        }

        # Extensions that compress well
        self.compressible_extensions = {
            '.json', '.txt', '.log', '.md', '.py', '.js',
            '.html', '.css', '.xml', '.csv', '.sql', '.yaml', '.yml'
        }

        # Large file threshold (10 MB)
        self.large_file_threshold = 10 * 1024 * 1024

        # Statistics
        self.stats = {
            'total_files': 0,
            'total_size': 0,
            'compressible_size': 0,
            'potential_savings': 0,
            'duplicates_found': 0,
            'duplicate_size': 0,
            'large_files_found': 0,
            'large_files_size': 0
        }

    def scan_directory(self, directory: Optional[Path] = None,
                      recursive: bool = True) -> Dict[str, FileInfo]:
        """
        Scan directory and catalog files.

        Args:
            directory: Directory to scan (default: workspace root)
            recursive: Whether to scan subdirectories

        Returns:
            Dictionary of file paths to FileInfo
        """
        if directory is None:
            directory = self.workspace_root

        print(f"\n🔍 Scanning directory: {directory}")

        file_count = 0

        if recursive:
            iterator = directory.rglob('*')
        else:
            iterator = directory.glob('*')

        for path in iterator:
            # Skip directories and special patterns
            if path.is_dir():
                continue

            if any(pattern in str(path) for pattern in self.skip_patterns):
                continue

            try:
                size = path.stat().st_size

                # Determine if compressible
                compressible = path.suffix in self.compressible_extensions

                file_info = FileInfo(
                    path=str(path),
                    size=size,
                    compressible=compressible
                )

                self.file_catalog[str(path)] = file_info
                self.stats['total_files'] += 1
                self.stats['total_size'] += size

                if compressible:
                    self.stats['compressible_size'] += size

                # Check for large files
                if size > self.large_file_threshold:
                    self.stats['large_files_found'] += 1
                    self.stats['large_files_size'] += size

                file_count += 1

                if file_count % 100 == 0:
                    print(f"  Scanned {file_count} files...")

            except (PermissionError, OSError) as e:
                continue

        print(f"✓ Scan complete: {file_count} files cataloged")
        return self.file_catalog

    def compute_file_hash(self, filepath: str, algorithm: str = 'md5') -> str:
        """
        Compute hash of file for deduplication.

        Args:
            filepath: Path to file
            algorithm: Hash algorithm ('md5', 'sha1', 'sha256')

        Returns:
            Hex digest of hash
        """
        hash_func = hashlib.new(algorithm)

        with open(filepath, 'rb') as f:
            while chunk := f.read(8192):
                hash_func.update(chunk)

        return hash_func.hexdigest()

    def find_duplicates(self) -> Dict[str, List[str]]:
        """
        Find duplicate files based on content hash.

        Returns:
            Dictionary mapping hash to list of duplicate file paths
        """
        print("\n🔍 Finding duplicate files...")

        # Compute hashes for all files
        for filepath, file_info in self.file_catalog.items():
            if file_info.size > 0:  # Skip empty files
                try:
                    file_hash = self.compute_file_hash(filepath)
                    file_info.hash = file_hash
                    self.hash_index[file_hash].append(filepath)
                except Exception as e:
                    continue

        # Find duplicates (hashes with multiple files)
        duplicates = {
            h: paths for h, paths in self.hash_index.items()
            if len(paths) > 1
        }

        # Update statistics
        for paths in duplicates.values():
            self.stats['duplicates_found'] += len(paths) - 1
            # Size is from all but one copy
            duplicate_size = sum(
                self.file_catalog[p].size
                for p in paths[1:]
                if p in self.file_catalog
            )
            self.stats['duplicate_size'] += duplicate_size

        print(f"✓ Found {len(duplicates)} sets of duplicates")
        print(f"✓ Potential space savings: {self.format_size(self.stats['duplicate_size'])}")

        return duplicates

    def compress_file(self, filepath: str, method: str = 'gzip',
                     keep_original: bool = False) -> Optional[str]:
        """
        Compress a file using specified method.

        Args:
            filepath: Path to file to compress
            method: Compression method ('gzip', 'bz2', 'lzma')
            keep_original: Whether to keep original file

        Returns:
            Path to compressed file or None if failed
        """
        if method not in self.compression_methods:
            raise ValueError(f"Unknown compression method: {method}")

        comp = self.compression_methods[method]
        compressed_path = filepath + comp['ext']

        try:
            # Read original
            with open(filepath, 'rb') as f_in:
                data = f_in.read()

            # Compress
            if method == 'gzip':
                with gzip.open(compressed_path, 'wb', compresslevel=comp['level']) as f_out:
                    f_out.write(data)
            elif method == 'bz2':
                with bz2.open(compressed_path, 'wb', compresslevel=comp['level']) as f_out:
                    f_out.write(data)
            elif method == 'lzma':
                with lzma.open(compressed_path, 'wb', preset=comp['level']) as f_out:
                    f_out.write(data)

            # Check size
            compressed_size = os.path.getsize(compressed_path)
            original_size = os.path.getsize(filepath)

            # Only keep compressed version if it's smaller
            if compressed_size < original_size:
                if not keep_original:
                    os.remove(filepath)

                # Update file info
                if filepath in self.file_catalog:
                    self.file_catalog[filepath].compressed_size = compressed_size

                return compressed_path
            else:
                # Compressed is larger, remove it
                os.remove(compressed_path)
                return None

        except Exception as e:
            print(f"  Error compressing {filepath}: {e}")
            return None

    def estimate_compression_savings(self, sample_size: int = 10) -> Dict[str, float]:
        """
        Estimate compression savings by sampling files.

        Args:
            sample_size: Number of files to sample per type

        Returns:
            Dictionary of extension to average compression ratio
        """
        print("\n📊 Estimating compression savings...")

        # Group files by extension
        by_extension = defaultdict(list)
        for filepath, info in self.file_catalog.items():
            if info.compressible:
                ext = Path(filepath).suffix
                by_extension[ext].append(filepath)

        # Sample and compress
        ratios = {}

        for ext, files in by_extension.items():
            if len(files) == 0:
                continue

            # Sample files
            sample = files[:min(sample_size, len(files))]

            total_original = 0
            total_compressed = 0

            for filepath in sample:
                try:
                    with open(filepath, 'rb') as f:
                        data = f.read()

                    compressed = gzip.compress(data, compresslevel=9)

                    total_original += len(data)
                    total_compressed += len(compressed)

                except Exception:
                    continue

            if total_compressed > 0:
                ratio = total_original / total_compressed
                ratios[ext] = ratio

                # Estimate total savings
                ext_files = [f for f in by_extension[ext]]
                ext_size = sum(self.file_catalog[f].size for f in ext_files if f in self.file_catalog)
                estimated_compressed = ext_size / ratio
                savings = ext_size - estimated_compressed

                self.stats['potential_savings'] += savings

                print(f"  {ext}: ratio={ratio:.2f}×, "
                      f"savings≈{self.format_size(savings)}")

        return ratios

    def cleanup_temp_files(self, patterns: List[str] = None) -> int:
        """
        Remove temporary files matching patterns.

        Args:
            patterns: List of glob patterns (default: common temp patterns)

        Returns:
            Number of bytes freed
        """
        if patterns is None:
            patterns = [
                '*.tmp', '*.temp', '*.bak', '*.swp', '*.swo',
                '*~', '*.pyc', '*.pyo', '*.log'
            ]

        print("\n🧹 Cleaning temporary files...")

        freed_space = 0
        removed_count = 0

        for pattern in patterns:
            for path in self.workspace_root.rglob(pattern):
                if path.is_file():
                    try:
                        size = path.stat().st_size
                        path.unlink()
                        freed_space += size
                        removed_count += 1
                    except Exception:
                        continue

        print(f"✓ Removed {removed_count} temp files")
        print(f"✓ Freed space: {self.format_size(freed_space)}")

        return freed_space

    def optimize_json_files(self, pretty: bool = False) -> int:
        """
        Optimize JSON files by removing whitespace.

        Args:
            pretty: If True, use pretty printing (less space efficient)

        Returns:
            Number of bytes saved
        """
        print("\n📄 Optimizing JSON files...")

        total_saved = 0
        optimized_count = 0

        json_files = [
            f for f, info in self.file_catalog.items()
            if f.endswith('.json')
        ]

        for filepath in json_files:
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)

                original_size = os.path.getsize(filepath)

                # Write optimized
                with open(filepath, 'w') as f:
                    if pretty:
                        json.dump(data, f, indent=2)
                    else:
                        json.dump(data, f, separators=(',', ':'))

                new_size = os.path.getsize(filepath)
                saved = original_size - new_size

                if saved > 0:
                    total_saved += saved
                    optimized_count += 1

            except Exception:
                continue

        print(f"✓ Optimized {optimized_count} JSON files")
        print(f"✓ Space saved: {self.format_size(total_saved)}")

        return total_saved

    def report_large_files(self, top_n: int = 20) -> List[Tuple[str, int]]:
        """
        Report largest files in workspace.

        Args:
            top_n: Number of files to report

        Returns:
            List of (filepath, size) tuples
        """
        print(f"\n📊 Top {top_n} largest files:")

        # Sort by size
        sorted_files = sorted(
            self.file_catalog.items(),
            key=lambda x: x[1].size,
            reverse=True
        )[:top_n]

        large_files = []
        for filepath, info in sorted_files:
            large_files.append((filepath, info.size))
            print(f"  {self.format_size(info.size):>10} - {Path(filepath).name}")

        return large_files

    def generate_report(self) -> Dict:
        """Generate comprehensive optimization report."""
        report = {
            'workspace': str(self.workspace_root),
            'total_files': self.stats['total_files'],
            'total_size': self.stats['total_size'],
            'total_size_formatted': self.format_size(self.stats['total_size']),
            'compressible_files_size': self.stats['compressible_size'],
            'compressible_size_formatted': self.format_size(self.stats['compressible_size']),
            'potential_compression_savings': self.stats['potential_savings'],
            'savings_formatted': self.format_size(self.stats['potential_savings']),
            'duplicates': {
                'sets': self.stats['duplicates_found'],
                'size': self.stats['duplicate_size'],
                'size_formatted': self.format_size(self.stats['duplicate_size'])
            },
            'large_files': {
                'count': self.stats['large_files_found'],
                'total_size': self.stats['large_files_size'],
                'size_formatted': self.format_size(self.stats['large_files_size'])
            }
        }

        return report

    @staticmethod
    def format_size(size_bytes: int) -> str:
        """Format size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"


def demonstrate_data_space_optimizer():
    """Demonstrate data space optimization."""
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║              L104 DATA SPACE OPTIMIZER                                    ║
║         Intelligent Compression & Storage Management                      ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)

    # Create optimizer
    optimizer = DataSpaceOptimizer()

    # === DEMO 1: Scan Workspace ===
    print("\n" + "="*80)
    print("DEMO 1: SCANNING WORKSPACE")
    print("="*80)

    optimizer.scan_directory()

    print(f"\n📊 Scan Results:")
    print(f"  Total files: {optimizer.stats['total_files']}")
    print(f"  Total size: {optimizer.format_size(optimizer.stats['total_size'])}")
    print(f"  Compressible: {optimizer.format_size(optimizer.stats['compressible_size'])}")

    # === DEMO 2: Find Duplicates ===
    print("\n" + "="*80)
    print("DEMO 2: FINDING DUPLICATE FILES")
    print("="*80)

    duplicates = optimizer.find_duplicates()

    if duplicates:
        print(f"\n📋 Duplicate sets found: {len(duplicates)}")
        for i, (hash_val, paths) in enumerate(list(duplicates.items())[:5]):
            if i < 5:  # Show first 5
                print(f"\n  Set {i+1} ({len(paths)} copies):")
                for path in paths[:3]:  # Show first 3 of each set
                    print(f"    - {Path(path).name}")

    # === DEMO 3: Estimate Compression ===
    print("\n" + "="*80)
    print("DEMO 3: ESTIMATING COMPRESSION SAVINGS")
    print("="*80)

    ratios = optimizer.estimate_compression_savings(sample_size=5)

    print(f"\n✓ Potential savings: {optimizer.format_size(optimizer.stats['potential_savings'])}")

    # === DEMO 4: Large Files ===
    print("\n" + "="*80)
    print("DEMO 4: IDENTIFYING LARGE FILES")
    print("="*80)

    large_files = optimizer.report_large_files(top_n=10)

    # === DEMO 5: Generate Report ===
    print("\n" + "="*80)
    print("DEMO 5: OPTIMIZATION REPORT")
    print("="*80)

    report = optimizer.generate_report()

    print(f"\n📊 WORKSPACE ANALYSIS:")
    print(f"  Files: {report['total_files']}")
    print(f"  Total Size: {report['total_size_formatted']}")
    print(f"  Compressible: {report['compressible_size_formatted']}")
    print(f"\n💾 POTENTIAL SAVINGS:")
    print(f"  From compression: {report['savings_formatted']}")
    print(f"  From deduplication: {report['duplicates']['size_formatted']}")
    print(f"  Total potential: {optimizer.format_size(report['potential_compression_savings'] + report['duplicates']['size'])}")
    print(f"\n📈 LARGE FILES:")
    print(f"  Count: {report['large_files']['count']}")
    print(f"  Size: {report['large_files']['size_formatted']}")

    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                 DATA SPACE OPTIMIZATION COMPLETE                          ║
║                                                                           ║
║  Analysis capabilities:                                                  ║
║    • Intelligent file scanning and cataloging                            ║
║    • Duplicate detection via content hashing                             ║
║    • Compression ratio estimation                                        ║
║    • Large file identification                                           ║
║    • JSON optimization                                                   ║
║    • Temporary file cleanup                                              ║
║                                                                           ║
║  Space optimization strategies ready for deployment.                     ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    demonstrate_data_space_optimizer()
