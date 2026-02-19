"""L104 Intellect — Hardware Adaptive Runtime & Platform Compatibility."""
import hashlib
import math
import os
import time
from typing import Dict

import numpy as np

from .numerics import PHI


class L104HardwareAdaptiveRuntime:
    """
    [HARDWARE] Adaptive runtime that profiles and tunes for the host system.

    Detects:
    - CPU architecture, core count, cache hierarchy
    - Memory pressure and allocation patterns
    - Thermal state and frequency scaling
    - GPU/Neural Engine capabilities
    - Storage I/O characteristics

    Tunes:
    - Batch sizes for reasoning workloads
    - Thread pool sizes for parallel operations
    - Cache eviction policies based on memory pressure
    - Prefetch depth for sequential access patterns
    - GC interval for memory-constrained environments
    """

    PHI = 1.618033988749895

    def __init__(self):
        self.profile = self._detect_hardware()
        self.perf_samples = []
        self.tuning_history = []
        self.current_tuning = {
            "batch_size": 32,
            "thread_pool_size": 4,
            "cache_policy": "lru",
            "prefetch_depth": 4,
            "gc_interval": 100,
            "precision": "float32",
            "memory_limit_mb": 512,
        }

    def _detect_hardware(self) -> Dict:
        """Detect host hardware capabilities."""
        import platform

        cpu_count = 1
        try:
            cpu_count = os.cpu_count() or 1
        except Exception:
            pass

        total_memory_mb = 4096  # Default fallback
        try:
            import psutil
            total_memory_mb = psutil.virtual_memory().total // (1024 * 1024)
        except ImportError:
            # macOS fallback
            try:
                import subprocess
                result = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
                total_memory_mb = int(result.stdout.strip()) // (1024 * 1024)
            except Exception:
                pass

        arch = platform.machine()
        system = platform.system()

        # Detect GPU
        has_gpu = False
        gpu_info = "none"
        try:
            import subprocess
            if system == "Darwin":
                result = subprocess.run(["system_profiler", "SPDisplaysDataType"], capture_output=True, text=True, timeout=5)
                if "Metal" in result.stdout:
                    has_gpu = True
                    gpu_info = "Metal-capable"
        except Exception:
            pass

        # Detect Neural Engine
        has_neural_engine = arch == "arm64" and system == "Darwin"

        return {
            "cpu_count": cpu_count,
            "cpu_arch": arch,
            "system": system,
            "total_memory_mb": total_memory_mb,
            "has_gpu": has_gpu,
            "gpu_info": gpu_info,
            "has_neural_engine": has_neural_engine,
            "python_version": platform.python_version(),
            "endianness": "little" if int.from_bytes(b'\x01\x00', 'little') == 1 else "big"
        }

    def get_memory_pressure(self) -> Dict:
        """Assess current memory pressure level."""
        import gc

        gc_stats = gc.get_stats() if hasattr(gc, 'get_stats') else []

        # Estimate process memory from gc
        gc_tracked = len(gc.get_objects()) if hasattr(gc, 'get_objects') else 0

        try:
            import psutil
            proc = psutil.Process(os.getpid())
            rss_mb = proc.memory_info().rss / (1024 * 1024)
            available_mb = psutil.virtual_memory().available / (1024 * 1024)
        except ImportError:
            rss_mb = gc_tracked * 0.001  # Very rough estimate
            available_mb = self.profile["total_memory_mb"] * 0.5

        pressure = "low"
        if available_mb < 512:
            pressure = "critical"
        elif available_mb < 1024:
            pressure = "high"
        elif available_mb < 2048:
            pressure = "moderate"

        return {
            "rss_mb": rss_mb,
            "available_mb": available_mb,
            "total_mb": self.profile["total_memory_mb"],
            "pressure": pressure,
            "gc_tracked_objects": gc_tracked,
            "gc_stats": gc_stats[:3] if gc_stats else []
        }

    def get_thermal_state(self) -> Dict:
        """Estimate thermal state (affects frequency scaling)."""
        # On macOS, we can try to read thermal state
        try:
            import subprocess
            result = subprocess.run(
                ["pmset", "-g", "therm"], capture_output=True, text=True, timeout=3
            )
            output = result.stdout
            if "CPU_Speed_Limit" in output:
                for line in output.splitlines():
                    if "CPU_Speed_Limit" in line:
                        limit = int(line.split("=")[-1].strip())
                        if limit >= 100:
                            return {"state": "nominal", "cpu_speed_limit": limit, "throttling": False}
                        elif limit >= 80:
                            return {"state": "fair", "cpu_speed_limit": limit, "throttling": True}
                        else:
                            return {"state": "critical", "cpu_speed_limit": limit, "throttling": True}
        except Exception:
            pass

        return {"state": "unknown", "cpu_speed_limit": 100, "throttling": False}

    def optimize_for_workload(self, workload_type: str = "reasoning") -> Dict:
        """
        Dynamically optimize runtime parameters for the given workload type.

        Workload types: reasoning, training, inference, io_bound, memory_intensive
        """
        mem = self.get_memory_pressure()
        thermal = self.get_thermal_state()
        cpus = self.profile["cpu_count"]
        total_mem = self.profile["total_memory_mb"]

        old_tuning = dict(self.current_tuning)

        if workload_type == "reasoning":
            # Balanced: moderate batch, most cores, LRU cache
            self.current_tuning["batch_size"] = max(8, min(64, total_mem // 128))
            self.current_tuning["thread_pool_size"] = max(2, cpus - 1)
            self.current_tuning["cache_policy"] = "lru"
            self.current_tuning["prefetch_depth"] = 4
            self.current_tuning["gc_interval"] = 200
            self.current_tuning["precision"] = "float32"
            self.current_tuning["memory_limit_mb"] = int(total_mem * 0.4)

        elif workload_type == "training":
            # Heavy compute: large batch, all cores, aggressive prefetch
            self.current_tuning["batch_size"] = max(16, min(128, total_mem // 64))
            self.current_tuning["thread_pool_size"] = cpus
            self.current_tuning["cache_policy"] = "lfu"
            self.current_tuning["prefetch_depth"] = 8
            self.current_tuning["gc_interval"] = 500
            self.current_tuning["precision"] = "float16" if self.profile.get("has_gpu") else "float32"
            self.current_tuning["memory_limit_mb"] = int(total_mem * 0.6)

        elif workload_type == "inference":
            # Low latency: small batch, moderate cores, write-through cache
            self.current_tuning["batch_size"] = max(1, min(16, total_mem // 256))
            self.current_tuning["thread_pool_size"] = max(2, cpus // 2)
            self.current_tuning["cache_policy"] = "lru"
            self.current_tuning["prefetch_depth"] = 2
            self.current_tuning["gc_interval"] = 100
            self.current_tuning["precision"] = "float32"
            self.current_tuning["memory_limit_mb"] = int(total_mem * 0.3)

        elif workload_type == "io_bound":
            # I/O: minimal compute, async-friendly
            self.current_tuning["batch_size"] = 4
            self.current_tuning["thread_pool_size"] = max(4, cpus * 2)  # More threads for I/O wait
            self.current_tuning["cache_policy"] = "write_back"
            self.current_tuning["prefetch_depth"] = 16
            self.current_tuning["gc_interval"] = 50
            self.current_tuning["precision"] = "float32"
            self.current_tuning["memory_limit_mb"] = int(total_mem * 0.2)

        elif workload_type == "memory_intensive":
            # Memory: small batch, aggressive GC, minimal cache
            self.current_tuning["batch_size"] = max(1, min(8, total_mem // 512))
            self.current_tuning["thread_pool_size"] = max(1, cpus // 2)
            self.current_tuning["cache_policy"] = "fifo"
            self.current_tuning["prefetch_depth"] = 1
            self.current_tuning["gc_interval"] = 25
            self.current_tuning["precision"] = "float32"
            self.current_tuning["memory_limit_mb"] = int(total_mem * 0.7)

        # Thermal throttling adjustment
        if thermal.get("throttling"):
            self.current_tuning["batch_size"] = max(1, self.current_tuning["batch_size"] // 2)
            self.current_tuning["thread_pool_size"] = max(1, self.current_tuning["thread_pool_size"] - 1)

        # Memory pressure adjustment
        if mem["pressure"] in ("high", "critical"):
            self.current_tuning["batch_size"] = max(1, self.current_tuning["batch_size"] // 2)
            self.current_tuning["memory_limit_mb"] = min(
                self.current_tuning["memory_limit_mb"],
                int(mem["available_mb"] * 0.5)
            )
            self.current_tuning["gc_interval"] = max(10, self.current_tuning["gc_interval"] // 2)

        self.tuning_history.append({
            "workload": workload_type,
            "timestamp": time.time(),
            "old_tuning": old_tuning,
            "new_tuning": dict(self.current_tuning),
            "memory_pressure": mem["pressure"],
            "thermal_state": thermal["state"]
        })

        return {
            "workload_type": workload_type,
            "tuning": dict(self.current_tuning),
            "memory_pressure": mem["pressure"],
            "thermal_state": thermal["state"],
            "adjustments_applied": {
                k: (old_tuning[k], self.current_tuning[k])
                for k in old_tuning if old_tuning[k] != self.current_tuning[k]
            }
        }

    def record_perf_sample(self, operation: str, duration_ms: float, memory_delta_mb: float = 0) -> None:
        """Record a performance sample for trend analysis."""
        self.perf_samples.append({
            "operation": operation,
            "duration_ms": duration_ms,
            "memory_delta_mb": memory_delta_mb,
            "timestamp": time.time(),
            "tuning_snapshot": dict(self.current_tuning)
        })
        # Keep last 1000 samples
        if len(self.perf_samples) > 1000:
            self.perf_samples = self.perf_samples[-1000:]

    def get_perf_trend(self, operation: str = None, window: int = 50) -> Dict:
        """Analyze performance trend for an operation."""
        samples = self.perf_samples[-window:]
        if operation:
            samples = [s for s in samples if s["operation"] == operation]

        if not samples:
            return {"trend": "insufficient_data", "samples": 0}

        durations = [s["duration_ms"] for s in samples]
        mean_d = sum(durations) / len(durations)

        # Simple linear trend
        if len(durations) >= 3:
            first_half = durations[:len(durations) // 2]
            second_half = durations[len(durations) // 2:]
            first_mean = sum(first_half) / len(first_half)
            second_mean = sum(second_half) / len(second_half)

            if second_mean > first_mean * 1.1:
                trend = "degrading"
            elif second_mean < first_mean * 0.9:
                trend = "improving"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"

        return {
            "operation": operation or "all",
            "samples": len(samples),
            "mean_duration_ms": mean_d,
            "min_duration_ms": min(durations),
            "max_duration_ms": max(durations),
            "trend": trend,
            "phi_weighted_avg": sum(d * (self.PHI ** (i / len(durations))) for i, d in enumerate(durations)) / sum(self.PHI ** (i / len(durations)) for i in range(len(durations)))
        }

    def get_runtime_status(self) -> Dict:
        """Get full hardware adaptive runtime status."""
        return {
            "profile": self.profile,
            "current_tuning": self.current_tuning,
            "memory_pressure": self.get_memory_pressure(),
            "thermal_state": self.get_thermal_state(),
            "perf_samples_count": len(self.perf_samples),
            "tuning_changes": len(self.tuning_history)
        }


class L104PlatformCompatibilityLayer:
    """
    [COMPAT] Platform compatibility detection and abstraction layer.

    Detects available modules, hardware features, and OS capabilities.
    Provides safe fallbacks and feature flags for cross-platform operation.
    Manages dynamic imports and optional dependency resolution.
    """

    # Module compatibility matrix
    OPTIONAL_MODULES = [
        "numpy", "scipy", "torch", "tensorflow", "transformers",
        "qiskit", "pennylane", "cirq", "sympy",
        "fastapi", "uvicorn", "pydantic",
        "psutil", "GPUtil",
        "PIL", "cv2", "matplotlib",
        "aiohttp", "httpx", "websockets",
        "cryptography", "nacl"
    ]

    def __init__(self):
        self.available_modules = self._detect_modules()
        self.feature_flags = self._compute_feature_flags()
        self.compatibility_warnings = []

    def _detect_modules(self) -> Dict[str, bool]:
        """Detect which optional modules are available."""
        results = {}
        for mod_name in self.OPTIONAL_MODULES:
            try:
                __import__(mod_name)
                results[mod_name] = True
            except ImportError:
                results[mod_name] = False
        return results

    def _compute_feature_flags(self) -> Dict[str, bool]:
        """Compute feature flags based on available capabilities."""
        return {
            "gpu_compute": self.available_modules.get("torch", False) or self.available_modules.get("tensorflow", False),
            "quantum_simulation": self.available_modules.get("qiskit", False) or self.available_modules.get("pennylane", False) or self.available_modules.get("cirq", False),
            "scientific_compute": self.available_modules.get("numpy", False) and self.available_modules.get("scipy", False),
            "neural_engine": self.available_modules.get("torch", False),  # CoreML via torch
            "web_server": self.available_modules.get("fastapi", False) and self.available_modules.get("uvicorn", False),
            "async_io": self.available_modules.get("aiohttp", False) or self.available_modules.get("httpx", False),
            "image_processing": self.available_modules.get("PIL", False) or self.available_modules.get("cv2", False),
            "visualization": self.available_modules.get("matplotlib", False),
            "symbolic_math": self.available_modules.get("sympy", False),
            "system_monitoring": self.available_modules.get("psutil", False),
            "encryption": self.available_modules.get("cryptography", False) or self.available_modules.get("nacl", False),
            "websocket": self.available_modules.get("websockets", False),
            "data_validation": self.available_modules.get("pydantic", False),
        }

    def safe_import(self, module_name: str, fallback=None):
        """Safely import a module with fallback."""
        try:
            return __import__(module_name)
        except ImportError:
            if fallback is not None:
                return fallback
            self.compatibility_warnings.append(f"Module '{module_name}' not available")
            return None

    def get_optimal_dtype(self) -> str:
        """Get optimal data type based on available hardware."""
        if self.feature_flags["gpu_compute"]:
            try:
                import torch
                if torch.cuda.is_available():
                    return "float16"  # GPU: use half precision
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    return "float32"  # Apple Silicon MPS
            except Exception:
                pass
        return "float32"  # CPU fallback

    def get_max_concurrency(self) -> int:
        """Get safe maximum concurrency level."""
        cpu_count = os.cpu_count() or 1

        if self.feature_flags["system_monitoring"]:
            try:
                import psutil
                available_mem_gb = psutil.virtual_memory().available / (1024 ** 3)
                # Limit concurrency if memory is tight
                if available_mem_gb < 1.0:
                    return max(1, cpu_count // 4)
                elif available_mem_gb < 2.0:
                    return max(2, cpu_count // 2)
            except Exception:
                pass

        return max(2, cpu_count - 1)

    def get_compatibility_report(self) -> Dict:
        """Generate full compatibility report."""
        return {
            "available_modules": {k: v for k, v in self.available_modules.items()},
            "feature_flags": self.feature_flags,
            "optimal_dtype": self.get_optimal_dtype(),
            "max_concurrency": self.get_max_concurrency(),
            "warnings": self.compatibility_warnings,
            "modules_available": sum(1 for v in self.available_modules.values() if v),
            "modules_missing": sum(1 for v in self.available_modules.values() if not v),
            "total_modules_checked": len(self.available_modules),
            "features_enabled": sum(1 for v in self.feature_flags.values() if v),
            "features_disabled": sum(1 for v in self.feature_flags.values() if not v),
            # v25.0 additions
            "performance_tier": self.classify_performance_tier(),
            "dependency_graph_size": len(self._build_dependency_graph()),
            "degradation_strategies": len(self._get_degradation_strategies()),
            "os_platform": self._detect_platform_details()["platform"],
        }

    # ═══════════════════════════════════════════════════════════════════
    # v25.0 DEPENDENCY RESOLUTION GRAPH
    # Maps inter-module dependencies to enable intelligent fallback
    # chains and detect which capabilities degrade together.
    # ═══════════════════════════════════════════════════════════════════

    # Module dependency relationships: module -> [required_by]
    DEPENDENCY_MAP = {
        "numpy": ["scipy", "torch", "tensorflow", "qiskit", "pennylane", "matplotlib"],
        "scipy": ["qiskit"],
        "torch": ["transformers"],
        "pydantic": ["fastapi"],
        "uvicorn": [],
        "aiohttp": [],
        "httpx": [],
        "websockets": [],
        "cryptography": [],
        "nacl": [],
        "PIL": [],
        "cv2": [],
        "sympy": [],
        "psutil": ["GPUtil"],
        "GPUtil": [],
        "qiskit": [],
        "pennylane": [],
        "cirq": [],
        "matplotlib": [],
        "tensorflow": [],
        "transformers": [],
    }

    def _build_dependency_graph(self) -> Dict[str, Dict]:
        """
        Build a full dependency graph with availability status.
        Returns adjacency map with transitive dependency resolution.
        """
        graph = {}
        for module, depends_on_by in self.DEPENDENCY_MAP.items():
            available = self.available_modules.get(module, False)
            # Find what this module depends on (reverse lookup)
            dependencies = [m for m, deps in self.DEPENDENCY_MAP.items()
                            if module in deps]
            # What depends on this module
            dependents = depends_on_by

            graph[module] = {
                "available": available,
                "dependencies": dependencies,  # What I need
                "dependents": dependents,       # What needs me
                "blocked": not available and any(
                    self.available_modules.get(d, False) for d in dependents
                ),  # True if I'm needed but missing
                "impact_score": len(dependents) + (2 if available else 0),
            }

        return graph

    def resolve_dependency_chain(self, target_feature: str) -> Dict:
        """
        Given a target feature flag, trace back through the dependency
        graph to identify what's needed and what's missing.
        """
        # Feature -> required modules mapping
        feature_requirements = {
            "gpu_compute": ["torch", "tensorflow", "numpy"],
            "quantum_simulation": ["qiskit", "pennylane", "cirq", "numpy"],
            "scientific_compute": ["numpy", "scipy"],
            "neural_engine": ["torch", "numpy"],
            "web_server": ["fastapi", "uvicorn", "pydantic"],
            "async_io": ["aiohttp", "httpx"],
            "image_processing": ["PIL", "cv2"],
            "visualization": ["matplotlib", "numpy"],
            "symbolic_math": ["sympy"],
            "system_monitoring": ["psutil"],
            "encryption": ["cryptography", "nacl"],
            "websocket": ["websockets"],
            "data_validation": ["pydantic"],
        }

        if target_feature not in feature_requirements:
            return {"error": f"Unknown feature: {target_feature}"}

        required = feature_requirements[target_feature]
        available = [m for m in required if self.available_modules.get(m, False)]
        missing = [m for m in required if not self.available_modules.get(m, False)]

        # Check transitive dependencies
        all_needed = set()
        for mod in required:
            all_needed.add(mod)
            for dep_mod, dep_list in self.DEPENDENCY_MAP.items():
                if mod in dep_list:
                    all_needed.add(dep_mod)

        return {
            "feature": target_feature,
            "enabled": self.feature_flags.get(target_feature, False),
            "direct_requirements": required,
            "available": available,
            "missing": missing,
            "transitive_dependencies": list(all_needed - set(required)),
            "can_enable": len(missing) == 0 or any(
                self.available_modules.get(m, False) for m in required
            ),
            "install_command": f"pip install {' '.join(missing)}" if missing else None,
        }

    # ═══════════════════════════════════════════════════════════════════
    # v25.0 FEATURE DEGRADATION STRATEGIES
    # When capabilities are unavailable, define graceful degradation
    # paths that preserve functionality at reduced fidelity.
    # ═══════════════════════════════════════════════════════════════════

    def _get_degradation_strategies(self) -> Dict[str, Dict]:
        """Get degradation strategies for all features."""
        return {
            "gpu_compute": {
                "full": "GPU acceleration via CUDA/MPS for tensor operations",
                "degraded": "CPU-only computation with reduced batch sizes",
                "minimal": "Pure Python loops with manual vectorization",
                "fallback_method": "_cpu_fallback_compute",
                "performance_ratio": {"full": 1.0, "degraded": 0.15, "minimal": 0.02},
            },
            "quantum_simulation": {
                "full": "Qiskit/Pennylane circuit simulation with noise models",
                "degraded": "Simplified state vector simulation (numpy-based)",
                "minimal": "Probabilistic classical approximation",
                "fallback_method": "_classical_quantum_approx",
                "performance_ratio": {"full": 1.0, "degraded": 0.4, "minimal": 0.1},
            },
            "scientific_compute": {
                "full": "NumPy/SciPy BLAS-accelerated linear algebra",
                "degraded": "NumPy-only basic operations",
                "minimal": "Pure Python math module",
                "fallback_method": "_pure_python_math",
                "performance_ratio": {"full": 1.0, "degraded": 0.3, "minimal": 0.01},
            },
            "web_server": {
                "full": "FastAPI + Uvicorn async ASGI server",
                "degraded": "http.server stdlib server",
                "minimal": "Socket-level request handling",
                "fallback_method": "_stdlib_http_server",
                "performance_ratio": {"full": 1.0, "degraded": 0.2, "minimal": 0.05},
            },
            "encryption": {
                "full": "Cryptography/NaCl with hardware-accelerated AES",
                "degraded": "hashlib-based HMAC/SHA256",
                "minimal": "XOR-based obfuscation (WARNING: not secure)",
                "fallback_method": "_hashlib_fallback",
                "performance_ratio": {"full": 1.0, "degraded": 0.6, "minimal": 0.1},
            },
            "system_monitoring": {
                "full": "psutil process/system monitoring with GPU stats",
                "degraded": "os.sysconf + /proc filesystem reading",
                "minimal": "os.cpu_count() and resource.getrusage()",
                "fallback_method": "_minimal_system_stats",
                "performance_ratio": {"full": 1.0, "degraded": 0.5, "minimal": 0.2},
            },
        }

    def get_degradation_level(self, feature: str) -> str:
        """Get current degradation level for a feature."""
        strategies = self._get_degradation_strategies()
        if feature not in strategies:
            return "unknown"

        if self.feature_flags.get(feature, False):
            return "full"

        # Check partial availability
        feature_modules = {
            "gpu_compute": ["torch", "tensorflow"],
            "quantum_simulation": ["qiskit", "pennylane", "cirq"],
            "scientific_compute": ["numpy", "scipy"],
            "web_server": ["fastapi", "uvicorn"],
            "encryption": ["cryptography", "nacl"],
            "system_monitoring": ["psutil"],
        }

        modules = feature_modules.get(feature, [])
        available_count = sum(1 for m in modules if self.available_modules.get(m, False))

        if available_count > 0:
            return "degraded"
        return "minimal"

    # ═══════════════════════════════════════════════════════════════════
    # v25.0 OS-SPECIFIC PLATFORM DETECTION & OPTIMIZATION
    # Detect platform details and recommend optimal configurations.
    # ═══════════════════════════════════════════════════════════════════

    def _detect_platform_details(self) -> Dict:
        """Detect detailed platform information for optimization."""
        import platform as plat
        import struct

        details = {
            "platform": plat.system(),           # Linux, Darwin, Windows
            "release": plat.release(),
            "machine": plat.machine(),           # x86_64, arm64, aarch64
            "python_version": plat.python_version(),
            "pointer_size": struct.calcsize("P") * 8,  # 32 or 64 bit
            "byte_order": "little" if struct.pack("H", 1)[0] == 1 else "big",
        }

        # CPU architecture classification
        machine = details["machine"].lower()
        if "arm" in machine or "aarch" in machine:
            details["arch_family"] = "ARM"
            details["simd_hint"] = "NEON"
        elif "x86" in machine or "amd64" in machine:
            details["arch_family"] = "x86"
            details["simd_hint"] = "AVX2"  # Conservative assumption
        elif "riscv" in machine:
            details["arch_family"] = "RISC-V"
            details["simd_hint"] = "V-extension"
        else:
            details["arch_family"] = "unknown"
            details["simd_hint"] = "none"

        # OS-specific capabilities
        system = details["platform"]
        if system == "Darwin":
            details["has_metal"] = True
            details["has_accelerate"] = True
            details["has_coreml"] = True
            details["recommended_backend"] = "mps"
        elif system == "Linux":
            details["has_metal"] = False
            details["has_accelerate"] = False
            details["has_coreml"] = False
            details["recommended_backend"] = "cuda" if self.available_modules.get("torch", False) else "cpu"
            # Check for WSL
            try:
                with open("/proc/version", "r") as f:
                    version_str = f.read().lower()
                    details["is_wsl"] = "microsoft" in version_str or "wsl" in version_str
            except Exception:
                details["is_wsl"] = False
        elif system == "Windows":
            details["has_metal"] = False
            details["has_accelerate"] = False
            details["has_coreml"] = False
            details["recommended_backend"] = "cuda" if self.available_modules.get("torch", False) else "cpu"
        else:
            details["recommended_backend"] = "cpu"

        return details

    def get_optimal_config_for_workload(self, workload_type: str) -> Dict:
        """
        Get platform-optimal configuration for a specific workload type.
        Workload types: inference, training, analysis, serving, storage
        """
        platform = self._detect_platform_details()
        concurrency = self.get_max_concurrency()
        dtype = self.get_optimal_dtype()

        base_config = {
            "workload": workload_type,
            "dtype": dtype,
            "backend": platform["recommended_backend"],
            "concurrency": concurrency,
            "arch_family": platform["arch_family"],
        }

        workload_configs = {
            "inference": {
                "batch_size": 1 if concurrency < 4 else 4,
                "precision": "float16" if platform.get("has_metal") or dtype == "float16" else "float32",
                "thread_pool_size": min(4, concurrency),
                "cache_enabled": True,
                "prefetch": True,
            },
            "training": {
                "batch_size": max(1, concurrency // 2),
                "precision": "float32",
                "gradient_accumulation": 4 if concurrency < 4 else 1,
                "thread_pool_size": concurrency,
                "cache_enabled": False,
                "checkpoint_interval": 100,
            },
            "analysis": {
                "batch_size": max(1, concurrency),
                "precision": "float64" if self.feature_flags["scientific_compute"] else "float32",
                "thread_pool_size": concurrency,
                "vectorize": platform["arch_family"] in ("x86", "ARM"),
                "simd_hint": platform["simd_hint"],
            },
            "serving": {
                "workers": max(2, concurrency - 1),
                "timeout_s": 30,
                "keep_alive": 5,
                "backlog": 128,
                "access_log": False,
                "limit_concurrency": concurrency * 4,
            },
            "storage": {
                "compression": "lz4" if concurrency > 2 else "gzip",
                "buffer_size": 65536 if concurrency > 4 else 8192,
                "sync_writes": False,
                "mmap_enabled": platform["pointer_size"] == 64,
            },
        }

        if workload_type in workload_configs:
            base_config.update(workload_configs[workload_type])

        return base_config

    # ═══════════════════════════════════════════════════════════════════
    # v25.0 PERFORMANCE TIER CLASSIFICATION
    # Classify the current system into a performance tier to guide
    # resource allocation and algorithm selection decisions.
    # ═══════════════════════════════════════════════════════════════════

    def classify_performance_tier(self) -> str:
        """
        Classify system into performance tiers:
        - 'sovereign': Full GPU + quantum + scientific + all features
        - 'advanced': GPU or quantum + scientific compute
        - 'standard': Scientific compute + web server capable
        - 'basic': Minimal — CPU only, limited modules
        - 'constrained': Very few modules, low resources
        """
        flags = self.feature_flags
        available_count = sum(1 for v in self.available_modules.values() if v)
        enabled_features = sum(1 for v in flags.values() if v)
        concurrency = self.get_max_concurrency()

        score = 0

        # Module availability (up to 30 points)
        score += min(30, available_count * 2)

        # Feature flags (up to 40 points)
        score += enabled_features * 3

        # GPU compute (20 points)
        if flags.get("gpu_compute"):
            score += 20

        # Quantum simulation (15 points)
        if flags.get("quantum_simulation"):
            score += 15

        # Concurrency (up to 15 points)
        score += min(15, concurrency * 2)

        if score >= 90:
            return "sovereign"
        elif score >= 60:
            return "advanced"
        elif score >= 35:
            return "standard"
        elif score >= 15:
            return "basic"
        else:
            return "constrained"

    def get_tier_recommendations(self) -> Dict:
        """Get optimization recommendations based on performance tier."""
        tier = self.classify_performance_tier()
        platform = self._detect_platform_details()

        recommendations = {
            "sovereign": {
                "tier": "sovereign",
                "max_model_params": "7B+",
                "recommended_batch_size": 32,
                "enable_features": list(self.feature_flags.keys()),
                "notes": "Full capability mode. Enable all subsystems.",
            },
            "advanced": {
                "tier": "advanced",
                "max_model_params": "1B-3B",
                "recommended_batch_size": 8,
                "enable_features": [k for k, v in self.feature_flags.items() if v],
                "notes": "Strong capability. Consider quantization for larger models.",
            },
            "standard": {
                "tier": "standard",
                "max_model_params": "500M",
                "recommended_batch_size": 4,
                "enable_features": [k for k, v in self.feature_flags.items() if v],
                "notes": "Moderate capability. Use CPU-optimized algorithms.",
            },
            "basic": {
                "tier": "basic",
                "max_model_params": "100M",
                "recommended_batch_size": 1,
                "enable_features": [k for k, v in self.feature_flags.items() if v],
                "notes": "Limited capability. Focus on lightweight operations.",
            },
            "constrained": {
                "tier": "constrained",
                "max_model_params": "10M",
                "recommended_batch_size": 1,
                "enable_features": [k for k, v in self.feature_flags.items() if v],
                "notes": "Severely limited. Use pure Python fallbacks only.",
            },
        }

        result = recommendations.get(tier, recommendations["basic"])
        result["platform"] = platform["platform"]
        result["arch"] = platform["arch_family"]
        result["backend"] = platform["recommended_backend"]

        return result


