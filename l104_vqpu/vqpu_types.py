"""L104 VQPU Package v14.0.0 — Data types for quantum circuits and results.

v14.0.0: Enhanced VQPUResult with pipeline metadata, timestamps, topology info.
"""

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional, List

from .constants import GOD_CODE


@dataclass
class QuantumGate:
    """A single quantum gate operation."""
    gate: str
    qubits: list
    parameters: Optional[list] = None


@dataclass
class QuantumJob:
    """A quantum circuit job for the vQPU."""
    circuit_id: str = ""
    num_qubits: int = 2
    operations: list = field(default_factory=list)
    shots: int = 1024
    priority: int = 1
    adapt: bool = False
    max_branches: Optional[int] = None
    prune_epsilon: Optional[float] = None
    topology: Optional[str] = None         # v14.0: target topology (linear, ring, heavy_hex, all_to_all)

    def __post_init__(self):
        if not self.circuit_id:
            self.circuit_id = f"l104-{uuid.uuid4().hex[:12]}"


@dataclass
class VQPUResult:
    """Result from the vQPU execution.

    v14.0: Enhanced with pipeline stage tracking, timestamps, and topology.
    """
    circuit_id: str
    probabilities: dict
    counts: Optional[dict] = None
    backend: str = "unknown"
    branch_count: int = 0
    t_gate_count: int = 0
    clifford_gate_count: int = 0
    execution_time_ms: float = 0.0
    num_qubits: int = 0
    god_code: float = GOD_CODE
    error: Optional[str] = None
    # v14.0: Pipeline metadata
    pipeline_stages: Optional[List[str]] = None       # Stages executed (transpile, compile, protect, execute, score)
    created_at: float = field(default_factory=time.time)  # Unix timestamp of creation
    topology: Optional[str] = None                    # Topology used for routing
    swap_count: int = 0                               # Number of SWAP gates inserted by router
    noise_model: Optional[str] = None                 # Noise model applied (if any)
    crosstalk_mitigated: bool = False                  # v14.0: Whether crosstalk mitigation was applied


__all__ = ["QuantumGate", "QuantumJob", "VQPUResult", "asdict"]
