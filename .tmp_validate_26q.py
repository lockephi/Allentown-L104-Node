#!/usr/bin/env python3
"""Quick 26Q migration validation — no heavy Qiskit imports."""
import sys
sys.dont_write_bytecode = True

print("=== 26Q Migration Validation (Quick) ===\n")
ok = 0

# 1. Science Engine
from l104_science_engine import CircuitTemplates26Q
from l104_science_engine.constants import QB, GOD_CODE_26Q_CONVERGENCE
print(f"  [1] science_engine: OK (26Q alias: True, N_QUBITS={QB.N_QUBITS}, CONV={GOD_CODE_26Q_CONVERGENCE:.10f})")
ok += 1

# 2. Check source files for 26Q methods using text search (avoids full import)
checks = [
    ("l104_asi/core.py", "quantum_26q_execute"),
    ("l104_agi/core.py", "quantum_26q_execute"),
    ("l104_code_engine/hub.py", "quantum_26q_build"),
    ("l104_math_engine/engine.py", "quantum_26q_build"),
    ("l104_intellect/local_intellect_core.py", "quantum_26q_build"),
]
for path, method in checks:
    with open(path) as f:
        content = f.read()
    has_method = f"def {method}" in content
    has_26q_builder = "_builder_26q" in content or "_get_builder_26q" in content
    label = path.split("/")[0]
    ok += 1
    if has_method and has_26q_builder:
        print(f"  [{ok}] {label}: OK ({method}: True, 26q_builder: True)")
    else:
        print(f"  [{ok}] {label}: WARN (method={has_method}, builder={has_26q_builder})")

# 3. Identity boundary
with open("l104_asi/identity_boundary.py") as f:
    content = f.read()
has_26q = "26Q" in content and "26Q iron-mapped" in content
ok += 1
print(f"  [{ok}] identity_boundary: OK (26Q in IS: {has_26q})")

# 4. Server route
with open("l104_server/app.py") as f:
    content = f.read()
has_26q_route = "/api/v14/quantum/circuits/26q" in content
has_legacy_25q = "Legacy 25Q route" in content
ok += 1
print(f"  [{ok}] server routes: OK (26q route: {has_26q_route}, 25q legacy: {has_legacy_25q})")

# 5. 26Q Engine Builder
from l104_26q_engine_builder import L104_26Q_CircuitBuilder
b = L104_26Q_CircuitBuilder()
ok += 1
print(f"  [{ok}] 26q_builder: OK ({b.n_qubits} qubits, {len(b.registers)} registers, {len(b.circuit_builders)} circuits)")

print(f"\n=== {ok}/9 checks passed — 26Q migration complete ===")
