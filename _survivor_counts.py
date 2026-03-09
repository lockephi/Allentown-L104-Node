#!/usr/bin/env python3
"""Extract actual survivor counts from all quantum brain subsystems."""
import sys, os, math, json
sys.path.insert(0, os.path.dirname(__file__))

GOD_CODE = 527.5184818492612
PHI = 1.618033988749895
TAU = 1.0 / PHI

from l104_quantum_engine import quantum_brain
from l104_quantum_engine.models import QuantumLink
from l104_quantum_engine.computation import (
    QuantumRegister, QuantumNeuron, QuantumCluster, QuantumCPU,
)
from l104_quantum_engine.math_core import QuantumMathCore

# Generate synthetic links (same as in the upgrade simulation)
links = []
for i in range(200):
    fidelity = 0.5 + 0.5 * math.sin(i * PHI)
    strength = 0.3 + 2.0 * abs(math.cos(i * GOD_CODE / 100))
    link = QuantumLink(
        source_file=f"sim_source_{i}.py", source_symbol=f"god_code_sim_{i}",
        source_line=i + 1, target_file=f"sim_target_{i}.py",
        target_symbol=f"god_code_recv_{i}", target_line=i + 10,
        link_type="god_code_derived",
        fidelity=max(0.01, min(1.0, fidelity)),
        strength=max(0.1, min(3.0, strength)),
        entanglement_entropy=abs(math.sin(i * TAU)) * math.log(2),
        noise_resilience=0.5 + 0.4 * math.cos(i * PHI),
    )
    links.append(link)

fids = [l.fidelity for l in links]
qmath = QuantumMathCore()
qle = quantum_brain.quantum_engine  # The quantum link experiments engine

P = print
P("=" * 70)
P("  QUANTUM BRAIN - FULL SURVIVOR COUNT REPORT")
P("  200 synthetic phi-distributed links")
P("=" * 70)

# 1) CPU Registers
cpu = QuantumCPU(qmath)
cpu_result = cpu.execute(links[:52])
total_r = cpu_result["total_registers"]
healthy = cpu_result["healthy"]
quaran = cpu_result["quarantined"]
P(f"\n  1. CPU REGISTERS (52 links)")
P(f"     Total:       {total_r}")
P(f"     Healthy:     {healthy}")
P(f"     Quarantined: {quaran}")
P(f"     SURVIVAL:    {healthy}/{total_r} = {healthy/total_r*100:.1f}%")

# 2) Cluster Entanglement
cluster = QuantumCluster(0, qmath)
registers = [QuantumRegister(link, qmath) for link in links[:20]]
processed = cluster.process_batch(registers)
entangled = [r for r in processed if r.is_entangled]
P(f"\n  2. CLUSTER ENTANGLEMENT (20 links)")
P(f"     Processed:   {cluster.registers_processed}")
P(f"     Entangled:   {len(entangled)}")
P(f"     Health:      {cluster.health:.4f}")
P(f"     ENTANGLED:   {len(entangled)}/{cluster.registers_processed} = {len(entangled)/max(cluster.registers_processed,1)*100:.1f}%")

# 3) Distillation
dist_result = quantum_brain.distiller.distill_links(links[:50])
P(f"\n  3. DISTILLATION (50 links)")
for k, v in dist_result.items():
    if isinstance(v, (int, float, str)):
        P(f"     {k}: {v}")

# 4) Quantum Zeno Stabilizer
fidelities_8 = [0.95, 0.88, 0.92, 0.78, 0.99, 0.85, 0.91, 0.87]
zeno = qle.quantum_zeno_stabilizer(fidelities_8)
P(f"\n  4. QUANTUM ZENO STABILIZER (8 fidelities)")
for k, v in zeno.items():
    if isinstance(v, (int, float)):
        P(f"     {k}: {v}")

# 5) Error Correction
ec = qle.quantum_error_correction(fidelities_8)
P(f"\n  5. ERROR CORRECTION")
for k, v in ec.items():
    if isinstance(v, (int, float)):
        P(f"     {k}: {v}")
    elif isinstance(v, dict):
        P(f"     {k}:")
        for k2, v2 in v.items():
            if isinstance(v2, (int, float)):
                P(f"       {k2}: {v2}")

# 6) Lindblad Decoherence
lb = qle.lindblad_decoherence_model(fidelities_8)
P(f"\n  6. LINDBLAD DECOHERENCE")
for k, v in lb.items():
    if isinstance(v, (int, float)):
        P(f"     {k}: {v}")

# 7) Entanglement Distillation (QLE)
ed = qle.entanglement_distillation(fidelities_8)
P(f"\n  7. ENTANGLEMENT DISTILLATION (QLE)")
for k, v in ed.items():
    if isinstance(v, (int, float)):
        P(f"     {k}: {v}")

# 8) Fe(26) Lattice Simulation
fe = qle.fe_lattice_simulation(n_sites=26)
P(f"\n  8. Fe(26) LATTICE SIMULATION")
for k, v in fe.items():
    if isinstance(v, (int, float)):
        P(f"     {k}: {v}")

# 9) Upgrade Engine
upgrade = quantum_brain.upgrader.auto_upgrade(links[:50])
P(f"\n  9. UPGRADE ENGINE (50 links)")
for k, v in upgrade.items():
    if isinstance(v, (int, float)):
        P(f"     {k}: {v}")

# 10) Repair Engine
repair = quantum_brain.repair.full_repair(links[:50])
P(f"\n  10. REPAIR ENGINE (50 links)")
for k, v in repair.items():
    if isinstance(v, (int, float, str)):
        P(f"     {k}: {v}")
    elif isinstance(v, dict):
        P(f"     {k}:")
        for k2, v2 in v.items():
            if isinstance(v2, (int, float)):
                P(f"       {k2}: {v2}")

# 11) Stress Tests
stress = quantum_brain.stress.run_stress_tests(links[:20])
P(f"\n  11. STRESS TESTS (20 links)")
for k, v in stress.items():
    if isinstance(v, (int, float)):
        P(f"     {k}: {v}")

# 12) Cross-Modal
cross = quantum_brain.cross_modal.full_analysis(links[:20])
P(f"\n  12. CROSS-MODAL ANALYSIS (20 links)")
for k, v in cross.items():
    if isinstance(v, (int, float)):
        P(f"     {k}: {v}")

# ══════════ SUMMARY ══════════
P("\n" + "=" * 70)
P("  SURVIVOR SUMMARY")
P("=" * 70)
P(f"  CPU Registers:         {healthy}/{total_r} ({healthy/total_r*100:.1f}%)")
P(f"  Entangled:             {len(entangled)}/{cluster.registers_processed} ({len(entangled)/max(cluster.registers_processed,1)*100:.1f}%)")
zs = zeno.get("mean_zeno_survival", zeno.get("survival_rate", "?"))
P(f"  Zeno survival:         {zs}")
cf = ec.get("composite_fidelity", "?")
P(f"  Error-corrected fid:   {cf}")
ur = upgrade.get("upgrade_rate", "?")
P(f"  Upgrade rate:          {ur}")
mff = upgrade.get("mean_final_fidelity", "?")
P(f"  Mean final fidelity:   {mff}")
prf = repair.get("post_repair_mean_fidelity", "?")
P(f"  Post-repair fidelity:  {prf}")
lp = stress.get("links_passed", 0)
tl = stress.get("tested_links", 1) or 1
P(f"  Stress passed:         {lp}/{stress.get('tested_links','?')} ({lp/tl*100:.1f}%)")
P("=" * 70)
