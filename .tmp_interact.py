#!/usr/bin/env python3
"""Quick supercomputer interaction — dial sweep + subsystems."""
import sys, time, os
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

# Suppress noisy imports
import logging
logging.disable(logging.WARNING)

from l104_quantum_mini_supercomputer import get_supercomputer

sc = get_supercomputer()

P = lambda *a, **k: (print(*a, **k), sys.stdout.flush())

P("=" * 72)
P("  L104 QUANTUM MINI SUPERCOMPUTER v2.1.0 — LIVE")
P("=" * 72)

# ── Dial Sweep ──────────────────────────────────────────────
P("\n  GOD_CODE Octave Ladder: G(0,0,0,d)")
P(f"  {'d':>3s} {'Freq Hz':>12s} {'Sacred':>8s} {'EntRev':>8s} {'Phi':>8s} {'GCFid':>8s} {'ms':>6s}")
P(f"  {'---':>3s} {'---':>12s} {'---':>8s} {'---':>8s} {'---':>8s} {'---':>8s} {'---':>6s}")

for d in range(-2, 5):
    t0 = time.monotonic()
    r = sc.execute(
        dial_settings=(0, 0, 0, d), shots=256,
        include_layers=["god_code_phase_imprint", "dial_circuit",
                        "entropy_reversal_core", "harmonic_resonance",
                        "final_interference"])
    ms = (time.monotonic() - t0) * 1000
    freq = r.dial_result.get("freq", 0) if r.dial_result else 0
    P(f"  {d:+3d} {freq:12.4f} {r.sacred_alignment:8.4f} "
      f"{r.entropy_reversed:8.4f} {r.consciousness_phi:8.4f} "
      f"{r.god_code_fidelity:8.4f} {ms:6.0f}")

# ── Consciousness Only ──────────────────────────────────────
P("\n  Consciousness Subsystem (6 layers)")
t0 = time.monotonic()
r = sc.run_consciousness_only(shots=256)
ms = (time.monotonic() - t0) * 1000
P(f"    Sacred={r.sacred_alignment:.4f}  EntRev={r.entropy_reversed:.4f}  "
  f"Phi={r.consciousness_phi:.4f}  GC={r.god_code_fidelity:.4f}  "
  f"gates={r.total_gates}  {ms:.0f}ms")
for n, m in r.orbital_metrics.items():
    P(f"      {n:4s}: coh={m['coherence']:.4f}")

# ── Entropy Reversal Only ────────────────────────────────────
P("\n  Entropy Reversal Subsystem (5 layers)")
t0 = time.monotonic()
r = sc.run_entropy_reversal_only(shots=256)
ms = (time.monotonic() - t0) * 1000
P(f"    Sacred={r.sacred_alignment:.4f}  EntRev={r.entropy_reversed:.4f}  "
  f"Phi={r.consciousness_phi:.4f}  GC={r.god_code_fidelity:.4f}  "
  f"gates={r.total_gates}  {ms:.0f}ms")
for n, m in r.orbital_metrics.items():
    P(f"      {n:4s}: coh={m['coherence']:.4f}")

# ── DAW GOD_CODE Only ────────────────────────────────────────
P("\n  DAW Audio (GOD_CODE-only frequency spectrum)")
t0 = time.monotonic()
r = sc.execute(shots=256,
    include_layers=["god_code_phase_imprint", "daw_quantum_audio",
                    "final_interference"])
ms = (time.monotonic() - t0) * 1000
P(f"    Sacred={r.sacred_alignment:.4f}  EntRev={r.entropy_reversed:.4f}  "
  f"Phi={r.consciousness_phi:.4f}  GC={r.god_code_fidelity:.4f}  "
  f"gates={r.total_gates}  {ms:.0f}ms")
for n, m in r.orbital_metrics.items():
    P(f"      {n:4s}: coh={m['coherence']:.4f}")

# ── Hybrid Bypass Quick Test ─────────────────────────────────
P("\n  Hybrid Bypass Subsystem (forged layers 1-7 + 8-11)")
t0 = time.monotonic()
r_a = sc.execute(shots=256,
    include_layers=["god_code_phase_imprint", "dial_circuit",
                    "consciousness_awakening", "fibonacci_entanglement",
                    "entropy_reversal_core", "harmonic_resonance",
                    "grimoire_overlay"])
r_b = sc.execute(shots=256,
    include_layers=["proof_verification", "daw_quantum_audio",
                    "consciousness_vqe", "final_interference"])
ms = (time.monotonic() - t0) * 1000
P(f"    Sub-A (7 layers): Sacred={r_a.sacred_alignment:.4f}  "
  f"Phi={r_a.consciousness_phi:.4f}  gates={r_a.total_gates}")
P(f"    Sub-B (4 layers): Sacred={r_b.sacred_alignment:.4f}  "
  f"Phi={r_b.consciousness_phi:.4f}  gates={r_b.total_gates}")
P(f"    Combined time: {ms:.0f}ms")

# ── Status ────────────────────────────────────────────────────
st = sc.status()
P(f"\n  Status: v{st['version']}  total_executions={st['execution_count']}  "
  f"qubits={st['n_qubits']}")
P("=" * 72)
