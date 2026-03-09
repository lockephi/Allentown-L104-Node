#!/usr/bin/env python3
"""Quick daemon data injection verification."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from l104_audio_simulation.engine_integration import VQPUState, boot_vqpu_engine

t0 = time.time()
vs = boot_vqpu_engine(527.5184818492612, 13, 180000, 1800000)
dt = time.time() - t0

print(f"Boot time: {dt:.2f}s")
print(f"sacred_score: {vs.sacred_score:.4f}")
print(f"concurrence: {vs.concurrence:.4f}")
print(f"daemon_connected: {vs.daemon_connected}")
print(f"three_engine_composite: {vs.three_engine_composite:.6f}")
print(f"three_engine_entropy_reversal: {vs.three_engine_entropy_reversal:.4f}")
print(f"three_engine_harmonic_resonance: {vs.three_engine_harmonic_resonance:.4f}")
print(f"three_engine_wave_coherence: {vs.three_engine_wave_coherence:.6f}")
print(f"daemon_phi_resonance: {vs.daemon_phi_resonance:.6f}")
print(f"daemon_god_code_alignment: {vs.daemon_god_code_alignment:.6f}")
print(f"wavetable_frames: {len(vs.wavetable_frames)}")
print(f"vqe_energy: {vs.vqe_energy:.4f}")
print(f"fm_concurrence: {vs.fm_concurrence:.4f}")

# Check daemon injection worked
if vs.daemon_connected:
    print("\n✓ DAEMON DATA INJECTION CONFIRMED")
    print(f"  Three-engine composite from daemon: {vs.three_engine_composite:.6f}")
    print(f"  Sacred score (daemon-boosted): {vs.sacred_score:.6f}")
else:
    print("\n⚠ Daemon not connected — using classical fallbacks")
