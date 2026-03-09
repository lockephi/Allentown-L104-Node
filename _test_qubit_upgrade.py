"""Test GOD_CODE qubit audio integration upgrade (v2.1.0)."""

from l104_audio_simulation import (
    GOD_CODE_PHASE, IRON_PHASE, QPU_FIDELITY,
    QubitState, EngineState,
    audio_suite,
)
from l104_audio_simulation.constants import (
    GOD_CODE_PHASE, IRON_PHASE, PHI_AUDIO_PHASE, VOID_AUDIO_PHASE,
    IRON_LATTICE_PHASE, QPU_FIDELITY, QPU_BACKEND,
    PHI_PHASE_CONTRIBUTION, OCTAVE_PHASE,
)
from l104_audio_simulation.engine_integration import (
    QubitState, boot_god_code_qubit,
)
from l104_audio_simulation.tone_generator import QuantumPureToneGenerator
from l104_audio_simulation.quantum_synth import _get_god_code_qubit, WaveShape
from l104_audio_simulation.quantum_sequencer import _get_seq_qubit

print("=== GOD_CODE Qubit Audio Integration Test ===")
print(f"GOD_CODE_PHASE    = {GOD_CODE_PHASE:.10f} rad")
print(f"IRON_PHASE        = {IRON_PHASE:.10f} rad")
print(f"PHI_AUDIO_PHASE   = {PHI_AUDIO_PHASE:.10f} rad")
print(f"VOID_AUDIO_PHASE  = {VOID_AUDIO_PHASE:.10f} rad")
print(f"IRON_LATTICE_PHASE= {IRON_LATTICE_PHASE:.10f} rad")
print(f"QPU_FIDELITY      = {QPU_FIDELITY}")
print(f"QPU_BACKEND       = {QPU_BACKEND}")
print()

# Test QubitState
qs = QubitState()
print(f"QubitState.god_code_phase = {qs.god_code_phase:.10f}")
print(f"QubitState.iron_phase     = {qs.iron_phase:.10f}")
print(f"QubitState.qpu_fidelity   = {qs.qpu_fidelity}")
print()

# Test boot_god_code_qubit
qs2 = boot_god_code_qubit([{
    "a": 0, "b": 0, "c": 0, "d": 0,
    "freq_ideal": 527.518, "bloch_phase": 0.0,
    "base_weight": 1.0, "bloch_vector": [0, 0, 1],
}])
print("boot_god_code_qubit:")
print(f"  phase           = {qs2.god_code_phase:.10f}")
print(f"  qpu_verified    = {qs2.qpu_verified}")
print(f"  decomposed      = {qs2.decomposed_phases}")
print(f"  dial_phases     = {qs2.dial_phases}")
print(f"  gate_matrix shape = {qs2.gate_matrix.shape}")
print()

# Test tone generator
gen = QuantumPureToneGenerator(sample_rate=44100)
status = gen.status()
print(f"ToneGenerator qubit_available = {status['god_code_qubit_available']}")
print(f"ToneGenerator qubit_phase     = {status['qubit_phase']:.10f}")

# Test qubit tone
result = gen.god_code_qubit_tone(duration=0.5)
print(f"god_code_qubit_tone: {len(result.samples)} samples, alignment={result.sacred_alignment}")
print(f"  qpu_verified = {result.quantum_metrics.get('qpu_verified')}")
print(f"  phase_gc     = {result.quantum_metrics.get('phase_gc', 'N/A')}")
print()

# Test qubit tone with dials
result2 = gen.god_code_qubit_tone(duration=0.3, dials=(0, 0, 0, 0))
print(f"god_code_qubit_tone(dials=(0,0,0,0)): {len(result2.samples)} samples")
print(f"  dial_phase   = {result2.quantum_metrics.get('dial_phase', 'N/A')}")
print()

# Test synth qubit lazy-load
qb = _get_god_code_qubit()
print(f"Synth qubit loaded = {qb is not None}")
if qb:
    print(f"  phase = {qb.phase:.10f}")
    print(f"  decomposed keys = {list(qb.decomposed.keys())}")

# Test sequencer qubit lazy-load
sq = _get_seq_qubit()
print(f"Sequencer qubit loaded = {sq is not None}")

# Test WaveShape.GOD_CODE_WAVE with qubit phases
from l104_audio_simulation.quantum_synth import QuantumOscillator, OscillatorState
osc = QuantumOscillator(
    state=OscillatorState(waveform=WaveShape.GOD_CODE_WAVE, frequency=527.518)
)
print(f"\nGOD_CODE_WAVE oscillator: wavetable len={len(osc._wavetable)}")
print(f"  max amplitude = {max(abs(osc._wavetable)):.4f}")

print()
print("=== ALL IMPORTS + TESTS PASSED ===")
