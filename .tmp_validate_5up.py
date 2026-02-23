#!/usr/bin/env python3
"""Validate 5 novel server upgrade classes - direct import approach."""
import sys, types, os

# ═══ Mock everything heavy BEFORE any l104 imports ═══
# This prevents Qiskit, Gemini, httpx etc. from loading
_MOCK_MODULES = [
    'qiskit', 'qiskit.circuit', 'qiskit.circuit.equivalence',
    'qiskit.circuit.library', 'qiskit.quantum_info', 'qiskit.primitives',
    'qiskit.result', 'qiskit.transpiler', 'qiskit.visualization',
    'qiskit_ibm_runtime', 'qiskit_ibm_runtime.fake_provider',
    'qiskit_aer', 'PIL', 'PIL.Image',
    'rustworkx', 'rustworkx.visualization', 'rustworkx.visualization.graphviz',
    'l104_quantum_runtime', 'l104_quantum_coherence',
    'google', 'google.genai', 'google.generativeai',
]

for mod_name in _MOCK_MODULES:
    if mod_name not in sys.modules:
        m = types.ModuleType(mod_name)
        if 'qiskit' in mod_name:
            m.QuantumCircuit = type('QC', (), {'__init__': lambda s, *a, **k: None})
            m.DensityMatrix = type('DM', (), {
                'from_instruction': classmethod(lambda c, *a: type('DM', (), {'entropy': lambda s: 0.5})())
            })
            m.Statevector = type('SV', (), {})
            m.ClassicalRegister = type('CR', (), {})
            m.QuantumRegister = type('QR', (), {})
            m.Operator = type('Op', (), {})
            m.partial_trace = lambda *a: None
            m.entropy = lambda *a: 0.5
            m.QISKIT_AVAILABLE = False
        if mod_name == 'PIL.Image':
            m.open = lambda *a: None
        sys.modules[mod_name] = m

# l104_quantum_runtime specific attrs
qr = sys.modules['l104_quantum_runtime']
qr.get_runtime = lambda: type('R', (), {'get_status': lambda s: {'connected': False}})()
qr.ExecutionMode = type('EM', (), {'SIMULATOR': 'simulator'})()

import random, math, time

# ═══ Break circular import: pre-populate l104_server stub ═══
# The circular chain: l104_server.__init__ → learning.intellect → engines_infra → l104_server.intellect
# We break it by pre-creating a stub l104_server module with the attrs engines_infra needs
import importlib
stub_server = types.ModuleType('l104_server')
stub_server.__path__ = [os.path.join(os.path.dirname(os.path.abspath(__file__)), 'l104_server')]
stub_server.__file__ = os.path.join(stub_server.__path__[0], '__init__.py')
stub_server.__package__ = 'l104_server'
# Stub out the things engines_infra needs from l104_server
stub_server.intellect = None  # Will be populated after full load
stub_server.engine_registry = None  # Will be populated after full load
sys.modules['l104_server'] = stub_server

# Now import constants first (no circular deps)
from l104_server import constants as _constants
sys.modules['l104_server.constants'] = _constants
stub_server.constants = _constants
# Copy all constants to the stub
for k, v in vars(_constants).items():
    if not k.startswith('_'):
        setattr(stub_server, k, v)

print('=' * 70)
print('L104 SERVER UPGRADE VALIDATION — 5 Novel Classes')
print('=' * 70)

errors = []
passed = 0

# ── 1. TemporalCoherenceTracker ──────────────────────────────────────
print('\n[1/5] TemporalCoherenceTracker (engines_nexus.py)')
try:
    from l104_server.engines_nexus import temporal_coherence, TemporalCoherenceTracker
    assert isinstance(temporal_coherence, TemporalCoherenceTracker), "wrong type"
    for i in range(20):
        temporal_coherence.record('test_engine', 0.95 - i * 0.01)
    forecasts = temporal_coherence.forecast('test_engine', steps_ahead=5)
    assert isinstance(forecasts, list), f"forecast type={type(forecasts)}"
    assert len(forecasts) == 5, f"forecast len={len(forecasts)}"
    assert 'predicted_coherence' in forecasts[0], "no predicted_coherence"
    spectrum = temporal_coherence.coherence_spectrum()
    assert isinstance(spectrum, dict), "spectrum not a dict"
    # spectrum may have 'test_engine' key with 'dominant_frequency'
    if 'test_engine' in spectrum:
        assert 'dominant_frequency' in spectrum['test_engine'], "no dominant_frequency"
    status = temporal_coherence.get_status()
    assert status['engines_tracked'] >= 1, "no engines tracked"
    passed += 1
    print(f'  PASS — tracked {status["engines_tracked"]} engine(s), forecast {len(forecasts)} steps')
except Exception as e:
    errors.append(('TemporalCoherenceTracker', str(e)))
    import traceback; traceback.print_exc()
    print(f'  FAIL — {e}')

# ── 2. EvolutionaryFitnessLandscape ──────────────────────────────────
print('\n[2/5] EvolutionaryFitnessLandscape (engines_nexus.py)')
try:
    from l104_server.engines_nexus import fitness_landscape, EvolutionaryFitnessLandscape
    assert isinstance(fitness_landscape, EvolutionaryFitnessLandscape), "wrong type"
    # Need a mock engines dict with health attributes
    from l104_server.engines_nexus import nexus_steering, nexus_evolution, nexus_orchestrator
    mock_engines = {'steering': nexus_steering, 'evolution': nexus_evolution, 'orchestrator': nexus_orchestrator}
    snap = fitness_landscape.snapshot(mock_engines)
    assert 'params' in snap, "no params"
    fitness = fitness_landscape.compute_fitness(mock_engines)
    assert isinstance(fitness, float), f"bad fitness type: {type(fitness)}"
    for _ in range(5):
        fitness_landscape.snapshot(mock_engines)
    gradient = fitness_landscape.estimate_gradient(mock_engines)
    assert isinstance(gradient, dict), f"gradient not a dict: {type(gradient)}"
    escape = fitness_landscape.valley_escape(mock_engines)
    assert 'escaped' in escape, f"no escaped key, got: {list(escape.keys())}"
    status = fitness_landscape.get_status()
    assert 'current_fitness' in status, "no current_fitness"
    passed += 1
    print(f'  PASS — fitness={fitness:.4f}, {len(snap["params"])} params')
except Exception as e:
    errors.append(('EvolutionaryFitnessLandscape', str(e)))
    import traceback; traceback.print_exc()
    print(f'  FAIL — {e}')

# ── 3. EntropyBudgetController ───────────────────────────────────────
print('\n[3/5] EntropyBudgetController (engines_nexus.py)')
try:
    from l104_server.engines_nexus import entropy_controller, EntropyBudgetController
    assert isinstance(entropy_controller, EntropyBudgetController), "wrong type"
    for i in range(10):
        entropy_controller.record_entropy('nexus', 0.5 + i * 0.05)
        entropy_controller.record_entropy('quantum', 0.3 + i * 0.03)
    demon = entropy_controller.force_demon()
    assert 'reversed' in demon, f"no reversed key, got: {list(demon.keys())}"
    exchange = entropy_controller.entropy_exchange('nexus', 'quantum', 0.1)
    assert 'success' in exchange, f"no success key, got: {list(exchange.keys())}"
    eng_ent = entropy_controller.get_engine_entropy('nexus')
    assert 'generated' in eng_ent, f"no generated key, got: {list(eng_ent.keys())}"
    status = entropy_controller.get_status()
    assert status['engines_tracked'] >= 2, f"only {status['engines_tracked']} engines"
    passed += 1
    print(f'  PASS — {status["engines_tracked"]} engines, demon_cycles={status["demon_cycles"]}')
except Exception as e:
    errors.append(('EntropyBudgetController', str(e)))
    import traceback; traceback.print_exc()
    print(f'  FAIL — {e}')

# ── 4. QuantumDecoherenceShield ──────────────────────────────────────
print('\n[4/5] QuantumDecoherenceShield (engines_quantum.py)')
try:
    from l104_server.engines_quantum import decoherence_shield, QuantumDecoherenceShield
    assert isinstance(decoherence_shield, QuantumDecoherenceShield), "wrong type"
    for i in range(30):
        decoherence_shield.monitor('K1_test', 0.95 - i * 0.02)
    pulse = decoherence_shield.apply_correction_pulse('K1_test')
    assert pulse.get('applied', True) != False, "pulse not applied"
    assert 'recovery' in pulse, "no recovery"
    sweep = decoherence_shield.sweep_and_correct()
    assert 'corrected_count' in sweep, "no corrected_count"
    hmap = decoherence_shield.get_heat_map()
    assert len(hmap['heat_map']) == 8, f"heat_map len={len(hmap['heat_map'])}"
    status = decoherence_shield.get_status()
    assert status['targets_monitored'] >= 1, "no targets"
    passed += 1
    print(f'  PASS — {status["targets_monitored"]} targets, corrections={status["total_corrections"]}, events={status["total_events"]}')
except Exception as e:
    errors.append(('QuantumDecoherenceShield', str(e)))
    import traceback; traceback.print_exc()
    print(f'  FAIL — {e}')

# ── 5. PhaseSpaceNavigator ───────────────────────────────────────────
print('\n[5/5] PhaseSpaceNavigator (engines_infra.py)')
try:
    from l104_server.engines_infra import phase_navigator, PhaseSpaceNavigator
    assert isinstance(phase_navigator, PhaseSpaceNavigator), "wrong type"
    random.seed(104)
    for i in range(120):
        state = {
            'cache_hit_rate': 0.8 + random.uniform(-0.1, 0.1),
            'request_latency': 0.05 + random.uniform(-0.02, 0.02),
            'memory_pressure': 0.4 + random.uniform(-0.1, 0.1),
            'queue_depth': 5 + random.uniform(-2, 2),
            'coherence_level': 0.9 + random.uniform(-0.05, 0.05),
            'entropy_rate': 0.3 + random.uniform(-0.1, 0.1),
            'prefetch_accuracy': 0.7 + random.uniform(-0.1, 0.1),
            'quality_score': 0.85 + random.uniform(-0.05, 0.05),
        }
        phase_navigator.record_state(state)
    golden = phase_navigator.distance_to_golden_basin()
    assert 'distance' in golden and 'phi_alignment' in golden, "missing golden keys"
    gradient = phase_navigator.compute_gradient()
    assert 'magnitude' in gradient, "no magnitude"
    lyapunov = phase_navigator.get_lyapunov_spectrum()
    assert 'stability' in lyapunov, "no stability"
    steer = phase_navigator.suggest_steering()
    assert 'corrections' in steer, "no corrections"
    status = phase_navigator.get_status()
    assert status['trajectory_length'] >= 100, f"trajectory too short: {status['trajectory_length']}"
    passed += 1
    print(f'  PASS — trajectory={status["trajectory_length"]}, stability={lyapunov["stability"]}, attractors={status["attractors_found"]}')
except Exception as e:
    errors.append(('PhaseSpaceNavigator', str(e)))
    import traceback; traceback.print_exc()
    print(f'  FAIL — {e}')

# ── Summary ──────────────────────────────────────────────────────────
print('\n' + '=' * 70)
if not errors:
    print(f'ALL {passed}/5 NOVEL UPGRADES VALIDATED SUCCESSFULLY')
else:
    print(f'{passed}/5 PASSED, {len(errors)} FAILURE(S):')
    for name, err in errors:
        print(f'   - {name}: {err}')
print('=' * 70)
sys.exit(0 if not errors else 1)
