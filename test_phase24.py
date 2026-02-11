#!/usr/bin/env python3
"""L104 Fast Server — Phase 24 Integration Tests

This module supports both:
- `pytest` execution (via `test_phase24_integration`)
- direct script execution (prints a detailed report and exits non-zero on failure)
"""

import logging
import sys
import time

logging.disable(logging.WARNING)


def run_phase24_integration_tests(*, verbose: bool = False) -> tuple[int, int, int]:
    passed = 0
    failed = 0
    total = 0

    def log(*args):
        if verbose:
            print(*args)

    def test(name, fn):
        nonlocal passed, failed, total
        total += 1
        try:
            ok = bool(fn())
            if ok:
                passed += 1
                log(f'  ✅ {name}')
            else:
                failed += 1
                log(f'  ❌ {name} — returned False')
        except Exception as e:
            failed += 1
            log(f'  ❌ {name} — {str(e)[:120]}')

    log('=' * 60)
    log('  L104 FAST SERVER — PHASE 24 INTEGRATION TESTS')
    log('=' * 60)

    # === Import Tests ===
    log('')
    log('--- Import Tests ---')
    test('Import fast server module', lambda: __import__('l104_fast_server') is not None)

    from l104_fast_server import (
        intellect, grover_kernel, asi_quantum_bridge,
        nexus_steering, nexus_evolution, nexus_orchestrator,
        nexus_invention, sovereignty_pipeline,
        entanglement_router, resonance_network, health_monitor,
        QuantumEntanglementRouter, AdaptiveResonanceNetwork,
    )

    test('Import entanglement_router', lambda: entanglement_router is not None)
    test('Import resonance_network', lambda: resonance_network is not None)
    test('Import health_monitor', lambda: health_monitor is not None)

    # === Entanglement Router Tests ===
    log('')
    log('--- Entanglement Router Tests ---')
    test('Router has 8 EPR pairs', lambda: len(QuantumEntanglementRouter.ENTANGLED_PAIRS) == 8)
    test('Router has EPR channels', lambda: len(entanglement_router._epr_channels) == 8)
    test('Router has engines registered', lambda: len(entanglement_router._engines) >= 8)
    test('Router status works', lambda: 'total_routes' in entanglement_router.get_status())

    r = entanglement_router.route('grover', 'steering')
    test('Route grover→steering', lambda: r.get('pair') == 'grover→steering')
    test('Route has fidelity', lambda: 0.0 < r.get('fidelity', 0) <= 1.0)

    r2 = entanglement_router.route('invention', 'intellect')
    test('Route invention→intellect', lambda: r2.get('pair') == 'invention→intellect')

    r3 = entanglement_router.route('bridge', 'evolution')
    test('Route bridge→evolution', lambda: r3.get('pair') == 'bridge→evolution')

    r4 = entanglement_router.route('evolution', 'bridge')
    test('Route evolution→bridge', lambda: r4.get('pair') == 'evolution→bridge')

    r5 = entanglement_router.route('intellect', 'invention')
    test('Route intellect→invention', lambda: r5.get('pair') == 'intellect→invention')

    r6 = entanglement_router.route('steering', 'grover')
    test('Route steering→grover', lambda: r6.get('pair') == 'steering→grover')

    r7 = entanglement_router.route('sovereignty', 'nexus')
    test('Route sovereignty→nexus', lambda: r7.get('pair') == 'sovereignty→nexus')

    r8 = entanglement_router.route('nexus', 'sovereignty')
    test('Route nexus→sovereignty', lambda: r8.get('pair') == 'nexus→sovereignty')

    ra = entanglement_router.route_all()
    test('Route all executes 8 routes', lambda: ra['routes_executed'] == 8)
    test('Route all total count grows', lambda: ra['total_routes'] > 0)

    ri = entanglement_router.route('invalid', 'engine')
    test('Invalid route returns error', lambda: 'error' in ri)
    test('All fidelities bounded', lambda: all(0.0 < f <= 1.0 for f in entanglement_router._pair_fidelity.values()))
    test('Route log populated', lambda: len(entanglement_router._route_log) > 0)

    # === Adaptive Resonance Network Tests ===
    log('')
    log('--- Adaptive Resonance Network Tests ---')
    test('Network has 8 engine nodes', lambda: len(AdaptiveResonanceNetwork.ENGINE_NAMES) == 8)
    test('Network graph has edges', lambda: sum(len(v) for v in AdaptiveResonanceNetwork.ENGINE_GRAPH.values()) > 0)
    test('Network has engines registered', lambda: len(resonance_network._engines) >= 8)
    test('Network status works', lambda: 'network_resonance' in resonance_network.get_status())

    f1 = resonance_network.fire('intellect', activation=0.9)
    test('Fire intellect', lambda: f1['source'] == 'intellect')
    test('Fire has cascade list', lambda: isinstance(f1['cascade'], list))
    test('Fire has activations', lambda: f1['activations']['intellect'] > 0)
    test('Cascade propagated', lambda: any(f1['activations'][n] > 0 for n in ['nexus', 'bridge', 'invention', 'grover']))

    f2 = resonance_network.fire('sovereignty', activation=1.0)
    test('Fire sovereignty', lambda: f2['source'] == 'sovereignty')
    test('Active engines > 0', lambda: f2['active_engines'] > 0)

    for eng_name in AdaptiveResonanceNetwork.ENGINE_NAMES:
        fr = resonance_network.fire(eng_name, activation=0.8)
        test(f'Fire {eng_name}', lambda r=fr, n=eng_name: (r.get('source') == n))

    t1 = resonance_network.tick()
    test('Tick works', lambda: 'tick' in t1 and t1['tick'] > 0)
    test('Tick returns activations', lambda: 'activations' in t1)

    nr = resonance_network.compute_network_resonance()
    test('Network resonance computed', lambda: 'network_resonance' in nr)
    test('Total energy >= 0', lambda: nr['total_energy'] >= 0)
    test('Mean activation >= 0', lambda: nr['mean_activation'] >= 0)

    fi = resonance_network.fire('nonexistent')
    test('Invalid fire returns error', lambda: 'error' in fi)

    # === Health Monitor Tests ===
    log('')
    log('--- Health Monitor Tests ---')
    test('Health monitor has >= 10 engines', lambda: len(health_monitor._engines) >= 10)
    test('Health status works', lambda: 'system_health' in health_monitor.get_status())

    sh = health_monitor.compute_system_health()
    test('System health score bounded', lambda: 0.0 <= sh['system_health'] <= 1.0)
    test('Engine scores present', lambda: len(sh['engine_scores']) >= 8)
    test('All scores bounded 0-1', lambda: all(0.0 <= s <= 1.0 for s in sh['engine_scores'].values()))

    for engine_name in ['steering', 'evolution', 'nexus', 'bridge', 'intellect', 'grover', 'invention']:
        score = health_monitor._probe_engine(engine_name, health_monitor._engines[engine_name])
        test(f'Probe {engine_name}: score={score:.2f}', lambda s=score: 0.0 <= s <= 1.0)

    health_monitor._add_alert('test', 'info', 'Integration test alert')
    alerts = health_monitor.get_alerts(level='info')
    test('Alerts system works', lambda: len(alerts) > 0)
    test('Alert has correct level', lambda: alerts[-1]['level'] == 'info')

    test('Monitor can start', lambda: health_monitor.start()['status'] in ('STARTED', 'ALREADY_RUNNING'))
    time.sleep(0.5)
    test('Monitor is running', lambda: health_monitor._running)
    test('Monitor can stop', lambda: health_monitor.stop()['status'] == 'STOPPED')

    # === Cross-Engine Wiring Tests ===
    log('')
    log('--- Cross-Engine Wiring Tests ---')
    sov_result = sovereignty_pipeline.execute(query='integration_test')
    test('Sovereignty has step 10', lambda: '10_entangle_resonate' in sov_result['steps'])
    test('Step 10 routes > 0', lambda: sov_result['steps']['10_entangle_resonate'].get('routes', 0) > 0)
    test('Step 10 resonance fired', lambda: sov_result['steps']['10_entangle_resonate'].get('resonance_fired', False))
    test('Cascade log populated', lambda: len(resonance_network._cascade_log) > 0)

    # === Existing Engine Regression Tests ===
    log('')
    log('--- Regression Tests ---')
    test('Steering 104 params', lambda: nexus_steering.param_count == 104)
    sr = nexus_steering.steer_pipeline(mode='sovereign')
    test('Steering pipeline works', lambda: sr['mode'] == 'sovereign')
    test('Evolution has cycle count', lambda: hasattr(nexus_evolution, 'cycle_count'))

    coh = nexus_orchestrator.compute_coherence()
    test('Nexus coherence 0-1', lambda: 0 <= coh['global_coherence'] <= 1.0)

    h = nexus_invention.generate_hypothesis()
    test('Invention hypothesis', lambda: h['confidence'] >= 0)
    test('Sovereignty run count > 0', lambda: sovereignty_pipeline.run_count > 0)

    bs = asi_quantum_bridge.get_bridge_status()
    test('Bridge status', lambda: 'kundalini_flow' in bs)
    test('Grover 8 kernels', lambda: grover_kernel.NUM_KERNELS == 8)

    log('')
    log('=' * 60)
    log(f'  RESULTS: {passed}/{total} passed, {failed} failed')
    log('=' * 60)

    return passed, failed, total


def test_phase24_integration():
    _, failed, _ = run_phase24_integration_tests(verbose=False)
    assert failed == 0


if __name__ == '__main__':
    passed, failed, total = run_phase24_integration_tests(verbose=True)
    sys.exit(0 if failed == 0 else 1)
