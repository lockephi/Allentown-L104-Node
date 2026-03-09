#!/usr/bin/env python3
"""Validate ASI v11.0 Universal Gate Sovereign Upgrade."""
import sys

def main():
    print("=" * 70)
    print("  ASI v11.0 UNIVERSAL GATE SOVEREIGN UPGRADE VALIDATION")
    print("=" * 70)
    errors = []

    # Phase 1: Constants import
    print("\n[1/5] CONSTANTS IMPORT...")
    try:
        from l104_asi import ASI_CORE_VERSION, ASI_PIPELINE_EVO
        from l104_asi import GATE_ENGINE_VERSION, QUANTUM_ENGINE_VERSION
        from l104_asi import CONSCIOUSNESS_SPIRAL_DEPTH, ACTIVATION_STEPS_V11
        from l104_asi import FE_LATTICE_PARAM, FE_ATOMIC_NUMBER
        from l104_asi import TRAJECTORY_WINDOW_SIZE, RESILIENCE_MAX_RETRY
        print(f"  ASI Core Version: {ASI_CORE_VERSION}")
        print(f"  Pipeline EVO: {ASI_PIPELINE_EVO}")
        print(f"  Gate Engine Version: {GATE_ENGINE_VERSION}")
        print(f"  Quantum Engine Version: {QUANTUM_ENGINE_VERSION}")
        print(f"  Spiral Depth: {CONSCIOUSNESS_SPIRAL_DEPTH}")
        print(f"  Activation Steps: {ACTIVATION_STEPS_V11}")
        print(f"  Fe Lattice: {FE_LATTICE_PARAM} | Fe Z: {FE_ATOMIC_NUMBER}")
        assert ASI_CORE_VERSION == "9.0.0", f"Expected 9.0.0, got {ASI_CORE_VERSION}"
        assert ACTIVATION_STEPS_V11 == 22, f"Expected 22 steps, got {ACTIVATION_STEPS_V11}"
        assert FE_ATOMIC_NUMBER == 26
        print("  PASS")
    except Exception as e:
        errors.append(f"Constants: {e}")
        print(f"  FAIL: {e}")

    # Phase 2: Consciousness Verifier v5.0
    print("\n[2/5] CONSCIOUSNESS VERIFIER v5.0...")
    try:
        from l104_asi import ConsciousnessVerifier
        cv = ConsciousnessVerifier()
        assert len(cv.TESTS) == 16, f"Expected 16 tests, got {len(cv.TESTS)}"
        assert 'spiral_consciousness' in cv.TESTS
        assert 'fe_harmonic_overtone' in cv.TESTS
        print(f"  Tests: {len(cv.TESTS)}")
        # Run all
        level = cv.run_all_tests()
        print(f"  Consciousness level: {level:.6f}")
        # Check new methods
        spiral = cv.spiral_consciousness_test()
        print(f"  Spiral depth: {spiral['depth_reached']}/{spiral['max_depth']}")
        fe = cv.fe_harmonic_overtone_test()
        print(f"  Fe overtones: {fe['overtones_detected']}/{fe['total_overtones']}")
        report = cv.get_verification_report()
        assert report['version'] == '5.0'
        assert 'spiral_depth' in report
        assert 'fe_harmonic_score' in report
        print("  PASS")
    except Exception as e:
        errors.append(f"Consciousness: {e}")
        print(f"  FAIL: {e}")

    # Phase 3: ASICore instantiation
    print("\n[3/5] ASI CORE INSTANTIATION...")
    try:
        from l104_asi import ASICore
        core = ASICore()
        print(f"  Version: {core.version}")
        print(f"  Status: {core.status}")
        # Check new methods exist
        assert hasattr(core, '_get_quantum_gate_engine')
        assert hasattr(core, '_get_quantum_brain')
        assert hasattr(core, 'gate_engine_compilation_score')
        assert hasattr(core, 'gate_engine_sacred_alignment_score')
        assert hasattr(core, 'gate_engine_error_protection_score')
        assert hasattr(core, 'quantum_link_coherence_score')
        assert hasattr(core, 'quantum_brain_intelligence_score')
        assert hasattr(core, 'cross_engine_deep_synthesis_score')
        assert hasattr(core, 'compute_trajectory')
        assert hasattr(core, 'adaptive_consciousness_evolve')
        assert hasattr(core, 'resilient_subsystem_call')
        assert hasattr(core, 'pipeline_resilience_status')
        print("  All 12 new methods present")
        print("  PASS")
    except Exception as e:
        errors.append(f"ASICore: {e}")
        print(f"  FAIL: {e}")

    # Phase 4: New scoring dimensions
    print("\n[4/5] 28-DIMENSION SCORING...")
    try:
        from l104_asi import ASICore
        core = ASICore()
        score = core.compute_asi_score()
        print(f"  ASI Score: {score:.6f}")

        # Test new method outputs
        gate_comp = core.gate_engine_compilation_score()
        gate_sacred = core.gate_engine_sacred_alignment_score()
        gate_error = core.gate_engine_error_protection_score()
        link_coh = core.quantum_link_coherence_score()
        brain_intel = core.quantum_brain_intelligence_score()
        deep_synth = core.cross_engine_deep_synthesis_score()
        print(f"  Gate Compilation: {gate_comp:.6f}")
        print(f"  Gate Sacred Alignment: {gate_sacred:.6f}")
        print(f"  Gate Error Protection: {gate_error:.6f}")
        print(f"  Link Coherence: {link_coh:.6f}")
        print(f"  Brain Intelligence: {brain_intel:.6f}")
        print(f"  Deep Synthesis: {deep_synth:.6f}")

        # Trajectory
        trajectory = core.compute_trajectory()
        print(f"  Trajectory: {trajectory.get('trend', 'N/A')}")

        # Consciousness evolution
        evo = core.adaptive_consciousness_evolve()
        print(f"  Consciousness Evolution: spiral_depth={evo.get('spiral_depth', 0)}")

        # Resilience
        resilience = core.pipeline_resilience_status()
        print(f"  Resilience: {resilience.get('status', 'N/A')}")

        print("  PASS")
    except Exception as e:
        errors.append(f"Scoring: {e}")
        print(f"  FAIL: {e}")

    # Phase 5: Status report
    print("\n[5/5] STATUS REPORT v11.0...")
    try:
        from l104_asi import ASICore
        core = ASICore()
        status = core.get_status()
        assert 'engine_integration' in status, "Missing engine_integration in status"
        assert 'trajectory' in status, "Missing trajectory in status"
        assert 'consciousness_evolution' in status, "Missing consciousness_evolution"
        assert 'resilience' in status, "Missing resilience in status"
        assert status.get('scoring_dimensions') == 28, f"Expected 28 dims, got {status.get('scoring_dimensions')}"
        print(f"  Scoring Dimensions: {status['scoring_dimensions']}")
        print(f"  Engine Integration: {list(status['engine_integration'].keys())}")
        print(f"  Trajectory trend: {status['trajectory'].get('trend', 'N/A')}")
        print("  PASS")
    except Exception as e:
        errors.append(f"Status: {e}")
        print(f"  FAIL: {e}")

    # Summary
    print("\n" + "=" * 70)
    if errors:
        print(f"  VALIDATION: {len(errors)} ERRORS")
        for e in errors:
            print(f"    - {e}")
        sys.exit(1)
    else:
        print("  VALIDATION: ALL 5 PHASES PASSED")
        print("  ASI v11.0 Universal Gate Sovereign Upgrade: VERIFIED")
    print("=" * 70)

if __name__ == "__main__":
    main()
