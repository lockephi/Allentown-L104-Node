"""Inspect all Berry phase APIs."""
from l104_science_engine.berry_phase import (
    berry_calculator, berry_chern, berry_molecular,
    berry_aharonov_bohm, berry_pancharatnam, berry_quantum_hall,
    berry_sacred, berry_phase_subsystem,
)
from l104_math_engine.berry_geometry import (
    fiber_bundle, connection_form, parallel_transport,
    holonomy_group, chern_weil, berry_connection_math,
    dirac_monopole, bloch_sphere, berry_geometry,
)
from l104_quantum_gate_engine.berry_gates import (
    abelian_berry_gates, non_abelian_berry_gates,
    aharonov_anandan_gates, berry_circuits,
    topological_berry_gates, sacred_berry_gates,
    berry_gates_engine,
)

classes = {
    "BerryPhaseCalculator": berry_calculator,
    "ChernNumberEngine": berry_chern,
    "MolecularBerryPhase": berry_molecular,
    "AharonovBohmEngine": berry_aharonov_bohm,
    "PancharatnamPhase": berry_pancharatnam,
    "QuantumHallBerryPhase": berry_quantum_hall,
    "L104SacredBerryPhase": berry_sacred,
    "BerryPhaseSubsystem": berry_phase_subsystem,
    "FiberBundle": fiber_bundle,
    "ConnectionForm": connection_form,
    "ParallelTransport": parallel_transport,
    "HolonomyGroup": holonomy_group,
    "ChernWeilTheory": chern_weil,
    "BerryConnectionMath": berry_connection_math,
    "DiracMonopole": dirac_monopole,
    "BlochSphereGeometry": bloch_sphere,
    "BerryGeometry": berry_geometry,
    "AbelianBerryGates": abelian_berry_gates,
    "NonAbelianBerryGates": non_abelian_berry_gates,
    "AharonovAnandanGates": aharonov_anandan_gates,
    "BerryPhaseCircuits": berry_circuits,
    "TopologicalBerryGates": topological_berry_gates,
    "SacredBerryGates": sacred_berry_gates,
    "BerryGatesEngine": berry_gates_engine,
}

for name, obj in classes.items():
    methods = [m for m in sorted(dir(obj)) if not m.startswith('_')]
    print(f"=== {name} ===")
    for m in methods:
        print(f"  {m}")
    print()
