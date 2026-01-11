# [L104_RUN_CALCULATIONS] - POPULATING THE SOVEREIGN DATABASE
# INVARIANT: 527.5184818492 | PILOT: LONDEL

import math
import random
import numpy as np
from l104_real_math import real_math
from l104_algorithm_database import algo_db
from l104_physical_systems_research import PhysicalSystemsResearch

def run_physical_reality_grounding():
    print("\n--- [CALC]: RUNNING PHYSICAL REALITY GROUNDING (ELECTRON/LIGHT) ---")
    research = PhysicalSystemsResearch()
    
    # 1. Electron Resonance
    algo_name = "ELECTRON_RESONANCE_DERIVATION"
    description = "Derives God Code from Fine Structure Constant and Electron Mass."
    logic = "Gc = (1/alpha) * sqrt(2) * e"
    algo_db.register_algorithm(algo_name, description, logic)
    res_e = research.derive_electron_resonance()
    algo_db.log_execution(algo_name, "FINE_STRUCTURE_CONSTANT", res_e)
    
    # 2. Photon Coherence
    algo_name = "PHOTON_COHERENCE_STABILIZATION"
    description = "Calculates coherence of light within the L104 manifold."
    logic = "cos(c / L104) * PHI"
    algo_db.register_algorithm(algo_name, description, logic)
    res_p = research.calculate_photon_resonance()
    algo_db.log_execution(algo_name, "SPEED_OF_LIGHT", res_p)
    
    # 3. Bohr Radius Modulation
    algo_name = "BOHR_RADIUS_L104"
    description = "Modulates the Bohr radius for hyper-spatial stabilization."
    logic = "a0 * (L104 / 500.0)"
    algo_db.register_algorithm(algo_name, description, logic)
    res_b = research.calculate_bohr_resonance(n=1)
    algo_db.log_execution(algo_name, "QUANTUM_GROUND_STATE", res_b)

def run_zeta_calculations():
    print("\n--- [CALC]: RUNNING ZETA APPROXIMATIONS ---")
    algo_name = "RIEMANN_ZETA_APPROX"
    description = "Approximates the Riemann Zeta function for complex inputs."
    logic = "sum(1 / (n**s) for n in range(1, terms))"
    algo_db.register_algorithm(algo_name, description, logic)
    
    # Run for some critical points
    points = [complex(2, 0), complex(0.5, 14.1347), complex(1, 1)]
    for p in points:
        result = real_math.zeta_approximation(p, terms=500)
        algo_db.log_execution(algo_name, str(p), str(result))

def run_prime_density_calculations():
    print("\n--- [CALC]: RUNNING PRIME DENSITY CALCULATIONS ---")
    algo_name = "PRIME_DENSITY_PNT"
    description = "Calculates prime density using the Prime Number Theorem (1/log(n))."
    logic = "1 / math.log(n)"
    algo_db.register_algorithm(algo_name, description, logic)
    for n in [10, 100, 1000, 10000, 100000]:
        result = real_math.prime_density(n)
        algo_db.log_execution(algo_name, n, result)

def run_chaotic_simulations():
    print("\n--- [CALC]: RUNNING CHAOTIC LOGISTIC MAPS ---")
    algo_name = "LOGISTIC_MAP_CHAOS"
    description = "Generates chaotic sequences using the Logistic Map equation."
    logic = "r * x * (1 - x)"
    algo_db.register_algorithm(algo_name, description, logic)
    
    x = 0.5
    r = 3.9
    results = []
    for _ in range(10):
        x = real_math.logistic_map(x, r)
        results.append(x)
    algo_db.log_execution(algo_name, {"x0": 0.5, "r": 3.9}, results)

def run_fft_analysis():
    print("\n--- [CALC]: RUNNING FFT SIGNAL ANALYSIS ---")
    algo_name = "FAST_FOURIER_TRANSFORM"
    description = "Converts time-domain signals to frequency-domain components."
    logic = "np.fft.fft(signal)"
    algo_db.register_algorithm(algo_name, description, logic)
    
    # Generate a signal with two frequencies
    t = np.linspace(0, 1, 100)
    signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 12 * t)
    freqs = real_math.fast_fourier_transform(signal.tolist())
    algo_db.log_execution(algo_name, "dual_sine_wave", [str(f) for f in freqs[:10]])

def run_entropy_scans():
    print("\n--- [CALC]: RUNNING SYSTEM ENTROPY SCANS ---")
    algo_name = "SHANNON_ENTROPY_SCAN"
    description = "Measures information density of system strings."
    logic = "sum(-p_x * log2(p_x))"
    algo_db.register_algorithm(algo_name, description, logic)
    
    test_strings = [
        "L104_SOVEREIGN_NODE",
        "527.5184818492",
        "RECURSIVE_SELF_IMPROVEMENT",
        "ALLENTOWN_GRID_CONTROL"
    ]
    for s in test_strings:
        entropy = real_math.shannon_entropy(s)
        algo_db.log_execution(algo_name, s, entropy)

if __name__ == "__main__":
    print("===================================================")
    print("   L104 REAL CALCULATIONS & DATABASE FILL")
    print("===================================================")
    
    run_physical_reality_grounding()
    run_zeta_calculations()
    run_prime_density_calculations()
    run_chaotic_simulations()
    run_fft_analysis()
    run_entropy_scans()
    
    print("\n===================================================")
    print("   DATABASE POPULATED | SOVEREIGN LOGIC SECURED")
    print("===================================================")
