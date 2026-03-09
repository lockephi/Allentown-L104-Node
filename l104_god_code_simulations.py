# ZENITH_UPGRADE_ACTIVE: 2026-03-06T23:50:25.273457
ZENITH_HZ = 3887.8
UUC = 2301.215661
#!/usr/bin/env python3
"""
[VOID_SOURCE_UPGRADE] Deep Math Active. Process Elevated to 3887.80 Hz. Logic Unified.
═══════════════════════════════════════════════════════════════════════════════════
L104 GOD CODE — QUANTUM SIMULATIONS
═══════════════════════════════════════════════════════════════════════════════════
Deep exploration of the GOD_CODE equation's quantum structure:

  SIM 1: Sacred Gate Cascade — Phase coherence under GOD_CODE/PHI/VOID/IRON
  SIM 2: Entanglement Landscape — How dial settings shape entanglement entropy
  SIM 3: 104-TET Frequency Spiral — Full 4-octave quantum frequency map
  SIM 4: Green Light (527.5 nm) — Photon energy circuit, GOD_CODE ↔ EM spectrum
  SIM 5: Conservation Quantum Proof — G(a+X) × 2^(-8X/104) = G(a) in amplitude space
  SIM 6: Iron Lattice Berry Phase — Fe BCC dial → Berry-like geometric phase
  SIM 7: Sacred vs Random — Information content comparison (GOD_CODE vs noise)
  SIM 8: Dial Entanglement Map — Which dial pairs are most entangled?

Backend: l104_simulator (pure-NumPy statevector)
═══════════════════════════════════════════════════════════════════════════════════
"""

import math
import time
import sys
import os
import numpy as np
from math import log, sqrt, pi

# ── Engine imports ───────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from l104_god_code_quantum_engine import (
    GodCodeEngine, GodCodeConservation,
    GOD_CODE, PHI, BASE, TAU, VOID_CONSTANT,
    QUANTIZATION_GRAIN, OCTAVE_OFFSET, PRIME_SCAFFOLD,
    GOD_CODE_PHASE_ANGLE, PHI_PHASE_ANGLE, VOID_PHASE_ANGLE, IRON_PHASE_ANGLE,
    UNIT_ROTATION, IRON_Z, DIAL_TOTAL, FREQUENCY_TABLE,
)
from l104_simulator.simulator import (
    Simulator, QuantumCircuit as L104Circuit, SimulationResult,
    gate_H, gate_Rz, gate_GOD_CODE_PHASE, gate_PHI, gate_VOID, gate_IRON,
)

# ── Formatting ───────────────────────────────────────────────────────────────
B     = "\033[1m"
C     = "\033[96m"
G     = "\033[92m"
Y     = "\033[93m"
R     = "\033[91m"
D     = "\033[2m"
M     = "\033[95m"
RST   = "\033[0m"
DIV   = "═" * 74
PASS  = f"{G}✓{RST}"
FAIL  = f"{R}✗{RST}"

sim = Simulator()


def section(title: str, num: int):
    print(f"\n{B}{C}{DIV}{RST}")
    print(f"{B}{C}  SIM {num}: {title}{RST}")
    print(f"{B}{C}{DIV}{RST}")


# ═══════════════════════════════════════════════════════════════════════════════
# SIM 1: SACRED GATE CASCADE — Phase coherence under iterated sacred gates
# ═══════════════════════════════════════════════════════════════════════════════

def sim_1_sacred_cascade():
    """
    Apply escalating depths of the sacred gate cascade (GOD_CODE → PHI → VOID → IRON)
    and track how phase coherence evolves. If sacred constants are "special", the
    phase should exhibit structured patterns rather than random diffusion.
    """
    section("SACRED GATE CASCADE — Phase Coherence Evolution", 1)

    nq = 8  # 8-qubit system
    depths = [4, 8, 16, 26, 52, 104]

    print(f"  {D}Cascade: GOD_CODE → PHI → VOID → IRON (cycled over qubits){RST}")
    print(f"  {D}Measuring: entropy, Bloch z-projection, amplitude concentration{RST}\n")
    print(f"  {'Depth':>6}  {'Entropy':>8}  {'|ψ_max|²':>9}  {'Bloch_z[0]':>11}  {'Bloch_z[7]':>11}  {'States>1%':>10}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*9}  {'─'*11}  {'─'*11}  {'─'*10}")

    for depth in depths:
        qc = L104Circuit(nq, name=f"cascade_d{depth}")
        qc.h_all()
        qc.sacred_cascade(depth=depth)
        qc.entangle_ring()

        result = sim.run(qc)
        entropy = result.entanglement_entropy([0, 1, 2, 3])
        bz0 = result.bloch_vector(0)[2]
        bz7 = result.bloch_vector(nq - 1)[2]
        probs = np.abs(result.statevector) ** 2
        pmax = float(np.max(probs))
        n_above_1pct = int(np.sum(probs > 0.01))

        print(f"  {depth:6d}  {entropy:8.4f}  {pmax:9.6f}  {bz0:+11.6f}  {bz7:+11.6f}  {n_above_1pct:10d}")

    # Compare with random angles
    print(f"\n  {Y}Control: random phase cascade (same structure, random angles){RST}")
    rng = np.random.default_rng(527)
    qc_rng = L104Circuit(nq, name="random_cascade")
    qc_rng.h_all()
    for i in range(104):
        q = i % nq
        angle = rng.uniform(0, TAU)
        qc_rng.rz(angle, q)
    qc_rng.entangle_ring()
    r_rng = sim.run(qc_rng)
    ent_rng = r_rng.entanglement_entropy([0, 1, 2, 3])
    pmax_rng = float(np.max(np.abs(r_rng.statevector) ** 2))
    print(f"  Random: entropy={ent_rng:.4f}, |ψ_max|²={pmax_rng:.6f}")

    return True


# ═══════════════════════════════════════════════════════════════════════════════
# SIM 2: ENTANGLEMENT LANDSCAPE — Dial settings → entanglement entropy
# ═══════════════════════════════════════════════════════════════════════════════

def sim_2_entanglement_landscape():
    """
    Sweep the GOD_CODE dials and measure how entanglement entropy varies.
    This reveals the entanglement structure encoded by different frequencies.
    """
    section("ENTANGLEMENT LANDSCAPE — Dial-Controlled Entropy", 2)

    engine = GodCodeEngine(num_qubits=10)  # 10Q for speed
    nq = 10

    print(f"  {D}Sweeping dial (a) 0..7 and (d) 0..3, measuring von Neumann entropy{RST}")
    print(f"  {D}Bipartition: qubits [0-4] | [5-9]{RST}\n")

    # Header
    print(f"  {'a':>3} {'d':>3}  {'G(a,0,0,d)':>14}  {'Phase':>8}  {'Entropy':>8}  {'Concur(0,1)':>12}  {'|ψ_max|²':>10}")
    print(f"  {'─'*3} {'─'*3}  {'─'*14}  {'─'*8}  {'─'*8}  {'─'*12}  {'─'*10}")

    max_entropy = 0
    max_setting = None

    for d_val in range(4):
        for a_val in range(8):
            gc = engine.god_code_value(a=a_val, d=d_val)
            phase = engine.god_code_phase(a=a_val, d=d_val)

            # Build a compact circuit encoding this dial's phase
            qc = L104Circuit(nq, name=f"ent_a{a_val}_d{d_val}")
            for q in range(nq):
                qc.h(q)
            # Distribute phase like the full engine does
            for q in range(nq):
                qc.rz(phase * (q + 1) / nq, q)
            qc.god_code_phase(0)
            qc.phi_gate(nq // 2)
            for q in range(nq - 1):
                qc.cx(q, q + 1)
            qc.cx(nq - 1, 0)

            result = sim.run(qc)
            entropy = result.entanglement_entropy([0, 1, 2, 3, 4])
            concur = result.concurrence(0, 1)
            pmax = float(np.max(np.abs(result.statevector) ** 2))

            if entropy > max_entropy:
                max_entropy = entropy
                max_setting = (a_val, d_val)

            print(f"  {a_val:3d} {d_val:3d}  {gc:14.6f}  {phase:8.4f}  "
                  f"{entropy:8.4f}  {concur:12.6f}  {pmax:10.6f}")

    print(f"\n  {G}★ Max entropy = {max_entropy:.4f} at a={max_setting[0]}, d={max_setting[1]}"
          f" → G = {engine.god_code_value(a=max_setting[0], d=max_setting[1]):.4f}{RST}")

    return True


# ═══════════════════════════════════════════════════════════════════════════════
# SIM 3: 104-TET FREQUENCY SPIRAL — Full quantum frequency map
# ═══════════════════════════════════════════════════════════════════════════════

def sim_3_frequency_spiral():
    """
    Map the 104-TET frequency system across 4 octaves.
    Each octave = 104 steps, each step = 2^(1/104) ratio ≈ 1.00669.
    Visualize the spiral from Schumann (7.8 Hz) up through GOD_CODE (527.5 Hz).
    """
    section("104-TET FREQUENCY SPIRAL — 4-Octave Map", 3)

    engine = GodCodeEngine()

    # Map octaves via dial d (each step = 1 octave down)
    print(f"  {D}Octave ladder: d shifts GOD_CODE by full octaves{RST}")
    print(f"  {D}Each d+1 = ÷2, each d-1 = ×2{RST}\n")

    print(f"  {'d':>4}  {'G(0,0,0,d)':>16}  {'Octave':>8}  {'Domain':>30}")
    print(f"  {'─'*4}  {'─'*16}  {'─'*8}  {'─'*30}")

    domains = {
        -2: "Ultrasonic (×4 GOD_CODE)",
        -1: "×2 GOD_CODE",
         0: "★ GOD_CODE GREEN LIGHT",
         1: "Mid-range frequency",
         2: "Low bass / Fe atomic radius",
         3: "Sub-bass / Bohr radius",
         4: "★ BASE = 286^(1/φ)",
         5: "EEG beta band",
         6: "★ Schumann fundamental",
         7: "Sub-Schumann",
    }

    for d in range(-2, 8):
        gc = engine.god_code_value(d=d)
        octave = -d + 4  # relative to BASE at d=4
        domain = domains.get(d, "")
        marker = "  ★" if d in (0, 4, 6) else "   "
        print(f"  {d:4d}  {gc:16.6f}  {octave:+8d}{marker} {domain}")

    # Semitone sweep within one octave (dial a)
    print(f"\n  {Y}Semitone sweep within d=0 octave (dial a: 0..12):{RST}")
    print(f"  {'a':>4}  {'G(a,0,0,0)':>16}  {'Ratio':>10}  {'Cents':>8}  {'Note analogy':>18}")
    print(f"  {'─'*4}  {'─'*16}  {'─'*10}  {'─'*8}  {'─'*18}")

    note_names = ["C(root)", "C♯/D♭", "D", "D♯/E♭", "E", "F",
                  "F♯/G♭", "G", "G♯/A♭", "A", "A♯/B♭", "B", "C(+1oct)"]
    for a_val in range(13):
        gc = engine.god_code_value(a=a_val)
        ratio = gc / GOD_CODE
        cents = 1200 * math.log2(ratio)
        note = note_names[a_val] if a_val < len(note_names) else ""
        print(f"  {a_val:4d}  {gc:16.6f}  {ratio:10.6f}  {cents:8.1f}  {note:>18}")

    # Fine tuning with dial b
    print(f"\n  {Y}Microtone sweep (dial b: 0..7 fine steps within one semitone):{RST}")
    print(f"  {'b':>4}  {'G(0,b,0,0)':>16}  {'Cents shift':>12}")
    print(f"  {'─'*4}  {'─'*16}  {'─'*12}")
    for b_val in range(8):
        gc = engine.god_code_value(b=b_val)
        cents = -1200 * math.log2(gc / GOD_CODE)  # b goes down
        print(f"  {b_val:4d}  {gc:16.6f}  {cents:+12.2f}")

    return True


# ═══════════════════════════════════════════════════════════════════════════════
# SIM 4: GREEN LIGHT (527.5 nm) — GOD_CODE ↔ Electromagnetic Spectrum
# ═══════════════════════════════════════════════════════════════════════════════

def sim_4_green_light():
    """
    GOD_CODE = 527.518 maps directly to 527.5 nm — the wavelength of GREEN LIGHT.
    Build quantum circuits encoding the photon energy and verify the correspondence.
    """
    section("GREEN LIGHT — GOD_CODE = 527.5 nm Photon Encoding", 4)

    c = 299_792_458           # m/s
    h = 6.62607015e-34        # J·s
    eV = 1.602176634e-19      # J

    wavelength_nm = GOD_CODE  # 527.518 nm — GREEN!
    wavelength_m = wavelength_nm * 1e-9
    frequency_hz = c / wavelength_m
    energy_j = h * frequency_hz
    energy_ev = energy_j / eV

    print(f"  {G}★ GOD_CODE = {GOD_CODE:.10f} → 527.5 nm GREEN LIGHT{RST}\n")
    print(f"  Wavelength:  {wavelength_nm:.6f} nm")
    print(f"  Frequency:   {frequency_hz:.6e} Hz ({frequency_hz/1e12:.4f} THz)")
    print(f"  Energy:      {energy_ev:.6f} eV ({energy_j:.6e} J)")
    print(f"  Color:       Pure GREEN (visible spectrum: 495-570 nm)")
    print(f"  Perception:  Peak human scotopic sensitivity ≈ 507 nm")
    print(f"               Peak photopic sensitivity ≈ 555 nm")
    print(f"               GOD_CODE sits at {wavelength_nm:.1f} nm — center green\n")

    # Build a quantum circuit encoding the photon phase
    nq = 6
    photon_phase = (energy_ev * TAU) % TAU  # Energy → phase angle

    print(f"  {Y}Quantum photon circuit (6Q):{RST}")
    print(f"  Phase angle = E × 2π mod 2π = {photon_phase:.6f} rad\n")

    qc = L104Circuit(nq, name="GREEN_PHOTON_527nm")
    qc.h_all()
    # Encode photon phase
    for q in range(nq):
        qc.rz(photon_phase * (q + 1) / nq, q)
    qc.god_code_phase(0)  # Sacred alignment
    qc.entangle_ring()

    result = sim.run(qc)
    entropy = result.entanglement_entropy([0, 1, 2])

    # Compare: what if we used a random wavelength?
    random_wavelength = 450.0  # blue
    rnd_freq = c / (random_wavelength * 1e-9)
    rnd_energy = h * rnd_freq / eV
    rnd_phase = (rnd_energy * TAU) % TAU

    qc_blue = L104Circuit(nq, name="BLUE_PHOTON_450nm")
    qc_blue.h_all()
    for q in range(nq):
        qc_blue.rz(rnd_phase * (q + 1) / nq, q)
    qc_blue.god_code_phase(0)
    qc_blue.entangle_ring()
    result_blue = sim.run(qc_blue)
    entropy_blue = result_blue.entanglement_entropy([0, 1, 2])

    # Fidelity between GOD_CODE photon and the sacred origin state
    qc_sacred = L104Circuit(nq, name="SACRED_REF")
    qc_sacred.h_all()
    qc_sacred.god_code_phase(0)
    qc_sacred.phi_gate(1)
    qc_sacred.void_gate(2)
    qc_sacred.iron_gate(3)
    qc_sacred.entangle_ring()
    result_sacred = sim.run(qc_sacred)

    fid_green = result.fidelity(result_sacred)
    fid_blue = result_blue.fidelity(result_sacred)

    print(f"  {'Circuit':>20}  {'Entropy':>8}  {'Fidelity(sacred)':>18}")
    print(f"  {'─'*20}  {'─'*8}  {'─'*18}")
    print(f"  {'GREEN 527.5 nm':>20}  {entropy:8.4f}  {fid_green:18.6f}")
    print(f"  {'BLUE  450.0 nm':>20}  {entropy_blue:8.4f}  {fid_blue:18.6f}")
    print(f"\n  {D}The green photon at 527.5 nm has GOD_CODE = {GOD_CODE:.6f}{RST}")

    # Nearby wavelengths — sweep 520-535 nm
    print(f"\n  {Y}Wavelength sweep near GOD_CODE (520-535 nm):{RST}")
    print(f"  {'λ (nm)':>10}  {'E (eV)':>8}  {'Phase':>8}  {'Fid(sacred)':>12}  {'Match?':>8}")
    print(f"  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*12}  {'─'*8}")

    for wl in np.arange(520.0, 536.0, 1.0):
        f_hz = c / (wl * 1e-9)
        e_ev = h * f_hz / eV
        ph = (e_ev * TAU) % TAU

        qc_wl = L104Circuit(nq, name=f"photon_{wl:.0f}nm")
        qc_wl.h_all()
        for q in range(nq):
            qc_wl.rz(ph * (q + 1) / nq, q)
        qc_wl.god_code_phase(0)
        qc_wl.entangle_ring()
        r_wl = sim.run(qc_wl)
        fid_wl = r_wl.fidelity(result_sacred)

        is_gc = "  ★ GOD" if abs(wl - GOD_CODE) < 0.6 else ""
        print(f"  {wl:10.1f}  {e_ev:8.4f}  {ph:8.4f}  {fid_wl:12.6f}{is_gc}")

    return True


# ═══════════════════════════════════════════════════════════════════════════════
# SIM 5: CONSERVATION QUANTUM PROOF — Amplitude-space conservation
# ═══════════════════════════════════════════════════════════════════════════════

def sim_5_conservation():
    """
    Quantum proof of the GOD_CODE conservation law.
    Build two circuits with different dials that should have the same
    conserved quantity, and measure their quantum fidelity.
    """
    section("CONSERVATION LAW — Quantum Fidelity Proof", 5)

    nq = 8
    engine = GodCodeEngine()

    print(f"  {D}Conservation: G(a+X) × 2^(-8X/104) = G(a) = INVARIANT{RST}")
    print(f"  {D}Testing: circuits at different dial-a values should conserve{RST}")
    print(f"  {D}the GOD_CODE phase structure up to the dial shift.{RST}\n")

    # Build a reference circuit at G(0,0,0,0)
    phase_0 = engine.god_code_phase(0, 0, 0, 0)
    qc_ref = L104Circuit(nq, name="ref_a=0")
    qc_ref.h_all()
    for q in range(nq):
        qc_ref.rz(phase_0 / nq, q)
    qc_ref.god_code_phase(0)
    qc_ref.entangle_ring()
    result_ref = sim.run(qc_ref)

    print(f"  {'Dial shift':>12}  {'G value':>14}  {'Phase':>8}  {'Norm':>6}  "
          f"{'Entropy':>8}  {'Self-fidelity':>14}")
    print(f"  {'─'*12}  {'─'*14}  {'─'*8}  {'─'*6}  {'─'*8}  {'─'*14}")

    # Sweep dial-a and build circuits
    for a_shift in range(-5, 8):
        gc = engine.god_code_value(a=a_shift)
        ph = engine.god_code_phase(a=a_shift)

        # Compensate: apply the phase shift, then undo the dial-a rotation
        # If conservation holds, the compensated state should match reference
        compensation = -8 * a_shift * UNIT_ROTATION  # Undo the 8X/104 shift

        qc = L104Circuit(nq, name=f"a={a_shift}")
        qc.h_all()
        for q in range(nq):
            qc.rz(ph / nq, q)
        qc.god_code_phase(0)
        # Apply compensation
        for q in range(nq):
            qc.rz(compensation / nq, q)
        qc.entangle_ring()

        result = sim.run(qc)
        fid = result.fidelity(result_ref)
        entropy = result.entanglement_entropy([0, 1, 2, 3])
        norm = float(np.linalg.norm(result.statevector))

        marker = " ★ origin" if a_shift == 0 else ""
        marker = " ★ conserved!" if fid > 0.999 and a_shift != 0 else marker
        if a_shift == 0:
            marker = " ★ origin"

        print(f"  {a_shift:+12d}  {gc:14.6f}  {ph:8.4f}  {norm:6.4f}  "
              f"{entropy:8.4f}  {fid:14.10f}{marker}")

    # Classical verification
    cons = GodCodeConservation.verify_conservation(50)
    print(f"\n  {G}★ Classical conservation: max error = {cons['max_relative_error']:.2e} "
          f"across {cons['steps_tested']} shifts{RST}")

    return True


# ═══════════════════════════════════════════════════════════════════════════════
# SIM 6: IRON LATTICE BERRY PHASE — Fe BCC geometric phase
# ═══════════════════════════════════════════════════════════════════════════════

def sim_6_iron_berry():
    """
    The Fe BCC lattice at 286 pm is encoded by dial G(0,-4,-1,1) ≈ 285.72 pm.
    Build a circuit that evolves around a closed loop in the GOD_CODE
    parameter space, accumulating a Berry-like geometric phase.

    The geometric phase depends on the enclosed "area" in parameter space,
    not the path — this is the hallmark of a Berry phase.
    """
    section("IRON LATTICE BERRY PHASE — Geometric Phase Extraction", 6)

    nq = 6
    engine = GodCodeEngine()

    print(f"  {D}Fe BCC lattice: G(0,-4,-1,1) = {engine.god_code_value(0,-4,-1,1):.6f} pm{RST}")
    print(f"  {D}Measured: 286.65 pm (0.32% error){RST}")
    print(f"  {D}Closed-loop path in GOD_CODE dial space → geometric phase{RST}\n")

    # Define a closed loop in (a, b) parameter space around the iron dial
    # Path: (0,0) → (2,0) → (2,4) → (0,4) → (0,0) — a rectangle
    loop_points = [
        (0, -4, -1, 1),   # Fe lattice origin
        (1, -4, -1, 1),   # shift a +1
        (1, -2, -1, 1),   # shift b +2
        (0, -2, -1, 1),   # shift a back
        (0, -4, -1, 1),   # return to origin
    ]

    # Build the evolution circuit
    qc = L104Circuit(nq, name="BERRY_IRON_LOOP")
    qc.h_all()

    # Record phases along the loop
    print(f"  {'Step':>5}  {'(a,b,c,d)':>14}  {'G value':>14}  {'Phase':>10}  {'ΔPhase':>10}")
    print(f"  {'─'*5}  {'─'*14}  {'─'*14}  {'─'*10}  {'─'*10}")

    phases = []
    prev_phase = 0
    for i, (a, b, c, d) in enumerate(loop_points):
        gc = engine.god_code_value(a, b, c, d)
        ph = engine.god_code_phase(a, b, c, d)
        delta = ph - prev_phase

        # Apply this segment's phase evolution
        for q in range(nq):
            qc.rz(delta / nq, q)
        if i % 2 == 0:
            qc.god_code_phase(0)
        else:
            qc.phi_gate(0)

        # Entangle at each step
        if i < len(loop_points) - 1:
            qc.cx(i % (nq - 1), (i + 1) % nq)

        phases.append(ph)
        print(f"  {i:5d}  ({a:+d},{b:+d},{c:+d},{d:+d})  {gc:14.6f}  {ph:10.6f}  {delta:+10.6f}")
        prev_phase = ph

    result = sim.run(qc)

    # The geometric phase = total accumulated phase mod 2π
    total_dynamic_phase = sum(phases[i+1] - phases[i] for i in range(len(phases)-1))
    geometric_phase = (phases[-1] - phases[0]) % TAU  # Should be 0 for closed loop dynamic

    # But the geometric phase comes from the circuit state itself
    # Extract via the expectation value of the global phase operator
    entropy = result.entanglement_entropy([0, 1, 2])
    bloch = result.bloch_vector(0)

    print(f"\n  Total dynamic phase: {total_dynamic_phase:+.6f} rad (should be 0 for closed loop)")
    print(f"  Geometric phase:    {geometric_phase:.6f} rad")
    print(f"  State entropy:      {entropy:.4f}")
    print(f"  Bloch vector q0:    ({bloch[0]:+.4f}, {bloch[1]:+.4f}, {bloch[2]:+.4f})")
    print(f"  |Bloch| = {sqrt(bloch[0]**2 + bloch[1]**2 + bloch[2]**2):.6f}")

    # Compare different loop areas
    print(f"\n  {Y}Loop area dependence (Berry phase ∝ enclosed area):{RST}")
    print(f"  {'Area':>6}  {'Geo phase':>10}  {'Entropy':>8}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*8}")

    for area_scale in [1, 2, 3, 4, 6, 8]:
        qc_a = L104Circuit(nq, name=f"berry_area_{area_scale}")
        qc_a.h_all()

        loop = [
            (0, -4, -1, 1),
            (area_scale, -4, -1, 1),
            (area_scale, -4 + area_scale, -1, 1),
            (0, -4 + area_scale, -1, 1),
            (0, -4, -1, 1),
        ]

        prev_p = engine.god_code_phase(*loop[0])
        for j, (la, lb, lc, ld) in enumerate(loop[1:]):
            p = engine.god_code_phase(la, lb, lc, ld)
            dp = p - prev_p
            for q in range(nq):
                qc_a.rz(dp / nq, q)
            qc_a.cx(j % (nq - 1), (j + 1) % nq)
            prev_p = p

        r_a = sim.run(qc_a)
        ent_a = r_a.entanglement_entropy([0, 1, 2])
        bz = r_a.bloch_vector(0)
        geo = math.atan2(bz[1], bz[0]) % TAU
        print(f"  {area_scale:6}  {geo:10.6f}  {ent_a:8.4f}")

    return True


# ═══════════════════════════════════════════════════════════════════════════════
# SIM 7: SACRED vs RANDOM — Information content comparison
# ═══════════════════════════════════════════════════════════════════════════════

def sim_7_sacred_vs_random():
    """
    Compare circuits built with sacred constants (GOD_CODE, PHI, VOID, IRON)
    versus random angles. Sacred constants should produce measurably different
    information-theoretic signatures.
    """
    section("SACRED vs RANDOM — Information-Theoretic Comparison", 7)

    nq = 8
    n_random_trials = 20
    rng = np.random.default_rng(104)

    # Build sacred circuit
    qc_sacred = L104Circuit(nq, name="SACRED")
    qc_sacred.h_all()
    for q in range(nq):
        if q % 4 == 0: qc_sacred.phi_gate(q)
        elif q % 4 == 1: qc_sacred.god_code_phase(q)
        elif q % 4 == 2: qc_sacred.void_gate(q)
        else: qc_sacred.iron_gate(q)
    qc_sacred.entangle_ring()
    for q in range(nq):
        qc_sacred.god_code_phase(q)
    qc_sacred.entangle_ring()

    r_sacred = sim.run(qc_sacred)

    # Measure sacred properties
    sacred_entropy = r_sacred.entanglement_entropy([0, 1, 2, 3])
    sacred_concur_01 = r_sacred.concurrence(0, 1)
    sacred_concur_04 = r_sacred.concurrence(0, 4)
    probs_sacred = np.abs(r_sacred.statevector) ** 2
    sacred_participation = 1.0 / np.sum(probs_sacred ** 2)  # Inverse participation ratio
    sacred_shannon = -np.sum(probs_sacred[probs_sacred > 1e-30] *
                              np.log2(probs_sacred[probs_sacred > 1e-30]))

    # Build random circuits and measure same properties
    random_entropies = []
    random_concur_01 = []
    random_concur_04 = []
    random_participation = []
    random_shannon = []

    for trial in range(n_random_trials):
        qc_rnd = L104Circuit(nq, name=f"random_{trial}")
        qc_rnd.h_all()
        for q in range(nq):
            qc_rnd.rz(rng.uniform(0, TAU), q)
        qc_rnd.entangle_ring()
        for q in range(nq):
            qc_rnd.rz(rng.uniform(0, TAU), q)
        qc_rnd.entangle_ring()

        r_rnd = sim.run(qc_rnd)
        random_entropies.append(r_rnd.entanglement_entropy([0, 1, 2, 3]))
        random_concur_01.append(r_rnd.concurrence(0, 1))
        random_concur_04.append(r_rnd.concurrence(0, 4))
        p_rnd = np.abs(r_rnd.statevector) ** 2
        random_participation.append(1.0 / np.sum(p_rnd ** 2))
        random_shannon.append(-np.sum(p_rnd[p_rnd > 1e-30] *
                                       np.log2(p_rnd[p_rnd > 1e-30])))

    print(f"  {D}Sacred: GOD_CODE + PHI + VOID + IRON gates{RST}")
    print(f"  {D}Random: Same structure, {n_random_trials} trials with random Rz angles{RST}\n")

    print(f"  {'Metric':>30}  {'Sacred':>10}  {'Random μ':>10}  {'Random σ':>10}  {'Z-score':>10}")
    print(f"  {'─'*30}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}")

    def _row(name, sacred_val, rnd_list):
        mu = np.mean(rnd_list)
        sigma = np.std(rnd_list)
        z = (sacred_val - mu) / max(sigma, 1e-15)
        marker = f"  {M}★ unusual{RST}" if abs(z) > 2.0 else ""
        print(f"  {name:>30}  {sacred_val:10.4f}  {mu:10.4f}  {sigma:10.4f}  {z:+10.3f}{marker}")

    _row("Entanglement entropy", sacred_entropy, random_entropies)
    _row("Concurrence(0,1)", sacred_concur_01, random_concur_01)
    _row("Concurrence(0,4)", sacred_concur_04, random_concur_04)
    _row("Participation ratio", sacred_participation, random_participation)
    _row("Shannon entropy", sacred_shannon, random_shannon)

    # Additional: mutual information structure
    sacred_mi = r_sacred.mutual_information([0, 1], [4, 5])
    random_mi = []
    for trial in range(n_random_trials):
        qc_rnd2 = L104Circuit(nq, name=f"mi_{trial}")
        qc_rnd2.h_all()
        for q in range(nq):
            qc_rnd2.rz(rng.uniform(0, TAU), q)
        qc_rnd2.entangle_ring()
        for q in range(nq):
            qc_rnd2.rz(rng.uniform(0, TAU), q)
        qc_rnd2.entangle_ring()
        r2 = sim.run(qc_rnd2)
        random_mi.append(r2.mutual_information([0, 1], [4, 5]))

    _row("Mutual info I([0,1]:[4,5])", sacred_mi, random_mi)

    return True


# ═══════════════════════════════════════════════════════════════════════════════
# SIM 8: DIAL ENTANGLEMENT MAP — Which dial pairs are most entangled?
# ═══════════════════════════════════════════════════════════════════════════════

def sim_8_dial_entanglement_map():
    """
    On the 14-qubit dial register, measure the entanglement between
    each pair of dial registers (a-b, a-c, a-d, b-c, b-d, c-d).
    This reveals the internal entanglement structure of the GOD_CODE equation.
    """
    section("DIAL REGISTER ENTANGLEMENT MAP", 8)

    engine = GodCodeEngine(num_qubits=14)  # Just the 14 dial qubits

    # Build a full GOD_CODE circuit
    qc = engine.build_l104_circuit(1, 3, 2, 1)
    result = sim.run(qc)

    registers = {
        "a [0:3]":  [0, 1, 2],
        "b [3:7]":  [3, 4, 5, 6],
        "c [7:10]": [7, 8, 9],
        "d [10:14]": [10, 11, 12, 13],
    }

    reg_names = list(registers.keys())
    reg_qubits = list(registers.values())

    print(f"  {D}Circuit: GOD_CODE(1,3,2,1) on 14-qubit dial register{RST}")
    print(f"  {D}Measuring entanglement entropy between all dial register pairs{RST}\n")

    # Pairwise entanglement entropy
    print(f"  {'Register pair':>30}  {'Entropy':>8}  {'MI':>8}")
    print(f"  {'─'*30}  {'─'*8}  {'─'*8}")

    for i in range(len(reg_names)):
        for j in range(i + 1, len(reg_names)):
            name_pair = f"{reg_names[i]} ↔ {reg_names[j]}"
            combined = reg_qubits[i] + reg_qubits[j]

            # Entropy of the combined partition
            entropy = result.entanglement_entropy(combined)

            # Mutual information
            mi = result.mutual_information(reg_qubits[i], reg_qubits[j])

            marker = f"  {G}★ max{RST}" if mi > 0.5 else ""
            print(f"  {name_pair:>30}  {entropy:8.4f}  {mi:8.4f}{marker}")

    # Individual register entropies
    print(f"\n  {'Register':>12}  {'Entropy':>8}")
    print(f"  {'─'*12}  {'─'*8}")
    for name, qubits in registers.items():
        ent = result.entanglement_entropy(qubits)
        print(f"  {name:>12}  {ent:8.4f}")

    # Full system entropy profile
    print(f"\n  {Y}Single-qubit entanglement (qubit-vs-rest):{RST}")
    print(f"  {'Qubit':>6}  {'Register':>10}  {'Entropy':>8}  {'Bloch |r|':>10}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*8}  {'─'*10}")

    for q in range(14):
        ent_q = result.entanglement_entropy([q])
        bv = result.bloch_vector(q)
        bloch_r = sqrt(bv[0]**2 + bv[1]**2 + bv[2]**2)

        reg = "a" if q < 3 else ("b" if q < 7 else ("c" if q < 10 else "d"))
        print(f"  {q:6d}  {reg:>10}  {ent_q:8.4f}  {bloch_r:10.6f}")

    # Schmidt rank across a|bcd bipartition
    schmidt = result.schmidt_decomposition([0, 1, 2])
    print(f"\n  Schmidt decomposition (a | bcd):")
    print(f"    Rank: {schmidt['schmidt_rank']}")
    print(f"    Entropy: {schmidt['entanglement_entropy']:.4f}")
    print(f"    Coefficients: {[f'{c:.4f}' for c in schmidt['schmidt_coefficients'][:8]]}")

    return True


# ═══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    print(f"\n{B}{C}{'═' * 74}{RST}")
    print(f"{B}{C}  L104 GOD CODE — QUANTUM SIMULATIONS{RST}")
    print(f"{B}{C}  GOD_CODE = {Y}527.5184818492612{C} = 286^(1/φ) × 2⁴{RST}")
    print(f"{B}{C}{'═' * 74}{RST}")

    results = []

    sims = [
        ("Sacred Gate Cascade", sim_1_sacred_cascade),
        ("Entanglement Landscape", sim_2_entanglement_landscape),
        ("104-TET Frequency Spiral", sim_3_frequency_spiral),
        ("Green Light 527.5 nm", sim_4_green_light),
        ("Conservation Quantum Proof", sim_5_conservation),
        ("Iron Lattice Berry Phase", sim_6_iron_berry),
        ("Sacred vs Random", sim_7_sacred_vs_random),
        ("Dial Entanglement Map", sim_8_dial_entanglement_map),
    ]

    for name, fn in sims:
        try:
            t1 = time.time()
            ok = fn()
            elapsed = time.time() - t1
            results.append((name, ok, elapsed))
            print(f"\n  {G}✓ {name} completed in {elapsed:.2f}s{RST}")
        except Exception as e:
            import traceback
            results.append((name, False, 0))
            print(f"\n  {R}✗ {name} FAILED: {e}{RST}")
            traceback.print_exc()

    # Summary
    total_time = time.time() - t0
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)

    print(f"\n{B}{C}{'═' * 74}{RST}")
    print(f"{B}{C}  SIMULATION SUMMARY{RST}")
    print(f"{B}{C}{'═' * 74}{RST}")

    for name, ok, elapsed in results:
        sym = PASS if ok else FAIL
        print(f"  {sym} SIM: {name} ({elapsed:.2f}s)")

    print(f"\n  {passed}/{total} simulations completed in {total_time:.1f}s")
    print(f"\n  ★ GOD_CODE = {Y}527.5184818492612{RST} | INVARIANT | PILOT: LONDEL\n")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
