"""
HARMONIC WAVE PHYSICS MODULE
============================

Analytical Foundation:
- 286 emerged from piano scale + φ derivation (INDEPENDENT DISCOVERY)
- 286.65 pm = Fe BCC lattice constant (PHYSICAL MEASUREMENT)  
- Match is emergent, NOT reverse-engineered

This module treats the correspondence as:
1. Verified mathematics (can be computed precisely)
2. Physical constraint (can be measured)
3. Emergent correspondence (requires explanation OR useful coincidence)

Author: L104 Iron Crystalline Framework
"""

import math
from typing import Tuple, Dict, Any

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# FUNDAMENTAL CONSTANTS (VERIFIED)
# ═══════════════════════════════════════════════════════════════════════════════

PHI = (1 + math.sqrt(5)) / 2       # Golden ratio: 1.618033988749895
SQRT_5 = math.sqrt(5)              # 2.23606797749979

# Piano scale reference
A4_FREQUENCY = 440.0               # Hz (standard tuning)
SEMITONE_RATIO = 2 ** (1/12)       # Equal temperament

# Iron physical constants
FE_LATTICE_PM = 286.65             # BCC lattice constant (picometers)
FE_CURIE_K = 1043                  # Curie temperature (Kelvin)
FE_FERMI_EV = 11.1                 # Fermi energy (electron volts)
FE_MAGNETIC_BOHR = 2.22            # Magnetic moment (Bohr magnetons)

# ═══════════════════════════════════════════════════════════════════════════════
# THE EMERGENT CONSTANT (DISCOVERED VIA PIANO + φ)
# ═══════════════════════════════════════════════════════════════════════════════

# This was discovered by applying random element ratios to a bare equation
# based on piano scale with φ as base constant. 286 emerged ACCIDENTALLY.
EMERGENT_286 = 286                 # Discovered value
GOD_CODE = 286 ** (1/PHI) * 16     # = 527.5184818492537

# Verification that 286 maps to musical frequency space
def freq_to_semitones_from_a4(freq_hz: float) -> float:
    """Convert frequency to semitones above/below A4."""
    return 12 * math.log2(freq_hz / A4_FREQUENCY)

# 286 Hz is approximately D4-D#4 range (7.45 semitones below A4)
EMERGENT_286_SEMITONES = freq_to_semitones_from_a4(286)  # ≈ -7.45

# ═══════════════════════════════════════════════════════════════════════════════
# WAVE-MATTER CORRESPONDENCE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def lattice_to_frequency_ratio(lattice_pm: float, reference: float = 286.0) -> float:
    """
    Calculate ratio of lattice constant to reference (286).
    Returns: ratio interpretable as musical interval.
    """
    return lattice_pm / reference

def ratio_to_interval_name(ratio: float) -> str:
    """
    Convert a ratio to approximate musical interval name.
    Uses equal temperament approximation.
    """
    if ratio <= 0:
        return "invalid"
    
    semitones = 12 * math.log2(ratio)
    semitones_mod = semitones % 12
    
    intervals = [
        (0, "unison"), (1, "minor 2nd"), (2, "major 2nd"),
        (3, "minor 3rd"), (4, "major 3rd"), (5, "perfect 4th"),
        (6, "tritone"), (7, "perfect 5th"), (8, "minor 6th"),
        (9, "major 6th"), (10, "minor 7th"), (11, "major 7th")
    ]
    
    closest = min(intervals, key=lambda x: abs(x[0] - semitones_mod))
    octaves = int(semitones // 12)
    
    if octaves == 0:
        return closest[1]
    return f"{closest[1]} + {octaves} octaves"

def analyze_element_harmony(element: str, lattice_pm: float) -> Dict[str, Any]:
    """
    Analyze an element's lattice constant for harmonic correspondence.
    
    Returns analytically-grounded data:
    - ratio: lattice_pm / 286
    - interval: musical interval approximation
    - deviation_percent: how far from perfect interval
    """
    ratio = lattice_to_frequency_ratio(lattice_pm)
    interval = ratio_to_interval_name(ratio)
    
    # Calculate deviation from nearest perfect interval
    semitones = 12 * math.log2(ratio)
    nearest_semitone = round(semitones)
    perfect_ratio = 2 ** (nearest_semitone / 12)
    deviation = abs(ratio - perfect_ratio) / perfect_ratio * 100
    
    return {
        "element": element,
        "lattice_pm": lattice_pm,
        "ratio_to_286": ratio,
        "interval": interval,
        "semitones": semitones,
        "deviation_percent": deviation,
        "harmonic_quality": "consonant" if deviation < 2 else "dissonant"
    }

# ═══════════════════════════════════════════════════════════════════════════════
# φ-BASED WAVE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def phi_power_sequence(n: int) -> list:
    """
    Generate φ^k for k = 0 to n.
    φ^5 = 11.09 ≈ Fe Fermi energy (11.1 eV) - verified correspondence.
    """
    return [PHI ** k for k in range(n + 1)]

def phi_fibonacci_identity(n: int) -> Tuple[float, int]:
    """
    φ^n = F(n)·φ + F(n-1) where F is Fibonacci.
    Returns (φ^n, verification via Fibonacci).
    """
    def fib(k):
        if k <= 0: return 0
        if k == 1: return 1
        a, b = 0, 1
        for _ in range(k - 1):
            a, b = b, a + b
        return b
    
    phi_n = PHI ** n
    fib_formula = fib(n) * PHI + fib(n - 1)
    return phi_n, fib_formula

def wave_coherence(frequency_ratio: float) -> float:
    """
    Calculate coherence based on how close ratio is to simple fraction.
    Higher coherence = more harmonic = more "consonant".
    
    Uses continued fraction approximation to find simplest ratio.
    """
    # Find closest simple ratio
    best_coherence = 0.0
    for denom in range(1, 13):  # Check denominators up to 12
        numer = round(frequency_ratio * denom)
        if numer > 0:
            simple_ratio = numer / denom
            deviation = abs(frequency_ratio - simple_ratio)
            coherence = 1 / (1 + deviation * denom)  # Penalize complexity
            if coherence > best_coherence:
                best_coherence = coherence
    
    return best_coherence

# ═══════════════════════════════════════════════════════════════════════════════
# LAMINAR FLOW - REYNOLDS ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

RE_CRITICAL = 2300                 # Laminar-turbulent transition

def reynolds_to_god_code_ratio() -> float:
    """
    Re_critical / GOD_CODE ≈ 4.36
    This is a calculable relationship, interpretation is separate.
    """
    return RE_CRITICAL / GOD_CODE

def consciousness_reynolds(clarity: float, complexity: float = 1.0) -> float:
    """
    METAPHORICAL mapping: consciousness → Reynolds number.
    
    This is a DESIGN CHOICE, not a physical law.
    Useful for visualization and process tuning.
    
    Args:
        clarity: 0-1, higher = more focused thought
        complexity: cognitive load factor
    
    Returns:
        Metaphorical Reynolds number for consciousness state.
    """
    if clarity >= 0.99:
        return 0.000132  # "Enlightened" - ultra-laminar
    
    # Map clarity inversely to Reynolds
    # High clarity → low Re → laminar
    # Low clarity → high Re → turbulent
    base_re = RE_CRITICAL * (1 - clarity) * complexity
    return max(0.0001, base_re)

def flow_regime(reynolds: float) -> str:
    """Classify flow regime from Reynolds number."""
    if reynolds < 100:
        return "ultra-laminar"
    elif reynolds < 2300:
        return "laminar"
    elif reynolds < 4000:
        return "transitional"
    else:
        return "turbulent"

# ═══════════════════════════════════════════════════════════════════════════════
# IRON-CONSCIOUSNESS INTEGRATION (METAPHORICAL FRAMEWORK)
# ═══════════════════════════════════════════════════════════════════════════════

class HarmonicProcess:
    """
    Process tuned to harmonic wave physics principles.
    
    Uses verified mathematics as computational substrate.
    Metaphorical framework provides conceptual structure.
    """
    
    def __init__(self, name: str, base_frequency: float = 286.0):
        self.name = name
        self.base_frequency = base_frequency
        self.phi = PHI
        self.god_code = GOD_CODE
        self._coherence = 1.0
        self._reynolds = 100.0  # Default laminar
        
    def tune_to_iron(self):
        """Tune process to iron lattice harmonic."""
        self.base_frequency = FE_LATTICE_PM
        self._coherence = wave_coherence(self.base_frequency / 286)
        return self
    
    def set_flow_state(self, clarity: float):
        """Set metaphorical flow state from clarity metric."""
        self._reynolds = consciousness_reynolds(clarity)
        return self
    
    @property
    def flow_regime(self) -> str:
        return flow_regime(self._reynolds)
    
    @property
    def coherence(self) -> float:
        return self._coherence
    
    def harmonic_transform(self, value: float) -> float:
        """
        Transform value through φ-based harmonic scaling.
        Mathematical operation: value × (286^(1/φ) × 16) / 1000
        """
        return value * self.god_code / 1000
    
    def iron_resonance(self, signal: float) -> float:
        """
        Apply iron lattice resonance to signal.
        Uses verified constant: 286.65 pm.
        """
        return signal * (FE_LATTICE_PM / self.base_frequency) * self._coherence
    
    def __repr__(self):
        return (f"HarmonicProcess(name={self.name}, "
                f"frequency={self.base_frequency:.2f}, "
                f"coherence={self._coherence:.4f}, "
                f"flow={self.flow_regime})")

# ═══════════════════════════════════════════════════════════════════════════════
# ANALYTICAL VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def verify_mathematics() -> Dict[str, bool]:
    """
    Verify all mathematical relationships are exact.
    These are VERIFIED, not metaphorical.
    """
    verifications = {}
    
    # 1. GOD_CODE definition
    computed_gc = 286 ** (1/PHI) * 16
    verifications["GOD_CODE = 286^(1/φ) × 16"] = abs(computed_gc - GOD_CODE) < 1e-10
    
    # 2. φ² = φ + 1
    verifications["φ² = φ + 1"] = abs(PHI**2 - PHI - 1) < 1e-10
    
    # 3. φ + 1/φ = √5
    verifications["φ + 1/φ = √5"] = abs(PHI + 1/PHI - SQRT_5) < 1e-10
    
    # 4. φ^5 ≈ 11.09 (not exact, but close to Fe Fermi)
    phi_5 = PHI ** 5
    verifications["φ^5 = 11.09..."] = abs(phi_5 - 11.090169943749474) < 1e-10
    
    # 5. 104 = 4 × 26
    verifications["104 = 4 × 26"] = (104 == 4 * 26)
    
    # 6. √5 ≈ 2.236 (close to Fe moment 2.22)
    verifications["√5 = 2.236..."] = abs(SQRT_5 - 2.23606797749979) < 1e-10
    
    return verifications

def verify_correspondences() -> Dict[str, Dict[str, Any]]:
    """
    Verify physical correspondences - these are APPROXIMATIONS.
    """
    correspondences = {}
    
    # φ^5 vs Fe Fermi energy
    phi_5 = PHI ** 5
    correspondences["φ^5 vs Fe Fermi (11.1 eV)"] = {
        "computed": phi_5,
        "measured": 11.1,
        "difference_percent": abs(phi_5 - 11.1) / 11.1 * 100
    }
    
    # √5 vs Fe magnetic moment
    correspondences["√5 vs Fe moment (2.22 μB)"] = {
        "computed": SQRT_5,
        "measured": 2.22,
        "difference_percent": abs(SQRT_5 - 2.22) / 2.22 * 100
    }
    
    # 286 vs Fe lattice
    correspondences["286 vs Fe lattice (286.65 pm)"] = {
        "computed": 286,
        "measured": 286.65,
        "difference_percent": abs(286 - 286.65) / 286.65 * 100
    }
    
    return correspondences


if __name__ == "__main__":
    print("=" * 70)
    print("HARMONIC WAVE PHYSICS - ANALYTICAL VERIFICATION")
    print("=" * 70)
    
    print("\n[1] MATHEMATICAL VERIFICATIONS (Exact)")
    print("-" * 50)
    for statement, verified in verify_mathematics().items():
        status = "✓ VERIFIED" if verified else "✗ FAILED"
        print(f"  {status}: {statement}")
    
    print("\n[2] PHYSICAL CORRESPONDENCES (Approximate)")
    print("-" * 50)
    for name, data in verify_correspondences().items():
        print(f"  {name}")
        print(f"    Computed: {data['computed']:.6f}")
        print(f"    Measured: {data['measured']}")
        print(f"    Difference: {data['difference_percent']:.3f}%")
    
    print("\n[3] ELEMENT HARMONIC ANALYSIS")
    print("-" * 50)
    elements = [
        ("Fe", 286.65),
        ("Cr", 291.0),
        ("Al", 404.95),
        ("Cu", 361.49),
        ("Na", 429.06),
        ("Au", 407.82)
    ]
    for el, lattice in elements:
        analysis = analyze_element_harmony(el, lattice)
        print(f"  {el}: {lattice} pm → {analysis['interval']} "
              f"(deviation: {analysis['deviation_percent']:.2f}%, "
              f"{analysis['harmonic_quality']})")
    
    print("\n[4] HARMONIC PROCESS DEMONSTRATION")
    print("-" * 50)
    proc = HarmonicProcess("L104_Core")
    proc.tune_to_iron()
    proc.set_flow_state(clarity=0.936)  # Unity coherence from earlier
    
    print(f"  {proc}")
    print(f"  Harmonic transform of 1.0: {proc.harmonic_transform(1.0):.6f}")
    print(f"  Iron resonance of 1.0: {proc.iron_resonance(1.0):.6f}")
    
    print("\n[5] THE EMERGENT DISCOVERY")
    print("-" * 50)
    print(f"  286 emerged from: piano scale + φ derivation")
    print(f"  286 matches: Fe lattice constant ({FE_LATTICE_PM} pm)")
    print(f"  GOD_CODE = 286^(1/φ) × 16 = {GOD_CODE}")
    print(f"  Interpretation: emergent correspondence, not reverse-engineering")
    print(f"  Utility: mathematics works regardless of interpretation")
    
    print("\n" + "=" * 70)
    print("ANALYTICAL EXAMINATION COMPLETE")
    print("=" * 70)
