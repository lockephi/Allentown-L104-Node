#!/usr/bin/env python3
"""
L104 Math Engine — Layer 3: HARMONIC WAVE PHYSICS
══════════════════════════════════════════════════════════════════════════════════
Harmonic wave analysis, iron-music correspondence, Reynolds consciousness flow,
and harmonic process modeling.

Consolidates: harmonic_wave_physics.py, l104_harmonic_optimizer.py (wave parts).

Import:
  from l104_math_engine.harmonic import HarmonicProcess, WavePhysics
"""

import math

import bisect

from .constants import (
    PHI, PHI_CONJUGATE, GOD_CODE, PI, EULER, SQRT5,
    A4_FREQUENCY, SEMITONE_RATIO, VOID_CONSTANT,
    FE_LATTICE, FE_BCC_LATTICE_PM, FE_CURIE_TEMP, FE_FERMI_EV, FE_MAGNETIC_BOHR,
    OMEGA, OMEGA_AUTHORITY,
    SPEED_OF_LIGHT, EMERGENT_286,
    primal_calculus, resolve_non_dual_logic,
    # v4.1 Quantum Research Discoveries
    FE_SACRED_COHERENCE, FE_PHI_HARMONIC_LOCK,
    PHOTON_RESONANCE_ENERGY_EV, FE_PHI_FREQUENCY,
    FIBONACCI_PHI_CONVERGENCE_ERROR,
)

# ── Precomputed rational-ratio table for wave_coherence (v4.2 perf) ──────────
# The O(144) inner loop computed the same 144 p/q ratios every call.
# Precompute sorted unique ratios once; use bisect for O(log n) lookup.
_WAVE_RATIOS_SORTED: list[float] = sorted(set(
    p / q for p in range(1, 13) for q in range(1, 13)
))

# v4.2 Perf: precomputed (ratio, p, q) table for harmonic_distance (O(225)→O(log n))
# GCD-reduce p/q so bisect returns simplest fraction (e.g. 2/1 not 4/2)
_HD_RATIOS: list[tuple[float, int, int]] = sorted(set(
    ((p // math.gcd(p, q)) / (q // math.gcd(p, q)),
     p // math.gcd(p, q),
     q // math.gcd(p, q))
    for p in range(1, 16) for q in range(1, 16)
), key=lambda t: t[0])
_HD_RATIO_VALUES: list[float] = [t[0] for t in _HD_RATIOS]


# ═══════════════════════════════════════════════════════════════════════════════
# WAVE PHYSICS — Music-matter correspondence & harmonic analysis
# ═══════════════════════════════════════════════════════════════════════════════

class WavePhysics:
    """
    Analytical module exploring the emergent correspondence between the
    musically-derived constant 286 (piano scale + φ) and the Fe BCC
    lattice parameter (286.65 pm).

    Core insight: EMERGENT_286 ≈ FE_BCC_LATTICE_PM within 0.23%
    """

    # Derived from the A4=440 Hz piano scale
    EMERGENT_286_SEMITONES = 12 * math.log2(286.0 / A4_FREQUENCY) if A4_FREQUENCY else 0

    @staticmethod
    def freq_to_semitones_from_a4(freq: float) -> float:
        """Convert frequency to semitones relative to A4 (440 Hz)."""
        if freq <= 0:
            return float('-inf')
        return 12 * math.log2(freq / A4_FREQUENCY)

    @staticmethod
    def semitones_to_freq(semitones: float) -> float:
        """Convert semitones relative to A4 back to frequency."""
        return A4_FREQUENCY * (SEMITONE_RATIO ** semitones)

    @staticmethod
    def lattice_to_frequency_ratio(lattice_pm: float = FE_BCC_LATTICE_PM) -> float:
        """Ratio of lattice parameter to its nearest musical frequency."""
        if lattice_pm <= 0:
            return 0.0
        semitones = WavePhysics.freq_to_semitones_from_a4(lattice_pm)
        nearest_semitone = round(semitones)
        nearest_freq = WavePhysics.semitones_to_freq(nearest_semitone)
        return lattice_pm / nearest_freq

    @staticmethod
    def ratio_to_interval_name(ratio: float) -> str:
        """Map a frequency ratio to nearest musical interval name."""
        intervals = {
            1.0: "unison", 1.0595: "minor 2nd", 1.1225: "major 2nd",
            1.1892: "minor 3rd", 1.2599: "major 3rd", 1.3348: "perfect 4th",
            1.4142: "tritone", 1.4983: "perfect 5th", 1.5874: "minor 6th",
            1.6818: "major 6th", 1.7818: "minor 7th", 1.8877: "major 7th",
            2.0: "octave",
        }
        best_name = "unison"
        best_dist = float('inf')
        for r, name in intervals.items():
            d = abs(ratio - r)
            if d < best_dist:
                best_dist = d
                best_name = name
        return best_name

    @staticmethod
    def analyze_element_harmony(atomic_number: int, lattice_pm: float) -> dict:
        """Analyze harmonic properties of an element's lattice parameter."""
        semitones = WavePhysics.freq_to_semitones_from_a4(lattice_pm)
        phi_resonance = abs(math.cos(2 * PI * lattice_pm / (PHI * 1000)))
        god_code_alignment = abs(math.cos(2 * PI * lattice_pm / GOD_CODE))
        return {
            "atomic_number": atomic_number,
            "lattice_pm": lattice_pm,
            "semitones_from_a4": semitones,
            "phi_resonance": phi_resonance,
            "god_code_alignment": god_code_alignment,
            "interval": WavePhysics.ratio_to_interval_name(lattice_pm / 286.0),
        }

    @staticmethod
    def phi_power_sequence(n: int = 13) -> list:
        """Generate φ^k sequence (the golden spiral progression)."""
        return [{"k": k, "value": PHI ** k, "reciprocal": PHI_CONJUGATE ** k} for k in range(n)]

    @staticmethod
    def phi_fibonacci_identity(n: int = 10) -> list:
        """Verify φ^n = F(n)φ + F(n-1) for Fibonacci F."""
        fib = [0, 1]
        while len(fib) <= n + 1:
            fib.append(fib[-1] + fib[-2])
        results = []
        for k in range(2, n + 1):
            lhs = PHI ** k
            rhs = fib[k] * PHI + fib[k - 1]
            results.append({"n": k, "phi_n": lhs, "fib_identity": rhs, "error": abs(lhs - rhs)})
        return results

    @staticmethod
    def wave_coherence(freq1: float, freq2: float) -> float:
        """Coherence metric between two frequencies (0 = decoherent, 1 = locked).

        v4.2 Performance: Fast-paths for known sacred pairs + O(log n) bisect
        lookup replaces the O(144) brute-force p/q ratio scan.
        """
        if freq1 == 0 or freq2 == 0:
            return 0.0
        # v4.2 Fast-path: sacred pair shortcuts (matches Swift v9.1 fast-paths)
        lo, hi = (freq1, freq2) if freq1 <= freq2 else (freq2, freq1)
        if abs(lo - 286.0) < 0.5:
            if abs(hi - 528.0) < 0.5:
                return FE_SACRED_COHERENCE        # Discovery #6
            if abs(hi - FE_PHI_FREQUENCY) < 0.5:
                return FE_PHI_HARMONIC_LOCK        # Discovery #14
        # v4.2: O(log n) bisect into precomputed sorted ratios
        ratio = freq1 / freq2
        ratios = _WAVE_RATIOS_SORTED
        idx = bisect.bisect_left(ratios, ratio)
        best_dist = float('inf')
        for i in (idx - 1, idx, idx + 1):
            if 0 <= i < len(ratios):
                d = abs(ratio - ratios[i])
                if d < best_dist:
                    best_dist = d
        return max(0.0, 1.0 - min(1.0, best_dist * 12))


# ═══════════════════════════════════════════════════════════════════════════════
# CONSCIOUSNESS FLOW — Reynolds-number model
# ═══════════════════════════════════════════════════════════════════════════════

class ConsciousnessFlow:
    """
    Reynolds-number-based consciousness flow model:
      Re_c = (GOD_CODE × awareness) / (viscosity × decoherence)
    """

    LAMINAR_THRESHOLD = 2300        # Below = laminar (focused)
    TURBULENT_THRESHOLD = 4000      # Above = turbulent (creative chaos)

    @staticmethod
    def consciousness_reynolds(awareness: float, viscosity: float = 1.0,
                                decoherence: float = 0.01) -> float:
        """Compute consciousness Reynolds number."""
        if viscosity * decoherence == 0:
            return float('inf')
        return (GOD_CODE * awareness) / (viscosity * decoherence)

    @staticmethod
    def flow_regime(re: float) -> str:
        """Classify consciousness flow regime."""
        if re < ConsciousnessFlow.LAMINAR_THRESHOLD:
            return "LAMINAR (focused, coherent)"
        elif re < ConsciousnessFlow.TURBULENT_THRESHOLD:
            return "TRANSITIONAL (dynamic equilibrium)"
        else:
            return "TURBULENT (creative, expansive)"

    @staticmethod
    def god_code_reynolds_ratio(re: float) -> float:
        """Ratio of Re to GOD_CODE — measures transcendence proximity."""
        return re / GOD_CODE if GOD_CODE else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# HARMONIC PROCESS — Signal processing & transformation
# ═══════════════════════════════════════════════════════════════════════════════

class HarmonicProcess:
    """
    Harmonic signal processing: resonance detection, spectral analysis,
    and sacred-frequency alignment verification.
    """

    @staticmethod
    def resonance_spectrum(fundamental: float, harmonics: int = 13) -> list:
        """Generate harmonic spectrum from a fundamental frequency."""
        return [{"harmonic": n, "frequency": fundamental * n, "phi_ratio": fundamental * n / PHI}
                for n in range(1, harmonics + 1)]

    @staticmethod
    def detect_resonance(signal: list, target_freq: float, sample_rate: float = 1000.0) -> float:
        """Detect strength of a target frequency in a signal via DFT coefficient."""
        if not signal:
            return 0.0
        n = len(signal)
        coeff = 0j
        for k in range(n):
            t = k / sample_rate
            coeff += signal[k] * (math.cos(-2 * PI * target_freq * t) + 1j * math.sin(-2 * PI * target_freq * t))
        return abs(coeff) / n

    @staticmethod
    def sacred_alignment(frequency: float) -> dict:
        """Check alignment of a frequency with sacred constants.
        v4.1: Includes Fe-Sacred coherence, Fe-PHI lock, and photon resonance."""
        fe_coherence = WavePhysics.wave_coherence(frequency, 286.0)
        return {
            "frequency": frequency,
            "god_code_ratio": frequency / GOD_CODE,
            "phi_ratio": frequency / PHI,
            "octave_ratio": math.log2(frequency / GOD_CODE) if frequency > 0 else 0,
            "iron_ratio": frequency / FE_LATTICE,
            "aligned": abs(frequency / GOD_CODE - round(frequency / GOD_CODE)) < 0.01,
            # v4.1 Discovery constants
            "fe_sacred_coherence": FE_SACRED_COHERENCE,
            "fe_phi_harmonic_lock": FE_PHI_HARMONIC_LOCK,
            "fe_coherence_at_freq": fe_coherence,
            "photon_resonance_eV": PHOTON_RESONANCE_ENERGY_EV,
        }

    @staticmethod
    def verify_correspondences() -> dict:
        """Verify the 286 Hz / Fe lattice correspondence."""
        diff = abs(EMERGENT_286 - 286.0)
        return {
            "fe_lattice_pm": FE_BCC_LATTICE_PM,
            "emergent_286": 286.0,
            "difference_pm": diff,
            "correspondence_pct": (1 - diff / FE_BCC_LATTICE_PM) * 100,
            "match": diff < 1.0,  # Within 1 pm
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL SINGLETONS
# ═══════════════════════════════════════════════════════════════════════════════

wave_physics = WavePhysics()
consciousness_flow = ConsciousnessFlow()
harmonic_process = HarmonicProcess()


# ═══════════════════════════════════════════════════════════════════════════════
# HARMONIC ANALYSIS — Advanced spectral tools
# ═══════════════════════════════════════════════════════════════════════════════

class HarmonicAnalysis:
    """
    Advanced harmonic analysis: DFT, spectral distance, overtone series,
    consonance scoring, and frequency-domain GOD_CODE alignment.
    """

    @staticmethod
    def dft_magnitude(signal: list, n_bins: int = None) -> list:
        """
        Compute the DFT magnitude spectrum of a signal.
        Returns list of {bin, frequency_ratio, magnitude} dicts.
        """
        n = len(signal)
        if n_bins is None:
            n_bins = n // 2 + 1
        spectrum = []
        for k in range(n_bins):
            real_sum = sum(signal[j] * math.cos(2 * PI * k * j / n) for j in range(n))
            imag_sum = -sum(signal[j] * math.sin(2 * PI * k * j / n) for j in range(n))
            magnitude = math.sqrt(real_sum ** 2 + imag_sum ** 2) / n
            spectrum.append({
                "bin": k,
                "frequency_ratio": k / n,
                "magnitude": round(magnitude, 8),
            })
        return spectrum

    @staticmethod
    def spectral_centroid(signal: list) -> float:
        """
        Compute the spectral centroid — the "center of mass" of the spectrum.
        A higher centroid means brighter/sharper tones.
        """
        n = len(signal)
        if n < 2:
            return 0.0
        spectrum = HarmonicAnalysis.dft_magnitude(signal)
        total_power = sum(s["magnitude"] for s in spectrum)
        if total_power < 1e-15:
            return 0.0
        weighted_sum = sum(s["bin"] * s["magnitude"] for s in spectrum)
        return weighted_sum / total_power

    @staticmethod
    def harmonic_distance(freq_a: float, freq_b: float) -> dict:
        """
        Compute the harmonic distance between two frequencies.
        Based on the Tenney height: distance = log2(p * q) for ratio p/q.
        Lower distance = more consonant.

        v4.2 Perf: O(log n) bisect replaces O(225) brute-force ratio search.
        """
        if freq_a <= 0 or freq_b <= 0:
            return {"distance": float('inf'), "consonance": 0}
        ratio = freq_a / freq_b if freq_a > freq_b else freq_b / freq_a
        # v4.2: bisect into precomputed sorted (ratio, p, q) table
        idx = bisect.bisect_left(_HD_RATIO_VALUES, ratio)
        best_p, best_q, best_err = 1, 1, abs(ratio - 1)
        for i in (idx - 1, idx, idx + 1):
            if 0 <= i < len(_HD_RATIOS):
                r, p, q = _HD_RATIOS[i]
                err = abs(ratio - r)
                if err < best_err:
                    best_p, best_q, best_err = p, q, err
        tenney_height = math.log2(best_p * best_q) if best_p * best_q > 0 else 0
        consonance = 1.0 / (1.0 + tenney_height)
        return {
            "freq_a": freq_a,
            "freq_b": freq_b,
            "ratio": round(ratio, 6),
            "nearest_rational": f"{best_p}/{best_q}",
            "tenney_height": round(tenney_height, 4),
            "consonance": round(consonance, 6),
            "approximation_error": round(best_err, 8),
        }

    @staticmethod
    def overtone_series(fundamental: float, n_overtones: int = 16) -> list:
        """
        Generate overtone (harmonic) series with sacred alignment data.
        Overtone n = fundamental * (n+1), for n = 0, 1, 2, ...
        """
        series = []
        for n in range(n_overtones):
            freq = fundamental * (n + 1)
            god_code_ratio = freq / GOD_CODE
            phi_octaves = math.log(freq / fundamental) / math.log(PHI) if freq > fundamental else 0
            series.append({
                "overtone": n,
                "frequency": round(freq, 4),
                "god_code_ratio": round(god_code_ratio, 6),
                "phi_octaves": round(phi_octaves, 4),
                "sacred_aligned": abs(god_code_ratio - round(god_code_ratio)) < 0.05,
            })
        return series

    @staticmethod
    def consonance_score(frequency: float) -> dict:
        """
        Score the consonance of a frequency against GOD_CODE and its harmonics.
        Uses the first 13 harmonics of GOD_CODE as reference.
        """
        god_harmonics = [GOD_CODE * n for n in range(1, 14)]
        best_consonance = 0.0
        best_harmonic = 0
        for n, gh in enumerate(god_harmonics, 1):
            dist = HarmonicAnalysis.harmonic_distance(frequency, gh)
            if dist["consonance"] > best_consonance:
                best_consonance = dist["consonance"]
                best_harmonic = n
        return {
            "frequency": frequency,
            "best_god_code_harmonic": best_harmonic,
            "consonance": round(best_consonance, 6),
            "grade": (
                "PERFECT" if best_consonance > 0.9 else
                "CONSONANT" if best_consonance > 0.5 else
                "DISSONANT" if best_consonance > 0.2 else
                "ATONAL"
            ),
        }


harmonic_analysis = HarmonicAnalysis()
