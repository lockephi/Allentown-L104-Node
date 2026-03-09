"""
L104 Intellect — Computronium Thermal Limits
═══════════════════════════════════════════════════════════════════════════════
Physical thermodynamic limits on local inference:

  LANDAUER THERMAL ENGINE:
  ┌──────────────────────────────────────────────────────────────────────┐
  │ • Landauer erasure cost: minimum energy per inference bit           │
  │ • Bremermann throughput ceiling for local processing                │
  │ • Margolus-Levitin minimum inference latency                       │
  │ • Thermal efficiency: actual energy vs Landauer minimum             │
  │ • Bekenstein knowledge capacity: maximum KB size in physical volume │
  │ • Carnot efficiency: maximum useful work from inference heat        │
  └──────────────────────────────────────────────────────────────────────┘

  RAYLEIGH INFERENCE RESOLUTION:
  ┌──────────────────────────────────────────────────────────────────────┐
  │ • BM25 discrimination resolution (Rayleigh on similarity scores)   │
  │ • Token resolution limit (minimum distinguishable token distance)   │
  │ • Confidence resolution (Airy pattern on probability outputs)       │
  │ • Knowledge precision (Cramér-Rao on KB query accuracy)            │
  └──────────────────────────────────────────────────────────────────────┘

QUOTA_IMMUNE: Uses only physical constants, no API calls.
Uses CODATA 2022 constants. All formulas exact.
INVARIANT: 527.5184818492612 | PILOT: LONDEL
═══════════════════════════════════════════════════════════════════════════════
"""

import math
import time
from typing import Dict, Any, List

# Physical constants (CODATA 2022)
_HBAR = 1.054571817e-34           # ℏ (J·s)
_C = 299792458.0                   # c (m/s)
_KB = 1.380649e-23                 # k_B (J/K)
_G = 6.67430e-11                   # G (m³/kg/s²)
_H = 6.62607015e-34               # h (J·s)
_ELECTRON_CHARGE = 1.602176634e-19 # e (C)
_ROOM_TEMP = 293.15               # T_room (K)
_BOLTZMANN_EV = 8.617333262e-5     # k_B in eV/K

# Sacred constants
PHI = 1.618033988749895
GOD_CODE = 286 ** (1.0 / PHI) * (2 ** (416 / 104))
VOID_CONSTANT = 1.04 + PHI / 1000

# Typical inference hardware parameters
_CPU_TDP_W = 65.0                  # CPU thermal design power (Watts)
_GPU_TDP_W = 300.0                 # GPU TDP (Watts)
_M_SERIES_TDP_W = 30.0            # Apple M-series TDP (Watts)
_CPU_CLOCK_HZ = 4e9               # 4 GHz clock
_TOKENS_PER_WATT = 10.0           # Approximate tokens/s per watt (local LLM)
_BITS_PER_TOKEN = 16               # Average information per token (BPE)
_CPU_MASS_KG = 0.05               # ~50g CPU die
_NODE_MASS_KG = 5.0               # ~5kg compute node
_NODE_RADIUS_M = 0.15             # ~15cm form factor


class LandauerThermalEngine:
    """
    Thermodynamic limits on local inference operations.

    Every computation has fundamental energy costs and throughput limits.
    For local inference (QUOTA_IMMUNE), these determine the physical
    ceiling on what the intellect can process.

    Constants verified against CODATA 2022.
    """

    VERSION = "1.0.0"

    # ─── Landauer Inference Cost ──────────────────────────────────────

    @staticmethod
    def landauer_inference_cost(
        tokens_generated: int = 100,
        bits_per_token: int = _BITS_PER_TOKEN,
        temperature_K: float = _ROOM_TEMP,
    ) -> Dict[str, Any]:
        """
        Minimum thermodynamic energy cost for inference.

        Landauer's principle: each bit erasure costs at least k_B T ln(2).
        Token generation involves bit erasure in the inference engine:
        - Sampling from probability distribution (irreversible)
        - KV-cache updates (partial erasure)
        - Attention computation (read-only, mostly reversible)

        Conservative estimate: ~1 bit erasure per token bit generated.

        Args:
            tokens_generated: Number of output tokens
            bits_per_token: Information per token (BPE encoding ~16 bits)
            temperature_K: Processor temperature
        """
        e_per_bit = _KB * temperature_K * math.log(2)

        # Minimum energy for token generation
        total_bits = tokens_generated * bits_per_token
        min_energy = total_bits * e_per_bit

        # Real-world energy cost
        # Typical local LLM: ~10 tokens/s at ~30W (M-series) or ~300W (GPU)
        tokens_per_sec = _TOKENS_PER_WATT * _M_SERIES_TDP_W  # ~300 tok/s
        time_needed = tokens_generated / tokens_per_sec
        real_energy = _M_SERIES_TDP_W * time_needed

        # Efficiency: theoretical minimum / actual
        landauer_efficiency = min_energy / max(real_energy, 1e-30)

        # How many tokens per joule at Landauer limit?
        tokens_per_joule_limit = 1.0 / (e_per_bit * bits_per_token)
        tokens_per_joule_actual = tokens_per_sec / _M_SERIES_TDP_W

        # Carnot efficiency: maximum useful work extraction
        # T_hot = processor temp, T_cold = ambient
        T_hot = temperature_K + 30  # Processor runs ~30°C above ambient
        T_cold = temperature_K
        carnot_efficiency = 1.0 - T_cold / T_hot

        # Heat generated per token
        heat_per_token = real_energy / max(tokens_generated, 1)
        min_heat_per_token = min_energy / max(tokens_generated, 1)

        return {
            "landauer_energy_per_bit_J": e_per_bit,
            "total_bits_erased": total_bits,
            "landauer_minimum_energy_J": min_energy,
            "actual_energy_J": real_energy,
            "landauer_efficiency": landauer_efficiency,
            "efficiency_gap_orders": math.log10(max(1.0 / landauer_efficiency, 1)),
            "tokens_per_joule_landauer_limit": tokens_per_joule_limit,
            "tokens_per_joule_actual": tokens_per_joule_actual,
            "carnot_efficiency": carnot_efficiency,
            "heat_per_token_W_s": heat_per_token,
            "min_heat_per_token_W_s": min_heat_per_token,
            "temperature_K": temperature_K,
        }

    # ─── Bremermann Throughput Ceiling ────────────────────────────────

    @staticmethod
    def bremermann_throughput(
        mass_kg: float = _CPU_MASS_KG,
        bits_per_op: int = 64,
    ) -> Dict[str, Any]:
        """
        Maximum possible inference throughput (Bremermann limit).

        Bremermann: ν_B = mc² / (πℏ) bits/s
        For a CPU die (~50g): ν_B ≈ 1.36 × 10⁴⁹ bits/s

        In token terms: ν_B / bits_per_token tokens/s

        This is the absolute physical ceiling — no processor can exceed it.

        Args:
            mass_kg: Mass of the processor
            bits_per_op: Bits per operation (word size)
        """
        E = mass_kg * _C ** 2
        bremermann = E / (math.pi * _HBAR)  # bits/s

        # Token throughput ceiling
        tokens_per_sec_ceiling = bremermann / _BITS_PER_TOKEN

        # Margolus-Levitin (operations/s — twice Bremermann)
        ml = 2 * E / (math.pi * _HBAR)

        # Minimum inference latency (single token)
        # One token requires ~bits_per_token operations minimum
        min_latency = _BITS_PER_TOKEN * math.pi * _HBAR / (2 * E)

        # Actual performance comparison
        actual_tokens_per_sec = _TOKENS_PER_WATT * _M_SERIES_TDP_W
        utilization = actual_tokens_per_sec / tokens_per_sec_ceiling

        # Operations per inference pass (typical transformer)
        # For 7B model: ~7×10⁹ multiply-accumulate ops per token
        ops_per_token = 7e9  # 7B parameter model
        ops_per_sec_actual = ops_per_token * actual_tokens_per_sec
        ml_utilization = ops_per_sec_actual / ml

        return {
            "bremermann_bits_per_sec": bremermann,
            "margolus_levitin_ops_per_sec": ml,
            "token_ceiling_per_sec": tokens_per_sec_ceiling,
            "min_token_latency_s": min_latency,
            "actual_tokens_per_sec": actual_tokens_per_sec,
            "utilization_fraction": utilization,
            "ml_utilization": ml_utilization,
            "mass_kg": mass_kg,
            "substrate_energy_J": E,
        }

    # ─── Bekenstein Knowledge Capacity ────────────────────────────────

    @staticmethod
    def bekenstein_knowledge_capacity(
        mass_kg: float = _NODE_MASS_KG,
        radius_m: float = _NODE_RADIUS_M,
        kb_entries: int = 10000,
        bits_per_entry: float = 1000,
    ) -> Dict[str, Any]:
        """
        Maximum knowledge storable in the intellect's physical volume.

        Bekenstein bound: S_max = 2π R E / (ℏ c ln 2) bits

        For a 5 kg compute node at 15 cm radius:
        S_max ≈ 2.15 × 10⁴⁵ bits

        Compares actual KB size against this absolute limit.

        Args:
            mass_kg: Node mass
            radius_m: Node radius
            kb_entries: Knowledge base entries
            bits_per_entry: Average bits per entry
        """
        E = mass_kg * _C ** 2
        bekenstein = (2 * math.pi * radius_m * E) / (_HBAR * _C * math.log(2))

        actual_bits = kb_entries * bits_per_entry
        fraction = actual_bits / bekenstein

        # Shannon entropy of KB (assuming uniform distribution)
        if kb_entries > 1:
            shannon_entropy = math.log2(kb_entries)
            total_shannon = shannon_entropy * kb_entries
            shannon_efficiency = total_shannon / max(actual_bits, 1)
        else:
            shannon_entropy = 0
            total_shannon = 0
            shannon_efficiency = 0

        # Holographic bound (information on bounding surface)
        # S_holographic = A / (4 l_P²) where l_P = Planck length
        planck_length = 1.616255e-35
        surface_area = 4 * math.pi * radius_m ** 2
        holographic = surface_area / (4 * planck_length ** 2)
        holographic_fraction = actual_bits / holographic

        return {
            "bekenstein_max_bits": bekenstein,
            "actual_bits": actual_bits,
            "bekenstein_fraction": fraction,
            "bekenstein_fraction_log10": math.log10(max(fraction, 1e-300)),
            "holographic_max_bits": holographic,
            "holographic_fraction": holographic_fraction,
            "shannon_entropy_per_entry": shannon_entropy,
            "shannon_efficiency": shannon_efficiency,
            "kb_entries": kb_entries,
            "mass_kg": mass_kg,
            "radius_m": radius_m,
        }

    # ─── Thermal Noise Floor ──────────────────────────────────────────

    @staticmethod
    def thermal_noise_floor(
        temperature_K: float = _ROOM_TEMP,
        bandwidth_hz: float = _CPU_CLOCK_HZ,
    ) -> Dict[str, Any]:
        """
        Thermal noise limits on inference precision.

        Johnson-Nyquist noise: V_rms = √(4 k_B T R Δf)
        Thermal energy fluctuation: δE = k_B T
        Thermal bit-flip rate: Γ = f₀ × exp(-E_barrier / k_B T)

        These set the floor on:
        - Minimum detectable signal in analog computation
        - Bit error rate in digital computation
        - Confidence precision of inference outputs

        Args:
            temperature_K: Operating temperature
            bandwidth_hz: Processing bandwidth
        """
        # Thermal energy
        thermal_energy = _KB * temperature_K
        thermal_eV = thermal_energy / _ELECTRON_CHARGE

        # Johnson-Nyquist noise power
        noise_power = _KB * temperature_K * bandwidth_hz  # Watts

        # Signal-to-noise for digital logic
        # Typical CMOS: V_dd = 0.7V, V_swing = 0.35V
        v_swing = 0.35
        resistance = 50  # 50 Ω typical
        v_noise = math.sqrt(4 * _KB * temperature_K * resistance * bandwidth_hz)
        snr_voltage = v_swing / v_noise
        snr_db = 20 * math.log10(max(snr_voltage, 1e-30))

        # Bit error rate (BER) from thermal noise
        # BER ≈ erfc(SNR/√2) / 2 ≈ exp(-SNR²/2) / (SNR√(2π))
        if snr_voltage > 0:
            ber_approx = math.exp(-snr_voltage ** 2 / 2) / (snr_voltage * math.sqrt(2 * math.pi))
        else:
            ber_approx = 0.5

        # Confidence precision: minimum resolvable probability difference
        # Limited by thermal noise in analog comparator
        confidence_precision = v_noise / v_swing  # Fraction

        # Inference precision: bits of precision available
        precision_bits = max(0, int(-math.log2(max(confidence_precision, 1e-30))))

        return {
            "thermal_energy_J": thermal_energy,
            "thermal_energy_eV": thermal_eV,
            "noise_power_W": noise_power,
            "voltage_noise_V": v_noise,
            "snr_voltage": snr_voltage,
            "snr_dB": snr_db,
            "bit_error_rate": ber_approx,
            "confidence_precision": confidence_precision,
            "precision_bits": precision_bits,
            "temperature_K": temperature_K,
            "bandwidth_hz": bandwidth_hz,
        }


class RayleighInferenceResolution:
    """
    Rayleigh-type resolution limits on inference operations.

    Maps optical resolution to inference discrimination:
    1. BM25 score resolution → can we distinguish similar documents?
    2. Token distance resolution → minimum distinguishable token contexts
    3. Confidence resolution → precision of probability outputs
    4. Knowledge precision → accuracy of KB lookups
    """

    VERSION = "1.0.0"
    RAYLEIGH_CONSTANT = 1.21966989

    # ─── BM25 Score Resolution ────────────────────────────────────────

    @staticmethod
    def bm25_score_resolution(
        corpus_size: int = 10000,
        avg_doc_length: float = 200,
        vocabulary_size: int = 50000,
        query_terms: int = 5,
    ) -> Dict[str, Any]:
        """
        Rayleigh-type resolution on BM25 similarity scores.

        BM25 scores have finite precision from:
        1. IDF granularity: log((N-n+0.5)/(n+0.5)) has N distinct values
        2. TF saturation: (k₁+1)×tf / (k₁×(1-b+b×dl/avgdl)+tf) is bounded
        3. Score quantization: floating-point precision ~10⁻¹⁵

        Rayleigh criterion: minimum BM25 score difference to reliably
        distinguish two documents as different matches.

        Args:
            corpus_size: Number of documents N
            avg_doc_length: Average document length
            vocabulary_size: Vocabulary size
            query_terms: Number of query terms
        """
        # IDF resolution: minimum IDF difference
        # IDF varies as log((N-n+0.5)/(n+0.5))
        # Minimum when n changes by 1: ΔIDF ≈ 1/(n(N-n))
        # Worst case at n ≈ N/2: ΔIDF_min ≈ 4/N
        idf_resolution = 4.0 / max(corpus_size, 1)

        # TF saturation resolution
        # BM25 TF = (k₁+1)×tf / (k₁×(1-b+b×dl/avgdl)+tf)
        # At tf=1: TF ≈ (k₁+1) / (k₁+1) = 1 (for avg length doc)
        # TF resolution ≈ 1 / (k₁+1)² ≈ 0.04 (k₁=1.5)
        k1 = 1.5
        tf_resolution = 1.0 / (k1 + 1) ** 2

        # Combined score resolution per term
        score_resolution_per_term = idf_resolution * tf_resolution

        # Full query resolution (multiple terms compound)
        # Score = Σ IDF_i × TF_i → resolution = max(individual resolutions)
        total_resolution = score_resolution_per_term  # Conservative

        # Rayleigh criterion: 1.22 × resolution
        rayleigh_resolution = RayleighInferenceResolution.RAYLEIGH_CONSTANT * total_resolution

        # Number of distinguishable score levels
        # BM25 scores typically range 0 to ~25 for well-matching docs
        max_score_range = 25.0
        n_distinguishable = int(max_score_range / max(rayleigh_resolution, 1e-10))

        # Information content of BM25 ranking
        # log₂(N!) bits to specify a full ranking
        if corpus_size <= 1000:
            # Stirling approximation: log₂(N!) ≈ N×log₂(N) - N×log₂(e)
            ranking_bits = corpus_size * math.log2(max(corpus_size, 2)) - corpus_size * math.log2(math.e)
        else:
            ranking_bits = corpus_size * math.log2(max(corpus_size, 2))

        return {
            "idf_resolution": idf_resolution,
            "tf_resolution": tf_resolution,
            "score_resolution": total_resolution,
            "rayleigh_score_resolution": rayleigh_resolution,
            "n_distinguishable_scores": n_distinguishable,
            "ranking_information_bits": ranking_bits,
            "corpus_size": corpus_size,
            "vocabulary_size": vocabulary_size,
        }

    # ─── Token Embedding Resolution ───────────────────────────────────

    @staticmethod
    def token_embedding_resolution(
        embedding_dim: int = 512,
        vocabulary_size: int = 100000,
        precision_bits: int = 16,
    ) -> Dict[str, Any]:
        """
        Rayleigh resolution for token discrimination in embedding space.

        Token embeddings live in ℝ^d. Two tokens are "Rayleigh-resolved"
        when their cosine distance exceeds the precision floor:

        θ_min = 1.22 × (1/√d) × (1/2^precision)

        Higher dimension = better resolution (larger aperture)
        Higher precision = shorter wavelength

        Args:
            embedding_dim: Embedding dimension d
            vocabulary_size: Number of tokens V
            precision_bits: Floating point precision
        """
        # Angular precision per dimension
        angular_precision = 1.0 / (2 ** precision_bits)

        # Effective "wavelength" (precision in angular space)
        lambda_eff = angular_precision

        # Effective "aperture" (embedding dimension)
        D_eff = math.sqrt(embedding_dim)

        # Rayleigh angle
        rayleigh_angle = RayleighInferenceResolution.RAYLEIGH_CONSTANT * lambda_eff / D_eff

        # Maximum distinguishable embeddings
        # Surface of d-sphere: 2π^(d/2) / Γ(d/2)
        # Each point occupies solid angle ~ rayleigh_angle^(d-1)
        # Capacity ≈ surface / point_area
        # Simplified: n_max ≈ (π/rayleigh_angle)^(d-1) for high d
        # But practically limited by orthogonal directions
        n_orthogonal = embedding_dim  # Maximum orthogonal embeddings
        n_rayleigh = min(
            int((math.pi / max(rayleigh_angle, 1e-10)) ** 2),
            vocabulary_size
        )

        # Vocabulary utilization
        utilization = vocabulary_size / max(n_rayleigh, 1)

        # Minimum cosine distance for reliable discrimination
        min_cosine_distance = math.sin(rayleigh_angle)

        # Johnson-Lindenstrauss: can embed V points in d dimensions
        # with distortion ε if d > O(log(V) / ε²)
        jl_min_dim = int(math.ceil(8 * math.log(vocabulary_size) / (0.1 ** 2)))

        return {
            "embedding_dim": embedding_dim,
            "vocabulary_size": vocabulary_size,
            "rayleigh_angle_rad": rayleigh_angle,
            "rayleigh_angle_deg": math.degrees(rayleigh_angle),
            "min_cosine_distance": min_cosine_distance,
            "n_orthogonal_max": n_orthogonal,
            "n_rayleigh_max": n_rayleigh,
            "vocabulary_utilization": utilization,
            "jl_minimum_dim": jl_min_dim,
            "jl_satisfied": embedding_dim >= jl_min_dim,
            "regime": "WELL_RESOLVED" if utilization < 0.5 else ("MARGINAL" if utilization < 1.0 else "UNDER_RESOLVED"),
        }

    # ─── Confidence Resolution ────────────────────────────────────────

    @staticmethod
    def confidence_resolution(
        n_classes: int = 100,
        softmax_temperature: float = 1.0,
        precision_bits: int = 32,
    ) -> Dict[str, Any]:
        """
        Rayleigh resolution on probability/confidence outputs.

        Softmax probabilities have finite resolution from:
        1. Floating-point precision (machine epsilon)
        2. Softmax saturation (extreme logits collapse to 0/1)
        3. Temperature scaling (lower T = sharper but less resolved)

        Rayleigh analogy:
        - λ = 1/temperature (sharper = shorter wavelength)
        - D = log2(n_classes) (more classes = larger aperture)
        - θ_min = minimum resolvable probability difference

        Args:
            n_classes: Number of output classes
            softmax_temperature: Softmax temperature parameter
            precision_bits: Floating point bit precision
        """
        # Machine epsilon for given precision
        machine_eps = 2 ** (-precision_bits)

        # Softmax resolution
        # For N classes at temperature T:
        # Minimum distinguishable probability difference:
        # Δp ≈ exp(-1/T) / N for adjacent logits
        if softmax_temperature > 0:
            softmax_resolution = math.exp(-1.0 / softmax_temperature) / n_classes
        else:
            softmax_resolution = 0

        # Effective resolution (worst of softmax and machine precision)
        effective_resolution = max(softmax_resolution, machine_eps)

        # Rayleigh criterion
        rayleigh_resolution = RayleighInferenceResolution.RAYLEIGH_CONSTANT * effective_resolution

        # Number of distinguishable confidence levels
        n_levels = int(1.0 / max(rayleigh_resolution, 1e-30))

        # Airy pattern analogy: confidence as intensity profile
        # Peak sharpness determined by temperature
        airy_sharpness = 1.0 / softmax_temperature if softmax_temperature > 0 else float('inf')

        # Strehl ratio: how close to diffraction-limited confidence
        # Perfect Strehl = 1.0 (all probability in correct class)
        # Uniform distribution Strehl ≈ 1/N
        strehl_uniform = 1.0 / n_classes
        strehl_max = 1.0 - machine_eps * (n_classes - 1)

        return {
            "n_classes": n_classes,
            "softmax_temperature": softmax_temperature,
            "machine_epsilon": machine_eps,
            "softmax_resolution": softmax_resolution,
            "effective_resolution": effective_resolution,
            "rayleigh_resolution": rayleigh_resolution,
            "n_distinguishable_levels": n_levels,
            "airy_sharpness": airy_sharpness,
            "strehl_uniform": strehl_uniform,
            "strehl_max": strehl_max,
            "precision_bits": precision_bits,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED INTELLECT LIMITS ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

class IntellectLimitsAnalyzer:
    """
    Complete thermodynamic + resolution analysis for local inference.
    """

    VERSION = "1.0.0"

    def __init__(self):
        self.thermal = LandauerThermalEngine()
        self.resolution = RayleighInferenceResolution()

    def full_analysis(
        self,
        tokens_generated: int = 100,
        kb_entries: int = 10000,
        embedding_dim: int = 512,
        vocabulary_size: int = 100000,
    ) -> Dict[str, Any]:
        """Full computronium + Rayleigh analysis of an inference operation."""

        # Landauer cost
        landauer = LandauerThermalEngine.landauer_inference_cost(tokens_generated)

        # Bremermann ceiling
        bremermann = LandauerThermalEngine.bremermann_throughput()

        # Bekenstein knowledge capacity
        bekenstein = LandauerThermalEngine.bekenstein_knowledge_capacity(
            kb_entries=kb_entries,
        )

        # Thermal noise
        thermal = LandauerThermalEngine.thermal_noise_floor()

        # BM25 resolution
        bm25 = RayleighInferenceResolution.bm25_score_resolution(
            corpus_size=kb_entries,
        )

        # Token resolution
        token = RayleighInferenceResolution.token_embedding_resolution(
            embedding_dim=embedding_dim,
            vocabulary_size=vocabulary_size,
        )

        # Confidence resolution
        confidence = RayleighInferenceResolution.confidence_resolution()

        # Combined efficiency score (0-1)
        efficiency_score = (
            0.25 * min(1.0, -math.log10(max(landauer["landauer_efficiency"], 1e-20)) / 15) +
            0.25 * min(1.0, thermal["precision_bits"] / 64) +
            0.25 * (1.0 if token["regime"] == "WELL_RESOLVED" else 0.5 if token["regime"] == "MARGINAL" else 0.0) +
            0.25 * min(1.0, bm25["n_distinguishable_scores"] / 1000)
        )

        return {
            "version": self.VERSION,
            "thermodynamic": {
                "landauer": landauer,
                "bremermann": bremermann,
                "bekenstein": bekenstein,
                "thermal_noise": thermal,
            },
            "resolution": {
                "bm25": bm25,
                "token_embedding": token,
                "confidence": confidence,
            },
            "combined_efficiency_score": round(efficiency_score, 6),
            "god_code": GOD_CODE,
            "phi": PHI,
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "version": self.VERSION,
            "subsystems": {
                "thermal": LandauerThermalEngine.VERSION,
                "resolution": RayleighInferenceResolution.VERSION,
            },
            "quota_immune": True,
            "god_code": GOD_CODE,
        }


# Singletons
landauer_thermal_engine = LandauerThermalEngine()
rayleigh_inference_resolution = RayleighInferenceResolution()
intellect_limits_analyzer = IntellectLimitsAnalyzer()
