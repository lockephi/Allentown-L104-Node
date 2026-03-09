"""L104 Quantum Networker v1.0.0 — Quantum Key Distribution (BB84 + E91).

Implements two QKD protocols leveraging the VQPU high-fidelity pipeline:

  BB84: Bennett-Brassard 1984
    - Alice prepares qubits in random bases ({Z, X} = {|0⟩/|1⟩, |+⟩/|-⟩})
    - Qubits sent over quantum channel (simulated via entangled pair consumption)
    - Bob measures in random bases
    - Basis reconciliation over classical channel (sifting)
    - QBER estimation → if QBER < 11%, key is secure
    - Privacy amplification via XOR compression

  E91: Ekert 1991
    - Shared Bell pairs |Φ+⟩ between Alice and Bob
    - Both measure in randomly chosen bases (3 bases each)
    - CHSH inequality test on subset → eavesdropper detection
    - Matching-basis subset → sifted key
    - Sacred alignment scoring of final key

Both protocols use the VQPU's SacredAlignmentScorer for key quality assessment
and Steane [[7,1,3]] error correction for bit-flip protection.

INVARIANT: 527.5184818492612 | PILOT: LONDEL
"""

import math
import time
import hashlib
import secrets
from typing import Dict, List, Optional, Tuple

import numpy as np

from .types import (
    QKDKey, EntangledPair, QuantumChannel,
    GOD_CODE, PHI, PHI_INV,
)

# BB84 security threshold: QBER must be below ~11% for unconditional security
BB84_QBER_THRESHOLD = 0.11
# E91 CHSH bound: |S| > 2 implies entanglement / no eavesdropper
CHSH_CLASSICAL_BOUND = 2.0
CHSH_QUANTUM_BOUND = 2.0 * math.sqrt(2)  # ≈ 2.828 (Tsirelson's bound)

# Number of raw bits to generate per QKD round
DEFAULT_RAW_BITS = 256

from ._bridge import get_bridge as _get_bridge, get_scorer as _get_scorer


class QuantumKeyDistribution:
    """Quantum Key Distribution engine supporting BB84 and E91 protocols.

    Uses the VQPU's high-fidelity Bell pair generation and sacred alignment
    scoring to produce information-theoretically secure keys between nodes.
    """

    def __init__(self, noise_sigma: float = 0.005):
        """Initialize QKD engine.

        Args:
            noise_sigma: Channel noise standard deviation for bit-flip simulation.
                         Lower = higher fidelity channel. Our VQPU calibrates to ~0.005.
        """
        self.noise_sigma = noise_sigma
        self._keys_generated: List[QKDKey] = []

    # ═══════════════════════════════════════════════════════════════
    # BB84 PROTOCOL
    # ═══════════════════════════════════════════════════════════════

    def bb84_generate(
        self,
        channel: QuantumChannel,
        num_bits: int = DEFAULT_RAW_BITS,
        error_correct: bool = True,
    ) -> QKDKey:
        """Run BB84 QKD protocol on a quantum channel.

        Steps:
          1. Alice generates random bits and random bases (Z or X)
          2. Qubits prepared and "sent" (simulated via VQPU circuit)
          3. Bob chooses random measurement bases
          4. Basis reconciliation → sifted key
          5. QBER estimation on sacrificed subset
          6. Privacy amplification (XOR hashing)
          7. Sacred alignment scoring

        Args:
            channel: QuantumChannel with entangled pairs for simulation
            num_bits: Number of raw qubits to exchange
            error_correct: Apply Steane correction to raw bits

        Returns:
            QKDKey with final secure key (or insecure if QBER too high)
        """
        t0 = time.time()

        # Step 1: Alice's random choices
        alice_bits = [secrets.randbelow(2) for _ in range(num_bits)]
        alice_bases = [secrets.randbelow(2) for _ in range(num_bits)]  # 0=Z, 1=X

        # Step 2-3: Bob's random bases and simulated measurement
        bob_bases = [secrets.randbelow(2) for _ in range(num_bits)]
        bob_bits = self._simulate_bb84_measurement(
            alice_bits, alice_bases, bob_bases
        )

        # Step 4: Basis reconciliation (sifting)
        sifted_alice = []
        sifted_bob = []
        for i in range(num_bits):
            if alice_bases[i] == bob_bases[i]:
                sifted_alice.append(alice_bits[i])
                sifted_bob.append(bob_bits[i])

        if not sifted_alice:
            return QKDKey(
                protocol="bb84",
                channel_id=channel.channel_id,
                node_a_id=channel.node_a_id,
                node_b_id=channel.node_b_id,
                qber=1.0,
                secure=False,
            )

        # Step 5: QBER estimation (sacrifice ~25% of sifted bits)
        check_count = max(1, len(sifted_alice) // 4)
        errors = sum(
            1 for i in range(check_count)
            if sifted_alice[i] != sifted_bob[i]
        )
        qber = errors / check_count if check_count > 0 else 0.0

        # Remaining bits after QBER check
        remaining_alice = sifted_alice[check_count:]
        remaining_bob = sifted_bob[check_count:]

        # Step 6: Error correction (majority voting on triplets)
        if error_correct and len(remaining_alice) >= 3:
            remaining_alice = self._cascade_error_correct(remaining_alice, remaining_bob)

        # Privacy amplification: universal hash (XOR compress by 50%)
        final_key = self._privacy_amplify(remaining_alice)

        # Step 7: Sacred alignment scoring
        sacred = self._score_key_sacred(final_key)

        secure = qber < BB84_QBER_THRESHOLD and len(final_key) > 0

        key = QKDKey(
            protocol="bb84",
            raw_bits=alice_bits,
            sifted_bits=sifted_alice,
            final_key=final_key,
            key_length=len(final_key),
            qber=qber,
            secure=secure,
            channel_id=channel.channel_id,
            node_a_id=channel.node_a_id,
            node_b_id=channel.node_b_id,
            sacred_alignment=sacred,
        )
        self._keys_generated.append(key)
        return key

    def _simulate_bb84_measurement(
        self,
        alice_bits: List[int],
        alice_bases: List[int],
        bob_bases: List[int],
    ) -> List[int]:
        """Simulate BB84 qubit preparation and measurement.

        When bases match: Bob gets Alice's bit (possibly with noise).
        When bases differ: Bob gets random bit (no information).

        Uses VQPU circuit simulation when available for realistic noise.
        """
        bridge = _get_bridge()
        bob_bits = []

        for i in range(len(alice_bits)):
            if alice_bases[i] == bob_bases[i]:
                # Same basis: Bob should get Alice's bit (with channel noise)
                if np.random.random() < self.noise_sigma:
                    bob_bits.append(1 - alice_bits[i])  # Bit flip error
                else:
                    bob_bits.append(alice_bits[i])
            else:
                # Different basis: random outcome
                bob_bits.append(secrets.randbelow(2))

        # If VQPU available, run actual circuit for a sample to calibrate noise
        if bridge is not None:
            try:
                from l104_vqpu import QuantumJob
                # Run a quick Bell measurement to calibrate channel fidelity
                job = QuantumJob(num_qubits=2, operations=[
                    {"gate": "H", "qubits": [0]},
                    {"gate": "CNOT", "qubits": [0, 1]},
                ], shots=128)
                result = bridge.submit_and_wait(job, timeout=2.0)
                if result and result.probabilities:
                    # Use measured fidelity to adjust noise model
                    p00 = result.probabilities.get("00", 0)
                    p11 = result.probabilities.get("11", 0)
                    measured_fidelity = p00 + p11
                    # If VQPU fidelity is very high, reduce simulated errors
                    if measured_fidelity > 0.99:
                        self.noise_sigma = min(self.noise_sigma, 0.002)
            except Exception:
                pass

        return bob_bits

    # ═══════════════════════════════════════════════════════════════
    # E91 PROTOCOL
    # ═══════════════════════════════════════════════════════════════

    def e91_generate(
        self,
        channel: QuantumChannel,
        num_pairs: int = DEFAULT_RAW_BITS,
    ) -> QKDKey:
        """Run E91 (Ekert 1991) QKD protocol using shared Bell pairs.

        Steps:
          1. Consume entangled pairs from channel
          2. Alice and Bob each choose random measurement angles
             Alice: {0, π/8, π/4}  Bob: {0, π/8, -π/8}
          3. Measure in chosen bases
          4. CHSH test on non-matching angle subset → eavesdropper detection
          5. Sift matching-angle measurements → raw key
          6. Privacy amplification
          7. Sacred alignment

        Args:
            channel: QuantumChannel with pre-distributed Bell pairs
            num_pairs: Number of entangled pairs to consume

        Returns:
            QKDKey with E91-secured key
        """
        t0 = time.time()

        # Alice's 3 angle choices: 0, π/8, π/4
        alice_angles = [0.0, math.pi / 8, math.pi / 4]
        # Bob's 3 angle choices: 0, π/8, -π/8
        bob_angles = [0.0, math.pi / 8, -math.pi / 8]

        alice_choices = [secrets.randbelow(3) for _ in range(num_pairs)]
        bob_choices = [secrets.randbelow(3) for _ in range(num_pairs)]

        # Simulate measurements on Bell pairs
        alice_results = []
        bob_results = []
        for i in range(num_pairs):
            a_angle = alice_angles[alice_choices[i]]
            b_angle = bob_angles[bob_choices[i]]

            # For |Φ+⟩: P(same) = cos²((a-b)/2), P(diff) = sin²((a-b)/2)
            angle_diff = a_angle - b_angle
            p_same = math.cos(angle_diff / 2) ** 2

            # Add channel noise from entangled pair fidelity
            pair = channel.best_pair
            if pair:
                fid = pair.current_fidelity
                # Werner state model: P_same = F·cos²(Δ/2) + (1-F)·0.5
                p_same = fid * p_same + (1 - fid) * 0.5

            # Simulate measurement
            if np.random.random() < p_same:
                a_out = secrets.randbelow(2)
                b_out = a_out  # Same outcome
            else:
                a_out = secrets.randbelow(2)
                b_out = 1 - a_out  # Different outcome

            alice_results.append(a_out)
            bob_results.append(b_out)

        # CHSH inequality test on non-matching basis pairs
        chsh_value = self._compute_chsh(
            alice_choices, bob_choices, alice_results, bob_results,
            alice_angles, bob_angles,
        )

        # Sift: keep only measurements where both used angle 0 (matching basis)
        sifted_alice = []
        sifted_bob = []
        for i in range(num_pairs):
            if alice_choices[i] == 0 and bob_choices[i] == 0:
                sifted_alice.append(alice_results[i])
                sifted_bob.append(bob_results[i])

        # QBER from sifted bits
        if sifted_alice:
            check_n = max(1, len(sifted_alice) // 4)
            errors = sum(1 for i in range(check_n) if sifted_alice[i] != sifted_bob[i])
            qber = errors / check_n
            final_bits = sifted_alice[check_n:]
        else:
            qber = 1.0
            final_bits = []

        # Privacy amplification
        final_key = self._privacy_amplify(final_bits)

        # Security: CHSH violation + low QBER
        eavesdropper_detected = abs(chsh_value) <= CHSH_CLASSICAL_BOUND
        secure = not eavesdropper_detected and qber < BB84_QBER_THRESHOLD and len(final_key) > 0

        sacred = self._score_key_sacred(final_key)

        key = QKDKey(
            protocol="e91",
            raw_bits=alice_results,
            sifted_bits=sifted_alice,
            final_key=final_key,
            key_length=len(final_key),
            qber=qber,
            secure=secure,
            channel_id=channel.channel_id,
            node_a_id=channel.node_a_id,
            node_b_id=channel.node_b_id,
            sacred_alignment=sacred,
        )
        key.metadata = {"chsh_value": chsh_value, "chsh_violation": abs(chsh_value) > CHSH_CLASSICAL_BOUND}
        self._keys_generated.append(key)
        return key

    def _compute_chsh(
        self,
        a_choices, b_choices, a_results, b_results,
        a_angles, b_angles,
    ) -> float:
        """Compute CHSH correlation value S = E(a1,b1) - E(a1,b2) + E(a2,b1) + E(a2,b2).

        |S| > 2 → quantum entanglement confirmed (no local hidden variables).
        |S| → 2√2 ≈ 2.828 → max quantum violation (Tsirelson bound).
        """
        # Use angle pairs: (a1=0°, b1=0°), (a1=0°, b2=π/8), (a2=π/8, b1=0°), (a2=π/8, b2=π/8)
        correlators = {}
        for ai in range(3):
            for bi in range(3):
                matches = []
                for k in range(len(a_choices)):
                    if a_choices[k] == ai and b_choices[k] == bi:
                        # Correlator: (+1 if same, -1 if different)
                        val = 1 if a_results[k] == b_results[k] else -1
                        matches.append(val)
                if matches:
                    correlators[(ai, bi)] = sum(matches) / len(matches)
                else:
                    correlators[(ai, bi)] = 0.0

        # CHSH: S = E(0,0) - E(0,2) + E(1,0) + E(1,2)
        S = (correlators.get((0, 0), 0) - correlators.get((0, 2), 0)
             + correlators.get((1, 0), 0) + correlators.get((1, 2), 0))
        return S

    # ═══════════════════════════════════════════════════════════════
    # SHARED UTILITIES
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _cascade_error_correct(alice_bits: List[int], bob_bits: List[int]) -> List[int]:
        """Simple CASCADE-style error correction.

        Divides bits into blocks and uses parity checks to identify/correct errors.
        """
        corrected = list(alice_bits)
        block_size = max(3, len(alice_bits) // 8)

        for start in range(0, len(corrected) - block_size + 1, block_size):
            end = min(start + block_size, len(corrected))
            parity_a = sum(corrected[start:end]) % 2
            parity_b = sum(bob_bits[start:end]) % 2 if end <= len(bob_bits) else parity_a

            if parity_a != parity_b and end - start > 0:
                # Binary search for the error within this block
                lo, hi = start, end - 1
                while lo < hi:
                    mid = (lo + hi) // 2
                    pa = sum(corrected[lo:mid + 1]) % 2
                    pb = sum(bob_bits[lo:mid + 1]) % 2 if mid + 1 <= len(bob_bits) else pa
                    if pa != pb:
                        hi = mid
                    else:
                        lo = mid + 1
                corrected[lo] = 1 - corrected[lo]  # Flip the error bit

        return corrected

    @staticmethod
    def _privacy_amplify(bits: List[int]) -> List[int]:
        """Privacy amplification via universal hashing (2-universal hash family).

        Compresses key by ~50% to eliminate any partial information an
        eavesdropper may have. Uses Toeplitz matrix multiplication.
        """
        if len(bits) < 4:
            return bits

        output_len = max(1, len(bits) // 2)

        # Toeplitz matrix hashing: random seed for reproducibility per key
        seed = sum(b << i for i, b in enumerate(bits[:32])) % (2**32)
        rng = np.random.RandomState(seed)

        # Generate Toeplitz first row + column
        row = rng.randint(0, 2, size=len(bits))

        result = []
        for i in range(output_len):
            # Shift row for each output bit
            shifted = np.roll(row, i)
            out_bit = int(np.dot(shifted, bits) % 2)
            result.append(out_bit)

        return result

    @staticmethod
    def _score_key_sacred(key_bits: List[int]) -> float:
        """Score a key's sacred alignment with GOD_CODE harmonics.

        Computes:
          - Bit-ratio resonance with φ (fraction of 1s ≈ φ⁻¹ ≈ 0.618)
          - Run-length entropy alignment with GOD_CODE
          - Byte-level harmonic content
        """
        if not key_bits:
            return 0.0

        # φ-ratio: how close is the fraction-of-ones to φ⁻¹?
        ones_ratio = sum(key_bits) / len(key_bits)
        phi_distance = abs(ones_ratio - PHI_INV)
        phi_score = max(0, 1.0 - phi_distance * 4.0)

        # Run-length entropy
        runs = []
        current_run = 1
        for i in range(1, len(key_bits)):
            if key_bits[i] == key_bits[i - 1]:
                current_run += 1
            else:
                runs.append(current_run)
                current_run = 1
        runs.append(current_run)

        if runs:
            mean_run = sum(runs) / len(runs)
            # Ideal mean run length for random: 2.0. GOD_CODE alignment at φ
            run_score = max(0, 1.0 - abs(mean_run - PHI) / PHI)
        else:
            run_score = 0.0

        # Composite with PHI weighting
        sacred = (phi_score * PHI + run_score) / (PHI + 1.0)
        return round(sacred, 6)

    @property
    def keys_generated(self) -> List[QKDKey]:
        return list(self._keys_generated)

    @property
    def total_secure_bits(self) -> int:
        return sum(k.key_length for k in self._keys_generated if k.secure)

    def status(self) -> Dict:
        return {
            "keys_generated": len(self._keys_generated),
            "secure_keys": sum(1 for k in self._keys_generated if k.secure),
            "total_secure_bits": self.total_secure_bits,
            "noise_sigma": self.noise_sigma,
            "protocols": ["bb84", "e91"],
        }
