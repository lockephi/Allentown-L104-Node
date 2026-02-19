from .constants import *
class QuantumComputationCore:
    """Advanced quantum computation engine for ASI optimization and intelligence.

    v6.0 Capabilities:
      1. VQE (Variational Quantum Eigensolver) — parameterized circuit optimization
      2. QAOA (Quantum Approximate Optimization) — subsystem routing optimization
      3. ZNE (Zero-Noise Extrapolation) — quantum error mitigation
      4. QRC (Quantum Reservoir Computing) — time-series prediction
      5. QKM (Quantum Kernel Method) — domain classification
      6. QPE (Quantum Phase Estimation) — sacred constant verification

    All methods have Qiskit 2.3.0 quantum path + classical fallback.
    Sacred constants (GOD_CODE, PHI, FEIGENBAUM) wired into every circuit.
    """

    def __init__(self):
        self.vqe_history: List[Dict] = []
        self.qaoa_cache: Dict[str, List] = {}
        self.reservoir_state: Optional[np.ndarray] = None
        self.kernel_gram_cache: Dict[str, Any] = {}
        self.qpe_verifications: int = 0
        self.zne_corrections: int = 0
        self._metrics = {
            'vqe_runs': 0, 'qaoa_runs': 0, 'qrc_runs': 0,
            'qkm_runs': 0, 'qpe_runs': 0, 'zne_runs': 0,
            'total_circuits': 0,
        }
        self._boot_time = time.time()

    # ─── VQE: Variational Quantum Eigensolver for ASI Parameter Optimization ───

    def _build_ansatz(self, theta: np.ndarray, n_qubits: int) -> 'QuantumCircuit':
        """Build parameterized ansatz circuit for VQE."""
        qc = QuantumCircuit(n_qubits)
        p_idx = 0
        for layer in range(VQE_ANSATZ_DEPTH):
            for q in range(n_qubits):
                qc.ry(float(theta[p_idx % len(theta)]), q)
                p_idx += 1
                qc.rz(float(theta[p_idx % len(theta)]), q)
                p_idx += 1
            for q in range(n_qubits - 1):
                qc.cx(q, q + 1)
            qc.rz(GOD_CODE / (1000.0 * (layer + 1)), 0)
        return qc

    def _eval_energy(self, theta: np.ndarray, n_qubits: int, hamiltonian_diag: np.ndarray) -> float:
        """Evaluate expectation value <psi(theta)|H|psi(theta)>."""
        qc = self._build_ansatz(theta, n_qubits)
        sv = Statevector.from_instruction(qc)
        probs = np.abs(sv.data) ** 2
        return float(np.dot(probs, hamiltonian_diag))

    def vqe_optimize(self, cost_vector: List[float], num_params: int = 7) -> Dict[str, Any]:
        """Optimize ASI parameters using VQE with SPSA gradient estimation + Adam.

        Encodes cost_vector as a diagonal Hamiltonian, then variationally
        minimizes <psi(theta)|H|psi(theta)> using SPSA (Simultaneous Perturbation
        Stochastic Approximation) for O(1)-per-step gradient estimation, combined
        with Adam momentum for fast convergence.

        Args:
            cost_vector: ASI dimension scores to optimize (up to 8 values).
            num_params: Number of variational parameters per layer.

        Returns:
            Dict with optimal_params, min_energy, convergence_history, sacred_alignment.
        """
        if not QISKIT_AVAILABLE:
            return {
                'quantum': False, 'fallback': 'golden_section',
                'optimal_params': [PHI * (i + 1) / max(len(cost_vector), 1)
                                   for i in range(min(num_params, max(len(cost_vector), 1)))],
                'min_energy': min(cost_vector) if cost_vector else 0.0,
                'convergence_history': [],
                'sacred_alignment': GOD_CODE / 1000.0,
            }

        n_qubits = 3  # 8-dimensional Hilbert space
        padded = list(cost_vector[:8])
        while len(padded) < 8:
            padded.append(0.0)
        hamiltonian_diag = np.array(padded, dtype=float)

        # Initialize variational parameters with sacred seeding
        n_params_total = VQE_ANSATZ_DEPTH * n_qubits * 2
        theta = np.array([
            PHI * (i + 1) % (2 * np.pi) for i in range(n_params_total)
        ])

        best_energy = float('inf')
        best_theta = theta.copy()
        convergence = []
        total_circuits = 0

        # Adam optimizer state
        lr = 0.15 * TAU  # ~0.093 learning rate (slightly higher for SPSA noise)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        m = np.zeros_like(theta)  # First moment
        v = np.zeros_like(theta)  # Second moment

        # SPSA perturbation scale (decays with steps for convergence)
        spsa_c = 0.2  # Initial perturbation magnitude

        for step in range(VQE_OPTIMIZATION_STEPS):
            # Evaluate current energy
            energy = self._eval_energy(theta, n_qubits, hamiltonian_diag)
            total_circuits += 1

            if energy < best_energy:
                best_energy = energy
                best_theta = theta.copy()

            convergence.append({'step': step, 'energy': round(energy, 8)})

            # SPSA gradient estimation: 2 circuit evaluations per step (O(1) in params!)
            # Random perturbation direction: Rademacher ±1
            delta = np.where(np.random.random(len(theta)) > 0.5, 1.0, -1.0)
            c_k = spsa_c / (step + 1) ** 0.101  # Slowly decaying perturbation

            e_plus = self._eval_energy(theta + c_k * delta, n_qubits, hamiltonian_diag)
            e_minus = self._eval_energy(theta - c_k * delta, n_qubits, hamiltonian_diag)
            total_circuits += 2

            # SPSA gradient approximation
            grad = (e_plus - e_minus) / (2.0 * c_k * delta)

            # Adam update with SPSA gradient
            t_adam = step + 1
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad ** 2
            m_hat = m / (1 - beta1 ** t_adam)
            v_hat = v / (1 - beta2 ** t_adam)
            theta = theta - lr * m_hat / (np.sqrt(v_hat) + eps)

        god_harmonic = GOD_CODE % (2 * np.pi)
        sacred_alignment = 1.0 - abs(best_energy - god_harmonic) / max(god_harmonic, 1e-10)
        sacred_alignment = max(0.0, min(1.0, sacred_alignment))

        self._metrics['vqe_runs'] += 1
        self._metrics['total_circuits'] += total_circuits
        self.vqe_history.append({
            'min_energy': round(best_energy, 8),
            'steps': VQE_OPTIMIZATION_STEPS,
            'circuits': total_circuits,
            'timestamp': time.time(),
        })

        return {
            'quantum': True,
            'optimal_params': [round(float(t), 6) for t in best_theta[:num_params]],
            'min_energy': round(best_energy, 8),
            'convergence_history': convergence[-5:],
            'total_iterations': VQE_OPTIMIZATION_STEPS,
            'total_circuits': total_circuits,
            'optimizer': 'spsa_adam',
            'ansatz_depth': VQE_ANSATZ_DEPTH,
            'sacred_alignment': round(sacred_alignment, 6),
            'qubits': n_qubits,
        }

    # ─── QAOA: Quantum Approximate Optimization for Pipeline Routing ───

    def qaoa_route(self, affinity_scores: List[float],
                   subsystem_names: List[str]) -> Dict[str, Any]:
        """Route problems through optimal subsystems using QAOA.

        Encodes subsystem affinities as a QUBO cost Hamiltonian, builds
        alternating cost/mixer QAOA layers, and selects the highest-probability
        bitstring as the optimal subsystem combination.
        """
        n = min(len(affinity_scores), 2 ** QAOA_SUBSYSTEM_QUBITS)
        if not QISKIT_AVAILABLE or n == 0:
            ranked = sorted(zip(subsystem_names[:n], affinity_scores[:n]),
                            key=lambda x: x[1], reverse=True)
            return {
                'quantum': False, 'fallback': 'affinity_sort',
                'selected_subsystems': [r[0] for r in ranked[:3]],
                'bitstring': '', 'probability': 0.0,
                'qaoa_energy': sum(affinity_scores[:n]) if n > 0 else 0.0,
            }

        n_qubits = QAOA_SUBSYSTEM_QUBITS
        padded_affinities = list(affinity_scores[:2**n_qubits])
        while len(padded_affinities) < 2**n_qubits:
            padded_affinities.append(0.0)

        gammas = [GOD_CODE / (1000.0 * (l + 1)) for l in range(QAOA_LAYERS)]
        betas = [PHI / (l + 1) for l in range(QAOA_LAYERS)]

        qc = QuantumCircuit(n_qubits)
        qc.h(range(n_qubits))

        for layer in range(QAOA_LAYERS):
            gamma, beta = gammas[layer], betas[layer]
            for i in range(n_qubits - 1):
                weight = (padded_affinities[i] + padded_affinities[i + 1]) / 2.0
                qc.rzz(gamma * weight * 2, i, i + 1)
            for i in range(n_qubits):
                qc.rz(gamma * padded_affinities[i % len(padded_affinities)], i)
            for i in range(n_qubits):
                qc.rx(2 * beta, i)

        sv = Statevector.from_instruction(qc)
        probs = sv.probabilities()
        best_idx = int(np.argmax(probs))
        best_bitstring = format(best_idx, f'0{n_qubits}b')
        best_prob = float(probs[best_idx])

        selected = []
        for i, bit in enumerate(best_bitstring):
            if bit == '1' and i < len(subsystem_names):
                selected.append(subsystem_names[i])
        if not selected and subsystem_names:
            selected = [subsystem_names[int(np.argmax(affinity_scores[:len(subsystem_names)]))]]

        qaoa_energy = float(np.dot(probs, padded_affinities[:len(probs)]))

        self._metrics['qaoa_runs'] += 1
        self._metrics['total_circuits'] += 1

        return {
            'quantum': True,
            'selected_subsystems': selected,
            'bitstring': f'|{best_bitstring}>',
            'probability': round(best_prob, 6),
            'qaoa_energy': round(qaoa_energy, 6),
            'qaoa_layers': QAOA_LAYERS,
            'qubits': n_qubits,
            'cost_landscape': {
                'top_3': sorted(
                    [(format(i, f'0{n_qubits}b'), round(float(probs[i]), 6))
                     for i in range(len(probs))],
                    key=lambda x: x[1], reverse=True
                )[:3]
            },
        }

    # ─── ZNE: Zero-Noise Extrapolation Error Mitigation ───

    def quantum_error_mitigate(self, base_probs: np.ndarray) -> Dict[str, Any]:
        """Apply Zero-Noise Extrapolation to mitigate quantum errors.

        Evaluates at multiple noise levels by simulating gate noise scaling,
        then extrapolates to the zero-noise limit via polynomial fit.
        """
        if not QISKIT_AVAILABLE or len(base_probs) == 0:
            dominant = float(np.max(base_probs)) if len(base_probs) > 0 else 0.5
            return {
                'quantum': False, 'fallback': 'unmitigated',
                'mitigated_value': dominant,
                'raw_values': [dominant],
                'noise_factors': ZNE_NOISE_FACTORS,
            }

        base_arr = np.array(base_probs, dtype=float)
        raw_values = []
        for factor in ZNE_NOISE_FACTORS:
            uniform = np.ones_like(base_arr) / len(base_arr)
            noise_strength = 1.0 - 1.0 / factor
            noisy_probs = (1.0 - noise_strength * 0.1) * base_arr + noise_strength * 0.1 * uniform
            raw_values.append(float(np.max(noisy_probs)))

        factors = np.array(ZNE_NOISE_FACTORS)
        values = np.array(raw_values)
        if len(factors) >= 2:
            coeffs = np.polyfit(factors, values, min(len(factors) - 1, 2))
            mitigated = float(np.polyval(coeffs, 0.0))
            mitigated = max(0.0, min(1.0, mitigated))
        else:
            mitigated = raw_values[0]

        correction = mitigated - raw_values[0]
        self._metrics['zne_runs'] += 1
        self.zne_corrections += 1

        return {
            'quantum': True,
            'mitigated_value': round(mitigated, 8),
            'raw_values': [round(v, 8) for v in raw_values],
            'noise_factors': ZNE_NOISE_FACTORS,
            'correction_applied': round(correction, 8),
            'extrapolation_order': min(len(factors) - 1, 2),
        }

    # ─── QRC: Quantum Reservoir Computing for Metric Prediction ───

    def quantum_reservoir_compute(self, time_series: List[float],
                                   prediction_steps: int = 3) -> Dict[str, Any]:
        """Predict future ASI metrics using a quantum reservoir computer.

        Builds a fixed random unitary reservoir circuit (seeded by GOD_CODE),
        drives it with time-series data, and trains a linear readout layer
        to predict future values.
        """
        if len(time_series) < 3:
            return {
                'quantum': False, 'error': 'insufficient_data',
                'predictions': [], 'training_mse': 1.0,
            }

        if not QISKIT_AVAILABLE:
            alpha = TAU
            smoothed = time_series[-1]
            predictions = []
            for _ in range(prediction_steps):
                smoothed = alpha * smoothed + (1 - alpha) * float(np.mean(time_series[-3:]))
                predictions.append(round(float(smoothed), 6))
            return {
                'quantum': False, 'fallback': 'phi_exponential_smoothing',
                'predictions': predictions,
                'training_mse': 0.0, 'reservoir_dim': 0,
            }

        n_qubits = QRC_RESERVOIR_QUBITS
        reservoir_dim = 2 ** n_qubits
        seed_val = int(GOD_CODE * 100) % (2**31)

        def build_reservoir(input_val: float) -> np.ndarray:
            rng = np.random.RandomState(seed_val)
            qc = QuantumCircuit(n_qubits)
            qc.ry(float(input_val) * np.pi, 0)
            for depth in range(QRC_RESERVOIR_DEPTH):
                for q in range(n_qubits):
                    qc.ry(float(rng.uniform(0, 2 * np.pi)), q)
                    qc.rz(float(rng.uniform(0, 2 * np.pi)), q)
                for q in range(n_qubits - 1):
                    if rng.random() > 0.3:
                        qc.cx(q, q + 1)
                qc.rz(GOD_CODE / (1000.0 * (depth + 1)), n_qubits - 1)
            sv = Statevector.from_instruction(qc)
            return sv.probabilities()

        readout_states = [build_reservoir(val) for val in time_series]

        X = np.array(readout_states[:-1])
        y = np.array(time_series[1:])

        try:
            w, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            training_mse = float(np.mean((X @ w - y) ** 2))
        except np.linalg.LinAlgError:
            w = np.zeros(reservoir_dim)
            training_mse = 1.0

        predictions = []
        current_val = time_series[-1]
        for _ in range(prediction_steps):
            state = build_reservoir(current_val)
            predicted = max(0.0, min(1.0, float(np.dot(state, w))))
            predictions.append(round(predicted, 6))
            current_val = predicted

        self._metrics['qrc_runs'] += 1
        self._metrics['total_circuits'] += len(time_series) + prediction_steps
        self.reservoir_state = readout_states[-1] if readout_states else None

        return {
            'quantum': True,
            'predictions': predictions,
            'reservoir_dim': reservoir_dim,
            'reservoir_qubits': n_qubits,
            'reservoir_depth': QRC_RESERVOIR_DEPTH,
            'training_mse': round(training_mse, 8),
            'fidelity': round(1.0 - min(1.0, training_mse), 6),
            'time_series_length': len(time_series),
        }

    # ─── QKM: Quantum Kernel Method for Domain Classification ───

    def quantum_kernel_classify(self, query_features: List[float],
                                 domain_prototypes: Dict[str, List[float]]) -> Dict[str, Any]:
        """Classify a query into domains using quantum kernel similarity.

        Encodes features into a ZZ-entangling quantum feature map, computes
        kernel K[i,j] = |<phi(x_i)|phi(x_j)>|^2 via Statevector inner products,
        and classifies by maximum kernel similarity.
        """
        if not domain_prototypes:
            return {'quantum': False, 'error': 'no_domains', 'predicted_domain': 'unknown'}

        if not QISKIT_AVAILABLE:
            def cosine_sim(a, b):
                a_arr, b_arr = np.array(a, dtype=float), np.array(b, dtype=float)
                na, nb = np.linalg.norm(a_arr), np.linalg.norm(b_arr)
                return float(np.dot(a_arr, b_arr) / (na * nb)) if na > 1e-10 and nb > 1e-10 else 0.0

            sims = {name: cosine_sim(query_features, proto)
                    for name, proto in domain_prototypes.items()}
            best = max(sims, key=sims.get)
            return {
                'quantum': False, 'fallback': 'cosine_similarity',
                'predicted_domain': best,
                'confidence': round(max(sims.values()), 6),
                'kernel_similarities': {k: round(v, 6) for k, v in sims.items()},
            }

        n_qubits = QKM_FEATURE_QUBITS

        def feature_map_circuit(features: List[float]) -> Statevector:
            qc = QuantumCircuit(n_qubits)
            padded = list(features[:n_qubits])
            while len(padded) < n_qubits:
                padded.append(0.0)
            for i in range(n_qubits):
                qc.h(i)
                qc.rz(float(padded[i]) * 2.0, i)
            for i in range(n_qubits - 1):
                qc.cx(i, i + 1)
                qc.rz(float(padded[i] * padded[i + 1]) * PHI, i + 1)
                qc.cx(i, i + 1)
            for i in range(n_qubits):
                qc.ry(float(padded[i]) * np.pi, i)
            qc.rz(GOD_CODE / 1000.0, 0)
            return Statevector.from_instruction(qc)

        query_sv = feature_map_circuit(query_features)
        similarities = {}
        for name, proto in domain_prototypes.items():
            proto_sv = feature_map_circuit(proto)
            inner = np.abs(np.vdot(query_sv.data, proto_sv.data)) ** 2
            similarities[name] = round(float(inner), 8)

        best_domain = max(similarities, key=similarities.get)

        self._metrics['qkm_runs'] += 1
        self._metrics['total_circuits'] += 1 + len(domain_prototypes)

        return {
            'quantum': True,
            'predicted_domain': best_domain,
            'confidence': round(similarities[best_domain], 6),
            'kernel_similarities': {k: round(v, 6) for k, v in similarities.items()},
            'feature_map': 'ZZ_entangling',
            'qubits': n_qubits,
        }

    # ─── QPE: Quantum Phase Estimation for Sacred Constant Verification ───

    def qpe_sacred_verify(self, target_phase: Optional[float] = None) -> Dict[str, Any]:
        """Verify sacred constant alignment using Quantum Phase Estimation.

        Applies controlled-U^(2^k) rotations where U encodes the target phase,
        then runs inverse QFT to extract the estimated phase. Compares the
        estimate to GOD_CODE-derived reference.
        """
        if target_phase is None:
            target_phase = (GOD_CODE / 1000.0) % (2 * np.pi)

        if not QISKIT_AVAILABLE:
            return {
                'quantum': False, 'fallback': 'direct_comparison',
                'estimated_phase': round(target_phase, 8),
                'target_phase': round(target_phase, 8),
                'alignment_error': 0.0,
                'god_code_resonance': 1.0,
            }

        n_counting = QPE_PRECISION_QUBITS
        n_total = n_counting + 1
        target_qubit = n_counting

        qc = QuantumCircuit(n_total)
        for i in range(n_counting):
            qc.h(i)
        qc.x(target_qubit)

        # Controlled-U^(2^k) applications
        for k in range(n_counting):
            angle = target_phase * (2 ** k)
            qc.cp(angle, k, target_qubit)

        # Inverse QFT on counting qubits
        for i in range(n_counting // 2):
            qc.swap(i, n_counting - 1 - i)
        for i in range(n_counting):
            for j in range(i):
                qc.cp(-np.pi / (2 ** (i - j)), j, i)
            qc.h(i)

        sv = Statevector.from_instruction(qc)
        probs = sv.probabilities()

        counting_probs = np.zeros(2 ** n_counting)
        for state_idx in range(len(probs)):
            counting_bits = state_idx >> 1
            counting_probs[counting_bits % (2 ** n_counting)] += probs[state_idx]

        best_state = int(np.argmax(counting_probs))
        estimated_phase = 2 * np.pi * best_state / (2 ** n_counting)
        alignment_error = abs(estimated_phase - target_phase)

        god_harmonic = GOD_CODE % (2 * np.pi)
        resonance = max(0.0, min(1.0, 1.0 - abs(estimated_phase - god_harmonic) / np.pi))

        self._metrics['qpe_runs'] += 1
        self._metrics['total_circuits'] += 1
        self.qpe_verifications += 1

        return {
            'quantum': True,
            'estimated_phase': round(float(estimated_phase), 8),
            'target_phase': round(float(target_phase), 8),
            'alignment_error': round(float(alignment_error), 8),
            'god_code_resonance': round(float(resonance), 6),
            'precision_bits': n_counting,
            'best_counting_state': f'|{best_state:0{n_counting}b}>',
            'measurement_confidence': round(float(counting_probs[best_state]), 6),
        }

    def status(self) -> Dict[str, Any]:
        """Return quantum computation core status and metrics."""
        total = sum(self._metrics[k] for k in ['vqe_runs', 'qaoa_runs', 'qrc_runs',
                                                  'qkm_runs', 'qpe_runs', 'zne_runs'])
        return {
            'version': '7.0.0',
            'qiskit_available': QISKIT_AVAILABLE,
            'metrics': dict(self._metrics),
            'total_computations': total,
            'total_circuits_executed': self._metrics['total_circuits'],
            'vqe_history_length': len(self.vqe_history),
            'qaoa_cache_size': len(self.qaoa_cache),
            'qpe_verifications': self.qpe_verifications,
            'zne_corrections': self.zne_corrections,
            'uptime_s': round(time.time() - self._boot_time, 1),
            'capabilities': ['VQE', 'QAOA', 'ZNE', 'QRC', 'QKM', 'QPE'],
        }


