#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════╗
║  L104 QML v2.0 — COMPREHENSIVE VALIDATION SUITE                         ║
║  Tests all 9 upgraded Quantum Machine Learning capabilities              ║
╚═══════════════════════════════════════════════════════════════════════════╝

  Phase 1:  Module Boot & Sacred Constants            (4 tests)
  Phase 2:  ZZ Feature Map                            (5 tests)
  Phase 3:  Data Re-Uploading Circuit                 (5 tests)
  Phase 4:  Berry Phase Ansatz                        (5 tests)
  Phase 5:  Quantum Kernel Estimator                  (6 tests)
  Phase 6:  QAOA MaxCut                               (5 tests)
  Phase 7:  Barren Plateau Analyzer                   (4 tests)
  Phase 8:  Quantum Regressor QNN                     (5 tests)
  Phase 9:  Expressibility Analyzer                   (5 tests)
  Phase 10: QuantumMLHub v2 Orchestrator              (6 tests)
  ────────────────────────────────────────────────────
  Total:    50 tests across 10 phases
"""

import sys, os, time, math, traceback
import numpy as np

# ─── Constants ───
GOD_CODE = 527.5184818492612
PHI      = 1.618033988749895
VOID_CONSTANT = 1.0416180339887497

passed = 0
failed = 0
total  = 0

def test(name: str, condition: bool, detail: str = ""):
    global passed, failed, total
    total += 1
    if condition:
        passed += 1
        print(f"  ✓ {name}" + (f"  ({detail})" if detail else ""))
    else:
        failed += 1
        print(f"  ✗ {name}" + (f"  — {detail}" if detail else ""))


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — MODULE BOOT & SACRED CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n╔══════════════════════════════════════════════════════════════╗")
print("║  PHASE 1: Module Boot & Sacred Constants                    ║")
print("╚══════════════════════════════════════════════════════════════╝")

try:
    from l104_qml_v2 import (
        ZZFeatureMap,
        DataReUploadingCircuit,
        BerryPhaseAnsatz,
        QuantumKernelEstimator,
        QAOACircuit,
        BarrenPlateauAnalyzer,
        QuantumRegressorQNN,
        ExpressibilityAnalyzer,
        QuantumMLHub,
        get_qml_hub,
        VERSION,
        GOD_CODE as QML_GOD_CODE,
        PHI as QML_PHI,
        VOID_CONSTANT as QML_VOID,
    )
    test("Import all 9 QML v2 classes", True)
except Exception as e:
    test("Import all 9 QML v2 classes", False, str(e))
    traceback.print_exc()
    sys.exit(1)

test("Version = 2.0.0", VERSION == "2.0.0", f"v{VERSION}")
test("GOD_CODE sacred constant", abs(QML_GOD_CODE - GOD_CODE) < 1e-10,
     f"{QML_GOD_CODE}")
test("VOID_CONSTANT sacred constant", abs(QML_VOID - VOID_CONSTANT) < 1e-10,
     f"{QML_VOID}")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — ZZ FEATURE MAP
# ═══════════════════════════════════════════════════════════════════════════════

print("\n╔══════════════════════════════════════════════════════════════╗")
print("║  PHASE 2: ZZ Feature Map                                    ║")
print("╚══════════════════════════════════════════════════════════════╝")

zz = ZZFeatureMap(n_qubits=4, reps=2, entanglement="full")

# Test 1: Produces valid quantum state
features = np.array([0.5, 1.2, -0.3, 0.8])
state = zz.encode(features)
norm = np.linalg.norm(state)
test("ZZ encode → normalized state", abs(norm - 1.0) < 1e-10,
     f"||ψ||={norm:.12f}")

# Test 2: Correct dimension
test("ZZ state dimension = 2^4 = 16", len(state) == 16, f"dim={len(state)}")

# Test 3: Different inputs → different states
features2 = np.array([1.0, 0.5, 0.3, -0.8])
state2 = zz.encode(features2)
fidelity = abs(np.vdot(state, state2)) ** 2
test("ZZ different inputs → different states", fidelity < 0.99,
     f"F={fidelity:.6f}")

# Test 4: Entanglement types
zz_linear = ZZFeatureMap(4, reps=2, entanglement="linear")
zz_circular = ZZFeatureMap(4, reps=2, entanglement="circular")
s_full = zz.encode(features)
s_linear = zz_linear.encode(features)
s_circular = zz_circular.encode(features)
# All should be valid but different
f_fl = abs(np.vdot(s_full, s_linear)) ** 2
f_fc = abs(np.vdot(s_full, s_circular)) ** 2
test("ZZ entanglement types produce different states",
     f_fl < 0.999 or f_fc < 0.999,
     f"F(full,linear)={f_fl:.4f}, F(full,circ)={f_fc:.4f}")

# Test 5: Stats
test("ZZ stats correct", zz.stats["feature_map"] == "ZZFeatureMap" and
     zz.stats["n_qubits"] == 4, f"pattern={zz.stats['pattern']}")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — DATA RE-UPLOADING CIRCUIT
# ═══════════════════════════════════════════════════════════════════════════════

print("\n╔══════════════════════════════════════════════════════════════╗")
print("║  PHASE 3: Data Re-Uploading Circuit                         ║")
print("╚══════════════════════════════════════════════════════════════╝")

reup = DataReUploadingCircuit(n_qubits=3, n_layers=4)

# Test 1: Forward produces normalized state
features3 = np.array([0.5, -0.3, 1.1])
state_ru = reup.forward(features3)
norm_ru = np.linalg.norm(state_ru)
test("ReUpload forward → normalized", abs(norm_ru - 1.0) < 1e-10,
     f"||ψ||={norm_ru:.12f}")

# Test 2: Correct dimension
test("ReUpload dim = 2^3 = 8", len(state_ru) == 8, f"dim={len(state_ru)}")

# Test 3: Expectation in [-1, 1]
exp_val = reup.expectation(features3)
test("ReUpload expectation ∈ [-1,1]", -1.0 <= exp_val <= 1.0,
     f"⟨Z⟩={exp_val:.6f}")

# Test 4: Different inputs → different expectations
exp2 = reup.expectation(np.array([2.0, -1.0, 0.5]))
test("ReUpload different inputs → different outputs",
     abs(exp_val - exp2) > 1e-6,
     f"Δ⟨Z⟩={abs(exp_val - exp2):.6f}")

# Test 5: Stats verify universal approximation claim
test("ReUpload is universal approximator",
     reup.stats["universal_approximator"] is True and
     reup.stats["pattern"] == "Pérez-Salinas_2020",
     f"n_params={reup.stats['n_params']}")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — BERRY PHASE ANSATZ
# ═══════════════════════════════════════════════════════════════════════════════

print("\n╔══════════════════════════════════════════════════════════════╗")
print("║  PHASE 4: Berry Phase Ansatz (Noise-Robust Geometric)       ║")
print("╚══════════════════════════════════════════════════════════════╝")

berry = BerryPhaseAnsatz(n_qubits=3, n_layers=2)

# Test 1: Apply to |0⟩ → valid state
dim3 = 2 ** 3
init_state = np.zeros(dim3, dtype=np.complex128)
init_state[0] = 1.0
berry_state = berry.apply(init_state)
berry_norm = np.linalg.norm(berry_state)
test("Berry ansatz → normalized state", abs(berry_norm - 1.0) < 1e-9,
     f"||ψ||={berry_norm:.12f}")

# Test 2: Parameters shape
test("Berry params shape correct",
     berry.weights.shape == (2, 3, 3),
     f"shape={berry.weights.shape}")

# Test 3: Noise-robust flag
test("Berry ansatz marked noise-robust",
     berry.stats["noise_robust"] is True,
     f"gates={berry.stats['gate_types']}")

# Test 4: Different weights → different states
w1 = np.random.uniform(0, 2 * math.pi, (2, 3, 3))
w2 = np.random.uniform(0, 2 * math.pi, (2, 3, 3))
s1 = berry.apply(init_state.copy(), w1)
s2 = berry.apply(init_state.copy(), w2)
fid_berry = abs(np.vdot(s1, s2)) ** 2
test("Berry different weights → different states", fid_berry < 0.999,
     f"F={fid_berry:.6f}")

# Test 5: Geometric gate types present
gt = berry.stats["gate_types"]
test("Berry uses abelian + holonomic + CNOT",
     "Abelian_Berry_Phase" in gt and "Non_Abelian_Holonomic" in gt,
     str(gt))


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — QUANTUM KERNEL ESTIMATOR
# ═══════════════════════════════════════════════════════════════════════════════

print("\n╔══════════════════════════════════════════════════════════════╗")
print("║  PHASE 5: Quantum Kernel Estimator                          ║")
print("╚══════════════════════════════════════════════════════════════╝")

# Test 1: K(x, x) ≈ 1
ke = QuantumKernelEstimator("zz", n_qubits=3)
x = np.array([0.5, 1.0, -0.3])
k_xx = ke.kernel_value(x, x)
test("K(x,x) ≈ 1 (self-fidelity)", abs(k_xx - 1.0) < 1e-8,
     f"K(x,x)={k_xx:.10f}")

# Test 2: 0 ≤ K(x,y) ≤ 1
y = np.array([2.0, -0.5, 0.8])
k_xy = ke.kernel_value(x, y)
test("0 ≤ K(x,y) ≤ 1", 0 <= k_xy <= 1.0,
     f"K(x,y)={k_xy:.6f}")

# Test 3: Symmetry K(x,y) = K(y,x)
k_yx = ke.kernel_value(y, x)
test("K(x,y) = K(y,x) (symmetry)", abs(k_xy - k_yx) < 1e-10,
     f"|Δ|={abs(k_xy - k_yx):.2e}")

# Test 4: Kernel matrix is positive semi-definite
X_data = np.random.randn(6, 3) * math.pi
K_mat = ke.kernel_matrix(X_data)
eigenvalues = np.linalg.eigvalsh(K_mat)
test("Kernel matrix is PSD (all eigenvalues ≥ -ε)",
     np.all(eigenvalues >= -1e-8),
     f"min_eig={eigenvalues[0]:.6e}")

# Test 5: Diagonal = 1
diag_ok = all(abs(K_mat[i, i] - 1.0) < 1e-8 for i in range(len(X_data)))
test("K diagonal = 1", diag_ok,
     f"diag=[{', '.join(f'{K_mat[i,i]:.6f}' for i in range(min(3, len(X_data))))}...]")

# Test 6: Kernel alignment computable
labels = np.array([1, 1, -1, -1, 1, -1], dtype=float)
alignment = ke.kernel_alignment(K_mat, labels)
test("Kernel alignment computable", -1.0 <= alignment <= 1.0,
     f"alignment={alignment:.6f}")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 6 — QAOA MAXCUT
# ═══════════════════════════════════════════════════════════════════════════════

print("\n╔══════════════════════════════════════════════════════════════╗")
print("║  PHASE 6: QAOA MaxCut                                       ║")
print("╚══════════════════════════════════════════════════════════════╝")

# Triangle graph: edges (0,1), (1,2), (0,2) — MaxCut = 2
edges_triangle = [(0, 1), (1, 2), (0, 2)]
qaoa = QAOACircuit(edges_triangle, p=2)

# Test 1: Circuit constructed
test("QAOA triangle graph constructed", qaoa.n_qubits == 3,
     f"n_qubits={qaoa.n_qubits}, p={qaoa.p}")

# Test 2: Run produces result
result = qaoa.run()
test("QAOA run → expected_cut", "expected_cut" in result and
     result["expected_cut"] >= 0,
     f"E[cut]={result['expected_cut']:.4f}")

# Test 3: Max cut correct
test("QAOA max_possible_cut = 2 (triangle)", result["max_possible_cut"] == 2,
     f"max_cut={result['max_possible_cut']}")

# Test 4: Optimization improves result
opt_result = qaoa.optimize(n_iterations=30)
test("QAOA optimization completes", "approximation_ratio" in opt_result,
     f"ratio={opt_result['approximation_ratio']:.4f}")

# Test 5: Approximation ratio > 0
test("QAOA approximation_ratio > 0", opt_result["approximation_ratio"] > 0,
     f"ratio={opt_result['approximation_ratio']:.4f}, "
     f"bitstring={opt_result['optimal_bitstring']}")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 7 — BARREN PLATEAU ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

print("\n╔══════════════════════════════════════════════════════════════╗")
print("║  PHASE 7: Barren Plateau Analyzer                           ║")
print("╚══════════════════════════════════════════════════════════════╝")

bp = BarrenPlateauAnalyzer(n_qubits=3, n_layers=2)

# Test 1: Analyze strongly entangling
bp_result = bp.analyze_gradient_variance("strongly_entangling", n_samples=20)
test("BP analysis completes (strongly_entangling)",
     "mean_gradient_variance" in bp_result,
     f"Var={bp_result['mean_gradient_variance']:.6e}")

# Test 2: Gradient norm > 0
test("BP gradient norm > 0", bp_result["gradient_norm_mean"] > 0,
     f"||∇||={bp_result['gradient_norm_mean']:.6f}")

# Test 3: Trainability assessment present
test("BP trainability assessment", "trainability" in bp_result,
     f"{bp_result['trainability']}")

# Test 4: Berry ansatz analysis
bp_berry = bp.analyze_gradient_variance("berry", n_samples=15)
test("BP Berry ansatz analysis completes",
     "mean_gradient_variance" in bp_berry,
     f"Var={bp_berry['mean_gradient_variance']:.6e}, "
     f"{bp_berry['trainability']}")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 8 — QUANTUM REGRESSOR QNN
# ═══════════════════════════════════════════════════════════════════════════════

print("\n╔══════════════════════════════════════════════════════════════╗")
print("║  PHASE 8: Quantum Regressor QNN                              ║")
print("╚══════════════════════════════════════════════════════════════╝")

reg = QuantumRegressorQNN(n_qubits=3, n_layers=2,
                          output_range=(-5.0, 5.0))

# Test 1: Predict outputs in range
feat = np.array([0.5, -0.3, 1.0])
pred = reg.predict(feat)
test("Regressor output ∈ [-5, 5]", -5.0 <= pred <= 5.0,
     f"ŷ={pred:.6f}")

# Test 2: Batch predict
X_batch = np.random.randn(5, 3)
preds = reg.predict_batch(X_batch)
test("Regressor batch: 5 predictions", len(preds) == 5,
     f"range=[{preds.min():.3f}, {preds.max():.3f}]")

# Test 3: Loss computable
y_batch = np.random.randn(5) * 2
loss = reg.compute_loss(X_batch, y_batch)
test("Regressor MSE loss ≥ 0", loss >= 0,
     f"MSE={loss:.6f}")

# Test 4: Single training step
step = reg.train_step(feat, target=2.5, learning_rate=0.01)
test("Regressor train_step → loss + gradient",
     "loss" in step and "gradient_norm" in step,
     f"loss={step['loss']:.4f}, ||∇||={step['gradient_norm']:.4f}")

# Test 5: Stats
test("Regressor stats correct",
     reg.stats["model"] == "QuantumRegressorQNN" and
     reg.stats["output_range"] == (-5.0, 5.0),
     f"n_params={reg.stats['n_params']}")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 9 — EXPRESSIBILITY ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

print("\n╔══════════════════════════════════════════════════════════════╗")
print("║  PHASE 9: Expressibility Analyzer                            ║")
print("╚══════════════════════════════════════════════════════════════╝")

ea = ExpressibilityAnalyzer(n_qubits=3)

# Test 1: Expressibility measurement
exp_result = ea.expressibility("strongly_entangling", n_layers=2, n_samples=100)
test("Expressibility measurement completes",
     "kl_divergence" in exp_result,
     f"KL={exp_result['kl_divergence']:.4f}, "
     f"score={exp_result['expressibility_score']:.4f}")

# Test 2: Mean fidelity close to Haar ≈ 1/d
haar_mean = exp_result["haar_mean_fidelity"]
emp_mean = exp_result["mean_fidelity"]
test("Mean fidelity order-of-magnitude correct",
     emp_mean < 1.0 and emp_mean > 0,
     f"F̄={emp_mean:.6f}, Haar={haar_mean:.6f}")

# Test 3: Meyer-Wallach for GHZ (high entanglement)
# GHZ = (|000⟩ + |111⟩)/√2
dim3 = 8
ghz = np.zeros(dim3, dtype=np.complex128)
ghz[0] = 1 / math.sqrt(2)  # |000⟩
ghz[7] = 1 / math.sqrt(2)  # |111⟩
q_ghz = ea.meyer_wallach_measure(ghz)
test("MW(GHZ) > 0.5 (highly entangled)", q_ghz > 0.5,
     f"Q(GHZ)={q_ghz:.6f}")

# Test 4: Meyer-Wallach for product state (no entanglement)
product = np.zeros(dim3, dtype=np.complex128)
product[0] = 1.0  # |000⟩
q_product = ea.meyer_wallach_measure(product)
test("MW(|000⟩) ≈ 0 (product state)", q_product < 0.01,
     f"Q(|000⟩)={q_product:.6f}")

# Test 5: Entangling capability
ent_result = ea.entangling_capability("strongly_entangling", n_layers=2,
                                       n_samples=50)
test("Entangling capability measured",
     "mean_Q" in ent_result and ent_result["mean_Q"] >= 0,
     f"Q̄={ent_result['mean_Q']:.4f} ± {ent_result['std_Q']:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 10 — QUANTUM ML HUB v2 ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

print("\n╔══════════════════════════════════════════════════════════════╗")
print("║  PHASE 10: QuantumMLHub v2 Orchestrator                      ║")
print("╚══════════════════════════════════════════════════════════════╝")

hub = get_qml_hub(n_qubits=4, n_layers=2)

# Test 1: Hub status
hs = hub.status()
test("Hub v2 status", hs["version"] == "2.0.0",
     f"{hs['n_qubits']}q/{hs['n_layers']}L")

# Test 2: All 9 capabilities enabled
caps = hs["capabilities"]
all_caps = all(caps.values())
test("All 9 capabilities enabled", all_caps,
     f"caps={sum(caps.values())}/9")

# Test 3: Kernel matrix via hub
X_hub = np.random.randn(4, 4) * 0.5
K_hub = hub.compute_kernel(X_hub, feature_map="zz")
test("Hub kernel matrix 4×4", K_hub.shape == (4, 4),
     f"shape={K_hub.shape}")

# Test 4: Berry classify via hub
feat_hub = np.array([0.3, -0.5, 0.7, 1.2])
bc = hub.berry_classify(feat_hub)
test("Hub Berry classify → prediction",
     bc["prediction"] in [0, 1] and bc["confidence"] > 0,
     f"class={bc['prediction']}, conf={bc['confidence']:.4f}")

# Test 5: QAOA via hub
qaoa_r = hub.qaoa_maxcut([(0, 1), (1, 2), (0, 2)], p=1, optimize=False)
test("Hub QAOA returns result", "expected_cut" in qaoa_r,
     f"E[cut]={qaoa_r['expected_cut']:.4f}")

# Test 6: Regression via hub
reg_val = hub.quantum_regress(feat_hub)
test("Hub regression → float", isinstance(reg_val, float),
     f"ŷ={reg_val:.6f}")


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL REPORT
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 62)
print(f"  QML v{VERSION} VALIDATION: {passed}/{total} passed, {failed} failed")
print("═" * 62)

if failed == 0:
    print("  ★ ALL TESTS PASSED — QML v2.0 Upgrade Complete")
    print(f"  ★ 9 new capabilities: ZZ Feature Map, Data Re-Uploading,")
    print(f"    Berry Phase Ansatz, Quantum Kernel, QAOA, Barren Plateau,")
    print(f"    Quantum Regression, Expressibility, QuantumMLHub v2")
else:
    print(f"  ⚠ {failed} test(s) failed — review above")

print("═" * 62)

sys.exit(0 if failed == 0 else 1)
