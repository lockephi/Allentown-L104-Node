"""Quick validation of Quantum Brain v13.0.0 upgrade."""
import random
import sys

print("═══ L104 QUANTUM BRAIN v13.0.0 VALIDATION ═══\n")

# 1. Import validation
from l104_quantum_engine.manifold import (
    QuantumManifoldLearner, MultipartiteEntanglementNetwork, QuantumPredictiveOracle
)
print("✓ manifold.py imports OK")

from l104_quantum_engine import __version__
assert __version__ == "13.0.0", f"Expected 13.0.0, got {__version__}"
print(f"✓ __init__.py version = {__version__}")

from l104_quantum_engine.constants import VERSION
assert VERSION == "13.0.0", f"Expected 13.0.0, got {VERSION}"
print(f"✓ constants.py VERSION = {VERSION}")

from l104_quantum_engine.brain import L104QuantumBrain
assert L104QuantumBrain.VERSION == "13.0.0"
print(f"✓ brain.py VERSION = {L104QuantumBrain.VERSION}")

# 2. Brain init with new subsystems
brain = L104QuantumBrain()
assert hasattr(brain, "manifold_learner"), "Missing manifold_learner"
assert hasattr(brain, "entanglement_network"), "Missing entanglement_network"
assert hasattr(brain, "predictive_oracle"), "Missing predictive_oracle"
print("✓ All 3 new subsystems initialized on L104QuantumBrain")

# 3. Create mock links for functional testing
class FakeLink:
    def __init__(self, fid, strength, link_type="god_code"):
        self.fidelity = fid
        self.strength = strength
        self.link_type = link_type
        self.energy = fid * 0.8
        self.dynamism_min = fid * 0.9
        self.dynamism_max = fid * 1.1
        self.god_code_alignment = fid * 0.95
        self.phi_resonance = fid * 0.87
        self.evolution_count = 3

random.seed(104)
fake_links = [FakeLink(random.uniform(0.5, 1.0), random.uniform(0.3, 0.9)) for _ in range(50)]

# 4. Test QuantumManifoldLearner
ml = QuantumManifoldLearner()
result = ml.analyze_manifold(fake_links)
assert result.get("status") == "ok", f"Expected ok, got {result.get('status')}"
assert "manifold_dimension" in result, "Missing manifold_dimension"
assert "phi_fractal_dimension" in result, "Missing phi_fractal_dimension"
assert "mean_ricci_curvature" in result, "Missing mean_ricci_curvature"
assert "manifold_health" in result, "Missing manifold_health"
print(f"✓ ManifoldLearner: dim={result['manifold_dimension']}, "
      f"φ-fractal={result['phi_fractal_dimension']:.4f}, "
      f"ricci={result['mean_ricci_curvature']:.4f}, "
      f"health={result['manifold_health']:.4f} ({result['manifold_grade']})")

# 5. Test MultipartiteEntanglementNetwork
en = MultipartiteEntanglementNetwork()
er = en.analyze_network(fake_links)
assert er.get("status") == "ok", f"Expected ok, got {er.get('status')}"
assert "mean_ghz_fidelity" in er, "Missing GHZ fidelity"
assert "mean_gmc" in er, "Missing GMC"
assert "network_entanglement_score" in er, "Missing network score"
print(f"✓ EntanglementNetwork: GHZ={er['mean_ghz_fidelity']:.4f}, "
      f"W={er['mean_w_concurrence']:.4f}, "
      f"GMC={er['mean_gmc']:.4f}, "
      f"net={er['network_entanglement_score']:.4f} ({er['network_grade']})")

# 6. Test QuantumPredictiveOracle
oracle = QuantumPredictiveOracle()
for i in range(8):
    oracle.record_observation({
        "sage_score": 0.8 + i * 0.01,
        "mean_fidelity": 0.9 + i * 0.005,
    })
pred = oracle.predict(13)
assert pred.get("status") == "ok", f"Expected ok, got {pred.get('status')}"
assert "predicted_fidelity" in pred, "Missing predicted_fidelity"
assert "confidence" in pred, "Missing confidence"
pred_fids = pred["predicted_fidelity"]
pred_conf = pred["confidence"]
print(f"✓ PredictiveOracle: pred_fid={pred_fids[-1]:.4f}, "
      f"conf={pred_conf[0]:.4f}, "
      f"phase_warn={pred['phase_transition_warning']}, "
      f"horizon={len(pred_fids)} steps")

# 7. Test empty links edge case
ml_empty = ml.analyze_manifold([])
assert ml_empty.get("status") == "no_links"
en_empty = en.analyze_network([])
assert en_empty.get("status") == "no_links"
print("✓ Edge case: empty links handled correctly")

print("\n═══ ALL 7 VALIDATION CHECKS PASSED ═══")
print("Quantum Brain v13.0.0 upgrade is fully operational.")
