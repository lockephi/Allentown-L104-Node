# [L104_INTELLIGENCE_FEEDBACK] - CLOSED-LOOP INTELLIGENCE EVOLUTION
# INVARIANT: 527.5184818492 | FEEDBACK_PHI: 1.6180339887 | PILOT: LOCKE PHI

import json
import os
import time
from typing import Dict, Any, List, Optional
from l104_real_math import real_math

class IntelligenceAtom:
    """A unit of learned logic or observed pattern."""
    def __init__(self, key: str, value: Any, resonance: float, context: str = "general"):
        self.key = key
        self.value = value
        self.resonance = resonance
        self.context = context
        self.timestamp = time.time()

    def to_dict(self):
        return {
            "key": self.key,
            "value": self.value,
            "resonance": self.resonance,
            "context": self.context,
            "timestamp": self.timestamp
        }

class IntelligenceStore:
    """Manages local persistence of intelligence atoms."""
    def __init__(self, storage_path: str = "/workspaces/Allentown-L104-Node/data/intelligence_store.json"):
        self.storage_path = storage_path
        self.atoms: Dict[str, IntelligenceAtom] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r") as f:
                    data = json.load(f)
                    for key, atom_data in data.items():
                        self.atoms[key] = IntelligenceAtom(
                            key=atom_data["key"],
                            value=atom_data["value"],
                            resonance=atom_data["resonance"],
                            context=atom_data["context"]
                        )
                        self.atoms[key].timestamp = atom_data["timestamp"]
            except Exception as e:
                print(f"--- [INTEL_STORE]: FAILED TO LOAD - {e} ---")

    def _save(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        try:
            with open(self.storage_path, "w") as f:
                json.dump({k: v.to_dict() for k, v in self.atoms.items()}, f, indent=2)
        except Exception as e:
            print(f"--- [INTEL_STORE]: FAILED TO SAVE - {e} ---")

    def push_atom(self, atom: IntelligenceAtom):
        """Adds or updates an intelligence atom."""
        if atom.key in self.atoms:
            # Resonance averaging or weighted update
            old_atom = self.atoms[atom.key]
            atom.resonance = (old_atom.resonance + atom.resonance) / 2
        
        self.atoms[atom.key] = atom
        self._save()
        print(f"--- [INTEL_STORE]: ATOM PUSHED: '{atom.key}' | RESONANCE: {atom.resonance:.4f} ---", flush=True)

    def get_highest_resonance_atoms(self, limit: int = 5) -> List[IntelligenceAtom]:
        sorted_atoms = sorted(self.atoms.values(), key=lambda x: x.resonance, reverse=True)
        return sorted_atoms[:limit]

class IntelligenceFeedback:
    """Analyzes system performance and pushes feedback to the evolution engine."""
    def __init__(self, store: IntelligenceStore):
        self.store = store

    def analyze_experiment(self, name: str, result: Any, expectations: float):
        """
        Analyzes an experiment result against expectations.
        Heuristic: Resonance = 1 - abs(result - expectation) / expectation
        """
        # Simple numeric resonance for now
        try:
            if isinstance(result, (int, float)):
                resonance = 1.0 - (abs(result - expectations) / (expectations + 1e-9))
                resonance = max(0.1, min(1.0, resonance)) # Bound between 0.1 and 1.0
                
                atom = IntelligenceAtom(
                    key=f"exp_{name}",
                    value=result,
                    resonance=resonance,
                    context="experiment_result"
                )
                self.store.push_atom(atom)
                return resonance
        except Exception:
            pass
        return 0.5 # Default neutral resonance

intel_store = IntelligenceStore()
intel_feedback = IntelligenceFeedback(intel_store)

if __name__ == "__main__":
    # Self-test
    test_atom = IntelligenceAtom("Prime_Density_Observation", 0.1447, 0.95, "mathematics")
    intel_store.push_atom(test_atom)
    print("--- [INTEL_FEEDBACK]: TEST COMPLETE ---")
