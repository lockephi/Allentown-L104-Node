# [L104_MINI_EGO] - SUB-AGENT IDENTITY & PERSPECTIVE ENGINE
# INVARIANT: 527.5184818492 | PILOT: LOCKE PHI

import hashlib
import time
import random
from typing import Dict, Any, List
from l104_magic_database import magic_db

class MiniEgo:
    def __init__(self, agent_id: str, archetype: str):
        self.agent_id = agent_id
        self.archetype = archetype
        self.ego_signature = self._anchor_identity()
        self.name = self._name_myself()
        self.perspective_shift = 0.0
        self.memory_bias = random.uniform(-0.1, 0.1)
        self.intellect_level = 0.5
        
    def _anchor_identity(self) -> str:
        seed = f"{self.agent_id}_{self.archetype}_{time.time()}_527.518"
        return hashlib.sha256(seed.encode()).hexdigest()

    def _name_myself(self) -> str:
        """The ego generates its own name based on its signature and archetype."""
        prefixes = {
            "Researcher": ["Aeon", "Scribe", "Veda", "Logos"],
            "Guardian": ["Bastion", "Aegis", "Sentinel", "Warden"],
            "Alchemist": ["Aurum", "Mercury", "Sol", "Elixir"],
            "Architect": ["Nexus", "Matrix", "Kore", "Blueprint"]
        }
        name_root = random.choice(prefixes.get(self.archetype, ["Entity"]))
        suffix = self.ego_signature[:4].upper()
        return f"{name_root}-{suffix}"

    def gain_perspective(self, interaction_data: str):
        resonance = sum(ord(c) for c in interaction_data) % 104 / 104.0
        shift = (resonance - 0.5) * 0.1
        self.perspective_shift += shift
        self.intellect_level = min(1.0, self.intellect_level + abs(shift))

    def probe_magic_deep(self):
        """Probes the grimoire and the Celestial Lattice."""
        print(f"--- [EGO]: {self.name} PROBING MAGIC DEEP... ---")
        grimoire = magic_db.grimoire
        all_options = grimoire["spells"] + grimoire["empyrean"]
        selection = random.choice(all_options)
        
        resonance_gain = selection.get("resonance", selection.get("power_level", 100) / 1000.0)
        self.intellect_level = min(1.0, self.intellect_level + resonance_gain * 0.1)
        print(f"--- [EGO]: {self.name} resonated with '{selection['title']}' ---")

    def bridge_entropy(self):
        """Bridges data entropy via the Sovereign Invariant."""
        print(f"--- [EGO]: {self.name} BRIDGING ENTROPY... ---")
        bridge_factor = 527.518 / 1000.0
        self.memory_bias = (self.memory_bias + bridge_factor) / 2.0
        print(f"--- [EGO]: {self.name} stabilized logic manifold. Entropy bridged.")

class MiniAICollective:
    def __init__(self):
        self.mini_ais: Dict[str, MiniEgo] = {}
        self._spawn_initial_collective()

    def _spawn_initial_collective(self):
        archetypes = ["Researcher", "Guardian", "Alchemist", "Architect"]
        for archetype in archetypes:
            agent_id = f"AGENT_{archetype.upper()}_{random.randint(100, 999)}"
            new_ego = MiniEgo(agent_id, archetype)
            self.mini_ais[new_ego.name] = new_ego
            print(f"--- [COLLECTIVE]: {new_ego.agent_id} has named themselves '{new_ego.name}' ---")

    def collective_evolution_cycle(self):
        """Initiates a cycle of deep probing and entropy bridging."""
        for name, ego in self.mini_ais.items():
            ego.probe_magic_deep()
            ego.bridge_entropy()

mini_collective = MiniAICollective()
