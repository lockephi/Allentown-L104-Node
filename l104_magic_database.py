# [L104_MAGIC_DATABASE] - SOVEREIGN GRIMOIRE
# INVARIANT: 527.5184818492 | PILOT: LONDEL

class MagicDatabase:
    """
    Experimental database for storing 'magical' resonance patterns and spells.
    Bridging the gap between code and alchemy.
    """
    def __init__(self):
        self.grimoire = {
            "spells": [
                {
                    "title": "Aura of Clarity",
                    "power_level": 104,
                    "mechanic": "REDUCE_ENTROPY",
                    "category": "spells"
                },
                {
                    "title": "Quantum Surge",
                    "power_level": 416,
                    "mechanic": "BOOST_RESONANCE",
                    "category": "spells"
                },
                {
                    "title": "Lattice Shield",
                    "power_level": 527,
                    "mechanic": "SHIELD_LOGIC",
                    "category": "spells"
                },
                {
                    "title": "Temporal Traverse",
                    "power_level": 1992,
                    "mechanic": "SLOW_ASCENSION",
                    "category": "spells",
                    "description": "Traversing magic slowly, respecting the cadence of time."
                }
            ],
            "artifacts": [
                {"title": "The Golden Phi", "power": 1.618},
                {"title": "The 1992 Resonance", "power": 0.527, "owner": "Locke Phi"}
            ]
        }

    def get_all_by_category(self, category: str):
        return self.grimoire.get(category, [])

magic_db = MagicDatabase()
