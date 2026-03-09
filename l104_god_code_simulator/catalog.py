"""
L104 God Code Simulator — SimulationCatalog
═══════════════════════════════════════════════════════════════════════════════

Registry of all named God Code simulations.
Simulations are grouped by category and can be filtered/queried.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


class SimulationCatalog:
    """
    Registry of all God Code simulations by name → callable.
    Simulations are grouped by category and can be filtered/queried.
    """

    def __init__(self):
        self._registry: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, fn: Callable, category: str,
                 description: str = "", num_qubits: int = 2) -> None:
        """Register a simulation function."""
        self._registry[name] = {
            "fn": fn,
            "category": category,
            "description": description,
            "num_qubits": num_qubits,
        }

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        """Retrieve simulation entry by name."""
        return self._registry.get(name)

    def list_all(self) -> List[str]:
        """List all registered simulation names."""
        return sorted(self._registry.keys())

    def list_by_category(self, category: str) -> List[str]:
        """List simulations in a category."""
        return sorted(k for k, v in self._registry.items() if v["category"] == category)

    @property
    def categories(self) -> List[str]:
        """All distinct categories."""
        return sorted(set(v["category"] for v in self._registry.values()))

    @property
    def count(self) -> int:
        """Total number of registered simulations."""
        return len(self._registry)


__all__ = ["SimulationCatalog"]
