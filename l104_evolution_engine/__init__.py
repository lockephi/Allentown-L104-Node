"""
L104 Evolution Engine Module

Auto-generated package initialization file.
Created to fix import structure.
"""

__version__ = "1.0.0"
__author__ = "L104 Autonomous System"
__package__ = "l104_evolution_engine"

print(f"[L104] {__package__} module loaded")

# Import evolution_engine from the module file (l104_evolution_engine.py)
# This handles the case where the package shadows the module
try:
    import importlib.util
    import os
    import sys

    # Find the module file (l104_evolution_engine.py in parent directory)
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    module_path = os.path.join(parent_dir, "l104_evolution_engine.py")

    if os.path.exists(module_path):
        # Load the module file directly using importlib
        spec = importlib.util.spec_from_file_location(
            "_l104_evolution_engine_impl", module_path
        )
        _evo_module = importlib.util.module_from_spec(spec)
        sys.modules["_l104_evolution_engine_impl"] = _evo_module
        spec.loader.exec_module(_evo_module)

        evolution_engine = _evo_module.evolution_engine
        EvolutionEngine = _evo_module.EvolutionEngine
    else:
        raise ImportError(f"Module file not found: {module_path}")

except Exception as e:
    print(f"[L104] Warning: Could not load evolution_engine: {e}")
    evolution_engine = None
    EvolutionEngine = None

# Export symbols
__all__ = ["evolution_engine", "EvolutionEngine"]
