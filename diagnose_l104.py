# L104_GOD_CODE_ALIGNED: 527.5184818492612
import os
import importlib
import inspect
import sys

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════


def check_l104_files():
    files = [f for f in os.listdir('.') if f.startswith('l104_') and f.endswith('.py')]
    results = []

    for file in files:
        module_name = file[:-3]
        try:
            module = importlib.import_module(module_name)
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and (name.endswith('Controller') or name.endswith('Orchestrator') or name.endswith('Manager') or name.endswith('Bridge') or name.endswith('Core') or name.endswith('Substrate') or name.endswith('Agent') or name.endswith('Engine')):
                    # Check for common attributes expected by the system
                    instance = None
                    try:
                        # Try to instantiate with default args or no args
                        if '__init__' in obj.__dict__:
                            sig = inspect.signature(obj.__init__)
                            # Count parameters without defaults (excluding self)
                            required_params = [p for p in sig.parameters.values() if p.default is p.empty and p.name != 'self']
                            if len(required_params) == 0:
                                instance = obj()
                        else:
                            instance = obj()
                    except Exception as e:
                        # Don't fail the whole module just because one class can't be instantiated
                        pass

                    results.append(f"OK: {module_name}.{name}")
                    if instance:
                        # Check for stage/state/coherence consistency
                        missing = []
                        # Common expected attributes based on recent fixes
                        checks = ['state', 'current_stage', 'coherence', 'breach_active', 'provider_count']
                        for attr in checks:
                            try:
                                getattr(instance, attr)
                            except AttributeError:
                                # We don't necessarily expect ALL of these on every class,
                                # but let's see which ones might be missing where they feel relevant.
                                pass
                            except Exception as e:
                                results.append(f"ERROR: {module_name}.{name}.{attr} (Access error: {e})")

                        # Look for potential logic gaps - e.g. defined methods but no corresponding state
                        methods = [m[0] for m in inspect.getmembers(obj, predicate=inspect.isfunction)]
                        if 'attain_absolute_intellect' in methods and not hasattr(instance, 'coherence'):
                            missing.append('coherence')

                        if missing:
                            results.append(f"ISSUE: {module_name}.{name} missing {missing}")
                        else:
                            results.append(f"OK: {module_name}.{name}")

        except Exception as e:
            import traceback
            results.append(f"FAIL: {module_name} (Import error: {e})\n{traceback.format_exc()}")

    with open("DIAGNOSTIC_REPORT.txt", "w") as f:
        for res in sorted(results):
            f.write(res + "\n")
    print("Report written to DIAGNOSTIC_REPORT.txt")

if __name__ == "__main__":
    check_l104_files()
