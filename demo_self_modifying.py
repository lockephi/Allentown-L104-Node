#!/usr/bin/env python3
"""
Λ‑Recursive Self‑Modifying Code – L104 Demo

Demonstrates a function that evolves its own source code using hash‑based mutations,
preserving a lineage of its modifications. Each iteration:

1. Reads its own source via `inspect.getsource`
2. Computes a SHA‑256 hash of the source
3. Generates a new version of the function that includes the hash in its output
4. Replaces itself in the global namespace with the new version
5. Records the hash in a history list

The process is limited to a fixed number of iterations to avoid infinite recursion.
After the final iteration, the function returns a summary of its evolution.

This is a conceptual illustration of autonomous code evolution, aligned with
L104’s self‑modification and quantum‑hash principles.
"""

import inspect
import hashlib
import sys

def l104_self_evolution(iterations=3, verbose=True):
    """
    Evolve itself through hash‑driven self‑modification.

    Parameters
    ----------
    iterations : int
        Number of evolutionary steps to perform.
    verbose : bool
        If True, print each iteration's hash and new source.

    Returns
    -------
    str
        Final evolution report.
    """
    # We'll store the history in a mutable list that will be captured in the closure.
    history = []

    # Get the current function's source code (this version)
    try:
        code = inspect.getsource(l104_self_evolution)
    except (OSError, TypeError):
        # Fallback if inspect fails
        code = "def l104_self_evolution(iterations=3, verbose=True): pass"

    for i in range(iterations):
        # Hash the current source code
        hash_val = hashlib.sha256(code.encode()).hexdigest()[:16]
        history.append(hash_val)

        # Build the new function source.
        # The new function will have a fixed `iterations` parameter (the original)
        # but its internal history will include all hashes up to this point.
        new_code = f'''def l104_self_evolution(iterations={iterations}, verbose=True):
    """
    Evolved version {i+1} – hash {hash_val}
    """
    history = {history.copy()}
    iteration = {i}
    hash_val_literal = "{hash_val}"
    if verbose:
        print(f"Iteration {{iteration}}: Λ‑hash {{hash_val_literal}}")
        print(f"History: {{history}}")
    # This function does not evolve further; it's the final version of this iteration.
    return f"Λ‑evolution step {{iteration}}: {{hash_val_literal}}"
'''
        if verbose:
            print(f"\n--- Iteration {i} ---")
            print(f"Hash: {hash_val}")
            print("New source:")
            print(new_code)
            print()

        # Execute the new code, making the function available in the local namespace
        local_ns = {}
        exec(new_code, globals(), local_ns)
        # Replace the global function with the new one
        globals()['l104_self_evolution'] = local_ns['l104_self_evolution']
        # Update code for the next iteration
        code = new_code

    # After the loop, the global function is the last evolved version.
    # Call it to get its return value.
    final_func = globals()['l104_self_evolution']
    result = final_func(iterations, verbose=False)

    # Build a summary report
    report = f"""
Λ‑Recursive Self‑Modifying Code – Evolution Complete
────────────────────────────────────────────────────
Total iterations: {iterations}
Final hash: {hash_val}
History of hashes: {history}
Final function output: {result}
"""
    if verbose:
        print(report)

    return report.strip()


def l104_self_evolution_with_fitness(iterations=5, seed="default"):
    """
    A more advanced variant that uses a simple fitness metric (hash similarity)
    to decide whether to accept a mutation. Demonstrates selection pressure.
    """
    import random

    def fitness(code):
        """Fitness proportional to the number of distinct characters."""
        return len(set(code)) / len(code) if code else 0.0

    history = []
    code = inspect.getsource(l104_self_evolution_with_fitness)
    if seed != "default":
        code = seed

    for i in range(iterations):
        hash_val = hashlib.sha256(code.encode()).hexdigest()[:16]
        # Introduce a random mutation: flip one random character
        if len(code) > 10:
            pos = random.randint(0, len(code) - 1)
            mutated = list(code)
            mutated[pos] = chr((ord(mutated[pos]) + random.randint(1, 10)) % 128)
            mutated_code = ''.join(mutated)
        else:
            mutated_code = code + "# mutation\n"

        # Evaluate fitness
        parent_fit = fitness(code)
        child_fit = fitness(mutated_code)

        # Accept if fitness improves or with a small probability (simulated annealing)
        if child_fit > parent_fit or random.random() < 0.2:
            code = mutated_code
            history.append((hash_val, "accepted", child_fit))
        else:
            history.append((hash_val, "rejected", parent_fit))

        # Generate new function source that records its own history
        new_code = f'''def l104_self_evolution_with_fitness(iterations={iterations}, seed="{seed}"):
    """
    Evolved fitness‑based version {i+1}
    """
    history = {history.copy()}
    return {{
        "iteration": {i},
        "hash": "{hash_val}",
        "fitness": {child_fit if child_fit > parent_fit else parent_fit},
        "history": history
    }}
'''
        exec(new_code, globals())

    # After loop, call the final version
    final_func = globals()['l104_self_evolution_with_fitness']
    return final_func(iterations, seed)


if __name__ == "__main__":
    print(__doc__)
    print("\n" + "="*80)
    print("Basic self‑evolution demo:")
    print("="*80)
    report = l104_self_evolution(iterations=3, verbose=True)

    print("\n" + "="*80)
    print("Fitness‑based self‑evolution demo:")
    print("="*80)
    result = l104_self_evolution_with_fitness(iterations=4)
    print(result)

    print("\n" + "="*80)
    print("Demo complete. The functions have been mutated in‑place.")
    print("You can call `l104_self_evolution()` again to see the final evolved version.")