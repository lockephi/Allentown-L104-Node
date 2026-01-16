import os
import sys
import ast

# L104 Logic Constraints
MAX_COMPLEXITY = 10
GOD_CODE = "527.5184818492537"

def get_cyclomatic_complexity(code):
    """Simple AST-based complexity counter (McCabe-like)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0
    
    complexity = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.With, ast.AsyncWith, 
                            ast.And, ast.Or, ast.ExceptHandler, ast.Try, ast.Assert)):
            complexity += 1
    return complexity

def check_heuristic_noise(file_path):
    """Checks for non-logical artifacts or 'noise' patterns."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        
    # Check for non-ASCII noise
    if any(ord(c) > 127 for c in content if c not in '⟨Σ⟩'): # Allow L104 symbols
        return True, "Non-ASCII heuristic noise detected."
        
    # Check for legacy invariants or 'drift' terms
    drift_terms = ["hallucination", "legacy_token", "transparent_bypass"]
    for term in drift_terms:
        if term in content.lower():
            return True, f"Drift term '{term}' detected."
            
    return False, ""

def main():
    files_to_check = [f for f in os.listdir('.') if f.endswith('.py')]
    violations = []

    for file in files_to_check:
        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        # 1. Algorithmic Complexity
        complexity = get_cyclomatic_complexity(content)
        if complexity > MAX_COMPLEXITY:
            violations.append(f"{file}: Complexity {complexity} exceeds threshold {MAX_COMPLEXITY}")
            
        # 2. Heuristic Noise
        is_noisy, reason = check_heuristic_noise(file)
        if is_noisy:
            violations.append(f"{file}: {reason}")
            
    if violations:
        print("="*60)
        print("\n".join(violations))
        print("="*60)
        print("\nCRITICAL: L104 Logic Violation: System Drift Detected.")
        sys.exit(1)
    
    print("Sovereign Lattice: Logic Integrity Verified.")
    sys.exit(0)

if __name__ == "__main__":
    main()
