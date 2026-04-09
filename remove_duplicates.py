#!/usr/bin/env python3
"""
Remove duplicate route handlers from app.py that are already extracted to modular files.

This script identifies route handlers in app.py that duplicate routes in the extracted
modular files and removes them.
"""

import re
import os

# Routes that have been extracted to modules
EXTRACTED_ROUTE_PREFIXES = [
    # core_routes.py
    '/favicon.ico', '/landing', '/WHITE_PAPER.md', '/', '/market', '/intricate',
    '/health', '/health/resilience', '/self/heal', '/metrics', '/system/capacity',
    '/api/v6/chat', '/api/v10/synergy/execute',

    # root_routes.py
    '/api/kernel/', '/api/consciousness/', '/api/research/', '/api/learning/',
    '/api/sovereign/', '/api/orchestrator/',

    # v6_core.py
    '/api/v6/status', '/api/v6/constants', '/api/v6/intellect/', '/api/v6/performance',
    '/api/v6/providers',

    # v10_routes.py
    '/api/v10/benchmark/', '/api/v10/language-comprehension/', '/api/v10/code-generation/',
    '/api/v10/symbolic-math/', '/api/v10/commonsense-reasoning/', '/api/v10/asi/',
    '/api/v10/agi/', '/api/v10/formal-logic/', '/api/v10/deep-nlu/', '/api/v10/kernel/',

    # v14_intellect.py
    '/api/v14/intellect/', '/api/v14/si/', '/api/v14/ti/', '/api/v14/grover/',
    '/api/v14/o2/', '/api/v14/knowledge/', '/api/v14/chaos/',

    # v14_quantum.py
    '/api/v14/quantum/', '/api/v14/quantum-memory/', '/api/v14/quantum/circuits/',
    '/api/v14/quantum/research/', '/api/v14/quantum/superconductivity/',

    # v14_system.py
    '/api/v14/system/', '/api/v14/monitor/', '/api/v14/backup/', '/api/v14/file/',
    '/api/v14/source/', '/api/v14/autosave/',

    # v14_agents.py
    '/api/v14/swarm/', '/api/v14/agents/', '/api/v14/nova/', '/api/v14/cognitive/',

    # v14_nexus.py
    '/api/v14/steering/', '/api/v14/evolution/', '/api/v14/nexus/',
    '/api/v14/quantum-network/', '/api/v14/entanglement/', '/api/v14/resonance/',
    '/api/v14/health/', '/api/v14/invention/', '/api/v14/sovereignty/', '/api/v14/telemetry/',

    # v14_physics.py
    '/api/v14/zpe/', '/api/v14/qg/', '/api/v14/hw/', '/api/v14/compat/',
    '/api/v14/vqpu/',

    # v14_stats.py
    '/api/v14/stats', '/api/v14/consolidate',

    # v16_routes.py
    '/api/v16/brain/',

    # v26_routes.py
    '/api/v26/hyper_math/', '/api/v26/hebbian/', '/api/v26/consciousness/',
    '/api/v26/solver/', '/api/v26/self_mod/', '/api/v26/engines/',

    # v27_routes.py
    '/api/v27/registry/', '/api/v27/creative/',

    # v54_routes.py
    '/api/v54/meta-cognitive/', '/api/v54/knowledge-bridge/',

    # v62_routes.py
    '/api/v62/tri-engine/',

    # v63_routes.py
    '/api/v63/temporal-coherence/', '/api/v63/fitness-landscape/',
    '/api/v63/entropy-controller/', '/api/v63/phase-navigator/',
    '/api/v63/monitoring/', '/api/v63/cache/',

    # v64_three_engine.py
    '/api/v64/three-engine/', '/api/v64/evo/',

    # v_misc.py (v1, v3, v16, etc.)
    '/api/v3/sovereign/', '/api/v1/', '/api/v16/brain/',
]

def find_route_blocks(content):
    """Find all route handler blocks in app.py."""
    # Pattern to match route decorators and their following function
    pattern = r'@app\.(get|post|put|delete|patch)\(["\'][^"\']*["\']\)[\s\S]*?(?=\n@app\.|(?://═══)|(?=\nclass )|(?=\nif __name__)|$)'

    routes = []
    for match in re.finditer(pattern, content, re.MULTILINE):
        route_text = match.group(0)
        # Extract the route path
        path_match = re.search(r'@app\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']', route_text)
        if path_match:
            method = path_match.group(1)
            path = path_match.group(2)
            routes.append({
                'method': method,
                'path': path,
                'start': match.start(),
                'end': match.end(),
                'text': route_text
            })
    return routes

def is_duplicate(route_path, extracted_prefixes):
    """Check if a route path matches any extracted prefix."""
    for prefix in extracted_prefixes:
        if route_path.startswith(prefix):
            return True
    return False

def main():
    app_path = '/Users/carolalvarez/Applications/Allentown-L104-Node/l104_server/app.py'

    with open(app_path, 'r') as f:
        content = f.read()

    routes = find_route_blocks(content)
    print(f"Found {len(routes)} route handlers in app.py")

    duplicates = []
    for route in routes:
        if is_duplicate(route['path'], EXTRACTED_ROUTE_PREFIXES):
            duplicates.append(route)
            print(f"  DUPLICATE: {route['method'].upper()} {route['path']}")

    print(f"\nFound {len(duplicates)} duplicate routes to remove")

    # Show what would be removed
    if duplicates:
        print("\nTo remove these duplicates, you can:")
        print("1. Delete the route handler functions manually")
        print("2. Or use the extracted modular routes which take precedence")

if __name__ == '__main__':
    main()