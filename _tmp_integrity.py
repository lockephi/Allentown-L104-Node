#!/usr/bin/env python3
"""Quick integrity check inspection."""
import json
from l104_asi.dual_layer import DualLayerEngine
dle = DualLayerEngine()
ic = dle.full_integrity_check(force=True)
for layer_name in ('thought_layer', 'physics_layer', 'bridge'):
    layer = ic.get(layer_name, {})
    checks = layer.get('checks', {})
    layer_pass = layer.get('all_passed', 'N/A')
    print(f'\n=== {layer_name} (all_passed={layer_pass}) ===')
    if isinstance(checks, dict):
        for check_name, check_data in checks.items():
            passed = check_data.get('passed', check_data.get('pass', check_data.get('ok', '?')))
            sym = 'PASS' if passed else 'FAIL'
            print(f'  [{sym}] {check_name}: {json.dumps(check_data, default=str)[:300]}')
    elif isinstance(checks, list):
        for chk in checks:
            print(f'  - {json.dumps(chk, default=str)[:300]}')
print(f'\nOverall: {ic.get("all_passed")}, {ic.get("checks_passed")}/{ic.get("total_checks")}')
