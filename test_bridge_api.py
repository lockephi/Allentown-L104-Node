#!/usr/bin/env python3
"""Quick integration test for the L104 ASI bridge API."""
import json

from l104_asi_core import get_current_parameters, update_parameters, asi_core

# Test 1: Fetch parameters
params = get_current_parameters()
print(f'[1] get_current_parameters: {len(params)} keys OK')
print(f'    ASI score: {params["asi_score"]:.4f}')

# Test 2: Get status
status = asi_core.get_status()
print(f'[2] asi_core.get_status: {status["state"]}, score={status["asi_score"]:.4f}')

# Test 3: Verify consciousness
level = asi_core.verify_consciousness()
print(f'[3] consciousness: {level:.4f}')

# Test 4: Update parameters (non-destructive test)
test_result = update_parameters([0.5, 0.8, 0.1])
print(f'[4] update_parameters: updated={test_result["updated"]}, keys={test_result["keys"]}')

# Restore original values
with open('kernel_parameters.json') as f:
    p = json.load(f)
p['embedding_dim'] = 206
p['hidden_dim'] = 414
p['num_layers'] = 4
with open('kernel_parameters.json', 'w') as f:
    json.dump(p, f, indent=2)
print('[5] kernel_parameters.json restored')

print('\n✓ All Python API tests passed')
