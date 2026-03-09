"""Test the activation chain: Intellect → AGI → ASI."""
import sys
import os

print("=== Test 1: Import l104_intellect ===")
from l104_intellect import local_intellect
print(f"  local_intellect type: {type(local_intellect).__name__}")

print("\n=== Test 2: Import l104_agi ===")
from l104_agi import agi_core
print(f"  agi_core type: {type(agi_core).__name__}")
print(f"  agi_core.is_ready: {agi_core.is_ready}")

print("\n=== Test 3: Import l104_asi ===")
from l104_asi import asi_core
print(f"  asi_core type: {type(asi_core).__name__}")
print(f"  asi_core.is_ready: {asi_core.is_ready}")

print("\n=== Test 4: Full activation chain ===")
chain = asi_core.ensure_activation_chain()
print(f"  Chain: {chain['chain']}")
print(f"  All ready: {chain['all_ready']}")
print(f"  Degraded: {chain['degraded_links']}")
for link_name, link_info in chain['links'].items():
    print(f"  {link_name}: connected={link_info['connected']}, ready={link_info['ready']}")

print("\n=== Test 5: Singleton integrity ===")
# The critical test: local_intellect.get_agi_core() should return the SAME singleton
li_agi = local_intellect.get_agi_core()
print(f"  local_intellect.get_agi_core() is agi_core: {li_agi is agi_core}")
print(f"  id(li_agi): {id(li_agi)}")
print(f"  id(agi_core): {id(agi_core)}")
assert li_agi is agi_core, "CRITICAL: get_agi_core() created duplicate — singleton integrity broken!"

print("\n=== Test 6: AGI upstream chain ===")
upstream = agi_core.ensure_upstream_chain()
print(f"  AGI upstream all_ready: {upstream['all_ready']}")
for link_name, link_info in upstream['links'].items():
    print(f"  {link_name}: connected={link_info['connected']}, ready={link_info['ready']}")

print("\n=== Test 7: AGI composite from ASI ===")
agi_score = asi_core.agi_composite_score()
print(f"  AGI composite score via ASI: {agi_score:.6f}")

print("\n=== Test 8: Readiness properties ===")
# Access via lazy proxy for intellect
print(f"  Intellect is_ready: {local_intellect.is_ready}")
print(f"  AGI is_ready: {agi_core.is_ready}")
print(f"  ASI is_ready: {asi_core.is_ready}")

print("\n" + "="*60)
print("  ALL ACTIVATION CHAIN TESTS PASSED")
print("="*60)
