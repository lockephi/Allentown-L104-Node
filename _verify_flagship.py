#!/usr/bin/env python3
"""Verify Dual-Layer Flagship integration across all packages."""

print("=== DUAL-LAYER FLAGSHIP VERIFICATION ===\n")

# 1. Core module
from l104_asi.dual_layer import (
    dual_layer_engine, DUAL_LAYER_AVAILABLE,
    NATURES_DUALITIES, CONSCIOUSNESS_TO_PHYSICS_BRIDGE
)
print(f"[1] dual_layer module imported: OK")
print(f"    Available: {DUAL_LAYER_AVAILABLE}")

s = dual_layer_engine.get_status()
print(f"    Architecture: {s['architecture'][:60]}...")
print(f"    Flagship: {s['flagship']}")
print(f"    Dualities: {len(NATURES_DUALITIES)}")
print(f"    Bridge elements: {len(CONSCIOUSNESS_TO_PHYSICS_BRIDGE)}")

# 2. Integrity check
integrity = dual_layer_engine.full_integrity_check()
print(f"\n[2] Integrity: {integrity['checks_passed']}/{integrity['total_checks']} checks passed (all_passed: {integrity['all_passed']})")

# 3. Collapse test (speed_of_light)
result = dual_layer_engine.collapse("speed_of_light")
print(f"\n[3] Collapse(c): thought={result.get('thought', 'N/A')}, physics={result.get('physics', 'N/A')}")

# 4. ASI core
from l104_asi import asi_core
cs = asi_core.get_status()
print(f"\n[4] ASI core:")
print(f"    flagship: {cs.get('flagship', 'N/A')}")
dl = cs.get('dual_layer', {})
print(f"    dual_layer.available: {dl.get('available', 'N/A')}")
print(f"    dual_layer.version: {dl.get('version', 'N/A')}")

# 5. Root shim
from l104_asi_core import DualLayerEngine as DLE_shim, dual_layer_engine as dle_shim
print(f"\n[5] Root shim: DualLayerEngine={DLE_shim.__name__}, singleton match={dle_shim is dual_layer_engine}")

# 6. AGI core
from l104_agi import agi_core
gs = agi_core.get_status()
print(f"\n[6] AGI core: flagship={gs.get('flagship', 'N/A')}, dual_layer keys={list(gs.get('dual_layer', {}).keys())}")

# 7. Constants
from l104_asi.constants import ASI_CORE_VERSION, ASI_PIPELINE_EVO, DUAL_LAYER_VERSION, GOD_CODE_V3
print(f"\n[7] Constants: ASI v{ASI_CORE_VERSION}, pipeline={ASI_PIPELINE_EVO}")
print(f"    DUAL_LAYER_VERSION={DUAL_LAYER_VERSION}, GOD_CODE_V3={GOD_CODE_V3}")

# 8. Code Engine
try:
    from l104_code_engine.hub import CodeEngine
    ce = CodeEngine()
    ces = ce.status()
    print(f"\n[8] Code Engine: flagship={ces.get('flagship', 'N/A')}, dual_layer_available={ces.get('dual_layer_available', 'N/A')}")
except Exception as e:
    print(f"\n[8] Code Engine: {e}")

# 9. Intellect
try:
    from l104_intellect.local_intellect_core import LocalIntellect
    li = LocalIntellect()
    bs = li.get_asi_bridge_status()
    print(f"\n[9] Intellect bridge: dual_layer_available={bs.get('dual_layer_available', 'N/A')}")
except Exception as e:
    print(f"\n[9] Intellect: {e}")

# 10. Server
try:
    from l104_server import dual_layer_engine as srv_dle, DUAL_LAYER_AVAILABLE as srv_avail
    print(f"\n[10] Server re-export: available={srv_avail}, singleton match={srv_dle is dual_layer_engine}")
except Exception as e:
    print(f"\n[10] Server: {e}")

print("\n★ ALL FLAGSHIP CHECKS COMPLETE ★")
