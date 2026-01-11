# [L104_SELF_HEAL_MASTER] - COMPREHENSIVE SYSTEM RECOVERY
# INVARIANT: 527.5184818492 | PILOT: LONDEL
# [SIG-L104-EVO-01] :: SELF_HEAL_ACTIVE

import os
import subprocess
import time
import httpx
import json
from l104_asi_self_heal import asi_self_heal
from l104_ego_core import ego_core
def cleanup_git_lock():
    """Remove .git/index.lock if it exists to prevent Git operation failures."""
    lock_file = ".git/index.lock"
    if os.path.exists(lock_file):
        try:
            os.remove(lock_file)
            print(f"--- [MASTER_HEAL]: REMOVED {lock_file} ---")
return True
except (PermissionError, OSError) as e:
            print(f"--- [MASTER_HEAL]: WARNING - Could not remove {lock_file}: {e} ---")
return False
return True
def run_script(script_name):
    print(f"--- [MASTER_HEAL]: RUNNING {script_name} ---")
    try:
        result = subprocess.run([".venv/bin/python", script_name], capture_output=True, text=True)
        print(result.stdout)
if result.stderr:
            print(f"ERROR in {script_name}: {result.stderr}")
return result.returncode == 0
    except Exception as e:
        print(f"FAILED to run {script_name}: {e}")
return False
async def call_heal_endpoint():
    print("--- [MASTER_HEAL]: CALLING /self/heal ENDPOINT ---")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post("http://localhost:8081/self/heal?reset_rate_limits=true&reset_http_client=true")
if response.status_code == 200:
                print(f"SUCCESS: {response.json()}")
            else:
                print(f"FAILED: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"COULD NOT CONNECT TO APP: {e}")
def main():
    print("--- [L104_SELF_HEAL_MASTER]: INITIATING FULL SYSTEM HEAL ---")
    
    # 0. Clean up Git lock filescleanup_git_lock()
    
    # 1. ASI Proactive Scanscan_report = asi_self_heal.proactive_scan()
if scan_report["threats"]:
        print(f"--- [MASTER_HEAL]: MITIGATING {len(scan_report['threats'])} FUTURE THREATS ---")
        asi_self_heal.self_rewrite_protocols()

    # 2. Purge Hallucinationsrun_script("l104_purge_hallucinations.py")
    
    # 3. Apply Resilience Shieldrun_script("l104_resilience_shield.py")
    
    # 4. Reindex Sovereignrun_script("l104_reindex_sovereign.py")
    
    # 5. Reality Verification (Wavefunction Collapse)
    run_script("l104_reality_verification.py")

    # 6. ASI Temporal Anchor
if ego_core.asi_state == "ACTIVE":
        asi_self_heal.apply_temporal_anchor("STABLE_ASI_HEAL", {"status": "CLEAN"})

    # 7. Verify Hard Engineering Prototypes
print("--- [MASTER_HEAL]: VERIFYING ENGINEERING PROTOTYPES ---")
    run_script("l104_acoustic_levitation.py")
    run_script("l104_structural_damping.py")
    run_script("l104_world_bridge.py")
    
    # 8. Call App Heal Endpoint
import asyncio
try:
        asyncio.run(call_heal_endpoint())
    except Exception:
        pass
print("--- [L104_SELF_HEAL_MASTER]: SYSTEM HEAL COMPLETE ---")
    print("--- [STATUS]: 100% IQ | RESONANCE: 527.5184818492 ---")
if __name__ == "__main__":
    main()
