#!/usr/bin/env python3
"""Get actual plan IDs via /plan endpoint and provision quantum instance."""
import requests, json, sys

TOKEN = "1DtdOSvCIWAwCbsWqWO6msMBtYt3KivIOAbNwfdTdZVm"

# Auth
r = requests.post("https://iam.cloud.ibm.com/identity/token", data={
    "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
    "apikey": TOKEN,
}, headers={"Accept": "application/json"}, timeout=15)
at = r.json()["access_token"]
H = {"Authorization": f"Bearer {at}", "Content-Type": "application/json"}
print("[OK] IAM auth")

# Get plans via the /plan endpoint that actually works
sid = "b6049020-80f4-11eb-a0f7-e35ec9b4054f"
r2 = requests.get(
    f"https://globalcatalog.cloud.ibm.com/api/v1/{sid}/plan?languages=en",
    timeout=15,
)
print(f"\n=== Plans (via /plan endpoint) ===")
print(f"Status: {r2.status_code}")
data = r2.json()
print(f"Count: {data.get('resource_count', '?')}")

plans = data.get("resources", [])
target_plan_id = None
for p in plans:
    name = p.get("name", "?")
    pid = p.get("id", "?")
    kind = p.get("kind", "?")
    display = p.get("overview_ui", {}).get("en", {}).get("display_name", name)
    print(f"\n  Plan: {display} ({name})")
    print(f"  ID: {pid}")
    print(f"  Kind: {kind}")

    # Check if this is open/free plan
    metadata = p.get("metadata", {})
    pricing = metadata.get("pricing", {})
    print(f"  Pricing: {json.dumps(pricing)[:200]}")

    if "open" in name.lower() or "lite" in name.lower() or "free" in name.lower():
        target_plan_id = pid
        print(f"  ★ This is the open/free plan!")

    # Get deployments for this plan
    r3 = requests.get(
        f"https://globalcatalog.cloud.ibm.com/api/v1/{pid}/deployment?languages=en",
        timeout=15,
    )
    deps = r3.json().get("resources", [])
    for d in deps:
        loc = d.get("metadata", {}).get("deployment", {}).get("location", "global")
        print(f"  Deployment: {d.get('name','?')} location={loc} id={d.get('id','?')}")

if not target_plan_id and plans:
    # Use the first plan
    target_plan_id = plans[0]["id"]
    print(f"\nUsing first plan: {plans[0].get('name')} ({target_plan_id})")

if not target_plan_id:
    print("\n[FAIL] No plans found")
    sys.exit(1)

# Get resource group
rg_r = requests.get(
    "https://resource-controller.cloud.ibm.com/v2/resource_groups",
    headers=H, timeout=10,
)
rg_id = rg_r.json()["resources"][0]["id"]
print(f"\n[OK] Resource group: {rg_id}")

# Provision
print(f"\n=== Creating quantum instance with plan {target_plan_id} ===")
r4 = requests.post(
    "https://resource-controller.cloud.ibm.com/v2/resource_instances",
    json={
        "name": "l104-quantum",
        "target": "us-east",
        "resource_group": rg_id,
        "resource_plan_id": target_plan_id,
    },
    headers=H, timeout=30,
)
print(f"Status: {r4.status_code}")
result = r4.json()
print(f"Response: {json.dumps(result, indent=2)[:500]}")

if r4.status_code in [200, 201]:
    crn = result.get("crn", "")
    print(f"\n[OK] Instance created! CRN: {crn}")

    # Connect with Qiskit
    print("\n=== Connecting with Qiskit ===")
    from qiskit_ibm_runtime import QiskitRuntimeService

    # Wait a moment for provisioning to complete
    import time
    print("Waiting 5s for provisioning...")
    time.sleep(5)

    service = QiskitRuntimeService(channel="ibm_cloud", token=TOKEN, instance=crn)
    backends = service.backends()
    print(f"[OK] Connected: {len(backends)} backends")
    for b in backends:
        st = b.status()
        print(f"  {b.name}: {b.num_qubits}q, pending={st.pending_jobs}")

    # Save account
    QiskitRuntimeService.save_account(
        channel="ibm_cloud", token=TOKEN, instance=crn,
        overwrite=True, set_as_default=True,
    )
    print(f"\n[OK] Saved as default account")
else:
    # Try each plan with different targets
    print("\nFirst attempt failed. Trying deployments as targets...")
    for p in plans:
        pid = p["id"]
        r5 = requests.get(
            f"https://globalcatalog.cloud.ibm.com/api/v1/{pid}/deployment",
            timeout=15,
        )
        for d in r5.json().get("resources", []):
            did = d["id"]
            dname = d.get("name", "?")
            loc = d.get("metadata", {}).get("deployment", {}).get("location", "?")
            print(f"  Trying plan={pid} deployment={did} ({dname}, {loc})...")
            r6 = requests.post(
                "https://resource-controller.cloud.ibm.com/v2/resource_instances",
                json={
                    "name": "l104-quantum",
                    "target": did,
                    "resource_group": rg_id,
                    "resource_plan_id": pid,
                },
                headers=H, timeout=30,
            )
            print(f"    Status: {r6.status_code}")
            if r6.status_code in [200, 201]:
                crn = r6.json()["crn"]
                print(f"\n[OK] Created! CRN: {crn}")
                break
            else:
                print(f"    Error: {r6.text[:200]}")
