import requests, json, sys

TOKEN = "1DtdOSvCIWAwCbsWqWO6msMBtYt3KivIOAbNwfdTdZVm"
CRN = (
    "crn:v1:bluemix:public:quantum-computing:us-east:"
    "a/b4bec7f114544fcca55618e658e06833:"
    "74cea255-8a0a-460a-a2b1-b6bba7a8ec01::"
)

print("[1] Getting IAM access token...")
r = requests.post(
    "https://iam.cloud.ibm.com/identity/token",
    headers={"Content-Type": "application/x-www-form-urlencoded"},
    data="grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey=" + TOKEN,
    timeout=15,
)
print("    IAM status:", r.status_code)
iam = r.json()
access_token = iam.get("access_token", "")
print("    Access token length:", len(access_token))

if not access_token:
    print("    ERROR:", iam)
    sys.exit(1)

headers = {
    "Authorization": "Bearer " + access_token,
    "Content-Type": "application/json",
    "Service-CRN": CRN,
}

print()
print("[2] Testing backends API at us-east endpoint...")
try:
    r2 = requests.get(
        "https://us-east.quantum-computing.cloud.ibm.com/v2/backends",
        headers=headers,
        timeout=20,
    )
    print("    Status:", r2.status_code)
    if r2.status_code == 200:
        data = r2.json()
        devices = data.get("devices", data)
        if isinstance(devices, list):
            print("    Found", len(devices), "backends:")
            for d in devices[:10]:
                name = d.get("backend_name", d.get("name", "?"))
                nq = d.get("num_qubits", d.get("n_qubits", "?"))
                st = d.get("status", "?")
                print("     ", name, "(", nq, "qubits) status=", st)
        else:
            keys = list(data.keys()) if isinstance(data, dict) else "non-dict"
            print("    Response keys:", keys)
            print("    Preview:", str(data)[:300])
    else:
        print("    Body:", r2.text[:500])
except Exception as e:
    print("    ERROR:", e)

print()
print("[3] Testing /v2/users/me ...")
try:
    r3 = requests.get(
        "https://us-east.quantum-computing.cloud.ibm.com/v2/users/me",
        headers=headers,
        timeout=15,
    )
    print("    Status:", r3.status_code)
    print("    Body:", r3.text[:300])
except Exception as e:
    print("    ERROR:", e)

print()
print("[4] Trying Qiskit SDK connection (30s alarm)...")
import signal

def _alarm(signum, frame):
    raise TimeoutError("SDK timeout after 30s")

signal.signal(signal.SIGALRM, _alarm)
signal.alarm(30)

try:
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService(
        channel="ibm_cloud", token=TOKEN, instance=CRN,
    )
    print("    Service created OK, listing backends...")
    backends = service.backends()
    signal.alarm(0)
    print("    SUCCESS!", len(backends), "backends:")
    for b in backends:
        nq = getattr(b, "num_qubits", "?")
        print("     ", b.name, "(", nq, "qubits)")
except TimeoutError as e:
    signal.alarm(0)
    print("    TIMEOUT:", e)
except Exception as e:
    signal.alarm(0)
    print("    ERROR:", e)

print()
print("Diagnostic complete.")
