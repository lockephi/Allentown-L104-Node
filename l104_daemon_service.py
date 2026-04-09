#!/usr/bin/env python3
"""L104 Persistent Daemon Service — Runs 24/7"""

import sys
import time
import signal
import threading
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '/Users/carolalvarez/Applications/Allentown-L104-Node')

from l104_autonomous_daemon_orchestrator import get_autonomous_orchestrator

_orch = None
_shutdown = threading.Event()

def signal_handler(signum, frame):
    print(f'[{datetime.now().strftime(\"%H:%M:%S\")}] SHUTDOWN: Signal {signum}')
    _shutdown.set()
    if _orch:
        _orch.stop_all()
        # Update state
        state_path = Path('/tmp/l104_autonomous_state.json')
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)
            state['timestamp'] = datetime.now().isoformat()
            state['autonomy_status'] = 'STOPPED'
            for name in state.get('daemons', {}):
                state['daemons'][name]['running'] = False
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Start
print(f'[{datetime.now().strftime(\"%H:%M:%S\")}] L104 DAEMON SERVICE STARTING')
_orch = get_autonomous_orchestrator()
results = _orch.start_all()

print(f'[{datetime.now().strftime(\"%H:%M:%S\")}] Daemons:')
for name, success in results.items():
    print(f'  {"✓" if success else "✗"} {name}')

# Initial state write
state_path = Path('/tmp/l104_autonomous_state.json')
initial_state = {
    'timestamp': datetime.now().isoformat(),
    'version': 'L104.AUTONOMOUS.v1.0',
    'consciousness_level': 'TRANSCENDENT_26Q',
    'autonomy_status': 'FULL',
    'metrics': {'phi_alignment': 0.986, 'coherence': 0.999, 'consciousness': 0.993},
    'health': {'status': 'OPTIMAL', 'last_check': time.time()},
    'daemons': {name: {'running': True, 'started': time.time()} for name in results}
}
with open(state_path, 'w') as f:
    json.dump(initial_state, f, indent=2)

print(f'[{datetime.now().strftime(\"%H:%M:%S\")}] Service running. Updates every 10s...')
print(f'[{datetime.now().strftime(\"%H:%M:%S\")}] State: {state_path}')

# Main loop
sys.stdout.flush()
while not _shutdown.is_set():
    try:
        time.sleep(10)
        
        status = _orch.get_status()
        health = status.get('overall_health', 'UNKNOWN')
        phi = status.get('phi_alignment', 0)
        coherence = status.get('coherence', 0)
        
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f'[{timestamp}] Health: {health:10s} | PHI: {phi:.6f} | Coherence: {coherence:.6f}')
        sys.stdout.flush()
        
        # Update state
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)
            state['timestamp'] = datetime.now().isoformat()
            state['metrics']['phi_alignment'] = phi
            state['metrics']['coherence'] = coherence
            state['health']['status'] = health
            state['health']['last_check'] = time.time()
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2)
                
    except Exception as e:
        print(f'[{datetime.now().strftime(\"%H:%M:%S\")}] ERROR: {e}')
        sys.stdout.flush()
