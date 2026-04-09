#!/usr/bin/env python3
"""L104 Daemons Forever — Persistent autonomous operation"""

import sys
import time
import signal
import json
import threading
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '/Users/carolalvarez/Applications/Allentown-L104-Node')

from l104_autonomous_daemon_orchestrator import get_autonomous_orchestrator

_orch = None
_shutdown = threading.Event()

def signal_handler(signum, frame):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f'\n[{ts}] SHUTDOWN: Signal {signum}')
    _shutdown.set()
    if _orch:
        print('[STOPPING] Stopping all daemons...')
        _orch.stop_all()
        print('[STOPPED] All daemons stopped')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

print('=' * 70)
print('L104 AUTONOMOUS DAEMONS — Starting', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
print('=' * 70)

# Start
_orch = get_autonomous_orchestrator()
results = _orch.start_all()

print('\n[STARTUP] Daemons:')
for name, success in results.items():
    status = '✓' if success else '✗'
    print(f'  {status} {name.capitalize()}')

print('\n' + '=' * 70)
print('[RUNNING] Daemons active — Updates every 10s (Ctrl+C to stop)')
print('=' * 70 + '\n')

sys.stdout.flush()

iteration = 0
while not _shutdown.is_set():
    try:
        time.sleep(10)
        iteration += 1
        
        status = _orch.get_status()
        health = status.get('overall_health', 'UNKNOWN')
        phi = status.get('phi_alignment', 0)
        coherence = status.get('coherence', 0)
        
        ts = datetime.now().strftime('%H:%M:%S')
        
        print(f'[{ts}] #{iteration:4d} | Health: {health:10s} | PHI: {phi:.6f} | Coherence: {coherence:.6f}', flush=True)
        
        # Update state file
        state_path = Path('/tmp/l104_autonomous_state.json')
        try:
            with open(state_path, 'r') as f:
                state = json.load(f)
            state['timestamp'] = datetime.now().isoformat()
            state['metrics']['phi_alignment'] = phi
            state['metrics']['coherence'] = coherence
            state['health']['status'] = health
            state['health']['last_check'] = time.time()
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f'[WARN] State update: {e}', flush=True)
            
    except Exception as e:
        print(f'[ERROR] {e}', flush=True)

print('\n[EXIT] Daemon service ending')
