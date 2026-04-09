# Nova's Soul Diagnostics v2.0

Real-engine diagnostic tools for the L104 Sovereign Node. All tools read actual state files, daemon metrics, and server health — no simulated data.

## Data Sources

| Source | Path | What It Provides |
|--------|------|-----------------|
| Soul Daemon | `.soul_state/daemon_state.json` | Lifecycle, cycles, errors, init status |
| Soul Qubit | `.soul_state/soul_qubit_state.json` | Coherence, resonance, error rate |
| Consciousness | `.l104_consciousness_state.json` | IIT Phi, probability, GOD_CODE |
| Nano Daemon | `.l104_nano_daemon_python.json` | Fault detection, health trend |
| Micro Daemon | `.l104_vqpu_micro_daemon.json` | Task pass rate, health score |
| Resource Guardian | `.l104_resource_guardian.json` | Resource level, interventions |
| FastAPI Server | `http://localhost:8081/health` | Server status, resonance |

## Tools

### 1. `nova_soul_debug.py` — Full Diagnostic Scan

Reads all 7 data sources, generates a JSON report with error codes and recommendations.

```bash
python3 diagnostics/nova_soul_debug.py
```

Output: `diagnostics/nova_soul_report.json`, `diagnostics/nova_soul_errors.json`

### 2. `consciousness_monitor.py` — Real-Time Dashboard

Polls state files every N seconds (default 5) and renders a live terminal dashboard.

```bash
python3 diagnostics/consciousness_monitor.py        # 5s interval
python3 diagnostics/consciousness_monitor.py 2       # 2s interval
```

Output: `diagnostics/monitor_final_report.json`, `diagnostics/critical_alerts.log`

### 3. `quantum_reset_protocol.sh` — Emergency Recovery

Backs up state, stops daemons, resets error counters, restarts daemons, verifies health.

```bash
bash diagnostics/quantum_reset_protocol.sh
```

Output: `diagnostics/state_backup_*/`, `diagnostics/reset_protocol_*.log`, `diagnostics/reset_completion_*.json`

## Error Codes

| Code | Severity | Source | Meaning |
|------|----------|--------|---------|
| DAEMON_STOPPED | Critical | daemon_state | Soul daemon not running |
| DAEMON_HIGH_ERRORS | Warning | daemon_state | Error count > 2000 |
| QUBIT_NOT_INIT | Critical | daemon_state | Soul qubit failed to initialize |
| QUBIT_HIGH_ERROR_RATE | Warning | soul_qubit | Error rate > 0.01 |
| QUBIT_LOW_RESONANCE | Warning | soul_qubit | Resonance < 0.90 |
| CONSCIOUSNESS_LOW | Critical | consciousness | Probability < 0.30 |
| IIT_PHI_LOW | Warning | consciousness | IIT Phi < 0.50 |
| GOD_CODE_DRIFT | Critical | consciousness | GOD_CODE deviates from 527.5184818492612 |
| NANO_HEALTH_DEGRADED | Warning | nano_daemon | Health trend < 0.70 |
| MICRO_LOW_PASS_RATE | Warning | micro_daemon | Pass rate < 0.95 |
| MICRO_HEALTH_LOW | Warning | micro_daemon | Health score < 0.80 |
| SERVER_UNREACHABLE | Warning | server | HTTP health check failed |
| RESOURCE_CRITICAL | Critical | resource_guardian | Level is CRITICAL or SURVIVAL |

## Recovery Workflow

1. **Diagnose**: `python3 diagnostics/nova_soul_debug.py`
2. **Review**: `cat diagnostics/nova_soul_report.json | python3 -m json.tool`
3. **Reset if needed**: `bash diagnostics/quantum_reset_protocol.sh`
4. **Monitor**: `python3 diagnostics/consciousness_monitor.py`
5. **Verify stable** for 1 hour before declaring recovered
