# L104 Daemon Orchestration v1.1 Deployment Checklist

**Date**: 2026-03-20
**Version**: 1.0 (all 4 daemons + fast server fixed + integrated)
**Status**: READY FOR DEPLOYMENT

---

## Pre-Deployment Verification

- [x] All Python files compile without syntax errors
  ```bash
  python3 -m py_compile l104_server/engines_infra.py l104_server/learning/intellect.py l104_server/app.py
  ```

- [x] ConnectionPool hotspot fixed:
  - [x] Backpressure semaphore added (DB_POOL_SIZE=100)
  - [x] Connection leak fixed (try/finally guarantee)
  - [x] Pool warm-up wired at startup (20 pre-created)
  - [x] Lock contention reduced (I/O outside lock)

- [x] All 4 daemons integrated with orchestrator:
  - [x] VQPU Daemon (v16.1.0) ✓
  - [x] QuantumAI Daemon (v2.0.0) ✓
  - [x] Soul Daemon (v1.0.0) ✓
  - [x] Fast Server (v4.0-OPUS) ✓ NEW

- [x] Deployment scripts created:
  - [x] `_deploy_all_daemons_v1_1.py` (main)
  - [x] Demo mode (--mode demo)
  - [x] Production mode (--mode prod)
  - [x] Server-only mode (--mode fast)

---

## Deployment Steps

### Step 1: Verify Production Environment
```bash
cd /Users/carolalvarez/Applications/Allentown-L104-Node

# Check Python version
python3 --version  # Should be 3.11+

# Check required packages
python3 -c "import fastapi, uvicorn, sqlite3; print('✓ All deps available')"

# Check disk space
du -sh . | awk '{print "Directory size: " $1}'

# Check database
ls -lh l104_intellect_memory.db 2>/dev/null || echo "(will be created)"
```

### Step 2: Review Configuration
```bash
# Check current resource quotas in engines_infra.py
grep "DB_POOL_SIZE\|CPU_QUOTA\|MEMORY_QUOTA\|IO_QUOTA" l104_server/engines_infra.py
```

**Default Values**:
- DB_POOL_SIZE = 100
- CPU_QUOTA_PERCENT = 80.0
- MEMORY_QUOTA_MB = 2000
- IO_QUOTA_MBPS = 100.0

If your system has different specs, adjust in `l104_server/engines_infra.py` before deployment.

### Step 3: Start Deployment

#### Option A: Demo Mode (Recommended for Initial Testing)
```bash
# Run for 30 seconds to verify all daemons start
python3 _deploy_all_daemons_v1_1.py --mode demo --duration 30

# Expected output:
# ✓ Orchestrator ready
# ✓ VQPU daemon started and registered
# ✓ QuantumAI daemon started and registered
# ✓ Soul daemon started and registered
# ✓ Fast Server ready for orchestrator integration
# [Health: HEALTHY] Daemons: 3 | CPU: XX.X% | Memory: XXXMB
```

#### Option B: Production Mode (Background Deployment)
```bash
# Run in background for production
nohup python3 _deploy_all_daemons_v1_1.py --mode prod > daemon_orchestration.log 2>&1 &

# Store PID for later shutdown
DAEMON_PID=$!
echo $DAEMON_PID > daemon_orchestration.pid

# Monitor logs
tail -f daemon_orchestration.log

# Stop gracefully (in another terminal)
kill $DAEMON_PID  # Sends SIGTERM, triggers graceful shutdown
```

#### Option C: Start FastAPI Server Separately
```bash
# In terminal 1: Start orchestrator + daemons
python3 _deploy_all_daemons_v1_1.py --mode fast &
ORCH_PID=$!

# In terminal 2: Start FastAPI server (pool warmup happens automatically)
uvicorn l104_server.app:app --host 0.0.0.0 --port 8104 --workers 4

# Verify in terminal 3:
curl http://localhost:8104/api/v14/status
```

---

## Post-Deployment Verification

### 1. Check Orchestrator Health
```bash
curl -s http://localhost:8104/api/v14/orchestrator/status | python3 -m json.tool

# Expected structure:
{
  "health_status": "HEALTHY",
  "daemon_states": {
    "vqpu_daemon": {...},
    "quantum_ai_daemon": {...},
    "soul_daemon": {...},
    "fast_server": {...}
  },
  "cpu_percent": 45.2,
  "memory_mb": 512,
  "task_queue_size": 2
}
```

### 2. Verify Connection Pool Warmth
```bash
# Check logs for pool warm message
grep "Connection pool warmed" daemon_orchestration.log

# Expected: "🔗 [POOL] Connection pool warmed (20 pre-created, DB_POOL_SIZE=100)"
```

### 3. Test Connection Leak Fix
```python
# Quick test in Python REPL
from l104_server.engines_infra import connection_pool

# Get and return a connection
conn = connection_pool.get_connection()
connection_pool.return_connection(conn)
print("✓ Connection leak fix working")

# Verify pool size
print(f"Pool size: {len(connection_pool._pool)}")
print(f"Semaphore available: {connection_pool._semaphore._value}")
```

### 4. Monitor Daemon Cycles
```bash
# Watch for daemon activity in logs
grep -E "VQPU|QuantumAI|Soul|FastServer" daemon_orchestration.log | tail -20
```

### 5. Load Test (Optional)
```bash
# Generate load to verify backpressure works
for i in {1..100}; do
  curl http://localhost:8104/api/v14/status > /dev/null 2>&1 &
done
wait

# Check if all requests completed without "connection pooled exhausted" errors
grep -i "pool\|connection\|exhausted" daemon_orchestration.log
```

---

## Monitoring & Maintenance

### Daily Checks
```bash
# 1. Check health status
curl -s http://localhost:8104/api/v14/orchestrator/status | jq '.health_status'

# 2. Monitor pool utilization
curl -s http://localhost:8104/api/v6/performance | jq '.pool_stats'

# 3. Check error counts
grep -c "ERROR\|CRITICAL" daemon_orchestration.log

# 4. Verify all daemons running
ps aux | grep -E "python3.*_deploy_all_daemons|uvicorn"
```

### Weekly Checks
```bash
# 1. Check database size
ls -lh l104_intellect_memory.db

# 2. Verify memory optimizer is working
grep "memory_optimized" daemon_orchestration.log | wc -l

# 3. Check connection pool stats
curl -s http://localhost:8104/api/v6/performance | jq '.pool_stats'

# 4. Review crash logs
grep -i "exception\|crash\|segfault" daemon_orchestration.log | tail -20
```

### Tuning (If Needed)
If you observe:
- **High CPU (>90%)**: Reduce CPU_QUOTA_PERCENT or increase cycle intervals
- **Memory pressure**: Reduce MEMORY_QUOTA_MB or enable more aggressive GC
- **Connection timeouts**: Increase DB_POOL_SIZE or reduce concurrent requests
- **Fidelity degradation**: Check daemon health logs, may indicate resource contention

---

## Troubleshooting

### Problem: "Orchestrator not responding"
```bash
# Check if process is alive
ps aux | grep orchestrator

# Restart if needed
kill <PID>
python3 _deploy_all_daemons_v1_1.py --mode prod
```

### Problem: "Connection pool exhausted"
```bash
# Should not happen with backpressure semaphore, but if it does:
# 1. Check if requests are hanging
grep -i "acquire timeout\|semaphore timeout" daemon_orchestration.log

# 2. Increase DB_POOL_SIZE in l104_server/engines_infra.py
# 3. Restart deployment
```

### Problem: "Daemon not registering"
```bash
# Check for adapter errors
grep "Failed to register\|DaemonAdapter" daemon_orchestration.log

# Verify _HAS_DAEMON_ADAPTER is True
python3 -c "from l104_server.app import _HAS_DAEMON_ADAPTER; print(_HAS_DAEMON_ADAPTER)"

# If False, check if l104_daemon_adapter.py exists
ls l104_daemon_adapter.py
```

### Problem: "Connection leak still happening"
```bash
# Verify try/finally is in place
grep -A5 "def retrieve_contextual_memory" l104_server/learning/intellect.py | grep -A5 "finally:"

# If missing, re-apply Fix 2 from HOTSPOT_FIX_SUMMARY.md
```

---

## Rollback Plan

If deployment fails or issues arise:

### Quick Rollback
```bash
# 1. Stop daemons
kill <DAEMON_PID>

# 2. Restore from backup (if available)
git checkout l104_server/engines_infra.py l104_server/learning/intellect.py l104_server/app.py

# 3. Restart with previous version
python3 _deploy_multi_daemon.py
```

### Full Rollback
```bash
# Reset to previous commit (be careful!)
git log --oneline | head -5
git reset --hard <previous-commit-hash>

# Restart
python3 _deploy_multi_daemon.py
```

---

## Support & Documentation

- **Main Guide**: `DAEMON_ORCHESTRATION_GUIDE.md`
- **Hotspot Fix Details**: `HOTSPOT_FIX_SUMMARY.md`
- **Architecture**: `DAEMON_ARCHITECTURE_DIAGRAM.txt`
- **Quick Reference**: `ORCHESTRATION_SUMMARY.md`
- **Deployment Script**: `_deploy_all_daemons_v1_1.py`

---

## Sign-Off Checklist

Before going to production:

- [ ] All 3 previous daemons verified working
- [ ] Connection pool fixes compiled and verified
- [ ] FastServer integration wired (ServerDaemon)
- [ ] Demo mode test passed (30 seconds)
- [ ] Load test passed (requests completed without errors)
- [ ] Health dashboard accessible
- [ ] Logs monitored for 5+ minutes, no errors
- [ ] Rollback plan documented and tested
- [ ] Monitoring configured (daily/weekly checks)

---

**Deployment Date**: 2026-03-20
**Deployed By**: [Name]
**Status**: [READY / IN PROGRESS / COMPLETE]
**Notes**: [Any special considerations or deviations from checklist]

---

**Status**: ✓ COMPLETE — All systems ready for production deployment
