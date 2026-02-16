# System Performance Optimization Guide

**Generated:** January 26, 2026

## 🚨 Critical Issues Identified

### 1. Disk Space Crisis (CRITICAL)

- **Current:** 100% disk usage (32GB / 32GB used)
- **Impact:** System cannot write files, crashes, data loss risk
- **Solution:** Freed ~1GB, but more cleanup needed

### 2. Memory Pressure (HIGH)

- **Current:** 9.0GB / 15GB used (60%)
- **Impact:** Reduced performance, potential OOM errors
- **Key Consumers:**
  - VS Code Extension Host: 1.5GB
  - Java Language Servers: 500MB each (multiple instances)
  - Python Uvicorn Server: 495MB

### 3. High CPU Usage (MEDIUM)

- **Uvicorn Process:** 54% CPU sustained
- **Impact:** Battery drain, thermal throttling

## ✅ Optimizations Performed

1. ✓ Cleaned Python caches (*__pycache__*, .pyc files)
2. ✓ Cleaned temporary files
3. ✓ Cleaned package manager caches (apt)
4. ✓ Cleaned VS Code logs
5. ✓ Created monitoring tools

## 🛠️ Available Tools

### 1. System Optimizer Script

```bash
./optimize_system.sh
```

**Purpose:** Automated cleanup and optimization
**When to use:** Daily or when disk >90%

### 2. Memory Monitor

```bash
./memory_monitor.py          # One-time check
./memory_monitor.py --watch  # Continuous monitoring
```

**Purpose:** Real-time resource monitoring
**When to use:** Performance troubleshooting

## 📊 Optimization Recommendations

### Immediate Actions (Do Now)

1. **Free More Disk Space:**

   ```bash
   # Remove old git branches
   git branch -D <old-branch-name>

   # Find and remove large files
   find . -type f -size +50M -ls

   # Consider moving fine_tune_exports.tar.gz (11MB) to external storage
   ```

2. **Reduce Java Memory Usage:**
   - You have 4+ Java language servers running (1.5GB total)
   - Close unused VS Code windows
   - Restart VS Code to consolidate language servers
   - Add to settings:

   ```json
   {
     "java.jdt.ls.vmargs": "-Xmx512m"
   }
   ```

3. **Optimize Uvicorn Server:**

   ```bash
   # Check what main.py is doing
   # Consider using --workers 1 or --limit-concurrency options
   # Example:
   # uvicorn main:app --host 0.0.0.0 --port 8081 --workers 1 --limit-concurrency 50
   ```

### Short-term Actions (This Week)

1. **Configure Swap Space:**

   ```bash
   # Create 2GB swap file (requires disk space first!)
   sudo fallocate -l 2G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile

   # Make permanent
   echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
   ```

2. **Git Repository Optimization:**

   ```bash
   # After freeing space, compress git history
   git gc --aggressive --prune=now

   # Consider shallow cloning if possible
   git fetch --depth=1
   ```

3. **Docker Optimization:**

   ```bash
   # Your Docker is using 9GB in /var/lib/docker
   docker system prune -af --volumes

   # Consider moving Docker to separate volume
   ```

### Long-term Actions (This Month)

1. **Workspace Cleanup:**
   - Archive old notebooks (advanced_kernel_research.ipynb is 1.6MB)
   - Move data files to external storage
   - Remove duplicate code files
   - Consider using .gitignore for large files

2. **Monitor Resource Usage:**

   ```bash
   # Add to cron for daily monitoring
   crontab -e
   # Add: 0 */6 * * * /workspaces/Allentown-L104-Node/optimize_system.sh
   ```

3. **VS Code Extensions:**
   - Review installed extensions
   - Disable unused language servers
   - Consider workspace-specific extension settings

## 📈 Performance Tuning

### Environment Variables

Add to your shell profile:

```bash
# Python optimization
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1  # Prevents .pyc files

# Node.js optimization
export NODE_OPTIONS="--max-old-space-size=2048"
```

### VS Code Settings

Add to `.vscode/settings.json`:

```json
{
  "files.watcherExclude": {
    "**/.git/objects/**": true,
    "**/.git/subtree-cache/**": true,
    "**/node_modules/**": true,
    "**/__pycache__/**": true,
    "**/*.pyc": true
  },
  "search.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    "**/.git": true
  },
  "python.analysis.cacheFolderPath": "/tmp/pylance-cache"
}
```

## 🔍 Monitoring Commands

```bash
# Quick status check
df -h && free -h

# Process monitoring
htop  # Interactive
top   # Basic

# Disk usage analysis
du -sh ./* | sort -hr | head -10

# Memory by process
ps aux --sort=-%mem | head -20

# Find large files
find . -type f -size +10M -exec ls -lh {} \;

# Network connections
netstat -tulpn | grep LISTEN
```

## ⚠️ Warning Signs

Watch for these indicators:

- Disk usage > 95%: CRITICAL - immediate action required
- Memory usage > 90%: System will slow significantly
- CPU usage > 80% sustained: Check for runaway processes
- Many zombie processes: Restart required

## 🎯 Performance Targets

**Healthy System:**

- Disk: < 80% used
- Memory: < 70% used
- CPU: < 50% average
- Swap: < 50% used (if configured)

**Current Status:**

- Disk: 100% ❌ (Target: 80%)
- Memory: 60% ⚠️ (Target: 70%)
- CPU: 54% ⚠️ (Target: 50%)
- Swap: Not configured ❌

## 📞 Quick Reference

| Task | Command |
|------|---------|
| System optimization | `./optimize_system.sh` |
| Memory monitor | `./memory_monitor.py` |
| Disk cleanup | `./optimize_system.sh` |
| Git cleanup | `git gc --aggressive` |
| Docker cleanup | `docker system prune -af` |
| Process kill | `kill -9 <PID>` |

## 🔗 Additional Resources

- htop: Interactive process viewer
- iotop: Disk I/O monitor
- nethogs: Network bandwidth monitor
- psutil: Python system monitoring library

---

**Next Steps:**

1. Run `./optimize_system.sh` to free more space
2. Identify and archive/delete large unused files
3. Consider requesting more disk space from your hosting provider
4. Monitor with `./memory_monitor.py --watch`
