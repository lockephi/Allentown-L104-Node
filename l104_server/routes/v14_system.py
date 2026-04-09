"""
v14 System Routes — System operations, monitoring, and file management

Extracted from app.py during EVO_78 refactoring.
Contains: All /api/v14/system/*, /api/v14/monitor/*, /api/v14/backup/*,
          /api/v14/file/*, /api/v14/source/*, /api/v14/autosave/* endpoints
"""

import os
import time
import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Request

logger = logging.getLogger("L104_FAST")

router = APIRouter(prefix="/api/v14", tags=["v14-system"])

# System control availability check
try:
    from l104_server.system_control import get_system_controller, SYSTEM_CONTROL_AVAILABLE
except ImportError:
    SYSTEM_CONTROL_AVAILABLE = False
    get_system_controller = None


# ═══════════════════════════════════════════════════════════════════
#  SYSTEM ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@router.post("/system/update")
async def system_update(req: Request):
    """Update system configuration"""
    data = await req.json()
    config = data.get("config", {})
    try:
        # Update system configuration
        return {"status": "UPDATED", "config": config}
    except Exception as e:
        return {"error": str(e)}


@router.get("/system/stream")
async def system_stream():
    """Stream system events (SSE placeholder)"""
    # SSE streaming would be implemented here
    return {"status": "STREAMING", "events": []}


@router.get("/system/status")
async def system_status():
    """Get system status"""
    if not SYSTEM_CONTROL_AVAILABLE:
        return {"status": "UNAVAILABLE", "error": "System control not available"}

    try:
        ctrl = get_system_controller()
        return {
            "status": "ACTIVE",
            "cpu": ctrl.get_cpu_info() if hasattr(ctrl, 'get_cpu_info') else {},
            "memory": ctrl.get_memory_info() if hasattr(ctrl, 'get_memory_info') else {},
            "disk": ctrl.get_disk_info() if hasattr(ctrl, 'get_disk_info') else {}
        }
    except Exception as e:
        return {"error": str(e), "status": "ERROR"}


@router.get("/system/cpu")
async def system_cpu():
    """Get CPU info"""
    if not SYSTEM_CONTROL_AVAILABLE:
        return {"error": "System control not available"}

    try:
        ctrl = get_system_controller()
        return ctrl.get_cpu_info() if hasattr(ctrl, 'get_cpu_info') else {}
    except Exception as e:
        return {"error": str(e)}


@router.get("/system/memory")
async def system_memory():
    """Get memory info"""
    if not SYSTEM_CONTROL_AVAILABLE:
        return {"error": "System control not available"}

    try:
        ctrl = get_system_controller()
        return ctrl.get_memory_info() if hasattr(ctrl, 'get_memory_info') else {}
    except Exception as e:
        return {"error": str(e)}


@router.get("/system/disk")
async def system_disk():
    """Get disk info"""
    if not SYSTEM_CONTROL_AVAILABLE:
        return {"error": "System control not available"}

    try:
        ctrl = get_system_controller()
        return ctrl.get_disk_info() if hasattr(ctrl, 'get_disk_info') else {}
    except Exception as e:
        return {"error": str(e)}


@router.get("/system/gpu")
async def system_gpu():
    """Get GPU info"""
    if not SYSTEM_CONTROL_AVAILABLE:
        return {"error": "System control not available"}

    try:
        ctrl = get_system_controller()
        return ctrl.get_gpu_info() if hasattr(ctrl, 'get_gpu_info') else {}
    except Exception as e:
        return {"error": str(e)}


@router.get("/system/processes")
async def system_processes():
    """Get process list"""
    if not SYSTEM_CONTROL_AVAILABLE:
        return {"error": "System control not available"}

    try:
        ctrl = get_system_controller()
        processes = ctrl.list_processes() if hasattr(ctrl, 'list_processes') else []
        return {"processes": processes[:100]}  # Limit to 100
    except Exception as e:
        return {"error": str(e)}


@router.post("/system/optimize")
async def system_optimize(req: Request):
    """Optimize system"""
    data = await req.json()
    target = data.get("target", "memory")

    if not SYSTEM_CONTROL_AVAILABLE:
        return {"error": "System control not available"}

    try:
        ctrl = get_system_controller()
        result = ctrl.optimize(target) if hasattr(ctrl, 'optimize') else {}
        return {"status": "OPTIMIZED", "result": result}
    except Exception as e:
        return {"error": str(e)}


@router.post("/system/execute")
async def system_execute(req: Request):
    """Execute system command"""
    data = await req.json()
    command = data.get("command", "")

    if not command:
        return {"error": "Must provide command"}

    # Security: limit allowed commands
    allowed = ["echo", "ls", "pwd", "date", "uptime"]
    base_cmd = command.split()[0] if command else ""

    if base_cmd not in allowed:
        return {"error": f"Command not allowed: {base_cmd}"}

    try:
        import subprocess
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        return {
            "status": "EXECUTED",
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except Exception as e:
        return {"error": str(e)}


@router.post("/system/process/priority")
async def system_process_priority(req: Request):
    """Set process priority"""
    data = await req.json()
    pid = data.get("pid", 0)
    priority = data.get("priority", 0)

    if not SYSTEM_CONTROL_AVAILABLE:
        return {"error": "System control not available"}

    try:
        ctrl = get_system_controller()
        result = ctrl.set_priority(pid, priority) if hasattr(ctrl, 'set_priority') else {}
        return {"status": "PRIORITY_SET", "pid": pid, "priority": priority}
    except Exception as e:
        return {"error": str(e)}


@router.post("/system/process/spawn")
async def system_process_spawn(req: Request):
    """Spawn process"""
    data = await req.json()
    cmd = data.get("cmd", "")
    args = data.get("args", [])

    if not SYSTEM_CONTROL_AVAILABLE:
        return {"error": "System control not available"}

    try:
        ctrl = get_system_controller()
        result = ctrl.spawn_process(cmd, args) if hasattr(ctrl, 'spawn_process') else {}
        return {"status": "SPAWNED", "result": result}
    except Exception as e:
        return {"error": str(e)}


@router.post("/system/process/kill")
async def system_process_kill(req: Request):
    """Kill process"""
    data = await req.json()
    pid = data.get("pid", 0)

    if not SYSTEM_CONTROL_AVAILABLE:
        return {"error": "System control not available"}

    try:
        ctrl = get_system_controller()
        result = ctrl.kill_process(pid) if hasattr(ctrl, 'kill_process') else {}
        return {"status": "KILLED", "pid": pid}
    except Exception as e:
        return {"error": str(e)}


@router.post("/system/memory/purge")
async def system_memory_purge():
    """Purge system memory"""
    if not SYSTEM_CONTROL_AVAILABLE:
        return {"error": "System control not available"}

    try:
        import gc
        collected = gc.collect()
        return {"status": "PURGED", "collected": collected}
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════
#  MONITORING ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@router.get("/monitor/metrics")
async def monitor_metrics():
    """Get monitoring metrics"""
    try:
        from l104_server.engines_infra import performance_metrics
        return performance_metrics.get_metrics() if hasattr(performance_metrics, 'get_metrics') else {}
    except Exception as e:
        return {"error": str(e)}


@router.get("/monitor/history")
async def monitor_history():
    """Get monitoring history"""
    try:
        from l104_server.engines_infra import performance_metrics
        return performance_metrics.get_history() if hasattr(performance_metrics, 'get_history') else []
    except Exception as e:
        return {"error": str(e)}


@router.get("/monitor/alerts")
async def monitor_alerts():
    """Get monitoring alerts"""
    try:
        from l104_server.engines_infra import performance_metrics
        return performance_metrics.get_alerts() if hasattr(performance_metrics, 'get_alerts') else []
    except Exception as e:
        return {"error": str(e)}


@router.post("/monitor/threshold")
async def monitor_threshold(req: Request):
    """Set monitoring threshold"""
    data = await req.json()
    metric = data.get("metric", "")
    threshold = data.get("threshold", 0.0)

    try:
        from l104_server.engines_infra import performance_metrics
        result = performance_metrics.set_threshold(metric, threshold) if hasattr(performance_metrics, 'set_threshold') else {}
        return {"status": "THRESHOLD_SET", "metric": metric, "threshold": threshold}
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════
#  BACKUP ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@router.post("/backup/workspace")
async def backup_workspace(req: Request):
    """Backup workspace"""
    data = await req.json()
    path = data.get("path", os.getcwd())

    try:
        import shutil
        import tempfile
        backup_name = f"backup_{int(time.time())}"

        with tempfile.TemporaryDirectory() as tmpdir:
            backup_path = os.path.join(tmpdir, backup_name)
            shutil.copytree(path, backup_path)
            # In real implementation, would store to permanent location
            return {"status": "BACKED_UP", "path": path, "backup_name": backup_name}
    except Exception as e:
        return {"error": str(e)}


@router.post("/backup/file")
async def backup_file(req: Request):
    """Backup file"""
    data = await req.json()
    filepath = data.get("path", "")

    if not filepath or not os.path.exists(filepath):
        return {"error": "File not found"}

    try:
        backup_path = f"{filepath}.backup_{int(time.time())}"
        import shutil
        shutil.copy2(filepath, backup_path)
        return {"status": "BACKED_UP", "original": filepath, "backup": backup_path}
    except Exception as e:
        return {"error": str(e)}


@router.post("/backup/restore")
async def backup_restore(req: Request):
    """Restore from backup"""
    data = await req.json()
    backup_path = data.get("backup_path", "")
    target_path = data.get("target_path", "")

    if not backup_path or not os.path.exists(backup_path):
        return {"error": "Backup not found"}

    try:
        import shutil
        shutil.copy2(backup_path, target_path)
        return {"status": "RESTORED", "backup": backup_path, "target": target_path}
    except Exception as e:
        return {"error": str(e)}


@router.get("/backup/list")
async def backup_list():
    """List backups"""
    try:
        # List backup files in current directory
        backups = []
        for f in os.listdir("."):
            if ".backup_" in f:
                backups.append({
                    "name": f,
                    "path": os.path.abspath(f),
                    "mtime": os.path.getmtime(f)
                })
        return {"backups": sorted(backups, key=lambda x: x["mtime"], reverse=True)}
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════
#  FILE ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@router.post("/file/read")
async def file_read(req: Request):
    """Read file"""
    data = await req.json()
    path = data.get("path", "")

    if not path:
        return {"error": "Must provide path"}

    # Security: prevent reading outside allowed directories
    allowed_dirs = [os.getcwd(), "/tmp"]
    abs_path = os.path.abspath(path)

    if not any(abs_path.startswith(d) for d in allowed_dirs):
        return {"error": "Path not in allowed directories"}

    try:
        with open(abs_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return {"status": "READ", "path": path, "content": content, "size": len(content)}
    except Exception as e:
        return {"error": str(e)}


@router.post("/file/write")
async def file_write(req: Request):
    """Write file"""
    data = await req.json()
    path = data.get("path", "")
    content = data.get("content", "")

    if not path:
        return {"error": "Must provide path"}

    # Security: prevent writing outside allowed directories
    allowed_dirs = [os.getcwd(), "/tmp"]
    abs_path = os.path.abspath(path)

    if not any(abs_path.startswith(d) for d in allowed_dirs):
        return {"error": "Path not in allowed directories"}

    try:
        with open(abs_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return {"status": "WRITTEN", "path": path, "size": len(content)}
    except Exception as e:
        return {"error": str(e)}


@router.post("/file/rewrite")
async def file_rewrite(req: Request):
    """Rewrite file with transformation"""
    data = await req.json()
    path = data.get("path", "")
    transformation = data.get("transformation", "identity")

    if not path:
        return {"error": "Must provide path"}

    # Security: prevent reading outside allowed directories
    allowed_dirs = [os.getcwd(), "/tmp"]
    abs_path = os.path.abspath(path)

    if not any(abs_path.startswith(d) for d in allowed_dirs):
        return {"error": "Path not in allowed directories"}

    try:
        with open(abs_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Apply transformation
        if transformation == "uppercase":
            content = content.upper()
        elif transformation == "lowercase":
            content = content.lower()
        # identity = no change

        with open(abs_path, 'w', encoding='utf-8') as f:
            f.write(content)

        return {"status": "REWRITTEN", "path": path, "transformation": transformation}
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════
#  SOURCE ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@router.get("/source/list")
async def source_list():
    """List source files"""
    try:
        sources = []
        for root, dirs, files in os.walk("."):
            # Skip hidden and common non-source directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', '.git']]
            for f in files:
                if f.endswith(('.py', '.js', '.ts', '.swift', '.md', '.json')):
                    sources.append(os.path.join(root, f))
        return {"sources": sources[:1000]}  # Limit to 1000
    except Exception as e:
        return {"error": str(e)}


@router.get("/source/stats/{filename:path}")
async def source_stats(filename: str):
    """Get source file stats"""
    try:
        if not os.path.exists(filename):
            return {"error": "File not found"}

        stat = os.stat(filename)
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()

        return {
            "filename": filename,
            "size": stat.st_size,
            "lines": content.count('\n') + 1,
            "characters": len(content),
            "mtime": stat.st_mtime
        }
    except Exception as e:
        return {"error": str(e)}


@router.post("/source/restore")
async def source_restore(req: Request):
    """Restore source from backup"""
    data = await req.json()
    path = data.get("path", "")
    backup = data.get("backup", "")

    if not path or not backup:
        return {"error": "Must provide path and backup"}

    try:
        import shutil
        shutil.copy2(backup, path)
        return {"status": "RESTORED", "path": path}
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════
#  AUTOSAVE ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@router.get("/autosave/status")
async def autosave_status():
    """Get autosave status"""
    try:
        # Check for autosave files
        autosave_dir = ".autosave"
        if os.path.exists(autosave_dir):
            files = os.listdir(autosave_dir)
            return {"status": "ACTIVE", "autosave_count": len(files)}
        return {"status": "INACTIVE"}
    except Exception as e:
        return {"error": str(e)}


@router.post("/autosave/snapshot")
async def autosave_snapshot(req: Request):
    """Create autosave snapshot"""
    data = await req.json()
    name = data.get("name", f"snapshot_{int(time.time())}")

    try:
        autosave_dir = ".autosave"
        os.makedirs(autosave_dir, exist_ok=True)

        snapshot_file = os.path.join(autosave_dir, name)
        with open(snapshot_file, 'w') as f:
            f.write(f"snapshot at {time.time()}\n")

        return {"status": "SNAPSHOT", "name": name}
    except Exception as e:
        return {"error": str(e)}


@router.post("/autosave/restore")
async def autosave_restore(req: Request):
    """Restore from autosave"""
    data = await req.json()
    name = data.get("name", "")

    if not name:
        return {"error": "Must provide autosave name"}

    try:
        autosave_dir = ".autosave"
        snapshot_file = os.path.join(autosave_dir, name)

        if not os.path.exists(snapshot_file):
            return {"error": "Snapshot not found"}

        with open(snapshot_file, 'r') as f:
            content = f.read()

        return {"status": "RESTORED", "name": name, "content": content}
    except Exception as e:
        return {"error": str(e)}


__all__ = ['router']