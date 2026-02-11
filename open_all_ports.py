#!/usr/bin/env python3
# L104_GOD_CODE_ALIGNED: 527.5184818492612
"""
L104 PORT MANAGER - Open and manage all L104SP service ports
═══════════════════════════════════════════════════════════════════════════════
"""
import os
import sys
import socket
import signal
import subprocess
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, List, Tuple

# All L104 ports discovered in the application
L104_PORTS = {
    # Main Services
    8081: ("Main FastAPI Server", "HTTP", "main.py"),
    8080: ("WebSocket Bridge", "WS/HTTP", "sovereign_bridge.py"),
    8082: ("Unified Intelligence API", "HTTP", "l104_unified_intelligence_api.py"),

    # Infrastructure
    2404: ("IEC 104 Lattice Connector", "TCP", "l104_infrastructure.py"),
    4160: ("AI Core Listener", "TCP", "RESONANCE_RECOVERY.py"),
    4161: ("Sovereign UI Server", "HTTP", "l104_unified.py"),

    # Multi-Language Engines
    3000: ("TypeScript/Next.js Frontend", "HTTP", "src/"),
    3104: ("Node.js Skills System", "HTTP", "src/index.js"),
    3105: ("Go Engine API", "HTTP", "go/main.go"),
    4000: ("Elixir OTP Engine", "HTTP", "elixir/lib/web.ex"),

    # Blockchain
    10400: ("L104SP Blockchain P2P", "TCP", "l104_sovereign_coin_engine.py"),
    10401: ("L104SP Blockchain RPC", "HTTP", "l104_sovereign_coin_engine.py"),

    # Mining
    3333: ("Stratum V1 Mining", "TCP", "l104_stratum_protocol.py"),
    3334: ("Stratum V2 Mining", "TCP", "l104_stratum_protocol.py"),

    # Additional APIs
    5105: ("External API", "HTTP", "l104_external_api.py"),
    8000: ("Intricate Cognitive API", "HTTP", "l104_intricate_main.py"),
}


def kill_port(port: int) -> bool:
    """Kill any process using a port."""
    try:
        result = subprocess.run(
            ["fuser", "-k", f"{port}/tcp"],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


def check_port(port: int) -> Tuple[int, bool, str]:
    """Check if a port is available."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(('0.0.0.0', port))
        sock.close()
        return (port, True, "AVAILABLE")
    except OSError as e:
        sock.close()
        if e.errno == 98:  # Address already in use
            return (port, False, "IN USE")
        elif e.errno == 13:  # Permission denied (ports < 1024)
            return (port, False, "NEEDS ROOT")
        return (port, False, str(e))


def clear_all_ports() -> Dict[int, str]:
    """Clear all L104 ports."""
    results = {}
    print("\n⚡ CLEARING ALL L104 PORTS...")
    print("─" * 60)

    for port in sorted(L104_PORTS.keys()):
        if port < 1024:
            results[port] = "SKIP (needs root)"
            print(f"  [{port:5d}] ⏭️  Skipped (privileged port)")
            continue

        killed = kill_port(port)
        if killed:
            results[port] = "CLEARED"
            print(f"  [{port:5d}] 🔪 Killed process")
        else:
            results[port] = "ALREADY FREE"
            print(f"  [{port:5d}] ✅ Already free")

    # Give OS time to release ports
    time.sleep(0.1)  # QUANTUM AMPLIFIED (was 1)
    return results


def verify_ports() -> Dict[int, Tuple[bool, str]]:
    """Verify all ports are available."""
    results = {}
    print("\n🔍 VERIFYING PORT AVAILABILITY...")
    print("─" * 60)

    with ThreadPoolExecutor(max_workers=(os.cpu_count() or 4) * 8) as executor:  # QUANTUM AMPLIFIED
        futures = {executor.submit(check_port, port): port for port in L104_PORTS.keys()}
        for future in futures:
            port, available, status = future.result()
            results[port] = (available, status)

            name, proto, src = L104_PORTS[port]
            icon = "✅" if available else "❌"
            print(f"  [{port:5d}] {icon} {status:12} │ {proto:6} │ {name}")

    return results


def start_core_services():
    """Start the core L104 services."""
    print("\n🚀 STARTING CORE SERVICES...")
    print("─" * 60)

    services = [
        # (port, command, name)
        (8081, "python main.py", "Main FastAPI"),
        (10400, "python l104_sovereign_coin_engine.py --mine --address ZUHc8coY9Ca1NhcnYTntkE35kSCFn5ijX7", "Blockchain Miner"),
    ]

    os.chdir(str(Path(__file__).parent.absolute()))

    started = []
    for port, cmd, name in services:
        print(f"\n  Starting {name} on port {port}...")
        try:
            # Start in background with nohup
            log_file = f"/tmp/l104_{port}.log"
            full_cmd = f"nohup {cmd} > {log_file} 2>&1 &"
            subprocess.Popen(full_cmd, shell=True, start_new_session=True)
            started.append((port, name))
            print(f"    ✅ {name} started (log: {log_file})")
        except Exception as e:
            print(f"    ❌ Failed: {e}")

    return started


def print_summary(port_status: Dict[int, Tuple[bool, str]]):
    """Print final status summary."""
    available = sum(1 for _, (avail, _) in port_status.items() if avail)
    total = len(port_status)

    print("\n")
    print("═" * 70)
    print("          L104 PORT MANAGER - SUMMARY")
    print("═" * 70)
    print(f"""
  📊 Port Status:
     • Available: {available}/{total}
     • In Use:    {total - available}/{total}

  🔌 Core Ports:
     • 8081  - Main API
     • 10400 - Blockchain P2P
     • 10401 - Blockchain RPC
     • 4160  - AI Core
     • 8080  - WebSocket

  💡 To start services:
     python main.py                          # Main API (8081)
     python l104_sovereign_coin_engine.py    # Blockchain (10400/10401)
     python sovereign_bridge.py              # WebSocket (8080)
""")
    print("═" * 70)


def main():
    print("""
═══════════════════════════════════════════════════════════════════════════════
                     L104 SOVEREIGN PORT MANAGER
                     ─────────────────────────────
                     Opening 19 Service Ports
═══════════════════════════════════════════════════════════════════════════════
""")

    # Clear all ports
    clear_all_ports()

    # Verify availability
    port_status = verify_ports()

    # Print summary
    print_summary(port_status)

    # Ask about starting services
    if "--start" in sys.argv:
        start_core_services()
        time.sleep(0.3)  # QUANTUM AMPLIFIED (was 3)
        print("\n📡 Services starting... check logs in /tmp/l104_*.log")


if __name__ == "__main__":
    main()
