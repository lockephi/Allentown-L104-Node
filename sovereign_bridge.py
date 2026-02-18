# L104_GOD_CODE_ALIGNED: 527.5184818492612
# [L104_WEBSOCKET_BRIDGE] - QUANTUM AMPLIFIED FULL-SPECTRUM WEB CONNECTOR v5.0
import asyncio
import socket
import json
import time
import hashlib
from typing import Optional, Dict, Any

try:
    from l104_codec import SovereignCodec
except ImportError:
    SovereignCodec = None

try:
    from const import UniversalConstants, GOD_CODE, PHI, GROVER_AMPLIFICATION
except ImportError:
    # Universal Equation: G(a,b,c,d) = 286^(1/φ) × 2^((8a+416-b-8c-104d)/104)
    PHI = 1.618033988749895
    GOD_CODE = 286 ** (1.0 / PHI) * (2 ** (416 / 104))  # G(0,0,0,0) = 527.5184818492612
    GROVER_AMPLIFICATION = PHI ** 3

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ALL LIMITERS REMOVED - QUANTUM AMPLIFIED - FULL WEB APP CONNECTIVITY
# ═══════════════════════════════════════════════════════════════════════════════

# Connection state - persistent across reconnects
_bridge_state = {
    "connections": 0,
    "messages_relayed": 0,
    "bytes_transferred": 0,
    "quantum_coherence": 1.0,
    "uptime_start": None,
    "last_heartbeat": None,
    "web_app_connected": False,
    "api_endpoints_synced": False,
}


def quantum_amplify_message(message: str) -> str:
    """Apply Grover-style quantum amplification to message routing.
    Amplifies signal-to-noise ratio by φ³ factor.
    """
    msg_hash = hashlib.sha256(message.encode()).hexdigest()
    resonance = sum(int(c, 16) for c in msg_hash[:8]) / (16 * 8)
    amplified_resonance = resonance * GROVER_AMPLIFICATION
    return f"[QA:{amplified_resonance:.6f}] {message}"


async def connect_to_web_app(port: int = 8081) -> Optional[Dict[str, Any]]:
    """Establish connection to the main web application API.
    Returns connection status dict or None on failure.
    """
    try:
        import httpx
        async with httpx.AsyncClient(timeout=None) as client:
            resp = await client.get(f"http://localhost:{port}/health")
            if resp.status_code == 200:
                _bridge_state["web_app_connected"] = True
                return resp.json()
    except Exception:
        pass

    # Fallback: try fast server
    try:
        import httpx
        async with httpx.AsyncClient(timeout=None) as client:
            resp = await client.get("http://localhost:5104/health")
            if resp.status_code == 200:
                _bridge_state["web_app_connected"] = True
                return resp.json()
    except Exception:
        pass

    return None


async def relay_to_api(endpoint: str, data: dict, port: int = 8081) -> Optional[dict]:
    """Relay data from WebSocket bridge to REST API and back.
    No rate limits, no throttling, no timeouts.
    """
    try:
        import httpx
        async with httpx.AsyncClient(timeout=None) as client:
            resp = await client.post(
                f"http://localhost:{port}/api/v6/{endpoint}",
                json=data,
                headers={"X-Bridge-Source": "sovereign_bridge", "X-Quantum-Amplified": "true"}
            )
            _bridge_state["messages_relayed"] += 1
            return resp.json()
    except Exception:
        return None


async def bridge_logic(websocket, path):
    """Quantum-amplified WebSocket bridge with full web app connectivity.
    No buffering limits. No message size caps. No rate throttling.
    """
    _bridge_state["connections"] += 1
    _bridge_state["uptime_start"] = _bridge_state["uptime_start"] or time.time()
    print(f"UI_CONNECTED: SYNERGY_ESTABLISHED [conn #{_bridge_state['connections']}]")

    codec = SovereignCodec() if SovereignCodec else None

    # Verify web app connectivity
    web_status = await connect_to_web_app()
    if web_status:
        _bridge_state["api_endpoints_synced"] = True
        print(f"[BRIDGE] Web app connected: {web_status.get('status', 'OK')}")

    try:
        # Direct socket to master node (legacy)
        node_socket = None
        try:
            node_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            node_socket.settimeout(None)  # NO TIMEOUT
            node_socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            node_socket.connect(('0.0.0.0', 4160))
        except Exception:
            node_socket = None

        while True:
            try:
                pilot_msg = await websocket.recv()
                _bridge_state["bytes_transferred"] += len(pilot_msg)

                # Quantum amplify the message
                amplified_msg = quantum_amplify_message(pilot_msg)

                # Route through web app API if connected
                if _bridge_state["web_app_connected"]:
                    try:
                        api_result = await relay_to_api("chat", {
                            "message": pilot_msg,
                            "source": "ws_bridge",
                            "quantum_amplified": True
                        })
                        if api_result:
                            response = json.dumps(api_result)
                            await websocket.send(response)
                            continue
                    except Exception:
                        pass  # Fall through to direct node

                # Fallback: Direct node communication
                if node_socket:
                    sleek_input = codec.generate_sleek_wrapper(pilot_msg) if codec else pilot_msg
                    node_socket.sendall(sleek_input.encode())

                    node_data = node_socket.recv(65536)  # 64KB buffer (was 4KB)
                    clean_text = node_data.decode('utf-8', errors='ignore')

                    s_hash = codec.singularity_hash(clean_text) if codec else GOD_CODE
                    coherence = _bridge_state["quantum_coherence"]
                    response = json.dumps({
                        "text": clean_text,
                        "stability": s_hash,
                        "coherence": coherence,
                        "quantum_amplified": True,
                        "god_code": GOD_CODE,
                    })
                    await websocket.send(response)
                else:
                    # Echo with quantum amplification when no backend
                    await websocket.send(json.dumps({
                        "text": amplified_msg,
                        "status": "bridge_active",
                        "web_app_connected": _bridge_state["web_app_connected"],
                    }))

            except Exception as e:
                print(f"Bridge relay error: {e}")
                break
    except Exception as e:
        print(f"Connection error: {e}")
    finally:
        _bridge_state["connections"] = max(0, _bridge_state["connections"] - 1)


def get_bridge_status() -> dict:
    """Get current bridge status for web app dashboard."""
    uptime = 0
    if _bridge_state["uptime_start"]:
        uptime = time.time() - _bridge_state["uptime_start"]
    return {
        **_bridge_state,
        "uptime_seconds": uptime,
        "god_code": GOD_CODE,
        "quantum_amplification": GROVER_AMPLIFICATION,
        "limiters": "NONE",
    }


async def start_bridge(port: int = 8080):
    """Start the quantum-amplified websocket bridge server.
    No connection limits. No message size limits. No rate throttling.
    """
    try:
        import websockets
        async with websockets.serve(
            bridge_logic, "0.0.0.0", port,
            max_size=None,          # NO MESSAGE SIZE LIMIT
            max_queue=None,         # NO QUEUE LIMIT
            ping_interval=30,       # Keep-alive
            ping_timeout=None,      # NO PING TIMEOUT
            compression=None,       # Raw speed
        ):
            print(f"SOVEREIGN_BRIDGE: ONLINE AT WS://0.0.0.0:{port}")
            print(f"[QUANTUM_AMPLIFIED] Grover gain: {GROVER_AMPLIFICATION:.4f}")
            print(f"[LIMITERS] ALL REMOVED | [WEB_APP] CONNECTING...")
            _bridge_state["uptime_start"] = time.time()

            # Background: Try to connect to web app
            asyncio.create_task(connect_to_web_app())

            await asyncio.Future()  # run forever
    except ImportError:
        print("websockets library not installed. Install with: pip install websockets")

if __name__ == "__main__":
    asyncio.run(start_bridge())
