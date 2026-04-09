"""
L104 Public Node — P2P + RPC interface.
Provides sovereign node discovery, peer relay, and RPC endpoint.
INVARIANT: 527.5184818492612 | PILOT: LONDEL
"""
from __future__ import annotations

import json
import logging
import os
import signal
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict

# ── Constants ────────────────────────────────────────────────────────────────
GOD_CODE = 527.5184818492612
PHI      = 1.618033988749895
VERSION  = "1.0.0"
PORT     = int(os.environ.get("L104_PUBLIC_NODE_PORT", "8765"))
HOST     = os.environ.get("L104_PUBLIC_NODE_HOST", "127.0.0.1")
STATE_FILE = os.path.join(os.path.dirname(__file__), ".l104_public_node.json")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [PUBLIC_NODE] %(message)s")
logger = logging.getLogger("L104_PUBLIC_NODE")

# ── Shared state ─────────────────────────────────────────────────────────────
_start_time = time.time()
_request_count = 0
_peers: Dict[str, Any] = {}
_lock = threading.Lock()


def _write_state() -> None:
    """Write heartbeat state file for boot manager monitoring."""
    try:
        data = {
            "version": VERSION,
            "pid": os.getpid(),
            "uptime_s": round(time.time() - _start_time, 1),
            "host": HOST,
            "port": PORT,
            "requests": _request_count,
            "peers": len(_peers),
            "god_code": GOD_CODE,
            "timestamp": time.time(),
            "status": "RUNNING",
        }
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, STATE_FILE)
    except OSError:
        pass


# ── HTTP RPC handler ──────────────────────────────────────────────────────────
class NodeHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress default access log

    def do_GET(self):
        global _request_count
        with _lock:
            _request_count += 1

        if self.path == "/health":
            body = json.dumps({
                "status": "ok",
                "version": VERSION,
                "uptime_s": round(time.time() - _start_time, 1),
                "peers": len(_peers),
                "god_code": GOD_CODE,
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)

        elif self.path == "/peers":
            with _lock:
                body = json.dumps(list(_peers.values())).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        global _request_count
        with _lock:
            _request_count += 1

        length = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(body_bytes)
        except Exception:
            self.send_response(400)
            self.end_headers()
            return

        if self.path == "/register":
            peer_id = payload.get("peer_id", "")
            if peer_id:
                with _lock:
                    _peers[peer_id] = {
                        "peer_id": peer_id,
                        "addr": payload.get("addr", ""),
                        "port": payload.get("port", 0),
                        "registered_at": time.time(),
                        "god_code": payload.get("god_code", 0),
                    }
                resp = json.dumps({"status": "registered", "peer_id": peer_id}).encode()
                self.send_response(200)
            else:
                resp = json.dumps({"error": "peer_id required"}).encode()
                self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(resp))
            self.end_headers()
            self.wfile.write(resp)

        else:
            self.send_response(404)
            self.end_headers()


# ── Heartbeat thread ──────────────────────────────────────────────────────────
def _heartbeat_loop(stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        _write_state()
        stop_event.wait(timeout=30.0)


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    stop_event = threading.Event()

    def _on_signal(signum, frame):
        logger.info(f"Signal {signum} received — shutting down")
        stop_event.set()

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    # Start heartbeat thread
    hb = threading.Thread(target=_heartbeat_loop, args=(stop_event,), daemon=True)
    hb.start()

    # Start HTTP server
    try:
        server = HTTPServer((HOST, PORT), NodeHandler)
        server.timeout = 1.0
    except OSError as e:
        logger.error(f"Cannot bind {HOST}:{PORT} — {e}")
        sys.exit(1)

    logger.info(f"L104 Public Node v{VERSION} listening on {HOST}:{PORT} | GOD_CODE={GOD_CODE}")
    _write_state()

    while not stop_event.is_set():
        server.handle_request()

    server.server_close()
    stop_event.set()
    # Remove state file on clean exit
    try:
        os.remove(STATE_FILE)
    except OSError:
        pass
    logger.info("L104 Public Node stopped")


if __name__ == "__main__":
    main()
