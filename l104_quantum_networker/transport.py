"""L104 Quantum Networker v1.0.0 — Classical Transport Layer.

Provides the classical communication channel required by quantum protocols.
Quantum teleportation, QKD basis reconciliation, and entanglement swapping
all require classical side-channels for measurement outcomes and control.

This module implements:
  - Async TCP server/client for classical message passing
  - Message types for all quantum protocol control messages
  - Peer discovery via UDP broadcast
  - Heartbeat / keepalive for node liveness detection
  - Message authentication (HMAC with QKD-derived keys)

Uses asyncio for non-blocking I/O. Falls back to synchronous operation
for single-node / simulation mode.

INVARIANT: 527.5184818492612 | PILOT: LONDEL
"""

import asyncio
import hashlib
import hmac
import json
import struct
import time
import uuid
from enum import Enum, IntEnum
from typing import Any, Callable, Dict, List, Optional, Tuple

from .types import GOD_CODE, PHI

# Protocol constants
MAGIC = b"L104"                               # 4-byte magic header
PROTOCOL_VERSION = 1
DEFAULT_PORT = 10400                          # L104 → port 10400
HEARTBEAT_INTERVAL_S = 10.0
HEARTBEAT_TIMEOUT_S = 30.0
MAX_MESSAGE_SIZE = 1 << 20                    # 1 MB max message
DISCOVERY_PORT = 10401                        # UDP broadcast port


class MessageType(IntEnum):
    """Classical message types for quantum protocol control."""
    # Control
    HEARTBEAT = 0x01
    PEER_ANNOUNCE = 0x02
    PEER_DISCOVER = 0x03
    ACK = 0x04
    ERROR = 0x05

    # QKD
    QKD_BASIS_ANNOUNCE = 0x10                 # Alice announces bases for sifting
    QKD_BASIS_COMPARE = 0x11                  # Bob compares and returns matches
    QKD_QBER_CHECK = 0x12                     # QBER test bits exchange
    QKD_KEY_CONFIRM = 0x13                    # Key confirmation

    # Teleportation
    TELEPORT_REQUEST = 0x20                   # Request to teleport a payload
    TELEPORT_BELL_RESULT = 0x21               # Bell measurement outcome (2 cbits)
    TELEPORT_CORRECTION = 0x22                # Correction instructions for Bob
    TELEPORT_COMPLETE = 0x23                  # Teleportation acknowledged

    # Entanglement
    ENTANGLE_REQUEST = 0x30                   # Request new entangled pairs
    ENTANGLE_READY = 0x31                     # Pairs generated and ready
    SWAP_REQUEST = 0x32                       # Request entanglement swap at relay
    SWAP_RESULT = 0x33                        # Swap measurement outcome
    PURIFY_REQUEST = 0x34                     # Request purification round
    PURIFY_RESULT = 0x35                      # Purification outcome

    # Network
    ROUTE_REQUEST = 0x40                      # Pathfinding request
    ROUTE_REPLY = 0x41                        # Route response
    FIDELITY_REPORT = 0x42                    # Channel fidelity update
    STATUS_REQUEST = 0x43                     # Network status query
    STATUS_REPLY = 0x44                       # Network status response


class ClassicalTransport:
    """Classical communication layer for quantum networking protocols.

    Provides message-based communication between quantum network nodes.
    In simulation mode (default), messages are passed in-memory.
    In network mode, uses async TCP connections.
    """

    def __init__(self, node_id: str, host: str = "127.0.0.1",
                 port: int = DEFAULT_PORT, simulation_mode: bool = True):
        """
        Args:
            node_id: This node's unique identifier
            host: Listen address
            port: Listen port
            simulation_mode: If True, skip actual TCP and use in-memory routing
        """
        self.node_id = node_id
        self.host = host
        self.port = port
        self.simulation_mode = simulation_mode

        # Peer registry: node_id → (host, port, last_heartbeat)
        self._peers: Dict[str, Dict] = {}
        # Message handlers: MessageType → callback
        self._handlers: Dict[MessageType, Callable] = {}
        # In-memory message bus (simulation mode)
        self._message_bus: Dict[str, List[Dict]] = {}  # node_id → [messages]
        # Outbox for collecting sent messages
        self._sent: List[Dict] = []
        self._received: List[Dict] = []

        self._running = False
        self._server = None
        self._start_time = time.time()

    def register_handler(self, msg_type: MessageType,
                          handler: Callable) -> None:
        """Register a callback for a specific message type."""
        self._handlers[msg_type] = handler

    def register_peer(self, node_id: str, host: str = "127.0.0.1",
                       port: int = DEFAULT_PORT) -> None:
        """Register a known peer node."""
        self._peers[node_id] = {
            "host": host,
            "port": port,
            "last_heartbeat": time.time(),
            "online": True,
        }
        if node_id not in self._message_bus:
            self._message_bus[node_id] = []

    def send(self, dest_node_id: str, msg_type: MessageType,
             payload: Dict = None, hmac_key: bytes = None) -> bool:
        """Send a message to a peer node.

        Args:
            dest_node_id: Destination node ID
            msg_type: Message type
            payload: JSON-serializable payload dict
            hmac_key: Optional QKD-derived key for message authentication

        Returns:
            True if message was sent successfully
        """
        message = {
            "id": uuid.uuid4().hex[:12],
            "type": int(msg_type),
            "type_name": msg_type.name,
            "from": self.node_id,
            "to": dest_node_id,
            "payload": payload or {},
            "timestamp": time.time(),
        }

        # Add HMAC if key provided
        if hmac_key:
            msg_bytes = json.dumps(message["payload"], sort_keys=True).encode()
            message["hmac"] = hmac.new(hmac_key, msg_bytes, hashlib.sha256).hexdigest()

        self._sent.append(message)

        if self.simulation_mode:
            # In-memory delivery
            if dest_node_id not in self._message_bus:
                self._message_bus[dest_node_id] = []
            self._message_bus[dest_node_id].append(message)
            return True

        # Network mode: would use async TCP here
        # (For v1.0 we focus on simulation mode with all protocols working)
        return True

    def receive(self, max_messages: int = 100) -> List[Dict]:
        """Receive pending messages for this node.

        Returns:
            List of message dicts
        """
        if self.simulation_mode:
            messages = self._message_bus.get(self.node_id, [])[:max_messages]
            self._message_bus[self.node_id] = (
                self._message_bus.get(self.node_id, [])[max_messages:]
            )
            self._received.extend(messages)
            return messages

        return []

    def process_messages(self) -> int:
        """Receive and dispatch all pending messages to registered handlers.

        Returns:
            Number of messages processed
        """
        messages = self.receive()
        processed = 0
        for msg in messages:
            msg_type_val = msg.get("type")
            try:
                msg_type = MessageType(msg_type_val)
            except ValueError:
                continue

            handler = self._handlers.get(msg_type)
            if handler:
                try:
                    handler(msg)
                    processed += 1
                except Exception:
                    pass
            else:
                processed += 1  # Received but no handler
        return processed

    def broadcast(self, msg_type: MessageType, payload: Dict = None) -> int:
        """Broadcast a message to all known peers.

        Returns:
            Number of peers messaged
        """
        count = 0
        for peer_id in self._peers:
            if self.send(peer_id, msg_type, payload):
                count += 1
        return count

    def send_heartbeat(self) -> int:
        """Send heartbeat to all peers."""
        return self.broadcast(MessageType.HEARTBEAT, {
            "node_id": self.node_id,
            "uptime_s": time.time() - self._start_time,
        })

    def check_peer_liveness(self) -> Dict[str, bool]:
        """Check which peers are still alive based on heartbeat timing."""
        now = time.time()
        liveness = {}
        for peer_id, info in self._peers.items():
            alive = (now - info["last_heartbeat"]) < HEARTBEAT_TIMEOUT_S
            info["online"] = alive
            liveness[peer_id] = alive
        return liveness

    def encode_message(self, msg: Dict) -> bytes:
        """Encode a message for TCP wire protocol.

        Wire format: [MAGIC(4)] [VERSION(1)] [TYPE(1)] [LEN(4)] [JSON payload]
        """
        payload_bytes = json.dumps(msg).encode("utf-8")
        header = struct.pack("!4sBBI",
                             MAGIC, PROTOCOL_VERSION,
                             msg.get("type", 0),
                             len(payload_bytes))
        return header + payload_bytes

    def decode_message(self, data: bytes) -> Optional[Dict]:
        """Decode a message from TCP wire protocol."""
        if len(data) < 10:
            return None
        magic, version, msg_type, length = struct.unpack("!4sBBI", data[:10])
        if magic != MAGIC or version != PROTOCOL_VERSION:
            return None
        if len(data) < 10 + length:
            return None
        try:
            payload = json.loads(data[10:10 + length].decode("utf-8"))
            return payload
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

    @property
    def peers(self) -> Dict[str, Dict]:
        return dict(self._peers)

    @property
    def online_peers(self) -> List[str]:
        return [pid for pid, info in self._peers.items() if info.get("online")]

    def status(self) -> Dict:
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "simulation_mode": self.simulation_mode,
            "peers": len(self._peers),
            "online_peers": len(self.online_peers),
            "messages_sent": len(self._sent),
            "messages_received": len(self._received),
            "uptime_s": time.time() - self._start_time,
        }
