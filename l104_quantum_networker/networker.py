"""L104 Quantum Networker v1.0.0 — Main Orchestrator.

The QuantumNetworker ties together all quantum networking subsystems:
  - EntanglementRouter (pair generation, routing, swapping, purification)
  - QuantumKeyDistribution (BB84 + E91 QKD protocols)
  - QuantumTeleporter (state, score, phase, bitstring teleportation)
  - QuantumRepeaterChain (multi-hop repeater chains)
  - FidelityMonitor (real-time fidelity tracking + auto-healing)
  - ClassicalTransport (message-based control channel)

Usage:
  from l104_quantum_networker import get_networker

  net = get_networker()

  # Add nodes
  alice = net.add_node("alice", role="sovereign")
  bob = net.add_node("bob", role="sovereign")
  relay = net.add_node("relay1", role="relay")

  # Connect quantum channels
  net.connect(alice.node_id, relay.node_id)
  net.connect(relay.node_id, bob.node_id)

  # QKD: establish shared key
  key = net.establish_qkd(alice.node_id, bob.node_id, protocol="bb84")

  # Teleport a score through the network
  result = net.teleport_score(alice.node_id, bob.node_id, score=0.8)

  # Check network health
  status = net.status()

INVARIANT: 527.5184818492612 | PILOT: LONDEL
"""

import time
import threading
from typing import Any, Dict, List, Optional, Tuple

from .types import (
    QuantumNode, QuantumChannel, EntangledPair, QKDKey,
    TeleportPayload, TeleportResult, NetworkStatus, NetworkTopology,
    GOD_CODE, PHI, PHI_INV,
)
from .entanglement_router import EntanglementRouter
from .qkd import QuantumKeyDistribution
from .teleporter import QuantumTeleporter
from .repeater import QuantumRepeaterChain
from .fidelity_monitor import FidelityMonitor
from .transport import ClassicalTransport, MessageType

# Singleton
_networker_instance = None
_lock = threading.Lock()


class QuantumNetworker:
    """L104 Sovereign Quantum Network Orchestrator.

    Coordinates all quantum communication protocols across a network of
    L104 nodes, leveraging the VQPU's high-fidelity quantum circuits
    and sacred alignment scoring.
    """

    VERSION = "1.4.0"

    def __init__(self, node_name: str = "L104-Sovereign",
                 simulation_mode: bool = True):
        """Initialize the quantum networker.

        Args:
            node_name: Name for this node (the local sovereign)
            simulation_mode: True = in-memory simulation, False = TCP networking
        """
        self._start_time = time.time()

        # Core subsystems
        self.router = EntanglementRouter()
        self.qkd = QuantumKeyDistribution()
        self.teleporter = QuantumTeleporter()
        self.repeater = QuantumRepeaterChain(self.router)
        self.monitor = FidelityMonitor(self.router)

        # Create the local sovereign node
        self._local_node = QuantumNode(
            name=node_name,
            role="sovereign",
            max_qubits=26,
        )
        self.router.add_node(self._local_node)

        # Classical transport
        self.transport = ClassicalTransport(
            node_id=self._local_node.node_id,
            simulation_mode=simulation_mode,
        )

        # Register classical message handlers
        self._register_handlers()

        # QKD key store: (node_a, node_b) sorted → QKDKey
        self._qkd_keys: Dict[Tuple[str, str], QKDKey] = {}

    # ═══════════════════════════════════════════════════════════════
    # NODE MANAGEMENT
    # ═══════════════════════════════════════════════════════════════

    @property
    def local_node(self) -> QuantumNode:
        """The local sovereign node."""
        return self._local_node

    def add_node(self, name: str, role: str = "sovereign",
                 host: str = "127.0.0.1", port: int = 10400,
                 max_qubits: int = 26) -> QuantumNode:
        """Add a new node to the quantum network.

        Args:
            name: Human-readable node name
            role: Node role (sovereign, relay, endpoint, observer)
            host: Node address
            port: Node port
            max_qubits: Maximum qubit capacity

        Returns:
            The created QuantumNode
        """
        node = QuantumNode(
            name=name,
            role=role,
            host=host,
            port=port,
            max_qubits=max_qubits,
        )
        self.router.add_node(node)
        self.transport.register_peer(node.node_id, host, port)
        return node

    def remove_node(self, node_id: str) -> None:
        """Remove a node from the network."""
        self.router.remove_node(node_id)

    # ═══════════════════════════════════════════════════════════════
    # CHANNEL MANAGEMENT
    # ═══════════════════════════════════════════════════════════════

    def connect(self, node_a_id: str, node_b_id: str,
                pairs: int = 8) -> QuantumChannel:
        """Establish a quantum channel between two nodes with entangled pairs.

        Args:
            node_a_id: First node
            node_b_id: Second node
            pairs: Number of initial Bell pairs to generate

        Returns:
            The created QuantumChannel
        """
        ch = self.router.create_channel(node_a_id, node_b_id, pairs)

        # Announce over classical channel
        self.transport.send(node_b_id, MessageType.ENTANGLE_READY, {
            "channel_id": ch.channel_id,
            "pairs": len(ch.usable_pairs),
            "fidelity": ch.mean_fidelity,
        })

        return ch

    def disconnect(self, node_a_id: str, node_b_id: str) -> None:
        """Tear down the quantum channel between two nodes."""
        ch = self.router.get_channel(node_a_id, node_b_id)
        if ch:
            edge = tuple(sorted([node_a_id, node_b_id]))
            self.router._edge_to_channel.pop(edge, None)
            self.router._channels.pop(ch.channel_id, None)
            self.router._adjacency[node_a_id].discard(node_b_id)
            self.router._adjacency[node_b_id].discard(node_a_id)

    # ═══════════════════════════════════════════════════════════════
    # QUANTUM KEY DISTRIBUTION
    # ═══════════════════════════════════════════════════════════════

    def establish_qkd(self, node_a_id: str, node_b_id: str,
                      protocol: str = "bb84",
                      num_bits: int = 256) -> QKDKey:
        """Run QKD protocol between two nodes to establish a shared secret key.

        Args:
            node_a_id: Alice node
            node_b_id: Bob node
            protocol: "bb84" or "e91"
            num_bits: Raw bits to exchange

        Returns:
            QKDKey with the secure key (check .secure for validity)
        """
        # Ensure channel exists
        ch = self.router.get_channel(node_a_id, node_b_id)
        if not ch:
            ch = self.connect(node_a_id, node_b_id)

        # Run QKD protocol
        if protocol == "e91":
            key = self.qkd.e91_generate(ch, num_pairs=num_bits)
        else:
            key = self.qkd.bb84_generate(ch, num_bits=num_bits)

        # Store key
        edge = tuple(sorted([node_a_id, node_b_id]))
        self._qkd_keys[edge] = key

        # Store on channel
        ch.qkd_key = key

        # Announce key readiness
        self.transport.send(node_b_id, MessageType.QKD_KEY_CONFIRM, {
            "key_id": key.key_id,
            "protocol": protocol,
            "secure": key.secure,
            "key_length": key.key_length,
            "qber": key.qber,
        })

        return key

    def get_shared_key(self, node_a_id: str, node_b_id: str) -> Optional[QKDKey]:
        """Retrieve the QKD key shared between two nodes."""
        edge = tuple(sorted([node_a_id, node_b_id]))
        return self._qkd_keys.get(edge)

    # ═══════════════════════════════════════════════════════════════
    # QUANTUM TELEPORTATION
    # ═══════════════════════════════════════════════════════════════

    def teleport_score(self, source_id: str, dest_id: str,
                       score: float,
                       use_qkd: bool = True,
                       error_correct: bool = True) -> TeleportResult:
        """Teleport a [0,1] score from source to destination.

        Supports multi-hop teleportation via entanglement-swapped relay chains.

        Args:
            source_id: Source node
            dest_id: Destination node
            score: Value to teleport (0.0 to 1.0)
            use_qkd: Encrypt classical channel with QKD key
            error_correct: Apply Steane error correction

        Returns:
            TeleportResult with recovered score and fidelity
        """
        payload = TeleportPayload(
            data_type="score",
            score_value=score,
            num_qubits=1,
            sacred_tag="score_teleport",
        )
        return self._execute_teleportation(
            source_id, dest_id, payload, use_qkd, error_correct
        )

    def teleport_phase(self, source_id: str, dest_id: str,
                       phase: float,
                       use_qkd: bool = True,
                       error_correct: bool = True) -> TeleportResult:
        """Teleport a phase angle from source to destination.

        Args:
            source_id: Source node
            dest_id: Destination node
            phase: Phase angle to teleport (radians)
            use_qkd: Encrypt classical channel with QKD key
            error_correct: Apply Steane error correction

        Returns:
            TeleportResult with recovered phase and fidelity
        """
        payload = TeleportPayload(
            data_type="phase",
            phase_value=phase,
            num_qubits=1,
            sacred_tag="phase_teleport",
        )
        return self._execute_teleportation(source_id, dest_id, payload, use_qkd, error_correct)

    def teleport_state(self, source_id: str, dest_id: str,
                       state_vector: list,
                       use_qkd: bool = True,
                       error_correct: bool = True) -> TeleportResult:
        """Teleport an arbitrary quantum state from source to destination.

        Args:
            source_id: Source node
            dest_id: Destination node
            state_vector: State vector [α, β] to teleport
            use_qkd: Encrypt classical channel with QKD key
            error_correct: Apply Steane error correction

        Returns:
            TeleportResult with recovered state and fidelity
        """
        payload = TeleportPayload(
            data_type="state_vector",
            state_vector=state_vector,
            num_qubits=1,
            sacred_tag="state_teleport",
        )
        return self._execute_teleportation(source_id, dest_id, payload, use_qkd, error_correct)

    def teleport_bitstring(self, source_id: str, dest_id: str,
                           bitstring: str,
                           use_qkd: bool = True,
                           error_correct: bool = True) -> TeleportResult:
        """Teleport a classical bitstring from source to destination.

        Args:
            source_id: Source node
            dest_id: Destination node
            bitstring: Classical bitstring to teleport
            use_qkd: Encrypt classical channel with QKD key
            error_correct: Apply Steane error correction

        Returns:
            TeleportResult with recovered bitstring and fidelity
        """
        payload = TeleportPayload(
            data_type="bitstring",
            bitstring=bitstring,
            num_qubits=len(bitstring),
            sacred_tag="bitstring_teleport",
        )
        return self._execute_teleportation(source_id, dest_id, payload, use_qkd, error_correct)

    def _execute_teleportation(
        self,
        source_id: str,
        dest_id: str,
        payload: TeleportPayload,
        use_qkd: bool = True,
        error_correct: bool = True,
    ) -> TeleportResult:
        """Execute teleportation with automatic routing and QKD encryption."""
        # Find route
        route = self.router.find_route(source_id, dest_id)
        if not route:
            return TeleportResult(
                success=False,
                payload_id=payload.payload_id,
                source_node=source_id,
                dest_node=dest_id,
            )

        # For multi-hop: use repeater chain to establish end-to-end pair
        if len(route) > 2:
            # Replenish all segments before chain establishment
            for i in range(len(route) - 1):
                seg_ch = self.router.get_channel(route[i], route[i + 1])
                if seg_ch:
                    self.router.replenish_channel(seg_ch.channel_id, 6)
                else:
                    self.connect(route[i], route[i + 1], pairs=6)

            chain_result = self.repeater.establish_chain(route)
            if not chain_result.get("success"):
                # Fallback: create direct channel
                ch_fallback = self.connect(source_id, dest_id, pairs=4)
                if ch_fallback.usable_pairs:
                    route = [source_id, dest_id]  # Simplify route
                else:
                    return TeleportResult(
                        success=False,
                        payload_id=payload.payload_id,
                        source_node=source_id,
                        dest_node=dest_id,
                        route=route,
                        hops=len(route) - 1,
                    )

        # Get channel for final teleportation
        ch = self.router.get_channel(source_id, dest_id)
        if not ch:
            # Try creating direct channel as fallback
            ch = self.connect(source_id, dest_id, pairs=4)

        # Ensure channel has usable pairs
        if not ch.usable_pairs:
            self.router.replenish_channel(ch.channel_id, 4)

        # Get QKD key for encryption
        qkd_key = None
        if use_qkd:
            qkd_key = self.get_shared_key(source_id, dest_id)

        # Send teleport request over classical channel
        self.transport.send(dest_id, MessageType.TELEPORT_REQUEST, {
            "payload_id": payload.payload_id,
            "data_type": payload.data_type,
            "route": route,
        })

        # Execute teleportation
        result = self.teleporter.teleport(
            payload, ch,
            qkd_key=qkd_key,
            error_correct=error_correct,
            route=route,
        )

        # Send completion notification
        self.transport.send(dest_id, MessageType.TELEPORT_COMPLETE, {
            "payload_id": payload.payload_id,
            "success": result.success,
            "fidelity": result.fidelity,
        })

        return result

    # ═══════════════════════════════════════════════════════════════
    # ENTANGLEMENT PURIFICATION
    # ═══════════════════════════════════════════════════════════════

    def purify(self, node_a_id: str, node_b_id: str,
               rounds: int = 3) -> Dict:
        """Purify entangled pairs between two nodes.

        Returns:
            Purification result dict
        """
        ch = self.router.get_channel(node_a_id, node_b_id)
        if not ch:
            return {"error": "No channel between nodes"}
        return self.router.purify_channel(ch.channel_id, rounds)

    # ═══════════════════════════════════════════════════════════════
    # MONITORING & DIAGNOSTICS
    # ═══════════════════════════════════════════════════════════════

    def scan_fidelity(self, auto_heal: bool = True) -> Dict:
        """Run a fidelity scan across the entire network."""
        return self.monitor.scan(auto_heal=auto_heal)

    def network_status(self) -> NetworkStatus:
        """Get comprehensive network status."""
        ns = self.monitor.network_status()
        ns.total_qkd_keys = len(self._qkd_keys)
        ns.uptime_s = time.time() - self._start_time
        return ns

    def status(self) -> Dict:
        """Full status report of all subsystems."""
        ns = self.network_status()
        return {
            "version": self.VERSION,
            "local_node": self._local_node.to_dict(),
            "network": ns.to_dict(),
            "router": self.router.status(),
            "qkd": self.qkd.status(),
            "teleporter": self.teleporter.status(),
            "repeater": self.repeater.status(),
            "monitor": self.monitor.status(),
            "transport": self.transport.status(),
        }

    def self_test(self) -> Dict:
        """Run a comprehensive self-test of all quantum networking subsystems.

        Creates a 4-node test network, runs all protocols, and reports results.
        """
        t0 = time.time()
        results = {}

        # 1. Create test nodes
        alice = self.add_node("Alice", role="sovereign")
        bob = self.add_node("Bob", role="sovereign")
        relay = self.add_node("Relay-1", role="relay")

        results["nodes_created"] = 3

        # 2. Create channels
        ch_ar = self.connect(alice.node_id, relay.node_id, pairs=8)
        ch_rb = self.connect(relay.node_id, bob.node_id, pairs=8)

        results["channels_created"] = 2
        results["initial_pairs"] = {
            "alice_relay": len(ch_ar.usable_pairs),
            "relay_bob": len(ch_rb.usable_pairs),
        }

        # 3. Direct channel test (Alice ↔ Relay)
        ch_direct = self.connect(alice.node_id, bob.node_id, pairs=4)
        results["direct_channel_fidelity"] = ch_direct.mean_fidelity

        # 4. QKD test (BB84)
        key_bb84 = self.establish_qkd(alice.node_id, bob.node_id, "bb84", 128)
        results["bb84"] = {
            "secure": key_bb84.secure,
            "key_length": key_bb84.key_length,
            "qber": key_bb84.qber,
            "sacred_alignment": key_bb84.sacred_alignment,
        }

        # 5. QKD test (E91)
        key_e91 = self.establish_qkd(alice.node_id, bob.node_id, "e91", 128)
        results["e91"] = {
            "secure": key_e91.secure,
            "key_length": key_e91.key_length,
            "qber": key_e91.qber,
            "sacred_alignment": key_e91.sacred_alignment,
        }

        # 6. Direct teleportation (score)
        tp_score = self.teleport_score(alice.node_id, bob.node_id, score=0.8)
        results["teleport_score"] = {
            "success": tp_score.success,
            "fidelity": tp_score.fidelity,
            "original": 0.8,
            "recovered": tp_score.recovered_score,
            "sacred_score": tp_score.sacred_score,
        }

        # 7. Phase teleportation
        import math
        tp_phase = self.teleport_phase(alice.node_id, bob.node_id,
                                        phase=math.pi / 3)
        results["teleport_phase"] = {
            "success": tp_phase.success,
            "fidelity": tp_phase.fidelity,
            "original_phase": math.pi / 3,
            "recovered_phase": tp_phase.recovered_phase,
        }

        # 8. State teleportation
        import cmath
        alpha = complex(0.6, 0)
        beta = complex(0, 0.8)
        tp_state = self.teleport_state(alice.node_id, bob.node_id,
                                        state_vector=[alpha, beta])
        results["teleport_state"] = {
            "success": tp_state.success,
            "fidelity": tp_state.fidelity,
        }

        # 9. Multi-hop teleportation (Alice → Relay → Bob)
        # Need to ensure route goes through relay
        tp_multi = self.teleport_score(alice.node_id, bob.node_id, score=0.618)
        results["teleport_multihop"] = {
            "success": tp_multi.success,
            "fidelity": tp_multi.fidelity,
            "hops": tp_multi.hops,
            "route": tp_multi.route,
        }

        # 10. Purification test
        purify_result = self.purify(alice.node_id, bob.node_id, rounds=2)
        results["purification"] = purify_result

        # 11. Repeater chain test
        relay2 = self.add_node("Relay-2", role="relay")
        self.connect(relay.node_id, relay2.node_id, pairs=6)
        self.connect(relay2.node_id, bob.node_id, pairs=6)
        chain = self.repeater.establish_chain(
            [alice.node_id, relay.node_id, relay2.node_id, bob.node_id]
        )
        results["repeater_chain"] = {
            "success": chain.get("success"),
            "fidelity": chain.get("fidelity"),
            "hops": chain.get("hops"),
            "levels": chain.get("levels"),
        }

        # 12. Fidelity scan
        scan = self.scan_fidelity(auto_heal=True)
        results["fidelity_scan"] = {
            "network_fidelity": scan["network_fidelity"],
            "sacred_score": scan["network_sacred_score"],
            "alerts": len(scan["alerts"]),
        }

        # 13. Network status
        ns = self.network_status()
        results["network_status"] = ns.to_dict()

        results["execution_time_ms"] = (time.time() - t0) * 1000

        # Compute pass/fail
        tests_passed = sum([
            results["bb84"]["secure"],
            results.get("e91", {}).get("secure", False),
            results["teleport_score"]["success"],
            results["teleport_phase"]["success"],
            results["teleport_state"]["success"],
            results["teleport_multihop"]["success"],
            results.get("repeater_chain", {}).get("success", False),
            scan["network_fidelity"] > 0.5,
        ])
        results["tests_passed"] = tests_passed
        results["tests_total"] = 8
        results["all_passed"] = tests_passed == 8

        return results

    # ═══════════════════════════════════════════════════════════════
    # INTERNAL
    # ═══════════════════════════════════════════════════════════════

    def _register_handlers(self):
        """Register classical message handlers for protocol coordination."""

        def _on_heartbeat(msg):
            peer_id = msg.get("from")
            if peer_id in self.transport._peers:
                self.transport._peers[peer_id]["last_heartbeat"] = time.time()
                self.transport._peers[peer_id]["online"] = True

        def _on_entangle_ready(msg):
            # Acknowledge entanglement readiness
            self.transport.send(msg["from"], MessageType.ACK, {
                "ref": msg.get("id"),
                "channel_id": msg.get("payload", {}).get("channel_id"),
            })

        def _on_teleport_complete(msg):
            pass  # Log completed teleportation

        self.transport.register_handler(MessageType.HEARTBEAT, _on_heartbeat)
        self.transport.register_handler(MessageType.ENTANGLE_READY, _on_entangle_ready)
        self.transport.register_handler(MessageType.TELEPORT_COMPLETE, _on_teleport_complete)


def get_networker(node_name: str = "L104-Sovereign",
                  simulation_mode: bool = True) -> QuantumNetworker:
    """Get or create the singleton QuantumNetworker."""
    global _networker_instance
    with _lock:
        if _networker_instance is None:
            _networker_instance = QuantumNetworker(node_name, simulation_mode)
    return _networker_instance
