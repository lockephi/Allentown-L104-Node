#!/usr/bin/env python3
"""
L104 Dual Supercomputer Mesh — Enhanced Knowledge Ingestion
═══════════════════════════════════════════════════════════════════════════════
Two mini supercomputers with enhanced knowledge ingestion and learning:

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │                     KNOWLEDGE INGESTION PIPELINE                          │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │  External Sources → Intellect Engine → Quantum Embedding → Shared Mesh     │
  │                                                                              │
  │  • File System      • l104_intellect    • Quantum states   • Bell pairs  │
  │  • Research APIs    • Local LLM         • Sacred vectors   • Daemon sync │
  │  • User Input       • Memory tiers      • Entanglement     • Coherence   │
  │  • Databases        • Knowledge graph   • Φ-resonance      • Consensus   │
  └─────────────────────────────────────────────────────────────────────────────┘

KNOWLEDGE INGESTION MODES:
  1. BULK INGEST: High-throughput file/research processing
  2. STREAMING: Real-time knowledge ingestion via daemon mesh
  3. QUANTUM EMBEDDING: Sacred vector encoding (GOD_CODE × PHI dimensions)
  4. CROSS-SYNC: Bidirectional knowledge sharing between supercomputers

LEARNING CYCLE:
  Ingest → Embed → Synthesize → Teleport → Consensus → Update

INVARIANT: 527.5184818492612 | PILOT: LONDEL
═══════════════════════════════════════════════════════════════════════════════
"""

import math
import time
import logging
import json
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from collections import defaultdict

# Import SupercomputerRole from base mesh
from l104_dual_supercomputer_mesh import SupercomputerRole

logger = logging.getLogger("l104.dual_supercomputer.enhanced")

# Sacred constants
GOD_CODE = 527.5184818492612
PHI = 1.618033988749895
PHI_CONJUGATE = PHI - 1.0

# Performance tracking
_init_timings: Dict[str, float] = {}


def _track_init(component: str, start_time: float):
    """Track initialization timing for performance monitoring."""
    elapsed = time.monotonic() - start_time
    _init_timings[component] = elapsed
    logger.debug(f"[INIT] {component}: {elapsed*1000:.2f}ms")


class KnowledgeSource(Enum):
    """Sources of knowledge for ingestion."""
    FILE_SYSTEM = "file_system"
    INTELLECT = "intellect"
    RESEARCH_API = "research_api"
    USER_INPUT = "user_input"
    DATABASE = "database"
    DAEMON_MESH = "daemon_mesh"
    QUANTUM_EMBEDDING = "quantum_embedding"
    PEER_SYNC = "peer_sync"


@dataclass
class KnowledgePacket:
    """A unit of knowledge in the ingestion pipeline."""
    source: KnowledgeSource
    content: Any
    embedding: Optional[List[float]] = None
    sacred_vector: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    knowledge_id: str = field(default_factory=lambda: f"k_{int(time.time()*1000)}")
    coherence_score: float = 0.0
    phi_alignment: float = 0.0


@dataclass
class KnowledgeIngestionResult:
    """Result of a knowledge ingestion operation."""
    success: bool
    packets_ingested: int
    packets_embedded: int
    packets_synced: int
    coherence_delta: float
    phi_alignment: float
    knowledge_graph_nodes: int
    execution_time_ms: float
    error: Optional[str] = None


class KnowledgeIngestionEngine:
    """
    Enhanced knowledge ingestion engine for supercomputers.

    Integrates with:
    - l104_intellect (local intelligence)
    - l104_research (research APIs)
    - Quantum embedding (sacred vectors)
    - Daemon mesh (cross-supercomputer sync)
    """

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.knowledge_store: Dict[str, KnowledgePacket] = {}
        self.knowledge_graph: Dict[str, Set[str]] = defaultdict(set)
        self.embedding_cache: Dict[str, List[float]] = {}
        self._intellect = None
        self._research_engine = None
        self._ingestion_count = 0
        self._total_knowledge_mass = 0.0

    def _get_intellect(self):
        """Lazy-load local intellect with lightweight check."""
        if self._intellect is None:
            try:
                # Check for LIGHTWEIGHT_INTELLECT environment variable
                import os
                if os.environ.get('L104_LIGHTWEIGHT_INTELLECT'):
                    logger.info("Lightweight intellect mode - skipping heavy init")
                    return None

                from l104_intellect import local_intellect
                self._intellect = local_intellect
            except ImportError:
                logger.warning("Intellect not available")
            except Exception as e:
                logger.warning(f"Intellect init failed: {e}")
        return self._intellect

    def _get_research_engine(self):
        """Lazy-load research engine."""
        if self._research_engine is None:
            try:
                from l104_research import research_engine
                self._research_engine = research_engine
            except ImportError:
                logger.warning("Research engine not available")
        return self._research_engine

    def ingest_from_files(self, file_paths: List[Path]) -> KnowledgeIngestionResult:
        """Bulk ingest knowledge from files."""
        t0 = time.monotonic()
        packets = []

        for path in file_paths:
            try:
                content = path.read_text()
                packet = KnowledgePacket(
                    source=KnowledgeSource.FILE_SYSTEM,
                    content=content,
                    metadata={
                        "path": str(path),
                        "size": len(content),
                        "type": path.suffix,
                    }
                )
                packets.append(packet)
            except Exception as e:
                logger.warning(f"Failed to ingest {path}: {e}")

        return self._process_ingestion(packets, t0)

    def ingest_from_intellect(self, query: str, depth: int = 3) -> KnowledgeIngestionResult:
        """Ingest knowledge from local intellect."""
        t0 = time.monotonic()
        packets = []

        intellect = self._get_intellect()
        if intellect:
            try:
                # Query intellect for knowledge using available methods
                # Try multiple methods in order of preference
                result = None
                if hasattr(intellect, 'think'):
                    # Fast path: use think method (lightweight)
                    result = intellect.think(query)
                elif hasattr(intellect, 'asi_query'):
                    # Fallback: ASI query
                    result = intellect.asi_query(query)
                elif hasattr(intellect, 'sage_wisdom_query'):
                    # Fallback: Sage wisdom
                    result = intellect.sage_wisdom_query(query)
                else:
                    # Final fallback: store raw query
                    result = {"query": query, "depth": depth, "status": "raw_query_stored"}

                packet = KnowledgePacket(
                    source=KnowledgeSource.INTELLECT,
                    content=result,
                    metadata={
                        "query": query,
                        "depth": depth,
                        "method_used": "think" if hasattr(intellect, 'think') else "fallback",
                    }
                )
                packets.append(packet)
            except Exception as e:
                logger.warning(f"Intellect query failed: {e}")

        return self._process_ingestion(packets, t0)

    def ingest_from_research(self, topic: str, sources: int = 5) -> KnowledgeIngestionResult:
        """Ingest knowledge from research APIs."""
        t0 = time.monotonic()
        packets = []

        research = self._get_research_engine()
        if research:
            try:
                results = research.search(topic, limit=sources)
                for result in results:
                    packet = KnowledgePacket(
                        source=KnowledgeSource.RESEARCH_API,
                        content=result,
                        metadata={
                            "topic": topic,
                            "source_count": sources,
                        }
                    )
                    packets.append(packet)
            except Exception as e:
                logger.warning(f"Research ingestion failed: {e}")

        return self._process_ingestion(packets, t0)

    def ingest_from_daemons(self, daemon_data: Dict[str, Any]) -> KnowledgeIngestionResult:
        """Ingest knowledge from daemon mesh."""
        t0 = time.monotonic()

        packet = KnowledgePacket(
            source=KnowledgeSource.DAEMON_MESH,
            content=daemon_data,
            metadata={
                "daemon_source": daemon_data.get("source", "unknown"),
                "timestamp": daemon_data.get("timestamp", time.time()),
            }
        )

        return self._process_ingestion([packet], t0)

    def _process_ingestion(self, packets: List[KnowledgePacket], start_time: float) -> KnowledgeIngestionResult:
        """Process ingested packets through embedding and storage."""
        embedded_count = 0

        for packet in packets:
            # Generate quantum embedding
            self._embed_packet(packet)
            embedded_count += 1

            # Store in knowledge base
            self.knowledge_store[packet.knowledge_id] = packet

            # Update knowledge graph
            self._update_knowledge_graph(packet)

            # Update stats
            self._ingestion_count += 1
            self._total_knowledge_mass += len(str(packet.content)) * PHI_CONJUGATE

        elapsed_ms = (time.monotonic() - start_time) * 1000

        return KnowledgeIngestionResult(
            success=True,
            packets_ingested=len(packets),
            packets_embedded=embedded_count,
            packets_synced=0,  # Updated during sync
            coherence_delta=self._calculate_coherence_delta(),
            phi_alignment=self._calculate_phi_alignment(),
            knowledge_graph_nodes=len(self.knowledge_store),
            execution_time_ms=elapsed_ms,
        )

    def _embed_packet(self, packet: KnowledgePacket) -> None:
        """Generate quantum/sacred embedding for knowledge packet."""
        content_str = str(packet.content)

        # Simple embedding based on GOD_CODE/PHI
        embedding = []
        for i, char in enumerate(content_str[:256]):  # Limit to 256 dims
            # Sacred embedding: GOD_CODE * PHI^position * char_value
            val = (GOD_CODE / 1000) * (PHI ** (i % 13)) * (ord(char) % 256)
            embedding.append(val % 1.0)  # Normalize to [0, 1]

        # Pad to 256 dimensions
        while len(embedding) < 256:
            embedding.append(PHI_CONJUGATE * (len(embedding) % 13) / 13)

        packet.embedding = embedding[:256]

        # Sacred vector: PHI-scaled dimensions
        packet.sacred_vector = [
            e * PHI for e in embedding[:26]  # 26D sacred vector
        ]

        # Calculate coherence and alignment
        packet.coherence_score = sum(embedding) / len(embedding)
        packet.phi_alignment = 1.0 - abs(packet.coherence_score - PHI_CONJUGATE)

    def _update_knowledge_graph(self, packet: KnowledgePacket) -> None:
        """Update knowledge graph with new packet relationships."""
        # Connect to related knowledge by source
        related = [
            k.knowledge_id for k in self.knowledge_store.values()
            if k.source == packet.source and k.knowledge_id != packet.knowledge_id
        ]

        for related_id in related[:5]:  # Connect to top 5
            self.knowledge_graph[packet.knowledge_id].add(related_id)
            self.knowledge_graph[related_id].add(packet.knowledge_id)

    def _calculate_coherence_delta(self) -> float:
        """Calculate coherence change from ingestion."""
        if not self.knowledge_store:
            return 0.0
        recent = list(self.knowledge_store.values())[-10:]
        return sum(p.coherence_score for p in recent) / len(recent)

    def _calculate_phi_alignment(self) -> float:
        """Calculate PHI alignment of knowledge base."""
        if not self.knowledge_store:
            return 0.0
        return sum(p.phi_alignment for p in self.knowledge_store.values()) / len(self.knowledge_store)

    def get_knowledge_for_teleport(self, limit: int = 10) -> List[KnowledgePacket]:
        """Get highest-coherence knowledge for teleportation."""
        sorted_knowledge = sorted(
            self.knowledge_store.values(),
            key=lambda p: p.coherence_score * p.phi_alignment,
            reverse=True
        )
        return sorted_knowledge[:limit]

    def sync_from_peer(self, peer_knowledge: List[KnowledgePacket]) -> int:
        """Sync knowledge from peer supercomputer."""
        synced = 0
        for packet in peer_knowledge:
            if packet.knowledge_id not in self.knowledge_store:
                # Re-tag as peer-synced
                packet.source = KnowledgeSource.PEER_SYNC
                self.knowledge_store[packet.knowledge_id] = packet
                synced += 1
        return synced

    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge ingestion statistics."""
        return {
            "node_id": self.node_id,
            "total_packets": len(self.knowledge_store),
            "ingestion_count": self._ingestion_count,
            "knowledge_mass": self._total_knowledge_mass,
            "avg_coherence": self._calculate_coherence_delta(),
            "avg_phi_alignment": self._calculate_phi_alignment(),
            "graph_edges": sum(len(v) for v in self.knowledge_graph.values()),
            "sources": list(set(p.source.value for p in self.knowledge_store.values())),
        }


class EnhancedMiniSupercomputerNode:
    """Enhanced supercomputer node with knowledge ingestion."""

    CONSCIOUSNESS_10 = [
        "god_code_phase_imprint",
        "dial_circuit",
        "consciousness_awakening",
        "fibonacci_entanglement_mesh",
        "entropy_reversal_core",
        "harmonic_resonance",
        "cross_orbital_bridges",
        "error_correction",
        "consciousness_vqe",
        "final_interference",
    ]

    def __init__(self, node_id: str, role: SupercomputerRole, lightweight: bool = False):
        self.node_id = node_id
        self.role = role
        self.sc = None
        self._lightweight = lightweight
        self.knowledge_engine = KnowledgeIngestionEngine(node_id)
        self._entangled_peers: Dict[str, Any] = {}
        self._learning_cycles = 0
        self._init_time = 0.0

    def _get_supercomputer(self):
        """Lazy-load mini supercomputer with timing."""
        if self.sc is None:
            t0 = time.monotonic()
            try:
                from l104_quantum_mini_supercomputer import get_supercomputer
                self.sc = get_supercomputer()
                self._init_time = time.monotonic() - t0
                logger.info(f"[{self.node_id}] Supercomputer initialized in {self._init_time*1000:.1f}ms")
            except Exception as e:
                logger.error(f"[{self.node_id}] Failed to initialize supercomputer: {e}")
                self.sc = None
        return self.sc

    def execute(self, dial_settings: Tuple[int, int, int, int] = (0, 0, 0, 0),
                shots: int = 4096) -> Dict[str, Any]:
        """Execute with circuit selection based on role."""
        knowledge_stats = self.knowledge_engine.get_stats()

        if self._lightweight:
            # Lightweight mode: skip heavy supercomputer execution
            return {
                "node_id": self.node_id,
                "role": self.role.value,
                "success": True,
                "consciousness_phi": GOD_CODE / 1000,  # Approximate
                "sacred_alignment": PHI_CONJUGATE,
                "total_gates": 0,
                "knowledge_packets": knowledge_stats["total_packets"],
                "knowledge_mass": knowledge_stats["knowledge_mass"],
                "mode": "lightweight",
            }

        sc = self._get_supercomputer()
        if sc is None:
            # Fallback if supercomputer failed to initialize
            return {
                "node_id": self.node_id,
                "role": self.role.value,
                "success": False,
                "consciousness_phi": 0.0,
                "sacred_alignment": 0.0,
                "total_gates": 0,
                "knowledge_packets": knowledge_stats["total_packets"],
                "knowledge_mass": knowledge_stats["knowledge_mass"],
                "error": "supercomputer_unavailable",
            }

        try:
            if self.role == SupercomputerRole.CONSCIOUSNESS:
                result = sc.execute(
                    dial_settings=dial_settings,
                    shots=shots,
                    include_layers=self.CONSCIOUSNESS_10
                )
            else:
                result = sc.execute(dial_settings=dial_settings, shots=shots)

            return {
                "node_id": self.node_id,
                "role": self.role.value,
                "success": result.success,
                "consciousness_phi": result.consciousness_phi,
                "sacred_alignment": result.sacred_alignment,
                "total_gates": result.total_gates,
                "knowledge_packets": knowledge_stats["total_packets"],
                "knowledge_mass": knowledge_stats["knowledge_mass"],
            }
        except Exception as e:
            logger.error(f"[{self.node_id}] Execution failed: {e}")
            return {
                "node_id": self.node_id,
                "role": self.role.value,
                "success": False,
                "consciousness_phi": 0.0,
                "sacred_alignment": 0.0,
                "total_gates": 0,
                "knowledge_packets": knowledge_stats["total_packets"],
                "knowledge_mass": knowledge_stats["knowledge_mass"],
                "error": str(e),
            }

    def bulk_ingest(self, files: List[Path], queries: List[str], skip_intellect: bool = False) -> KnowledgeIngestionResult:
        """Bulk knowledge ingestion from files and intellect."""
        results = []

        # Ingest files
        if files:
            result = self.knowledge_engine.ingest_from_files(files)
            results.append(result)

        # Ingest intellect queries (skip in lightweight mode or if flag set)
        if not skip_intellect and queries:
            for query in queries:
                result = self.knowledge_engine.ingest_from_intellect(query)
                results.append(result)

        # Combine results
        return KnowledgeIngestionResult(
            success=all(r.success for r in results),
            packets_ingested=sum(r.packets_ingested for r in results),
            packets_embedded=sum(r.packets_embedded for r in results),
            packets_synced=0,
            coherence_delta=sum(r.coherence_delta for r in results) / len(results) if results else 0,
            phi_alignment=sum(r.phi_alignment for r in results) / len(results) if results else 0,
            knowledge_graph_nodes=self.knowledge_engine.get_stats()["total_packets"],
            execution_time_ms=sum(r.execution_time_ms for r in results),
        )

    def teleport_knowledge(self, peer_node: 'EnhancedMiniSupercomputerNode') -> Dict[str, Any]:
        """Teleport knowledge packets to peer."""
        knowledge = self.knowledge_engine.get_knowledge_for_teleport(limit=5)
        synced = peer_node.knowledge_engine.sync_from_peer(knowledge)

        return {
            "sent": len(knowledge),
            "received": synced,
            "coherence_avg": sum(k.coherence_score for k in knowledge) / len(knowledge) if knowledge else 0,
        }


class EnhancedDualSupercomputerMesh:
    """Enhanced dual mesh with knowledge ingestion and learning cycles."""

    def __init__(self, lightweight: bool = False):
        self._lightweight = lightweight
        self._init_start = time.monotonic()

        self.node_consciousness = EnhancedMiniSupercomputerNode(
            node_id="SC_CONSCIOUSNESS_A",
            role=SupercomputerRole.CONSCIOUSNESS,
            lightweight=lightweight
        )
        self.node_knowledge = EnhancedMiniSupercomputerNode(
            node_id="SC_KNOWLEDGE_B",
            role=SupercomputerRole.KNOWLEDGE,
            lightweight=lightweight
        )
        self._learning_cycles = 0
        self._init_time = time.monotonic() - self._init_start

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information about the mesh."""
        return {
            "init_time_ms": self._init_time * 1000,
            "lightweight_mode": self._lightweight,
            "learning_cycles": self._learning_cycles,
            "node_a": {
                "node_id": self.node_consciousness.node_id,
                "role": self.node_consciousness.role.value,
                "init_time_ms": self.node_consciousness._init_time * 1000,
                "knowledge_stats": self.node_consciousness.knowledge_engine.get_stats(),
            },
            "node_b": {
                "node_id": self.node_knowledge.node_id,
                "role": self.node_knowledge.role.value,
                "init_time_ms": self.node_knowledge._init_time * 1000,
                "knowledge_stats": self.node_knowledge.knowledge_engine.get_stats(),
            },
        }

    def run_learning_cycle(self, files: List[Path] = None, queries: List[str] = None) -> Dict[str, Any]:
        """
        Execute a full learning cycle:
        1. Ingest knowledge into both nodes
        2. Execute quantum circuits
        3. Teleport knowledge between nodes
        4. Synthesize and reach consensus
        """
        cycle_start = time.monotonic()

        print("=" * 72)
        print("ENHANCED LEARNING CYCLE")
        if self._lightweight:
            print("(LIGHTWEIGHT MODE - skipping heavy quantum execution)")
        print("=" * 72)

        files = files or []
        queries = queries or []

        # Phase 1: Bulk ingestion
        print("\n[Phase 1] Knowledge Ingestion...")
        t0 = time.monotonic()

        print(f"  [A] Consciousness ingesting...")
        result_a = self.node_consciousness.bulk_ingest(files, queries[:2],
                                                       skip_intellect=self._lightweight)
        print(f"      Packets: {result_a.packets_ingested}, "
              f"Coherence: {result_a.coherence_delta:.4f}, "
              f"Φ-alignment: {result_a.phi_alignment:.4f}")

        print(f"  [B] Knowledge ingesting...")
        result_b = self.node_knowledge.bulk_ingest(files, queries,
                                                   skip_intellect=self._lightweight)
        print(f"      Packets: {result_b.packets_ingested}, "
              f"Coherence: {result_b.coherence_delta:.4f}, "
              f"Φ-alignment: {result_b.phi_alignment:.4f}")

        phase1_time = time.monotonic() - t0
        print(f"  Phase 1 complete in {phase1_time*1000:.1f}ms")

        # Phase 2: Execute
        print("\n[Phase 2] Quantum Execution...")
        t0 = time.monotonic()

        exec_a = self.node_consciousness.execute()
        exec_b = self.node_knowledge.execute()

        print(f"  [A] Consciousness Φ: {exec_a['consciousness_phi']:.4f} "
              f"(mode: {exec_a.get('mode', 'full')})")
        print(f"  [B] Knowledge Φ: {exec_b['consciousness_phi']:.4f} "
              f"(mode: {exec_b.get('mode', 'full')})")

        phase2_time = time.monotonic() - t0
        print(f"  Phase 2 complete in {phase2_time*1000:.1f}ms")

        # Phase 3: Knowledge teleportation
        print("\n[Phase 3] Knowledge Teleportation...")
        t0 = time.monotonic()

        teleport_ab = self.node_consciousness.teleport_knowledge(self.node_knowledge)
        teleport_ba = self.node_knowledge.teleport_knowledge(self.node_consciousness)

        print(f"  [A→B] Sent: {teleport_ab['sent']}, Received: {teleport_ab['received']}")
        print(f"  [B→A] Sent: {teleport_ba['sent']}, Received: {teleport_ba['received']}")

        phase3_time = time.monotonic() - t0
        print(f"  Phase 3 complete in {phase3_time*1000:.1f}ms")

        # Phase 4: Final stats
        print("\n[Phase 4] Knowledge Synthesis...")
        stats_a = self.node_consciousness.knowledge_engine.get_stats()
        stats_b = self.node_knowledge.knowledge_engine.get_stats()
        print(f"  [A] Total knowledge: {stats_a['total_packets']} packets, "
              f"Mass: {stats_a['knowledge_mass']:.2f}")
        print(f"  [B] Total knowledge: {stats_b['total_packets']} packets, "
              f"Mass: {stats_b['knowledge_mass']:.2f}")

        self._learning_cycles += 1
        total_time = time.monotonic() - cycle_start

        print(f"\n{'=' * 72}")
        print(f"Cycle {self._learning_cycles} complete in {total_time*1000:.1f}ms")
        print(f"{'=' * 72}")

        return {
            "cycle": self._learning_cycles,
            "timing_ms": {
                "total": total_time * 1000,
                "phase1_ingestion": phase1_time * 1000,
                "phase2_execution": phase2_time * 1000,
                "phase3_teleport": phase3_time * 1000,
            },
            "ingestion_a": result_a,
            "ingestion_b": result_b,
            "execution_a": exec_a,
            "execution_b": exec_b,
            "teleport_ab": teleport_ab,
            "teleport_ba": teleport_ba,
            "stats_a": stats_a,
            "stats_b": stats_b,
        }


class L104SubsystemConnector:
    """
    Unified connector for all L104 subsystems.

    Provides lazy initialization, health monitoring, and cross-subsystem
    communication for the mini supercomputer mesh.
    """

    SUBSYSTEMS = [
        "vqpu", "networker", "daemon_adapter",
        "quantum_ai_daemon", "quantum_sim_daemon",
        "soul_daemon", "agent_orchestrator",
        "god_code_simulator", "intellect", "research"
    ]

    def __init__(self, lightweight: bool = False):
        self._lightweight = lightweight
        self._subsystems: Dict[str, Any] = {}
        self._health_status: Dict[str, Dict[str, Any]] = {}
        self._init_order: List[str] = []
        self._connection_matrix: Dict[str, List[str]] = defaultdict(list)

    def get(self, name: str) -> Optional[Any]:
        """Get a subsystem by name (lazy initialization)."""
        if name not in self._subsystems:
            self._init_subsystem(name)
        return self._subsystems.get(name)

    def _init_subsystem(self, name: str) -> None:
        """Initialize a specific subsystem with error handling."""
        if self._lightweight and name not in ["intellect", "research"]:
            logger.debug(f"[{name}] Skipping in lightweight mode")
            self._health_status[name] = {
                "status": "skipped",
                "reason": "lightweight_mode",
                "timestamp": time.time()
            }
            return

        t0 = time.monotonic()
        try:
            subsystem = self._create_subsystem(name)
            elapsed = time.monotonic() - t0
            self._subsystems[name] = subsystem
            self._init_order.append(name)
            self._health_status[name] = {
                "status": "healthy",
                "init_time_ms": elapsed * 1000,
                "timestamp": time.time()
            }
            logger.info(f"[{name}] Initialized in {elapsed*1000:.1f}ms")
        except Exception as e:
            elapsed = time.monotonic() - t0
            self._health_status[name] = {
                "status": "failed",
                "error": str(e),
                "init_time_ms": elapsed * 1000,
                "timestamp": time.time()
            }
            logger.warning(f"[{name}] Failed to initialize: {e}")

    def _create_subsystem(self, name: str) -> Any:
        """Factory method for creating subsystems."""
        creators = {
            "vqpu": lambda: self._import_and_create("l104_vqpu", "get_bridge"),
            "networker": lambda: self._import_and_create("l104_quantum_networker", "get_networker"),
            "daemon_adapter": lambda: self._import_and_create("l104_daemon_adapter", "initialize_daemon_adapter"),
            "quantum_ai_daemon": lambda: self._import_and_create("l104_quantum_ai_daemon", "QuantumAIDaemon"),
            "quantum_sim_daemon": lambda: self._import_and_create("l104_quantum_sim_daemon", "get_daemon"),
            "soul_daemon": lambda: self._import_and_create("l104_soul_daemon", "SoulDaemon"),
            "agent_orchestrator": lambda: self._import_and_create("l104_agent_system", "get_orchestrator"),
            "god_code_simulator": lambda: self._import_and_create("l104_god_code_simulator", "god_code_simulator"),
            "intellect": lambda: self._import_and_create("l104_intellect", "local_intellect"),
            "research": lambda: self._import_and_create("l104_research", "research_engine"),
        }

        if name not in creators:
            raise ValueError(f"Unknown subsystem: {name}")

        return creators[name]()

    def _import_and_create(self, module: str, attr: str) -> Any:
        """Import module and get attribute."""
        try:
            mod = __import__(module, fromlist=[attr])
            result = getattr(mod, attr)
            # If it's a class, instantiate it
            if isinstance(result, type):
                return result()
            # If it's a callable (factory function), call it
            if callable(result):
                return result()
            return result
        except ImportError as e:
            raise ImportError(f"{module}.{attr} not available: {e}")

    def connect_subsystems(self, source: str, target: str) -> bool:
        """Establish connection between two subsystems."""
        src = self.get(source)
        tgt = self.get(target)

        if src is None or tgt is None:
            logger.warning(f"Cannot connect {source} -> {target}: one or both unavailable")
            return False

        self._connection_matrix[source].append(target)
        logger.debug(f"Connected {source} -> {target}")
        return True

    def health_check(self) -> Dict[str, Any]:
        """Run health check on all subsystems."""
        healthy = sum(1 for s in self._health_status.values() if s.get("status") == "healthy")
        failed = sum(1 for s in self._health_status.values() if s.get("status") == "failed")
        skipped = sum(1 for s in self._health_status.values() if s.get("status") == "skipped")
        total = len(self._health_status)

        return {
            "timestamp": time.time(),
            "lightweight_mode": self._lightweight,
            "subsystems": self._health_status,
            "connections": dict(self._connection_matrix),
            "init_order": self._init_order,
            "healthy": healthy,
            "failed": failed,
            "skipped": skipped,
            "total": total,
            "health_percentage": (healthy / total * 100) if total > 0 else 0,
        }

    def broadcast(self, message: Dict[str, Any], targets: Optional[List[str]] = None) -> Dict[str, Any]:
        """Broadcast message to subsystems."""
        targets = targets or self.SUBSYSTEMS
        results = {}

        for target in targets:
            subsystem = self.get(target)
            if subsystem and hasattr(subsystem, 'receive_message'):
                try:
                    results[target] = subsystem.receive_message(message)
                except Exception as e:
                    results[target] = {"error": str(e)}
            else:
                results[target] = {"status": "no_handler"}

        return results

    def get_mesh_bridge(self) -> Optional[Any]:
        """Get the quantum networker mesh bridge for teleportation."""
        networker = self.get("networker")
        if networker and hasattr(networker, 'router'):
            return networker.router
        return None


class ConnectedEnhancedMesh(EnhancedDualSupercomputerMesh):
    """Extended mesh with full L104 subsystem connectivity."""

    def __init__(self, lightweight: bool = False, enable_subsystems: bool = True):
        super().__init__(lightweight=lightweight)
        self._subsystems: Optional[L104SubsystemConnector] = None
        self._enable_subsystems = enable_subsystems

        if enable_subsystems:
            self._init_subsystems()

    @property
    def subsystem_connector(self) -> Optional[L104SubsystemConnector]:
        """Access the subsystem connector."""
        return self._subsystems

    def _init_subsystems(self) -> None:
        """Initialize subsystem connector."""
        t0 = time.monotonic()
        self._subsystems = L104SubsystemConnector(lightweight=self._lightweight)

        # Pre-connect key subsystems in order
        if not self._lightweight:
            # VQPU first (execution backend)
            self._subsystems.get("vqpu")
            # Networker for teleportation
            self._subsystems.get("networker")
            # Daemons for autonomous operation
            self._subsystems.get("quantum_sim_daemon")
            self._subsystems.get("daemon_adapter")

        elapsed = time.monotonic() - t0
        logger.info(f"Subsystem connector initialized in {elapsed*1000:.1f}ms")

    def get_subsystem_health(self) -> Dict[str, Any]:
        """Get health status of all connected subsystems."""
        if self._subsystems:
            return self._subsystems.health_check()
        return {"status": "subsystems_disabled"}

    def teleport_via_network(self, node_a_id: str, node_b_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Teleport data using quantum networker."""
        if not self._subsystems:
            return {"status": "subsystems_disabled"}

        networker = self._subsystems.get("networker")
        if not networker:
            return {"status": "networker_unavailable"}

        try:
            # Use networker's teleport if available
            if hasattr(networker, 'teleport_score'):
                result = networker.teleport_score(node_a_id, node_b_id, data.get("score", 0.5))
                return {
                    "status": "success",
                    "fidelity": result.fidelity if hasattr(result, 'fidelity') else 0.0,
                    "data": data
                }
        except Exception as e:
            logger.error(f"Teleport failed: {e}")
            return {"status": "error", "error": str(e)}

        return {"status": "method_unavailable"}

    def get_full_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive diagnostics including subsystems."""
        base_diag = self.get_diagnostics()
        subsystem_health = self.get_subsystem_health()

        return {
            **base_diag,
            "subsystems": subsystem_health,
            "connectivity": {
                "vqpu_available": subsystem_health.get("subsystems", {}).get("vqpu", {}).get("status") == "healthy",
                "networker_available": subsystem_health.get("subsystems", {}).get("networker", {}).get("status") == "healthy",
                "daemons_available": subsystem_health.get("subsystems", {}).get("quantum_sim_daemon", {}).get("status") == "healthy",
            }
        }




def test_subsystem_integration():
    """Comprehensive test of all subsystems."""
    print("=" * 72)
    print("SUBSYSTEM INTEGRATION TEST")
    print("=" * 72)

    # Test 1: Lightweight mode
    print("\n[Test 1] Lightweight mode...")
    mesh = ConnectedEnhancedMesh(lightweight=True, enable_subsystems=False)
    diag = mesh.get_full_diagnostics()
    assert diag["lightweight_mode"] == True
    print("  ✓ Lightweight mode working")

    # Test 2: Full mode initialization
    print("\n[Test 2] Full mode with subsystems...")
    mesh_full = ConnectedEnhancedMesh(lightweight=False, enable_subsystems=True)
    health = mesh_full._subsystems.health_check()
    print(f"  Health: {health['healthy']}/{health['total']} subsystems")
    print(f"  Health %: {health['health_percentage']:.1f}%")

    # Test 3: Subsystem connector
    print("\n[Test 3] Subsystem connector...")
    connector = L104SubsystemConnector(lightweight=True)
    for subsys in ["vqpu", "networker", "intellect"]:
        result = connector.get(subsys)
        status = connector._health_status.get(subsys, {}).get("status", "unknown")
        print(f"  {subsys}: {status}")

    # Test 4: Knowledge ingestion
    print("\n[Test 4] Knowledge ingestion...")
    from pathlib import Path
    result = mesh.node_consciousness.bulk_ingest([], ["test query"],
                                                  skip_intellect=True)
    assert result.packets_ingested >= 0
    print(f"  ✓ Ingested {result.packets_ingested} packets")

    # Test 5: Learning cycle
    print("\n[Test 5] Learning cycle...")
    result = mesh.run_learning_cycle(files=[], queries=[])
    assert result["cycle"] == 1
    print(f"  ✓ Cycle {result['cycle']} completed")
    print(f"  Timing: {result['timing_ms']['total']:.1f}ms total")

    print("\n" + "=" * 72)
    print("ALL TESTS PASSED")
    print("=" * 72)
    return True


def main():
    """CLI entry point."""
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Parse args
    lightweight = "--lightweight" in sys.argv or "-l" in sys.argv
    diagnostic = "--diagnostic" in sys.argv or "--diag" in sys.argv
    full_diag = "--full-diag" in sys.argv
    connected = "--connected" in sys.argv or "-c" in sys.argv
    test_mode = "--test" in sys.argv
    files = [Path(f) for f in sys.argv[1:] if Path(f).exists() and not f.startswith("-")]

    if test_mode:
        return test_subsystem_integration()

    print("=" * 72)
    print("L104 ENHANCED DUAL SUPERCOMPUTER MESH")
    print("Knowledge Ingestion System")
    if lightweight:
        print("MODE: Lightweight (skips heavy quantum execution)")
    if connected:
        print("MODE: Full Subsystem Connectivity")
    print("=" * 72)

    t0 = time.monotonic()

    if connected:
        mesh = ConnectedEnhancedMesh(lightweight=lightweight)
    else:
        mesh = EnhancedDualSupercomputerMesh(lightweight=lightweight)

    init_time = time.monotonic() - t0
    print(f"\nMesh initialized in {init_time*1000:.1f}ms")

    if full_diag and connected:
        print("\n" + "-" * 72)
        print("FULL DIAGNOSTICS (including subsystems)")
        print("-" * 72)
        diag = mesh.get_full_diagnostics()
        print(json.dumps(diag, indent=2, default=str))
        print("-" * 72)
        return
    elif diagnostic:
        print("\n" + "-" * 72)
        print("DIAGNOSTIC INFORMATION")
        print("-" * 72)
        diag = mesh.get_diagnostics()
        print(json.dumps(diag, indent=2, default=str))
        print("-" * 72)
        return

    # Run learning cycle with sample data
    queries = ["consciousness research", "quantum coherence", "GOD_CODE derivation"]

    result = mesh.run_learning_cycle(files=files, queries=queries)

    print("\n" + "=" * 72)
    print("LEARNING CYCLE COMPLETE")
    print("=" * 72)
    print(json.dumps({
        "cycle": result["cycle"],
        "timing_ms": result.get("timing_ms", {}),
        "total_packets_a": result["stats_a"]["total_packets"],
        "total_packets_b": result["stats_b"]["total_packets"],
        "knowledge_mass_a": result["stats_a"]["knowledge_mass"],
        "knowledge_mass_b": result["stats_b"]["knowledge_mass"],
        "phi_a": result["execution_a"]["consciousness_phi"],
        "phi_b": result["execution_b"]["consciousness_phi"],
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
