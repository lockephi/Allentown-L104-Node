#!/usr/bin/env python3
"""
L104 MULTI-DAEMON DEPLOYMENT FRAMEWORK v1.0
Enables deployment of additional physical daemon nodes with unified configuration,
lifecycle management, and cross-regional networking capabilities.

Supports:
  • Multi-region daemon node deployment
  • Container-based orchestration (Docker/Kubernetes)
  • Automatic node discovery and mesh formation
  • Cross-region quantum key distribution
  • Centralized telemetry & monitoring
"""

import sys
import json
import time
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))

print("=" * 100)
print("L104 MULTI-DAEMON DEPLOYMENT FRAMEWORK v1.0")
print("=" * 100)

# ═══════════════════════════════════════════════════════════════════
# DAEMON NODE SPECIFICATIONS
# ═══════════════════════════════════════════════════════════════════

class DaemonRegion(str, Enum):
    """Physical or logical region for daemon deployment."""
    US_EAST = "us-east-1"
    US_WEST = "us-west-1"
    EU_CENTRAL = "eu-central-1"
    EU_WEST = "eu-west-1"
    ASIA_PACIFIC = "ap-northeast-1"
    ASIA_SOUTH = "ap-south-1"
    SA_EAST = "sa-east-1"
    AF_SOUTH = "af-south-1"
    LOCAL = "localhost"


class DaemonRole(str, Enum):
    """Daemon operational role in the network."""
    SOVEREIGN = "sovereign"          # Primary computation node
    RELAY = "relay"                  # Entanglement relay node
    QKD_DISTRIBUTOR = "qkd_dist"     # QKD key distribution hub
    OBSERVER = "observer"             # Network observer (telemetry only)
    HYPER_NODE = "hyper_node"        # High-capacity compute cluster


@dataclass
class DaemonNodeSpec:
    """Complete specification for a daemon node."""
    name: str
    node_id: str
    region: DaemonRegion
    role: DaemonRole
    host: str
    port: int
    max_qubits: int
    quantum_topology: str = "all_to_all"
    enable_qkd: bool = True
    enable_error_correction: bool = True
    telemetry_port: int = 9090
    ipc_path: str = "/tmp/l104_bridge/multi_daemon"
    state_file: Optional[str] = None


@dataclass
class DaemonCluster:
    """Multi-region daemon cluster specification."""
    cluster_id: str
    cluster_name: str
    nodes: List[DaemonNodeSpec] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def add_node(self, spec: DaemonNodeSpec) -> None:
        """Add a daemon node to the cluster."""
        self.nodes.append(spec)

    def nodes_by_region(self, region: DaemonRegion) -> List[DaemonNodeSpec]:
        """Get all nodes in a specific region."""
        return [n for n in self.nodes if n.region == region]

    def nodes_by_role(self, role: DaemonRole) -> List[DaemonNodeSpec]:
        """Get all nodes with a specific role."""
        return [n for n in self.nodes if n.role == role]

    def to_dict(self) -> Dict:
        """Serialize cluster to dictionary."""
        return {
            "cluster_id": self.cluster_id,
            "cluster_name": self.cluster_name,
            "total_nodes": len(self.nodes),
            "regions": list(set(n.region.value for n in self.nodes)),
            "roles": list(set(n.role.value for n in self.nodes)),
            "total_qubits": sum(n.max_qubits for n in self.nodes),
            "nodes": [asdict(n) for n in self.nodes],
        }


# ═══════════════════════════════════════════════════════════════════
# STANDARD CLUSTER TEMPLATES
# ═══════════════════════════════════════════════════════════════════

def create_massive_cluster() -> DaemonCluster:
    """Create a 32-node massive distributed daemon cluster."""

    cluster = DaemonCluster(
        cluster_id="massive-cluster-001",
        cluster_name="L104 Massive Global Intelligence Network"
    )

    regions = [
        DaemonRegion.US_EAST, DaemonRegion.US_WEST,
        DaemonRegion.EU_CENTRAL, DaemonRegion.EU_WEST,
        DaemonRegion.ASIA_PACIFIC, DaemonRegion.ASIA_SOUTH,
        DaemonRegion.SA_EAST, DaemonRegion.AF_SOUTH
    ]

    for i, region in enumerate(regions):
        base_port = 10400 + (i * 10)
        base_tel = 9090 + (i * 10)
        
        # Add 4 nodes per region
        cluster.add_node(DaemonNodeSpec(
            name=f"Sovereign-{region.value}-1",
            node_id=f"daemon-{region.value}-001",
            region=region,
            role=DaemonRole.SOVEREIGN,
            host=f"10.{i}.1.10",
            port=base_port + 1,
            max_qubits=26,
            telemetry_port=base_tel + 1,
        ))

        cluster.add_node(DaemonNodeSpec(
            name=f"Relay-{region.value}-1",
            node_id=f"daemon-{region.value}-002",
            region=region,
            role=DaemonRole.RELAY,
            host=f"10.{i}.1.20",
            port=base_port + 2,
            max_qubits=16,
            telemetry_port=base_tel + 2,
        ))

        cluster.add_node(DaemonNodeSpec(
            name=f"QKD-Hub-{region.value}-1",
            node_id=f"daemon-{region.value}-003",
            region=region,
            role=DaemonRole.QKD_DISTRIBUTOR,
            host=f"10.{i}.1.30",
            port=base_port + 3,
            max_qubits=20,
            telemetry_port=base_tel + 3,
        ))

        cluster.add_node(DaemonNodeSpec(
            name=f"HyperNode-{region.value}-1",
            node_id=f"daemon-{region.value}-004",
            region=region,
            role=DaemonRole.HYPER_NODE,
            host=f"10.{i}.1.40",
            port=base_port + 4,
            max_qubits=32,
            telemetry_port=base_tel + 4,
        ))

    return cluster


def create_docker_deployment_manifest(cluster: DaemonCluster) -> Dict:
    """Generate Docker Compose manifest for cluster deployment."""

    services = {}

    for node in cluster.nodes:
        service_name = node.name.lower().replace("-", "_").replace(".", "_")

        services[service_name] = {
            "image": "l104/quantum-daemon:latest",
            "container_name": node.name,
            "hostname": node.name,
            "environment": {
                "L104_NODE_ID": node.node_id,
                "L104_NODE_NAME": node.name,
                "L104_REGION": node.region.value,
                "L104_ROLE": node.role.value,
                "L104_MAX_QUBITS": str(node.max_qubits),
                "L104_QUANTUM_TOPOLOGY": node.quantum_topology,
                "L104_ENABLE_QKD": str(node.enable_qkd),
                "L104_ENABLE_ERROR_CORRECTION": str(node.enable_error_correction),
                "L104_PORT": str(node.port),
                "L104_TELEMETRY_PORT": str(node.telemetry_port),
                "L104_CONCURRENCY_LIMIT": "256",
            },
            "ports": [
                f"{node.port}:{node.port}",
                f"{node.telemetry_port}:{node.telemetry_port}",
            ],
            "networks": ["l104_quantum_network"],
            "restart_policy": {
                "condition": "on-failure",
                "delay": "5s",
                "max_attempts": 5,
            },
            "healthcheck": {
                "test": ["CMD", "curl", "-f", f"http://localhost:{node.telemetry_port}/health"],
                "interval": "30s",
                "timeout": "10s",
                "retries": 3,
            },
        }

    manifest = {
        "version": "3.9",
        "services": services,
        "networks": {
            "l104_quantum_network": {
                "driver": "bridge",
            }
        },
        "volumes": {
            "l104_state": {},
        },
    }

    return manifest


# ═══════════════════════════════════════════════════════════════════
# DAEMON DEPLOYMENT & DISCOVERY
# ═══════════════════════════════════════════════════════════════════

print("\n[PHASE 1] DEFINE MASSIVE CLUSTER")
print("-" * 100)

cluster = create_massive_cluster()

# Show node breakdown
print(f"\nNode Deployment Plan:")
for region in DaemonRegion:
    region_nodes = cluster.nodes_by_region(region)
    if region_nodes:
        print(f"\n  {region.value.upper()}:")
        for node in region_nodes:
            print(f"    • {node.name:35} (role: {node.role.value:15} qubits: {node.max_qubits})")

# ═══════════════════════════════════════════════════════════════════
# PHASE 2: DOCKER DEPLOYMENT CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

print("\n[PHASE 2] GENERATE DOCKER DEPLOYMENT MANIFEST")
print("-" * 100)

docker_manifest = create_docker_deployment_manifest(cluster)
print(f"✓ Generated Docker Compose manifest with {len(docker_manifest['services'])} services")

# Save Docker Compose file
docker_compose_file = "/Users/carolalvarez/Applications/Allentown-L104-Node/docker-compose.l104.yml"
try:
    with open(docker_compose_file, 'w') as f:
        import yaml
        yaml.dump(docker_manifest, f, default_flow_style=False)
    print(f"✓ Docker Compose file written: {docker_compose_file}")
except ImportError:
    # Fallback to JSON if yaml not available
    with open(docker_compose_file.replace('.yml', '.json'), 'w') as f:
        json.dump(docker_manifest, f, indent=2)
    print(f"✓ Docker Compose config written (JSON format)")
except Exception as e:
    print(f"⚠ Docker Compose file save skipped: {e}")

# ═══════════════════════════════════════════════════════════════════
# PHASE 3: CLUSTER CONFIGURATION PERSISTENCE
# ═══════════════════════════════════════════════════════════════════

print("\n[PHASE 3] PERSIST CLUSTER CONFIGURATION")
print("-" * 100)

cluster_config_file = "/Users/carolalvarez/Applications/Allentown-L104-Node/.l104_cluster_config.json"
try:
    with open(cluster_config_file, 'w') as f:
        json.dump(cluster.to_dict(), f, indent=2)
    print(f"✓ Cluster configuration persisted")
    print(f"  File: {cluster_config_file}")
    print(f"  Nodes: {len(cluster.nodes)}")
except Exception as e:
    print(f"✗ Failed to persist cluster config: {e}")

# ═══════════════════════════════════════════════════════════════════
# PHASE 4: NETWORK TOPOLOGY MAPPING
# ═══════════════════════════════════════════════════════════════════

print("\n[PHASE 4] MULTI-REGION NETWORK TOPOLOGY")
print("-" * 100)

# Build inter-region links
topology = {
    "nodes": [{"id": n.node_id, "name": n.name, "region": n.region.value} for n in cluster.nodes],
    "channels": []
}

# Create full-mesh within each region
for region in set(n.region for n in cluster.nodes):
    region_nodes = [n for n in cluster.nodes if n.region == region]
    if len(region_nodes) > 1:
        for i, node_a in enumerate(region_nodes):
            for node_b in region_nodes[i+1:]:
                topology["channels"].append({
                    "source": node_a.node_id,
                    "dest": node_b.node_id,
                    "region": region.value,
                    "type": "intra-region"
                })

# Create inter-region links (QKD hubs & relays)
qkd_nodes = cluster.nodes_by_role(DaemonRole.QKD_DISTRIBUTOR)
sovereign_nodes = cluster.nodes_by_role(DaemonRole.SOVEREIGN)

for i, qkd_a in enumerate(qkd_nodes):
    for qkd_b in qkd_nodes[i+1:]:
        topology["channels"].append({
            "source": qkd_a.node_id,
            "dest": qkd_b.node_id,
            "region": f"{qkd_a.region.value}-{qkd_b.region.value}",
            "type": "inter-region-qkd"
        })

print(f"Network Topology:")
print(f"  Nodes: {len(topology['nodes'])}")
print(f"  Channels: {len(topology['channels'])}")

# Count channel types
intra_count = sum(1 for c in topology["channels"] if c["type"] == "intra-region")
inter_count = sum(1 for c in topology["channels"] if c["type"] == "inter-region-qkd")
print(f"  Intra-region channels: {intra_count}")
print(f"  Inter-region QKD links: {inter_count}")

# ═══════════════════════════════════════════════════════════════════
# PHASE 5: DEPLOYMENT INSTRUCTIONS
# ═══════════════════════════════════════════════════════════════════

print("\n[PHASE 5] DEPLOYMENT INSTRUCTIONS")
print("-" * 100)

print("\nDOCKER DEPLOYMENT:")
print("  1. Build base image:")
print("     docker build -f Dockerfile.l104 -t l104/quantum-daemon:latest .")
print("\n  2. Start cluster:")
print("     docker-compose -f docker-compose.l104.yml up -d")
print("\n  3. Monitor nodes:")
print("     docker-compose -f docker-compose.l104.yml logs -f")
print("\n  4. Health check:")
print("     for node in $(docker ps --format '{{.Names}}' | grep daemon); do")
print("       echo \"$node:\"; curl http://localhost:909x/health 2>/dev/null | jq .")
print("     done")

print("\nKUBERNETES DEPLOYMENT:")
print("  1. Convert Docker Compose to Kubernetes manifests:")
print("     kompose convert -f docker-compose.l104.yml -o k8s/")
print("\n  2. Deploy to cluster:")
print("     kubectl apply -f k8s/")
print("\n  3. Monitor pods:")
print("     kubectl get pods -l app=l104-daemon")
print("     kubectl logs -f deployment/l104-daemon")

print("\nMANUAL DEPLOYMENT (LOCAL):")
print("  For each daemon node:")
print("    export L104_NODE_ID=daemon-region-001")
print("    export L104_REGION=us-east-1")
print("    export L104_ROLE=sovereign")
print("    export L104_MAX_QUBITS=26")
print("    python -m l104_vqpu.micro_daemon --start")

# ═══════════════════════════════════════════════════════════════════
# COMPLETION
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 100)
print("✓ MULTI-DAEMON DEPLOYMENT FRAMEWORK CONFIGURED")
print("=" * 100)

print("\nDeployment Summary:")
print(f"  • {len(cluster.nodes)} daemon nodes across {len(set(n.region for n in cluster.nodes))} regions")
print(f"  • {sum(n.max_qubits for n in cluster.nodes)} total qubits")
print(f"  • {len(qkd_nodes)} QKD distribution hubs for cross-region key exchange")
print(f"  • Full-mesh intra-region channels + inter-region QKD links")
print(f"  • Docker Compose + Kubernetes support")
print(f"  • Health monitoring on all nodes")

print("\nNext Steps:")
print("  1. Deploy nodes using Docker Compose or Kubernetes")
print("  2. Run network stability tests (use _network_stability_test.py)")
print("  3. Establish cross-regional QKD links")
print("  4. Enable quantum error correction on all 26Q circuits")

print("=" * 100)
