"""Quick topology upgrade verification."""
from l104_vqpu.quantum_network import QuantumNetworkMesh

for topo in ['linear', 'ring', 'heavy_hex', 'all_to_all']:
    m = QuantumNetworkMesh(node_ids=['n1','n2','n3','n4','n5','n6'], topology=topo)
    r = m.establish_channels()
    d = m.detect_topology()
    a = m.topology_analysis()
    print(f"{topo:12s}: ch={r['channels']:2d} detect={d:12s} "
          f"deg=[{a['min_degree']},{a['max_degree']}] dia={a['diameter']} "
          f"sacred={a['sacred_topology_score']:.4f}")

# Test set_topology
m2 = QuantumNetworkMesh(node_ids=['a','b','c','d'], topology='all_to_all')
m2.establish_channels()
rt = m2.set_topology('ring')
print(f"\nset_topology: {rt['old_topology']} -> {rt['new_topology']} "
      f"ch={rt['old_channels']}->{rt['new_channels']}")

# Test recommendation
rec = m2.topology_recommendation()
print(f"recommend: current={rec['current_topology']} best={rec['recommended_topology']}")

# Test network_health includes topology
health = m2.network_health()
print(f"\nhealth topology={health['topology']} detected={health['detected_topology']} "
      f"score={health['network_score']:.4f} sacred_topo={health['sacred_topology_score']:.4f}")

# Test add_node with topology
m3 = QuantumNetworkMesh(node_ids=['a','b','c'], topology='ring')
m3.establish_channels()
r3 = m3.add_node('d')
print(f"\nadd_node (ring): added={r3['added']} topology={r3['topology']} "
      f"new_ch={r3['new_channels']} total_ch={r3['total_channels']}")

print("\nAll topology tests PASSED")
