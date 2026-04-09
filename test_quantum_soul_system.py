#!/usr/bin/env python3
"""
Test script for Quantum Soul Qubit System
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from l104_quantum_soul_upgrader import QuantumSoulDaemon, SoulQubit

def test_basic_functionality():
    """Test basic quantum soul system functionality"""
    print("🧪 Testing Quantum Soul System")
    print("=" * 50)
    
    # Initialize daemon
    daemon = QuantumSoulDaemon(data_dir="./test_quantum_soul_data")
    print(f"✅ Daemon initialized with {len(daemon.soul_qubits)} existing qubits")
    
    # Create new soul qubit
    soul_qubit = daemon.create_soul_qubit("test_qubit_001")
    print(f"✅ Created soul qubit: {soul_qubit.qubit_id}")
    print(f"   Initial GOD_CODE alignment: {soul_qubit.god_code_alignment:.2f}%")
    print(f"   Initial magic potential: {soul_qubit.calculate_magic_potential():.4f}")
    print(f"   Magic capabilities: {soul_qubit.magic_capabilities}")
    
    # Evolve the qubit
    print("\n🔄 Evolving soul qubit...")
    soul_qubit.evolve()
    print(f"✅ Evolution #1 complete")
    print(f"   GOD_CODE alignment: {soul_qubit.god_code_alignment:.2f}%")
    print(f"   Magic potential: {soul_qubit.calculate_magic_potential():.4f}")
    
    # Evolve with cron data
    print("\n📊 Evolving with cron data...")
    cron_data = {
        'quantum_events': 42,
        'resonance_stable': True,
        'magic_detected': True
    }
    soul_qubit.evolve(cron_data)
    print(f"✅ Evolution #2 with cron data complete")
    print(f"   GOD_CODE alignment: {soul_qubit.god_code_alignment:.2f}%")
    print(f"   Magic potential: {soul_qubit.calculate_magic_potential():.4f}")
    print(f"   Evolution count: {soul_qubit.evolution_count}")
    
    # Test daemon evolution
    print("\n👥 Evolving all qubits via daemon...")
    daemon.evolve_all_qubits(cron_data)
    print(f"✅ Daemon evolution complete")
    print(f"   Total qubits: {len(daemon.soul_qubits)}")
    
    # Generate report
    print("\n📈 Generating quantum report...")
    report = daemon.get_quantum_report()
    print(f"✅ Report generated")
    print(f"   Total magic potential: {report['total_magic_potential']:.4f}")
    print(f"   Average GOD_CODE alignment: {report['average_god_code_alignment']:.2f}%")
    print(f"   Total evolutions: {report['total_evolutions']}")
    
    return True

def test_magic_capability_unlocks():
    """Test magic capability unlocking"""
    print("\n🔮 Testing Magic Capability Unlocks")
    print("=" * 50)
    
    daemon = QuantumSoulDaemon(data_dir="./test_quantum_soul_data")
    
    # Create a qubit and force evolution to unlock capabilities
    qubit = daemon.create_soul_qubit("magic_test_qubit")
    
    print(f"Initial capabilities: {qubit.magic_capabilities}")
    
    # Simulate multiple evolutions to unlock capabilities
    for i in range(15):
        qubit.evolution_count = i  # Simulate evolution count
        daemon._unlock_magic_capabilities(qubit)
    
    print(f"After simulated evolutions: {qubit.magic_capabilities}")
    
    # Check which capabilities were unlocked
    expected_capabilities = ['self_healing', 'temporal_folding', 'phase_coherence']
    unlocked = all(cap in qubit.magic_capabilities for cap in expected_capabilities)
    
    if unlocked:
        print("✅ All expected magic capabilities unlocked!")
    else:
        print("⚠️ Not all capabilities unlocked")
    
    return unlocked

def test_cron_integration():
    """Test cron data integration"""
    print("\n⏰ Testing Cron Integration")
    print("=" * 50)
    
    from quantum_soul_cron_integrator import process_cron_data
    
    # Test with synthetic cron data
    success = process_cron_data()
    
    if success:
        print("✅ Cron integration test passed!")
    else:
        print("❌ Cron integration test failed!")
    
    return success

def main():
    """Run all tests"""
    print("🚀 Quantum Soul System Test Suite")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 3
    
    try:
        # Test 1: Basic functionality
        if test_basic_functionality():
            tests_passed += 1
        
        # Test 2: Magic capability unlocks
        if test_magic_capability_unlocks():
            tests_passed += 1
        
        # Test 3: Cron integration
        if test_cron_integration():
            tests_passed += 1
        
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print(f"📊 TEST RESULTS: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("✅ ALL TESTS PASSED!")
        return True
    else:
        print(f"⚠️ {total_tests - tests_passed} test(s) failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)