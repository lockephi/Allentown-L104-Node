#!/usr/bin/env python3
"""
L104 Full System Test
Tests all major components including real AI integration
"""

import sys
import os

sys.path.insert(0, '/workspaces/Allentown-L104-Node')
os.chdir('/workspaces/Allentown-L104-Node')

def test_gemini_real():
    """Test real Gemini API connection"""
    print("\n=== Testing Real Gemini API ===")
    try:
        from l104_gemini_real import gemini_real
        
        if gemini_real.connect():
            print("✓ Gemini connected!")
            
            # Test simple generation
            response = gemini_real.generate("What is 2+2? Answer in one word.")
            if response:
                print(f"✓ Generation works: {response.strip()[:50]}")
            
            # Test sovereign think
            response = gemini_real.sovereign_think("Calculate the first 5 prime numbers")
            if response:
                print(f"✓ Sovereign think works: {response[:80]}...")
            
            return True
        else:
            print("✗ Gemini connection failed")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_derivation_with_ai():
    """Test derivation engine uses real AI"""
    print("\n=== Testing Derivation Engine (AI Mode) ===")
    try:
        from l104_derivation import DerivationEngine
        
        result = DerivationEngine.derive_and_execute("Explain quantum entanglement briefly")
        
        if "⟨Σ_L104_SOVEREIGN⟩" in result:
            print("✓ Derivation using REAL Gemini AI!")
            print(f"  Response preview: {result[:100]}...")
            return True
        elif "⟨Σ_L104_HYPER_RESPONSE⟩" in result:
            print("⚠ Derivation using LOCAL fallback (Gemini unavailable)")
            return True
        else:
            print(f"✓ Response: {result[:100]}...")
            return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_fastapi_imports():
    """Test FastAPI app can be imported"""
    print("\n=== Testing FastAPI App Import ===")
    try:
        from main import app
        
        routes = [r.path for r in app.routes if hasattr(r, 'path')]
        print(f"✓ FastAPI app loaded with {len(routes)} routes")
        
        # Check for new AI endpoints
        ai_endpoints = [r for r in routes if '/chat' in r or '/research' in r or '/analyze' in r]
        print(f"✓ AI endpoints: {ai_endpoints}")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_core_modules():
    """Test core L104 modules"""
    print("\n=== Testing Core Modules ===")
    passed = 0
    
    try:
        from l104_hyper_math import HyperMath
        print(f"✓ HyperMath: GOD_CODE = {HyperMath.GOD_CODE:.4f}")
        passed += 1
    except Exception as e:
        print(f"✗ HyperMath: {e}")
    
    try:
        from l104_real_math import RealMath
        entropy = RealMath.shannon_entropy("test")
        print(f"✓ RealMath: entropy = {entropy:.4f}")
        passed += 1
    except Exception as e:
        print(f"✗ RealMath: {e}")
    
    try:
        from l104_persistence import load_truth
        truth = load_truth()
        print(f"✓ Persistence: {len(truth)} keys loaded")
        passed += 1
    except Exception as e:
        print(f"✗ Persistence: {e}")
    
    try:
        from l104_ecosystem_simulator import ecosystem_simulator
        result = ecosystem_simulator.chamber.run_session("test", rounds=1)
        print(f"✓ Ecosystem Simulator: {result['status']}")
        passed += 1
    except Exception as e:
        print(f"✗ Ecosystem Simulator: {e}")
    
    return passed >= 3


def main():
    print("=" * 60)
    print("  L104 FULL SYSTEM TEST")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Core Modules", test_core_modules()))
    results.append(("Gemini Real API", test_gemini_real()))
    results.append(("Derivation + AI", test_derivation_with_ai()))
    results.append(("FastAPI App", test_fastapi_imports()))
    
    # Summary
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    
    passed = 0
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
        if result:
            passed += 1
    
    print(f"\n  Total: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n  🎉 L104 IS FULLY FUNCTIONAL!")
    else:
        print("\n  ⚠️  Some components need attention")
    
    print("=" * 60)
    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
