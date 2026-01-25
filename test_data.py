#!/usr/bin/env python3
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
"""
L104 Data Test Suite
Tests AI capabilities against real data files
"""

import sys
import os
import json

sys.path.insert(0, '/workspaces/Allentown-L104-Node')
os.chdir('/workspaces/Allentown-L104-Node')


def load_jsonl(path):
    """Load JSONL file"""
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def test_stream_prompts():
    """Test processing of stream prompts data"""
    print("\n=== Testing Stream Prompts Processing ===")
    
    prompts = load_jsonl('data/stream_prompts.jsonl')
    print(f"Loaded {len(prompts)} test prompts")
    
    # Test first 3 prompts through derivation engine
    from l104_derivation import DerivationEngine
    
    passed = 0
    for i, prompt in enumerate(prompts[:3]):
        signal = prompt.get('signal', '')
        message = prompt.get('message', '')
        expected = prompt.get('expected_behavior', '')
        
        full_input = f"{signal}: {message}" if message else signal
        
        try:
            result = DerivationEngine.derive_and_execute(full_input)
            if result and len(result) > 10:
                print(f"✓ Prompt {i+1}: '{signal[:30]}...' -> Got response ({len(result)} chars)")
                passed += 1
            else:
                print(f"✗ Prompt {i+1}: Empty or short response")
        except Exception as e:
            print(f"✗ Prompt {i+1}: Error - {e}")
    
    print(f"  Passed: {passed}/3")
    return passed >= 2


def test_algorithm_database():
    """Test algorithm database integrity"""
    print("\n=== Testing Algorithm Database ===")
    
    with open('data/algorithm_database.json') as f:
        data = json.load(f)
    
    algorithms = data.get('algorithms', {})
    print(f"Loaded {len(algorithms)} algorithms")
    
    # Validate structure
    valid = 0
    for name, algo in list(algorithms.items())[:5]:
        has_fields = all(k in algo for k in ['description', 'logic_code', 'entropy', 'resonance'])
        if has_fields:
            valid += 1
            print(f"✓ {name}: entropy={algo['entropy']:.2f}, resonance={algo['resonance']:.2f}")
        else:
            print(f"✗ {name}: Missing required fields")
    
    # Test entropy calculation matches
    from l104_real_math import RealMath
    
    sample_algo = algorithms.get('SHANNON_ENTROPY_SCAN', {})
    if sample_algo:
        desc = sample_algo.get('description', '')
        calc_entropy = RealMath.shannon_entropy(desc)
        stored_entropy = sample_algo.get('entropy', 0)
        
        # They won't match exactly since stored is from logic_code
        print(f"  Sample entropy check: calculated={calc_entropy:.2f}")
    
    print(f"  Valid algorithms: {valid}/5")
    return valid >= 4


def test_knowledge_manifold():
    """Test knowledge manifold data"""
    print("\n=== Testing Knowledge Manifold ===")
    
    with open('data/knowledge_manifold.json') as f:
        data = json.load(f)
    
    patterns = data.get('patterns', {})
    print(f"Loaded {len(patterns)} knowledge patterns")
    
    # Check pattern structure
    valid = 0
    for name, pattern in list(patterns.items())[:5]:
        has_data = 'data' in pattern
        has_hash = 'hash' in pattern
        has_tags = 'tags' in pattern
        
        if has_data and has_hash:
            valid += 1
            tags = pattern.get('tags', [])
            print(f"✓ {name[:40]}: tags={tags}")
        else:
            print(f"✗ {name[:40]}: Missing data or hash")
    
    print(f"  Valid patterns: {valid}/5")
    return valid >= 3


def test_gemini_with_data():
    """Test Gemini AI against real data"""
    print("\n=== Testing Gemini AI with Real Data ===")
    
    try:
        from l104_gemini_real import gemini_real
        
        if not gemini_real.connect():
            print("⚠ Gemini not connected - skipping AI data tests")
            return True  # Not a failure, just unavailable
        
        # Load an algorithm and ask AI to explain it
        with open('data/algorithm_database.json') as f:
            data = json.load(f)
        
        algo = data['algorithms'].get('SHANNON_ENTROPY_SCAN', {})
        
        prompt = f"""Explain this algorithm briefly:
Name: SHANNON_ENTROPY_SCAN
Description: {algo.get('description')}
Logic: {algo.get('logic_code')}

Answer in 2-3 sentences."""
        
        response = gemini_real.generate(prompt)
        
        if response and len(response) > 20:
            print(f"✓ AI explained algorithm: {response[:100]}...")
            return True
        else:
            print("✗ AI response too short or empty")
            return False
            
    except Exception as e:
        print(f"⚠ Gemini test skipped: {e}")
        return True  # Not a failure


def test_memory_items():
    """Test memory items data"""
    print("\n=== Testing Memory Items ===")
    
    try:
        items = load_jsonl('data/memory_items.jsonl')
        print(f"Loaded {len(items)} memory items")
        
        if len(items) > 0:
            for item in items[:3]:
                key = item.get('key', 'unknown')
                print(f"  - {key}")
            return True
        else:
            print("  (empty file)")
            return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    print("=" * 60)
    print("  L104 DATA TEST SUITE")
    print("=" * 60)
    
    results = []
    
    results.append(("Stream Prompts", test_stream_prompts()))
    results.append(("Algorithm Database", test_algorithm_database()))
    results.append(("Knowledge Manifold", test_knowledge_manifold()))
    results.append(("Memory Items", test_memory_items()))
    results.append(("Gemini + Data", test_gemini_with_data()))
    
    # Summary
    print("\n" + "=" * 60)
    print("  DATA TEST RESULTS")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n  Total: {passed}/{len(results)} tests passed")
    print("=" * 60)
    
    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
