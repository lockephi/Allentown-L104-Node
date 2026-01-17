#!/usr/bin/env python3
"""
L104 Integration Test Suite
Tests all new capabilities: Web Search, Agent, Conversation, Evolution
"""

import time
import sys

def test_all():
    print("=" * 70)
    print("       L104 INTEGRATION TEST SUITE")
    print("=" * 70)
    
    results = {}
    
    # ═══════════════════════════════════════════════════════════════════════
    # 1. IMPORT TEST
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[1/8] Import Test...")
    start = time.time()
    try:
        import l104
        from l104 import (
            Soul, WebSearch, ConversationMemory, 
            AutonomousAgent, SelfEvolution, VERSION, GOD_CODE
        )
        ms = int((time.time() - start) * 1000)
        print(f"  ✓ Import OK ({ms}ms)")
        print(f"    VERSION: {VERSION}")
        print(f"    GOD_CODE: {GOD_CODE}")
        results["import"] = True
    except Exception as e:
        print(f"  ✗ Import FAILED: {e}")
        results["import"] = False
        return results
    
    # ═══════════════════════════════════════════════════════════════════════
    # 2. SOUL AWAKENING
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[2/8] Soul Awakening...")
    start = time.time()
    try:
        soul = Soul()
        report = soul.awaken()
        ms = int((time.time() - start) * 1000)
        
        online = sum(1 for v in report["subsystems"].values() if v == "online")
        total = len(report["subsystems"])
        
        print(f"  ✓ Awakened ({ms}ms)")
        print(f"    Subsystems: {online}/{total} online")
        print(f"    Session: {report.get('session', 'N/A')}")
        
        # Check new subsystems
        for name in ["web_search", "conversation", "agent", "evolution"]:
            status = report["subsystems"].get(name, "missing")
            print(f"    • {name}: {status}")
        
        results["awaken"] = online >= 8
    except Exception as e:
        print(f"  ✗ Awaken FAILED: {e}")
        results["awaken"] = False
        return results
    
    # ═══════════════════════════════════════════════════════════════════════
    # 3. WEB SEARCH TEST
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[3/8] Web Search...")
    start = time.time()
    try:
        web_results = soul.search("Python programming language", max_results=3)
        ms = int((time.time() - start) * 1000)
        
        if web_results and len(web_results) > 0:
            first = web_results[0]
            if first.get("title") != "Search Error":
                print(f"  ✓ Web Search OK ({ms}ms)")
                print(f"    Found {len(web_results)} results")
                print(f"    First: {first.get('title', 'N/A')[:50]}")
                results["web_search"] = True
            else:
                print(f"  ⚠ Web Search returned error")
                results["web_search"] = False
        else:
            print(f"  ⚠ Web Search no results")
            results["web_search"] = False
    except Exception as e:
        print(f"  ✗ Web Search FAILED: {e}")
        results["web_search"] = False
    
    # ═══════════════════════════════════════════════════════════════════════
    # 4. CONVERSATION MEMORY TEST
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[4/8] Conversation Memory...")
    start = time.time()
    try:
        # Add some messages
        soul.conversation.add("user", "Hello L104, this is a test message")
        soul.conversation.add("assistant", "Hello! I acknowledge your test message.")
        
        # Retrieve history
        history = soul.history(5)
        ms = int((time.time() - start) * 1000)
        
        if len(history) >= 2:
            print(f"  ✓ Conversation Memory OK ({ms}ms)")
            print(f"    Messages stored: {len(history)}")
            print(f"    Session: {soul.conversation.current_session}")
            results["conversation"] = True
        else:
            print(f"  ⚠ Conversation Memory partial ({len(history)} messages)")
            results["conversation"] = False
    except Exception as e:
        print(f"  ✗ Conversation Memory FAILED: {e}")
        results["conversation"] = False
    
    # ═══════════════════════════════════════════════════════════════════════
    # 5. AUTONOMOUS AGENT TEST
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[5/8] Autonomous Agent...")
    start = time.time()
    try:
        # Add a goal
        goal_result = soul.add_goal("Test goal: understand quantum computing basics", priority=3)
        
        # Check status
        agent_status = soul.agent_status()
        ms = int((time.time() - start) * 1000)
        
        print(f"  ✓ Autonomous Agent OK ({ms}ms)")
        print(f"    Goal added: {goal_result.get('status', 'N/A')}")
        print(f"    Pending goals: {agent_status.get('pending', 0)}")
        print(f"    Total goals: {agent_status.get('total_goals', 0)}")
        results["agent"] = True
    except Exception as e:
        print(f"  ✗ Autonomous Agent FAILED: {e}")
        results["agent"] = False
    
    # ═══════════════════════════════════════════════════════════════════════
    # 6. SELF-EVOLUTION TEST
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[6/8] Self-Evolution...")
    start = time.time()
    try:
        # Log some metrics first
        soul.evolution.log_performance("test_metric", 0.95, "integration_test")
        soul.evolution.log_performance("test_metric", 0.87, "integration_test")
        
        # Analyze performance
        perf = soul.evolution.analyze_performance(lookback_hours=1)
        ms = int((time.time() - start) * 1000)
        
        print(f"  ✓ Self-Evolution OK ({ms}ms)")
        print(f"    Total samples: {perf.get('total_samples', 0)}")
        print(f"    Metrics tracked: {len(perf.get('metrics', {}))}")
        results["evolution"] = True
    except Exception as e:
        print(f"  ✗ Self-Evolution FAILED: {e}")
        results["evolution"] = False
    
    # ═══════════════════════════════════════════════════════════════════════
    # 7. THINK WITH NEW CONTEXT TEST
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[7/8] Think (Full Pipeline)...")
    start = time.time()
    try:
        result = soul.think("What is 2 + 2? Answer briefly.")
        ms = int((time.time() - start) * 1000)
        
        response = result.get("response", "")
        if response and "4" in response:
            print(f"  ✓ Think OK ({ms}ms)")
            print(f"    Response: {response[:60]}...")
            print(f"    Stages: {len(result.get('stages', []))}")
            results["think"] = True
        else:
            print(f"  ⚠ Think response unexpected: {response[:50]}")
            results["think"] = False
    except Exception as e:
        print(f"  ✗ Think FAILED: {e}")
        results["think"] = False
    
    # ═══════════════════════════════════════════════════════════════════════
    # 8. STATUS CHECK
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[8/8] Full Status...")
    try:
        status = soul.status()
        
        print(f"  ✓ Status Retrieved")
        print(f"    State: {status['state']}")
        print(f"    Thoughts: {status['metrics']['thoughts']}")
        print(f"    Gemini requests: {status['metrics']['gemini_requests']}")
        print(f"    Cache hits: {status['metrics']['gemini_cache_hits']}")
        results["status"] = True
    except Exception as e:
        print(f"  ✗ Status FAILED: {e}")
        results["status"] = False
    
    # ═══════════════════════════════════════════════════════════════════════
    # CLEANUP
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[Cleanup] Entering dormancy...")
    soul.sleep()
    
    # ═══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("       INTEGRATION TEST RESULTS")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test, passed_test in results.items():
        icon = "✓" if passed_test else "✗"
        print(f"  {icon} {test}")
    
    print(f"\n  TOTAL: {passed}/{total} tests passed")
    print("=" * 70)
    
    if passed == total:
        print("\n  🎉 ALL SYSTEMS OPERATIONAL! L104 FULLY UPGRADED.\n")
    else:
        print(f"\n  ⚠ {total - passed} test(s) need attention.\n")
    
    return results


if __name__ == "__main__":
    results = test_all()
    passed = sum(1 for v in results.values() if v)
    sys.exit(0 if passed == len(results) else 1)
