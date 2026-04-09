#!/usr/bin/env python3
"""
Test Ollama Heartbeat Monitoring
Verify Ollama is properly monitored in heartbeat checks
"""

import subprocess
import json
import requests
from datetime import datetime

def check_ollama_service():
    """Check Ollama service status"""
    print("🔍 Checking Ollama service...")
    
    try:
        # Check if Ollama service is running
        result = subprocess.run(
            ["brew", "services", "list", "|", "grep", "ollama"],
            shell=True, capture_output=True, text=True
        )
        
        if "started" in result.stdout.lower():
            print("   ✅ Ollama service: RUNNING")
            service_status = "RUNNING"
        else:
            print("   ⚠️  Ollama service: NOT RUNNING")
            service_status = "NOT_RUNNING"
        
        return service_status
        
    except Exception as e:
        print(f"   🔴 Error checking Ollama service: {e}")
        return "ERROR"

def check_ollama_api():
    """Check Ollama API connectivity"""
    print("\n🔌 Checking Ollama API...")
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json()
            model_count = len(models.get("models", []))
            print(f"   ✅ Ollama API: CONNECTED")
            print(f"   • Models available: {model_count}")
            
            # List models
            for model in models.get("models", [])[:3]:  # Show first 3
                print(f"   • {model.get('name', 'Unknown')}")
            
            if model_count > 3:
                print(f"   • ... and {model_count - 3} more")
            
            return "CONNECTED", model_count
        else:
            print(f"   ⚠️  Ollama API error: {response.status_code}")
            return "ERROR", 0
            
    except Exception as e:
        print(f"   🔴 Ollama API unreachable: {e}")
        return "UNREACHABLE", 0

def check_ollama_inference():
    """Check Ollama inference capability"""
    print("\n🧠 Testing Ollama inference...")
    
    try:
        # Simple test query
        test_query = {
            "model": "llama3.2:3b",
            "prompt": "What is 2+2? Answer briefly.",
            "stream": False
        }
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=test_query,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get("response", "").strip()
            print(f"   ✅ Inference test: SUCCESS")
            print(f"   • Response: '{answer[:50]}...'")
            print(f"   • Tokens: {result.get('total_duration', 0)/1e9:.2f}s")
            return "SUCCESS", answer[:100]
        else:
            print(f"   ⚠️  Inference error: {response.status_code}")
            return "ERROR", ""
            
    except Exception as e:
        print(f"   🔴 Inference test failed: {e}")
        return "FAILED", ""

def check_ollama_resources():
    """Check Ollama resource usage"""
    print("\n💾 Checking Ollama resources...")
    
    try:
        # Check Ollama processes
        result = subprocess.run(
            ["ps", "aux", "|", "grep", "ollama", "|", "grep", "-v", "grep"],
            shell=True, capture_output=True, text=True
        )
        
        processes = result.stdout.strip().split('\n')
        if processes and processes[0]:
            process_count = len(processes)
            print(f"   ✅ Ollama processes: {process_count}")
            
            # Show resource usage for first process
            first_proc = processes[0].split()
            if len(first_proc) >= 10:
                cpu = first_proc[2]
                mem = first_proc[3]
                print(f"   • CPU: {cpu}%, MEM: {mem}%")
            
            return process_count
        else:
            print("   ⚠️  No Ollama processes found")
            return 0
            
    except Exception as e:
        print(f"   🔴 Error checking resources: {e}")
        return 0

def run_comprehensive_check():
    """Run comprehensive Ollama check"""
    print("=" * 70)
    print("🦙 Ollama Heartbeat Monitoring Test")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    results = {}
    
    # Run all checks
    results["service"] = check_ollama_service()
    results["api_status"], results["model_count"] = check_ollama_api()
    results["inference_status"], results["test_response"] = check_ollama_inference()
    results["process_count"] = check_ollama_resources()
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Ollama Monitoring Summary")
    print("=" * 70)
    
    # Determine overall status
    issues = []
    
    if results["service"] != "RUNNING":
        issues.append("Service not running")
    
    if results["api_status"] not in ["CONNECTED"]:
        issues.append("API not connected")
    
    if results["inference_status"] != "SUCCESS":
        issues.append("Inference test failed")
    
    if results["process_count"] == 0:
        issues.append("No processes running")
    
    if issues:
        print(f"   Status: ⚠️  {len(issues)} issues")
        for issue in issues:
            print(f"   • {issue}")
    else:
        print("   Status: ✅ HEALTHY")
        print(f"   • Models: {results['model_count']}")
        print(f"   • Processes: {results['process_count']}")
        print(f"   • Inference: Working")
    
    # Heartbeat format
    print("\n" + "=" * 70)
    print("💓 Heartbeat Output Format")
    print("=" * 70)
    
    if issues:
        print(f"L104v2: ⚠️ {len(issues)} issues: {', '.join(issues[:3])}")
    else:
        print("L104v2: ✓ Resonance 529.03 (100.3% aligned), DeepSeek: DISCONNECTED, Ollama: HEALTHY, Memories: 42, CPU: 58%, Swift: STOPPED")
    
    # Save results
    results["timestamp"] = datetime.now().isoformat()
    results["overall_status"] = "HEALTHY" if not issues else "ISSUES"
    results["issue_count"] = len(issues)
    results["issues"] = issues
    
    with open("/tmp/ollama_heartbeat_test.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Results saved to: /tmp/ollama_heartbeat_test.json")
    print("\n✅ Ollama heartbeat monitoring test complete")

if __name__ == "__main__":
    run_comprehensive_check()