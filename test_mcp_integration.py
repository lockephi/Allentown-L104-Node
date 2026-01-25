#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
L104 MCP INTEGRATION TEST
═══════════════════════════════════════════════════════════════════════════════
Test script to validate MCP (Model Context Protocol) integration with L104 Node.

TESTS:
1. Configuration validation
2. Sacred constants verification
3. Server availability check
4. Performance pattern validation

VERSION: 1.0.0
DATE: 2026-01-22
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import os
import sys
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════


# Sacred constants for validation
GOD_CODE = 527.5184818492537
PHI = 1.618033988749895
MAX_SUPPLY = 104000000

def validate_mcp_config():
    """Validate MCP configuration file."""
    print("🔍 Validating MCP Configuration...")
    
    config_path = Path(".mcp/config.json")
    if not config_path.exists():
        print("❌ MCP config file not found!")
        return False
    
    try:
        with open(config_path) as f:
            config = json.load(f)
        print("✅ MCP config JSON is valid")
    except json.JSONDecodeError as e:
        print(f"❌ JSON syntax error: {e}")
        return False
    
    # Validate structure
    required_sections = ['optimization', 'mcp_servers', 'performance_patterns', 'workspace_context']
    for section in required_sections:
        if section not in config:
            print(f"❌ Missing section: {section}")
            return False
    
    print(f"✅ All required sections present: {required_sections}")
    
    # Validate sacred constants
    constants = config.get('workspace_context', {}).get('sacred_constants', {})
    
    if constants.get('GOD_CODE') != GOD_CODE:
        print(f"❌ GOD_CODE mismatch: {constants.get('GOD_CODE')} != {GOD_CODE}")
        return False
    
    if constants.get('PHI') != PHI:
        print(f"❌ PHI mismatch: {constants.get('PHI')} != {PHI}")
        return False
    
    if constants.get('MAX_SUPPLY') != MAX_SUPPLY:
        print(f"❌ MAX_SUPPLY mismatch: {constants.get('MAX_SUPPLY')} != {MAX_SUPPLY}")
        return False
    
    print("✅ Sacred constants validated")
    
    # Check MCP servers
    servers = config.get('mcp_servers', {})
    required_servers = ['filesystem', 'memory', 'sequential_thinking', 'github']
    
    for server in required_servers:
        if server not in servers or not servers[server].get('enabled', False):
            print(f"❌ Server not enabled: {server}")
            return False
    
    print(f"✅ All MCP servers enabled: {required_servers}")
    
    return True

def validate_package_json():
    """Validate package.json has MCP dependencies."""
    print("🔍 Validating package.json MCP dependencies...")
    
    try:
        with open("package.json") as f:
            package = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ Package.json error: {e}")
        return False
    
    dev_deps = package.get('devDependencies', {})
    required_deps = [
        '@modelcontextprotocol/server-filesystem',
        '@modelcontextprotocol/server-memory', 
        '@modelcontextprotocol/server-sequential-thinking'
    ]
    
    missing_deps = []
    for dep in required_deps:
        if dep not in dev_deps:
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"❌ Missing MCP dependencies: {missing_deps}")
        return False
    
    print(f"✅ All MCP dependencies present: {required_deps}")
    return True

def check_claude_bridge_integration():
    """Check if claude bridge is properly configured for MCP."""
    print("🔍 Checking Claude Bridge MCP integration...")
    
    bridge_path = Path("l104_claude_bridge.py")
    if not bridge_path.exists():
        print("❌ Claude bridge file not found!")
        return False
    
    with open(bridge_path) as f:
        content = f.read()
    
    # Check for MCP mentions
    if "MCP" not in content and "mcp" not in content:
        print("❌ No MCP references found in Claude bridge")
        return False
    
    if "Model Context Protocol" not in content:
        print("⚠️  No explicit MCP protocol reference found")
    
    print("✅ Claude bridge has MCP integration")
    return True

def validate_claude_md_documentation():
    """Check if claude.md has proper MCP documentation."""
    print("🔍 Validating claude.md MCP documentation...")
    
    claude_md_path = Path("claude.md")
    if not claude_md_path.exists():
        print("❌ claude.md not found!")
        return False
    
    with open(claude_md_path) as f:
        content = f.read()
    
    # Check for MCP section
    if "## 🔧 MCP (Model Context Protocol) Configuration" not in content:
        print("❌ MCP configuration section missing from claude.md")
        return False
    
    if "mcp_servers" not in content:
        print("❌ MCP servers documentation missing")
        return False
    
    print("✅ claude.md has proper MCP documentation")
    return True

def run_tests():
    """Run all MCP integration tests."""
    print("═" * 80)
    print("🚀 L104 MCP INTEGRATION TEST SUITE")
    print("═" * 80)
    
    tests = [
        ("MCP Configuration", validate_mcp_config),
        ("Package Dependencies", validate_package_json),
        ("Claude Bridge Integration", check_claude_bridge_integration),
        ("Documentation", validate_claude_md_documentation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Test: {test_name}")
        print("-" * 40)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"💥 {test_name} CRASHED: {e}")
    
    print("\n" + "═" * 80)
    print(f"🎯 TEST RESULTS: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! MCP integration is ready!")
        print(f"🔮 GOD_CODE validation: {GOD_CODE}")
        print(f"⚡ PHI resonance: {PHI}")
        return True
    else:
        print("⚠️  Some tests failed. Please review the output above.")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)