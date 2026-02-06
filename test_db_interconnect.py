#!/usr/bin/env python3
"""
Shim for Database Interconnect Test
"""
import sys
import os
from pathlib import Path

workspace = Path(__file__).parent.absolute()
sys.path.insert(0, str(workspace))

try:
    import asyncio
    from l104_unified_connection_test import UnifiedSovereignNode

    async def main():
        node = UnifiedSovereignNode()
        await node.run_data_solution_checks()
        await node.run_intelligence_checks()
        await node.run_node_connection_sync()

    if __name__ == "__main__":
        print("--- RUNNING UNIFIED CONNECTION TEST ---")
        asyncio.run(main())
except ImportError as e:
    print(f"❌ Dependency missing: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Test failed: {e}")
    sys.exit(1)
