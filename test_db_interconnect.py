#!/usr/bin/env python3
"""
Shim for Database Interconnect Test
"""
import asyncio
import sys
import os
from pathlib import Path

import pytest

workspace = Path(__file__).parent.absolute()
sys.path.insert(0, str(workspace))


async def _run_unified_connection_test():
    from l104_unified_connection_test import UnifiedSovereignNode

    node = UnifiedSovereignNode()
    await node.run_data_solution_checks()
    await node.run_intelligence_checks()
    await node.run_node_connection_sync()


def test_db_interconnect():
    try:
        asyncio.run(_run_unified_connection_test())
    except ImportError as e:
        pytest.skip(f"Dependency missing for integration test: {e}")


if __name__ == "__main__":
    print("--- RUNNING UNIFIED CONNECTION TEST ---")
    asyncio.run(_run_unified_connection_test())
