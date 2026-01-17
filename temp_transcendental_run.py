import asyncio
import os
import sys

sys.path.append(os.getcwd())

from l104_asi_core import asi_core

async def run():
    await asi_core.ignite_sovereignty()
    await asi_core.run_unbound_cycle()
    stats = asi_core.get_status()
    print(f"\n[IQ]: {stats['intellect_index']}")
    print(f"[SOLVED]: {asi_core.impossible_problems_solved}")

if __name__ == "__main__":
    asyncio.run(run())
