import asyncio
import logging
from l104_choice_engine import choice_engine

logging.basicConfig(level=logging.INFO)

async def test_delegation():
    print("--- [TEST]: TRIGGERING CLOUD DELEGATION ---")
    result = await choice_engine._delegate_to_cloud()
    print(f"--- [TEST]: RESULT: {result}")

if __name__ == "__main__":
    asyncio.run(test_delegation())
