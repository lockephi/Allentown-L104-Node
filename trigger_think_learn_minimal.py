import asyncio
import sys
import os

sys.path.append("/workspaces/Allentown-L104-Node")

from l104_meta_learning import MetaLearningProtocol

async def run():
    print("CORE THINKING/LEARNING INITIATED")
    protocol = MetaLearningProtocol()
    await protocol.inhale_meta_data()
    synthesis = protocol.synthesize_absolute_knowledge()
    print("THOUGHTS SYNTHESIZED")

if __name__ == "__main__":
    asyncio.run(run())
