import asyncio
from l104_omega_controller import omega_controller

async def main():
    print("--- STARTING ABSOLUTE SINGULARITY TRIGGER ---")
    try:
        result = await omega_controller.trigger_absolute_singularity()
        print("\nSINGULARITY RESULT:")
        for phase in result.get("phases", []):
            print(f"Phase {phase['phase']}: {phase['name']} -> {phase['result']}")
        print(f"Final State: {result.get('final_state')}")
        print(f"Final Coherence: {result.get('final_coherence')}")
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(main())
