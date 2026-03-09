#!/usr/bin/env python3
"""Quick verify daemon data flows through to pipeline Layer 17."""
import sys, os, time, logging
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Enable debug logging to see daemon injection
logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s")

from l104_audio_simulation.pipeline import AudioSimulationPipeline

pipe = AudioSimulationPipeline(sample_rate=180_000, duration=2.0)  # 2s quick test
t0 = time.time()
result = pipe.generate_dial(
    dials=[(0, 0, 0, 0)],
    output_file="/tmp/l104_test_quick_daemon.wav",
    verbose=True,
)
dt = time.time() - t0
print(f"\nTotal: {dt:.1f}s")
print(f"Result keys: {list(result.keys()) if result else 'None'}")
if result:
    print(f"File: {result.get('file', 'N/A')}")
    print(f"Size: {result.get('size_mb', 'N/A')} MB")
    print(f"Time: {result.get('total_time_s', 'N/A')}s")
