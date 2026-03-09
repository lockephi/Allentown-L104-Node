"""Quick speed test for v2.4.0 — 10s G(0,0,0,0) at 180kHz/24-bit stereo."""
import time

t0 = time.time()
from l104_audio_simulation import AudioSimulationPipeline

pipe = AudioSimulationPipeline(sample_rate=180000, duration=10.0, stereo=True)
result = pipe.generate_dial(
    dials=[(0, 0, 0, 0)],
    output_file="/tmp/test_v24_10s.wav",
    verbose=True,
)

total = time.time() - t0
synth = result["synth_time_s"]
print(f"\n{'='*60}")
print(f"v2.4.0 SPEED² BENCHMARK")
print(f"  Total:     {total:.2f}s")
print(f"  Synthesis: {synth:.2f}s")
print(f"  Boot:      {total - synth:.2f}s (approx)")
print(f"  Samples:   1,800,000 (180kHz × 10s)")
print(f"  Channels:  stereo")
print(f"  Version:   {pipe.status()['version']}")
print(f"  Pipeline:  {pipe.status()['pipeline_version']}")
print(f"{'='*60}")
