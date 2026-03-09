#!/usr/bin/env python3
"""Profile audio pipeline bottlenecks — find where the time goes."""
import sys, os, time, cProfile, pstats, io
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from l104_audio_simulation.pipeline import AudioSimulationPipeline

pipe = AudioSimulationPipeline(sample_rate=180_000, duration=10.0)

pr = cProfile.Profile()
pr.enable()
result = pipe.generate_dial(
    dials=[(0, 0, 0, 0)],
    output_file="/tmp/l104_profile_test.wav",
    verbose=True,
)
pr.disable()

s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
ps.print_stats(40)
print(s.getvalue())

# Also print top by tottime
s2 = io.StringIO()
ps2 = pstats.Stats(pr, stream=s2).sort_stats("tottime")
ps2.print_stats(40)
print("\n=== BY TOTAL TIME ===")
print(s2.getvalue())
