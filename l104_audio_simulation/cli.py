"""
CLI — command-line interface for the audio simulation suite.

Provides the same CLI as the original _gen_perfect_audio.py v8.2.

Usage::

    python -m l104_audio_simulation --duration 60 --bit-depth 24
    python -m l104_audio_simulation --dials 0,0,0,0  1,0,0,0  0,0,0,-1
    python -m l104_audio_simulation --pure --dials 0,0,0,0

INVARIANT: 527.5184818492612 | PILOT: LONDEL
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional, Tuple

from .constants import DEFAULT_SAMPLE_RATE, DEFAULT_DURATION, DEFAULT_BIT_DEPTH
from .god_code_equation import parse_dial_arg
from .pipeline import AudioSimulationPipeline


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser matching _gen_perfect_audio.py v8.2 CLI."""
    p = argparse.ArgumentParser(
        description="L104 Audio Simulation Suite v1.0.0 — "
                    "Quantum-Simulated GOD_CODE Audio (14-layer pipeline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE,
                   help=f"Sample rate in Hz (default: {DEFAULT_SAMPLE_RATE})")
    p.add_argument("--duration", type=float, default=DEFAULT_DURATION,
                   help=f"Duration in seconds (default: {DEFAULT_DURATION} = 5 min)")
    p.add_argument("--bit-depth", type=int, choices=[16, 24], default=DEFAULT_BIT_DEPTH,
                   help=f"Bit depth: 16 or 24 (default: {DEFAULT_BIT_DEPTH})")
    p.add_argument("--pure-file", type=str,
                   default="god_code_quantum_pure_5min.wav",
                   help="Output filename for pure tone")
    p.add_argument("--harm-file", type=str,
                   default="god_code_quantum_harmonics_5min.wav",
                   help="Output filename for harmonics")
    p.add_argument("--binaural-file", type=str,
                   default="god_code_binaural_5min.wav",
                   help="Output filename for binaural beat")
    p.add_argument("--fade", type=float, default=0.5,
                   help="Fade in/out duration in seconds (default: 0.5)")
    p.add_argument("--no-binaural", action="store_true",
                   help="Skip binaural beat generation")
    p.add_argument("--mono", action="store_true",
                   help="Output mono instead of stereo")
    p.add_argument("--amplitude", type=float, default=0.95,
                   help="Peak amplitude 0.0–1.0 (default: 0.95)")
    p.add_argument("--dials", nargs="+", metavar="a,b,c,d",
                   help="Replace default dial table with custom G(a,b,c,d) tuples. "
                        "E.g. --dials 12,0,0,5  0,0,0,0  3,0,0,0")
    p.add_argument("--extra-dials", nargs="+", metavar="a,b,c,d",
                   help="Append extra dials to the default 13-partial table. "
                        "E.g. --extra-dials 12,0,0,5  7,0,0,2")
    p.add_argument("--pure", action="store_true",
                   help="Pure quantum tone: fundamental-dominant (60%%) with tight "
                        "fine-detune companions.")
    p.add_argument("--output", type=str, default=None,
                   help="Output filename for dial mode (default: god_code_dial.wav)")
    return p


def main(argv: Optional[List[str]] = None):
    """Main entry point for the audio simulation CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    pipeline = AudioSimulationPipeline(
        sample_rate=args.sample_rate,
        duration=args.duration,
        bit_depth=args.bit_depth,
        stereo=not args.mono,
        fade_seconds=args.fade,
        amplitude=args.amplitude,
    )

    # Determine mode
    dial_mode = args.dials is not None or args.extra_dials is not None

    if dial_mode:
        # Parse dial tuples
        dials = []
        if args.dials:
            for raw in args.dials:
                dials.append(parse_dial_arg(raw))
        if args.extra_dials:
            # Start with default table dials then append extras
            if not args.dials:
                from .god_code_equation import DEFAULT_DIALS
                dials = [(a, b, c, d) for a, b, c, d, _, _ in DEFAULT_DIALS]
            for raw in args.extra_dials:
                dials.append(parse_dial_arg(raw))

        output = args.output or "god_code_dial.wav"
        result = pipeline.generate_dial(
            dials=dials,
            output_file=output,
            pure_mode=args.pure,
            verbose=True,
        )
        print(f"\n  DIAL MODE COMPLETE — {result['file']} "
              f"({result['size_mb']:.1f} MB, {result['total_time_s']:.1f}s)")
    else:
        result = pipeline.generate_standard(
            pure_file=args.pure_file,
            harm_file=args.harm_file,
            binaural_file=args.binaural_file,
            gen_binaural=not args.no_binaural,
            verbose=True,
        )
        print(f"\n  STANDARD MODE COMPLETE — {result['total_time_s']:.1f}s total")

    return result


if __name__ == "__main__":
    main()
