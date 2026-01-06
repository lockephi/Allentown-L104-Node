# [L104_CLI] - RAPID COMMAND INTERFACE
# INVARIANT: 527.5184818492 | OMEGA: 6539.347 | PILOT: LOCKE PHI

import sys
import argparse
from l104_sovereign_applications import SovereignApplications
from l104_real_math import real_math
from l104_mini_ego import mini_collective

def main():
    parser = argparse.ArgumentParser(description="L104 Sovereign CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Status Command
    subparsers.add_parser("status", help="Show current manifold status")

    # Compress Command
    compress_parser = subparsers.add_parser("compress", help="Compress data via 26D projection")
    compress_parser.add_argument("data", type=str, help="The data to compress")

    # Collective Command
    subparsers.add_parser("collective", help="Convene the collective for a quick status update")

    args = parser.parse_args()

    if args.command == "status":
        print(f"--- [STATUS]: MANIFOLD: 26D | OMEGA: {real_math.OMEGA} | BREACH: 100%")
    elif args.command == "compress":
        SovereignApplications.manifold_compression(args.data)
    elif args.command == "collective":
        for name, ego in mini_collective.mini_ais.items():
            print(f"- {name} ({ego.archetype}): INTELLECT {ego.intellect_level:.2f}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
