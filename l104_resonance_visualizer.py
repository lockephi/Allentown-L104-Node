# L104_GOD_CODE_ALIGNED: 527.5184818492611
"""
L104 RESONANCE VISUALIZER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Terminal-based visualization of GOD_CODE resonance patterns.

G(X) = 286^(1/φ) × 2^((416-X)/104)
Conservation: G(X) × 2^(X/104) = 527.5184818492611

INVARIANT: 527.5184818492611 | PILOT: LONDEL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import math
import time
import sys
from typing import List, Tuple

# Sacred Constants
GOD_CODE = 527.5184818492611
PHI = 1.618033988749895
HARMONIC_BASE = 286
L104 = 104
OCTAVE_REF = 416

# Terminal colors
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    GOLD = "\033[38;5;220m"
    CYAN = "\033[38;5;51m"
    MAGENTA = "\033[38;5;201m"
    GREEN = "\033[38;5;46m"
    RED = "\033[38;5;196m"
    WHITE = "\033[38;5;255m"
    DIM = "\033[2m"


def calculate_god_code(X: float) -> float:
    """G(X) = 286^(1/φ) × 2^((416-X)/104)"""
    base = HARMONIC_BASE ** (1 / PHI)
    exponent = (OCTAVE_REF - X) / L104
    return base * (2 ** exponent)


def calculate_weight(X: float) -> float:
    """Weight = 2^(X/104)"""
    return 2 ** (X / L104)


def verify_conservation(X: float) -> Tuple[float, float]:
    """Returns (invariant, deviation)"""
    g_x = calculate_god_code(X)
    w = calculate_weight(X)
    invariant = g_x * w
    deviation = abs(invariant - GOD_CODE)
    return invariant, deviation


def draw_wave(X: float, width: int = 60) -> str:
    """Draw a wave pattern based on X value."""
    g_x = calculate_god_code(X)
    normalized = (g_x / GOD_CODE) * 0.5  # 0 to 0.5+ range
    
    # Create wave pattern
    wave = ""
    for i in range(width):
        phase = (i / width) * 4 * math.pi + (X / 10)
        amplitude = math.sin(phase) * normalized * 10
        
        if amplitude > 0.5:
            wave += f"{Colors.GOLD}█{Colors.RESET}"
        elif amplitude > 0.2:
            wave += f"{Colors.CYAN}▓{Colors.RESET}"
        elif amplitude > -0.2:
            wave += f"{Colors.DIM}░{Colors.RESET}"
        elif amplitude > -0.5:
            wave += f"{Colors.MAGENTA}▒{Colors.RESET}"
        else:
            wave += f"{Colors.RED}█{Colors.RESET}"
    
    return wave


def draw_conservation_bar(deviation: float, width: int = 40) -> str:
    """Draw a bar showing conservation accuracy."""
    if deviation < 1e-12:
        fill = width
        color = Colors.GREEN
    elif deviation < 1e-10:
        fill = int(width * 0.95)
        color = Colors.GOLD
    elif deviation < 1e-6:
        fill = int(width * 0.7)
        color = Colors.CYAN
    else:
        fill = int(width * 0.3)
        color = Colors.RED
    
    bar = f"{color}{'█' * fill}{Colors.DIM}{'░' * (width - fill)}{Colors.RESET}"
    return bar


def visualize_spectrum(X_range: Tuple[float, float] = (-416, 416), steps: int = 30):
    """Visualize G(X) spectrum across X values."""
    print(f"\n{Colors.BOLD}{Colors.GOLD}╔══════════════════════════════════════════════════════════════════════════╗{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GOLD}║              L104 RESONANCE SPECTRUM VISUALIZER                          ║{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GOLD}╚══════════════════════════════════════════════════════════════════════════╝{Colors.RESET}")
    print()
    print(f"  {Colors.CYAN}G(X) = 286^(1/φ) × 2^((416-X)/104){Colors.RESET}")
    print(f"  {Colors.MAGENTA}Conservation: G(X) × 2^(X/104) = {GOD_CODE}{Colors.RESET}")
    print()
    
    X_min, X_max = X_range
    step_size = (X_max - X_min) / steps
    
    print(f"  {'X':>8}  {'G(X)':>12}  {'Weight':>10}  {'Conservation':>12}  Resonance Pattern")
    print(f"  {'-'*8}  {'-'*12}  {'-'*10}  {'-'*12}  {'-'*40}")
    
    for i in range(steps + 1):
        X = X_min + i * step_size
        g_x = calculate_god_code(X)
        weight = calculate_weight(X)
        invariant, deviation = verify_conservation(X)
        
        # Highlight key X values
        if abs(X) < 0.01:
            x_str = f"{Colors.GOLD}{X:>8.1f}{Colors.RESET}"
        elif X % 104 == 0:
            x_str = f"{Colors.CYAN}{X:>8.1f}{Colors.RESET}"
        else:
            x_str = f"{X:>8.1f}"
        
        # Color G(X) based on value
        if g_x > GOD_CODE:
            gx_str = f"{Colors.RED}{g_x:>12.4f}{Colors.RESET}"
        elif g_x > GOD_CODE/2:
            gx_str = f"{Colors.GOLD}{g_x:>12.4f}{Colors.RESET}"
        else:
            gx_str = f"{Colors.CYAN}{g_x:>12.4f}{Colors.RESET}"
        
        # Conservation status
        if deviation < 1e-10:
            cons_str = f"{Colors.GREEN}{invariant:>12.6f}{Colors.RESET}"
        else:
            cons_str = f"{Colors.RED}{invariant:>12.6f}{Colors.RESET}"
        
        # Mini wave
        wave = ""
        wave_width = 20
        for j in range(wave_width):
            phase = (j / wave_width) * 2 * math.pi
            amp = math.sin(phase + X/50) * (g_x / GOD_CODE)
            if amp > 0.3:
                wave += f"{Colors.GOLD}█{Colors.RESET}"
            elif amp > 0:
                wave += f"{Colors.CYAN}▓{Colors.RESET}"
            elif amp > -0.3:
                wave += f"{Colors.MAGENTA}░{Colors.RESET}"
            else:
                wave += f"{Colors.RED}▒{Colors.RESET}"
        
        print(f"  {x_str}  {gx_str}  {weight:>10.4f}  {cons_str}  {wave}")
    
    print()


def animate_resonance(duration: float = 10.0, fps: float = 10):
    """Animate real-time resonance patterns."""
    print(f"\n{Colors.BOLD}{Colors.GOLD}╔══════════════════════════════════════════════════════════════════════════╗{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GOLD}║              L104 LIVE RESONANCE ANIMATION                               ║{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GOLD}╚══════════════════════════════════════════════════════════════════════════╝{Colors.RESET}")
    print(f"\n  Press Ctrl+C to stop\n")
    
    start = time.time()
    frame = 0
    
    try:
        while (time.time() - start) < duration:
            # Calculate dynamic X based on time (oscillating)
            t = time.time() - start
            X = 208 * math.sin(t * PHI)  # Oscillate between -208 and 208
            
            g_x = calculate_god_code(X)
            weight = calculate_weight(X)
            invariant, deviation = verify_conservation(X)
            
            # Build frame
            wave = draw_wave(X, 60)
            conservation_bar = draw_conservation_bar(deviation)
            
            # Clear line and print
            sys.stdout.write("\r" + " " * 100 + "\r")  # Clear line
            
            frame_output = (
                f"  {Colors.CYAN}X={X:>8.2f}{Colors.RESET} | "
                f"{Colors.GOLD}G(X)={g_x:>10.4f}{Colors.RESET} | "
                f"{Colors.GREEN}Conservation: {invariant:.10f}{Colors.RESET}"
            )
            sys.stdout.write(frame_output)
            sys.stdout.flush()
            
            frame += 1
            time.sleep(1/fps)
            
    except KeyboardInterrupt:
        pass
    
    print(f"\n\n  {Colors.GREEN}✓ Animation complete. {frame} frames rendered.{Colors.RESET}\n")


def show_factor_13_tree():
    """Display the Factor 13 sacred geometry."""
    print(f"""
{Colors.BOLD}{Colors.GOLD}╔══════════════════════════════════════════════════════════════════════════╗
║                         FACTOR 13 SACRED GEOMETRY                        ║
╚══════════════════════════════════════════════════════════════════════════╝{Colors.RESET}

                           {Colors.CYAN}13{Colors.RESET} (7th Fibonacci)
                          ╱   ╲
                    {Colors.GOLD}286{Colors.RESET}        {Colors.MAGENTA}104{Colors.RESET}        {Colors.GREEN}416{Colors.RESET}
                   ╱   ╲      │         │
               {Colors.GOLD}2×11×13{Colors.RESET}   {Colors.MAGENTA}8×13{Colors.RESET}      {Colors.GREEN}32×13{Colors.RESET}
                   │         │         │
              {Colors.GOLD}÷13=22{Colors.RESET}   {Colors.MAGENTA}÷13=8{Colors.RESET}    {Colors.GREEN}÷13=32{Colors.RESET}

    {Colors.WHITE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Colors.RESET}
    
    {Colors.GOLD}GOD_CODE = 286^(1/φ) × 2^(416/104){Colors.RESET}
             {Colors.GOLD}= 286^(1/φ) × 2^4{Colors.RESET}
             {Colors.GOLD}= 286^(1/φ) × 16{Colors.RESET}
             {Colors.BOLD}= {GOD_CODE}{Colors.RESET}
    
    {Colors.CYAN}Conservation Law:{Colors.RESET}
    G(X) × 2^(X/104) = {GOD_CODE} = INVARIANT
    
    {Colors.MAGENTA}The whole never changes. Only the rate of change varies.{Colors.RESET}
    {Colors.MAGENTA}X increasing → Magnetic compaction (gravity){Colors.RESET}
    {Colors.MAGENTA}X decreasing → Electric expansion (light){Colors.RESET}
""")


def main():
    """CLI interface for resonance visualizer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="L104 Resonance Visualizer")
    parser.add_argument("--spectrum", action="store_true", help="Show G(X) spectrum")
    parser.add_argument("--animate", type=float, default=0, help="Animate for N seconds")
    parser.add_argument("--factor13", action="store_true", help="Show Factor 13 tree")
    parser.add_argument("--X", type=float, default=None, help="Calculate for specific X")
    
    args = parser.parse_args()
    
    if args.X is not None:
        X = args.X
        g_x = calculate_god_code(X)
        weight = calculate_weight(X)
        invariant, deviation = verify_conservation(X)
        print(f"\n  X = {X}")
        print(f"  G(X) = {g_x}")
        print(f"  Weight = {weight}")
        print(f"  Invariant = {invariant}")
        print(f"  Deviation = {deviation}")
        print(f"  Conservation: {'✓ VERIFIED' if deviation < 1e-10 else '✗ BROKEN'}\n")
    elif args.spectrum:
        visualize_spectrum()
    elif args.animate > 0:
        animate_resonance(duration=args.animate)
    elif args.factor13:
        show_factor_13_tree()
    else:
        # Default: show everything
        show_factor_13_tree()
        visualize_spectrum(steps=20)


if __name__ == "__main__":
    main()
