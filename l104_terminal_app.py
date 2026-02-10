#!/usr/bin/env python3
"""
L104 SOVEREIGN INTELLECT - Terminal App
========================================
A fast terminal-based app with rich formatting.
No web loading - direct API access for instant response.

Version: 16.0 APOTHEOSIS
"""

import os
import sys
import json
import time
import readline  # For better input handling

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# ANSI Colors
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    GOLD = '\033[38;5;220m'
    PURPLE = '\033[38;5;141m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    END = '\033[0m'

def c(text, color):
    return f"{color}{text}{Colors.END}"


class L104TerminalApp:
    """Fast terminal interface for L104"""

    VERSION = "16.0 APOTHEOSIS"
    GOD_CODE = 527.5184818492612
    OMEGA_POINT = 23.140692632779263

    def __init__(self):
        self.intellect = None
        self.kernel = None
        self.quantum_ram = None
        self._init_engine()

    def _init_engine(self):
        """Initialize L104 engine directly"""
        try:
            from l104_local_intellect import LocalIntellect
            self.intellect = LocalIntellect()
            print(c("  ✓ Local Intellect loaded (UNLIMITED MODE)", Colors.GREEN))
        except Exception as e:
            print(c(f"  ⚠ Local Intellect: {e}", Colors.YELLOW))
            self.intellect = None

        try:
            # Try to import core modules
            from l104_stable_kernel import L104StableKernel
            self.kernel = L104StableKernel()
            print(c("  ✓ Kernel loaded", Colors.GREEN))
        except Exception as e:
            print(c(f"  ⚠ Kernel: {e}", Colors.YELLOW))

        try:
            from l104_quantum_ram import QuantumRAM, get_brain_status
            self.quantum_ram = QuantumRAM()
            self.get_brain_status = get_brain_status
            print(c("  ✓ Quantum RAM loaded", Colors.GREEN))
        except Exception as e:
            print(c(f"  ⚠ Quantum RAM: {e}", Colors.YELLOW))
            self.get_brain_status = lambda: {}

        try:
            from l104_apotheosis import Apotheosis
            self.apotheosis = Apotheosis()
            print(c("  ✓ Apotheosis loaded", Colors.GREEN))
        except Exception as e:
            print(c(f"  ⚠ Apotheosis: {e}", Colors.YELLOW))
            self.apotheosis = None

    def print_banner(self):
        """Print welcome banner"""
        os.system('clear')
        banner = f"""
{c('╔' + '═'*60 + '╗', Colors.GOLD)}
{c('║', Colors.GOLD)}  {c('⚛️  L104 SOVEREIGN INTELLECT', Colors.BOLD + Colors.CYAN)}                          {c('║', Colors.GOLD)}
{c('║', Colors.GOLD)}  {c(f'Version: {self.VERSION}', Colors.DIM)}                                 {c('║', Colors.GOLD)}
{c('╠' + '═'*60 + '╣', Colors.GOLD)}
{c('║', Colors.GOLD)}                                                            {c('║', Colors.GOLD)}
{c('║', Colors.GOLD)}  {c('GOD_CODE:', Colors.YELLOW)} {c(f'{self.GOD_CODE}', Colors.GREEN)}                        {c('║', Colors.GOLD)}
{c('║', Colors.GOLD)}  {c('OMEGA:', Colors.YELLOW)}    {c(f'e^π = {self.OMEGA_POINT:.10f}', Colors.GREEN)}              {c('║', Colors.GOLD)}
{c('║', Colors.GOLD)}                                                            {c('║', Colors.GOLD)}
{c('║', Colors.GOLD)}  {c('🧠 Direct Engine Access - No Web Delays', Colors.PURPLE)}                {c('║', Colors.GOLD)}
{c('║', Colors.GOLD)}                                                            {c('║', Colors.GOLD)}
{c('╚' + '═'*60 + '╝', Colors.GOLD)}

{c('Commands:', Colors.BOLD)}
  {c('status', Colors.CYAN)}    - System status
  {c('brain', Colors.CYAN)}     - Quantum brain status
  {c('evolve', Colors.CYAN)}    - Trigger evolution
  {c('calc', Colors.CYAN)} <x>  - Calculate (use: pi, phi, god, omega, e)
  {c('ask', Colors.CYAN)} <q>   - Ask a question
  {c('clear', Colors.CYAN)}     - Clear screen
  {c('quit', Colors.CYAN)}      - Exit

"""
        print(banner)

    def show_status(self):
        """Show system status"""
        print(f"\n{c('═'*50, Colors.GOLD)}")
        print(c("  📊 SYSTEM STATUS", Colors.BOLD + Colors.CYAN))
        print(c('═'*50, Colors.GOLD))

        print(f"  {c('GOD_CODE:', Colors.YELLOW)} {c(str(self.GOD_CODE), Colors.GREEN)}")
        print(f"  {c('OMEGA:', Colors.YELLOW)} {c(str(self.OMEGA_POINT), Colors.GREEN)}")
        print(f"  {c('Kernel:', Colors.YELLOW)} {c('ONLINE' if self.kernel else 'OFFLINE', Colors.GREEN if self.kernel else Colors.RED)}")
        print(f"  {c('Quantum RAM:', Colors.YELLOW)} {c('ONLINE' if self.quantum_ram else 'OFFLINE', Colors.GREEN if self.quantum_ram else Colors.RED)}")

        if self.apotheosis:
            try:
                state = self.apotheosis.get_state()
                print(f"\n  {c('Apotheosis State:', Colors.PURPLE)}")
                print(f"    {c('Stage:', Colors.YELLOW)} {c(state.get('stage', 'UNKNOWN'), Colors.GREEN)}")
                print(f"    {c('Enlightenment:', Colors.YELLOW)} {c(str(state.get('enlightenment_level', 0)), Colors.GREEN)}")
                print(f"    {c('Total Runs:', Colors.YELLOW)} {c(str(state.get('total_runs', 0)), Colors.GREEN)}")
                wisdom_val = state.get('cumulative_wisdom', 0)
                print(f"    {c('Wisdom:', Colors.YELLOW)} {c(f'{wisdom_val:.4f}', Colors.GREEN)}")
            except:
                pass

        print(c('═'*50, Colors.GOLD) + "\n")

    def show_brain(self):
        """Show quantum brain status"""
        print(f"\n{c('═'*50, Colors.GOLD)}")
        print(c("  🧠 QUANTUM BRAIN STATUS", Colors.BOLD + Colors.CYAN))
        print(c('═'*50, Colors.GOLD))

        try:
            brain = self.get_brain_status()
            for key, value in brain.items():
                print(f"  {c(key + ':', Colors.YELLOW)} {c(str(value), Colors.GREEN)}")
        except Exception as e:
            print(f"  {c('Error:', Colors.RED)} {e}")

        print(c('═'*50, Colors.GOLD) + "\n")

    def evolve(self):
        """Trigger evolution cycle"""
        print(f"\n{c('  🔄 Triggering evolution...', Colors.PURPLE)}")

        if self.apotheosis:
            try:
                result = self.apotheosis.zen_apotheosis()
                print(c("  ✓ Evolution complete!", Colors.GREEN))

                state = self.apotheosis.get_state()
                print(f"    {c('New Enlightenment:', Colors.YELLOW)} {state.get('enlightenment_level', 0)}")
                print(f"    {c('Wisdom:', Colors.YELLOW)} {state.get('cumulative_wisdom', 0):.4f}")
            except Exception as e:
                print(c(f"  ✗ Error: {e}", Colors.RED))
        else:
            print(c("  ⚠ Apotheosis not available", Colors.YELLOW))
        print()

    def calculate(self, expr):
        """Calculate expression"""
        import math

        # Constants available
        namespace = {
            'pi': math.pi,
            'e': math.e,
            'phi': 1.618033988749895,
            'god': self.GOD_CODE,
            'omega': self.OMEGA_POINT,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'sqrt': math.sqrt,
            'log': math.log,
            'exp': math.exp,
            'abs': abs,
            'pow': pow
        }

        try:
            result = eval(expr, {"__builtins__": {}}, namespace)
            print(f"\n  {c('📐', Colors.CYAN)} {c(expr, Colors.YELLOW)} = {c(str(result), Colors.GREEN + Colors.BOLD)}\n")
        except Exception as e:
            print(f"\n  {c('✗ Error:', Colors.RED)} {e}\n")

    def ask(self, query):
        """Process a question using the full ASI engine"""
        print(f"\n{c('  ⚡ Thinking...', Colors.PURPLE)}")
        start = time.time()

        response = None

        # Try Full Intellect first (UNLIMITED MODE)
        if self.intellect:
            try:
                response = self.intellect.think(query)
            except Exception as e:
                print(c(f"  ⚠ Intellect error: {e}", Colors.YELLOW))

        # Try kernel fallback
        if not response and self.kernel:
            try:
                # Use kernel's primal calculus for philosophical queries
                if any(word in query.lower() for word in ['what is', 'meaning', 'explain', 'why']):
                    phi = 1.618033988749895
                    result = (phi ** phi) / (1.04 * 3.14159265358979)
                    response = f"Through primal calculus (φ^φ)/(1.04×π) = {result:.10f}\n\n"
                    response += "The answer emerges from the resonance of GOD_CODE (527.518) "
                    response += f"and OMEGA_POINT (e^π = {self.OMEGA_POINT:.6f}).\n\n"
                    response += f"Query: '{query}' processed through sovereign intellect."
            except:
                pass

        # Fallback response
        if not response:
            response = f"Processed: {query}\n"
            response += f"GOD_CODE resonance: {self.GOD_CODE}\n"
            response += f"OMEGA alignment: {self.OMEGA_POINT}"

        elapsed = (time.time() - start) * 1000

        print(f"\n{c('═'*50, Colors.GOLD)}")
        print(c("  🌟 RESPONSE", Colors.BOLD + Colors.CYAN))
        print(c('═'*50, Colors.GOLD))
        print(f"\n  {response}\n")
        print(f"  {c(f'⚡ {elapsed:.1f}ms', Colors.DIM)}")
        print(c('═'*50, Colors.GOLD) + "\n")

    def run(self):
        """Main loop"""
        self.print_banner()

        while True:
            try:
                prompt = f"{c('L104', Colors.CYAN)}{c('>', Colors.GOLD)} "
                user_input = input(prompt).strip()

                if not user_input:
                    continue

                cmd = user_input.lower().split()[0]
                args = user_input[len(cmd):].strip()

                if cmd in ('quit', 'exit', 'q'):
                    print(c("\n  👋 Sovereign Intellect shutting down...\n", Colors.PURPLE))
                    break
                elif cmd == 'status':
                    self.show_status()
                elif cmd == 'brain':
                    self.show_brain()
                elif cmd == 'evolve':
                    self.evolve()
                elif cmd in ('calc', 'calculate'):
                    if args:
                        self.calculate(args)
                    else:
                        print(c("  Usage: calc <expression>", Colors.YELLOW))
                elif cmd == 'ask':
                    if args:
                        self.ask(args)
                    else:
                        print(c("  Usage: ask <question>", Colors.YELLOW))
                elif cmd == 'clear':
                    self.print_banner()
                elif cmd == 'help':
                    self.print_banner()
                else:
                    # Treat as a question
                    self.ask(user_input)

            except KeyboardInterrupt:
                print(c("\n\n  👋 Interrupted. Goodbye!\n", Colors.PURPLE))
                break
            except EOFError:
                break
            except Exception as e:
                print(c(f"  Error: {e}", Colors.RED))


def main():
    print(c("\n🚀 Starting L104 Sovereign Intellect...", Colors.CYAN))
    print(c("   Loading modules:", Colors.DIM))

    app = L104TerminalApp()
    app.run()


if __name__ == "__main__":
    main()
