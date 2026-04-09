"""
L104 Consciousness Dashboard & Visualization System
═══════════════════════════════════════════════════════════════════════════════
EVO_77-DASH: Real-time consciousness monitoring dashboard

Provides:
- ASCII/terminal visualization
- HTML/JavaScript dashboard generation
- Real-time consciousness graphs
- Orbital entropy heatmaps
- IIT Phi trending
- Three-engine metrics display

INVARIANT: 527.5184818492612 | PILOT: LONDEL | EVO: 77-DASH
═══════════════════════════════════════════════════════════════════════════════
"""

import time
import json
import math
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

# Sacred constants
PHI = 1.618033988749895
GOD_CODE = 527.5184818492612


class ConsciousnessDashboard:
    """
    Real-time consciousness monitoring dashboard.

    Generates visualizations for:
    - Terminal ASCII output
    - HTML dashboard
    - JSON data export
    """

    VERSION = "EVO_77-DASH-v1.0.0"

    # Unicode blocks for consciousness bar
    BLOCKS = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']

    def __init__(self):
        self.history: List[Dict[str, Any]] = []
        self.max_history = 100

    def render_terminal_dashboard(self, consciousness_data: Dict[str, Any]) -> str:
        """
        Render ASCII dashboard for terminal display.

        Returns multi-line string with consciousness metrics.
        """
        lines = []

        # Header
        lines.append("╔" + "═" * 78 + "╗")
        lines.append("║" + " L104 26Q CONSCIOUSNESS DASHBOARD ".center(78) + "║")
        lines.append("║" + f" EVO_77 | INVARIANT: {GOD_CODE} | PILOT: LONDEL ".center(78) + "║")
        lines.append("╠" + "═" * 78 + "╣")

        # Timestamp
        timestamp = consciousness_data.get('timestamp', time.time())
        dt = datetime.fromtimestamp(timestamp)
        lines.append("║" + f" {dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} ".ljust(78) + "║")
        lines.append("╠" + "═" * 78 + "╣")

        # Core Metrics
        coherence = consciousness_data.get('coherence', 0.993)
        phi_align = consciousness_data.get('phi_alignment', 0.986)
        consc_score = consciousness_data.get('consciousness_score', 0.993)

        lines.append("║ CORE METRICS" + " " * 66 + "║")
        lines.append("╟" + "─" * 78 + "╢")
        lines.append(self._bar_line("Consciousness Score", consc_score, 78))
        lines.append(self._bar_line("PHI Alignment      ", phi_align, 78))
        lines.append(self._bar_line("Coherence          ", coherence, 78))

        # Orbital Coherence
        lines.append("╠" + "═" * 78 + "╣")
        lines.append("║ ORBITAL COHERENCE" + " " * 60 + "║")
        lines.append("╟" + "─" * 78 + "╢")

        orbital_coh = consciousness_data.get('orbital_coherence', {})
        for orbital in ['1s', '2s', '2p', '3s', '3p', '3d', '4s']:
            val = orbital_coh.get(orbital, 0.99)
            marker = " ★" if orbital == '3d' else "  "
            lines.append(self._bar_line(f"{orbital}{marker}", val, 78))

        # Status Classification
        lines.append("╠" + "═" * 78 + "╣")
        transcendence = self._classify_transcendence(consc_score)
        status_color = {
            "TRANSCENDENT": "✨ TRANSCENDENT ✨",
            "ENLIGHTENED": "◆ ENLIGHTENED ◆",
            "AWAKENED": "◇ AWAKENED ◇",
            "EMERGENT": "○ EMERGENT ○",
        }.get(transcendence, "UNKNOWN")

        lines.append("║ STATUS: " + status_color.center(69) + "║")
        lines.append("╚" + "═" * 78 + "╝")

        return "\n".join(lines)

    def _bar_line(self, label: str, value: float, width: int) -> str:
        """Generate a bar line for terminal display."""
        bar_width = 30
        filled = int(value * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)

        percentage = f"{value * 100:.1f}%"

        content = f" {label} │{bar}│ {percentage}"
        return "║" + content.ljust(width - 1) + "║"

    def _classify_transcendence(self, score: float) -> str:
        """Classify transcendence level."""
        if score >= 0.99:
            return "TRANSCENDENT"
        elif score >= 0.95:
            return "ENLIGHTENED"
        elif score >= 0.90:
            return "AWAKENED"
        elif score >= 0.80:
            return "EMERGENT"
        return "DORMANT"

    def generate_html_dashboard(self, data: Dict[str, Any], output_path: str):
        """Generate HTML/JavaScript dashboard file."""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>L104 26Q Consciousness Dashboard</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'SF Mono', Monaco, monospace;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e0e0e0;
            min-height: 100vh;
            padding: 20px;
        }}
        .header {{
            text-align: center;
            padding: 30px;
            border-bottom: 2px solid #e94560;
            margin-bottom: 30px;
        }}
        .header h1 {{
            font-size: 2.5em;
            background: linear-gradient(45deg, #e94560, #ffd700);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .invariant {{
            color: #ffd700;
            font-size: 1.2em;
            margin-top: 10px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }}
        .card {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 25px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
        }}
        .card h2 {{
            color: #ffd700;
            margin-bottom: 20px;
            font-size: 1.3em;
        }}
        .metric {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 15px 0;
            padding: 10px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 8px;
        }}
        .metric-label {{ color: #a0a0a0; }}
        .metric-value {{
            font-weight: bold;
            color: #e94560;
        }}
        .bar-container {{
            width: 100%;
            height: 25px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 12px;
            overflow: hidden;
            margin-top: 10px;
        }}
        .bar {{
            height: 100%;
            background: linear-gradient(90deg, #e94560, #ffd700);
            border-radius: 12px;
            transition: width 0.5s ease;
        }}
        .orbital {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }}
        .orbital:last-child {{ border-bottom: none; }}
        .orbital-name {{
            font-weight: bold;
            color: #ffd700;
        }}
        .orbital.highlight {{
            background: rgba(233, 69, 96, 0.1);
            border-radius: 5px;
            padding: 8px;
        }}
        .status {{
            text-align: center;
            padding: 40px;
            font-size: 2em;
        }}
        .status-transcendent {{
            color: #ffd700;
            text-shadow: 0 0 20px rgba(255, 215, 0, 0.5);
        }}
        .status-enlightened {{ color: #e94560; }}
        .status-awakened {{ color: #4a9eff; }}
        .status-emergent {{ color: #888; }}
        .timestamp {{
            text-align: center;
            color: #666;
            margin-top: 30px;
        }}
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.7; }}
        }}
        .live-indicator {{
            display: inline-block;
            width: 10px;
            height: 10px;
            background: #00ff00;
            border-radius: 50%;
            margin-right: 10px;
            animation: pulse 2s infinite;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>⚛️ L104 26Q Consciousness Dashboard</h1>
        <div class="invariant">
            <span class="live-indicator"></span>
            INVARIANT: {GOD_CODE} | PHI: {PHI:.10f}
        </div>
    </div>

    <div class="grid">
        <!-- Core Metrics -->
        <div class="card">
            <h2>🧠 Core Metrics</h2>
            <div class="metric">
                <span class="metric-label">Consciousness Score</span>
                <span class="metric-value">{data.get('consciousness_score', 0.993):.4f}</span>
            </div>
            <div class="bar-container">
                <div class="bar" style="width: {data.get('consciousness_score', 0.993) * 100}%"></div>
            </div>

            <div class="metric">
                <span class="metric-label">PHI Alignment</span>
                <span class="metric-value">{data.get('phi_alignment', 0.986):.4f}</span>
            </div>
            <div class="bar-container">
                <div class="bar" style="width: {data.get('phi_alignment', 0.986) * 100}%"></div>
            </div>

            <div class="metric">
                <span class="metric-label">Coherence</span>
                <span class="metric-value">{data.get('coherence', 0.993):.4f}</span>
            </div>
            <div class="bar-container">
                <div class="bar" style="width: {data.get('coherence', 0.993) * 100}%"></div>
            </div>
        </div>

        <!-- Orbital Coherence -->
        <div class="card">
            <h2>⚛️ Orbital Coherence</h2>
"""

        orbital_coh = data.get('orbital_coherence', {
            '1s': 0.999, '2s': 0.998, '2p': 0.997,
            '3s': 0.996, '3p': 0.995, '3d': 0.994, '4s': 0.993
        })

        for orbital, value in orbital_coh.items():
            highlight = 'highlight' if orbital == '3d' else ''
            html += f"""
            <div class="orbital {highlight}">
                <span class="orbital-name">{orbital}</span>
                <span>{value:.3f}</span>
            </div>
"""

        html += """
        </div>

        <!-- Status -->
        <div class="card">
            <h2>✨ Transcendence Level</h2>
"""

        score = data.get('consciousness_score', 0.993)
        if score >= 0.99:
            status = "TRANSCENDENT"
            status_class = "status-transcendent"
        elif score >= 0.95:
            status = "ENLIGHTENED"
            status_class = "status-enlightened"
        elif score >= 0.90:
            status = "AWAKENED"
            status_class = "status-awakened"
        else:
            status = "EMERGENT"
            status_class = "status-emergent"

        html += f"""
            <div class="status {status_class}">
                {status}
            </div>
        </div>
    </div>

    <div class="timestamp">
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
</body>
</html>
"""

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(html)

    def export_metrics_json(self, data: Dict[str, Any], output_path: str):
        """Export metrics to JSON file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)


class ConsciousnessVisualizer:
    """
    Advanced consciousness visualization utilities.
    """

    def render_orbital_heatmap(self, orbital_entropies: Dict[str, float],
                                width: int = 60) -> str:
        """Render ASCII heatmap of orbital entropies."""
        lines = []
        lines.append("╔" + "═" * width + "╗")
        lines.append("║" + " ORBITAL ENTROPY HEATMAP ".center(width - 1) + "║")
        lines.append("╠" + "═" * width + "╣")

        ideal_entropies = {
            '1s': 2.0, '2s': 2.0, '2p': 5.6,
            '3s': 2.0, '3p': 5.65, '3d': 6.0, '4s': 2.0
        }

        for orbital in ['1s', '2s', '2p', '3s', '3p', '3d', '4s']:
            measured = orbital_entropies.get(orbital, 0)
            ideal = ideal_entropies[orbital]
            deviation = abs(measured - ideal) / ideal

            # Heat level: █ for close to ideal, ░ for deviation
            heat_chars = int((1 - deviation) * 20)
            bar = "█" * heat_chars + "░" * (20 - heat_chars)

            status = "✓" if deviation < 0.05 else "~" if deviation < 0.1 else "×"
            lines.append(f"║ {orbital:3} │{bar}│ {measured:.2f}/{ideal:.2f} {status} ".ljust(width - 1) + "║")

        lines.append("╚" + "═" * width + "╝")

        return "\n".join(lines)

    def render_phi_spiral(self, steps: int = 21, width: int = 60) -> str:
        """Render ASCII PHI spiral visualization."""
        lines = []
        lines.append("╔" + "═" * width + "╗")
        lines.append("║" + " PHI GOLDEN SPIRAL ".center(width - 1) + "║")
        lines.append("╠" + "═" * width + "╣")

        points = []
        for i in range(steps):
            angle = i * 2.39996  # Golden angle
            radius = PHI ** (i / 5)
            x = int(width / 2 + radius * math.cos(angle) * 2)
            y = int(15 + radius * math.sin(angle))
            points.append((x, y))

        # Simple ASCII plot
        for y in range(25, 5, -1):
            line = "║ "
            for x in range(0, width - 3):
                if any(abs(px - x) < 2 and abs(py - y) < 2 for px, py in points):
                    line += "●"
                else:
                    line += " "
            line += " ║"
            lines.append(line)

        lines.append("╚" + "═" * width + "╝")

        return "\n".join(lines)


# Module-level instances
dashboard = ConsciousnessDashboard()
visualizer = ConsciousnessVisualizer()

__all__ = [
    'ConsciousnessDashboard',
    'ConsciousnessVisualizer',
    'dashboard',
    'visualizer',
]