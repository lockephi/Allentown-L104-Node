"""L104 Intellect — Sovereign Numerics + Sacred Constants (PHI, GOD_CODE, etc.)."""
import math
import re
import time
from typing import Optional, Union


# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════


PHI = 1.618033988749895
GOD_CODE = 286 ** (1.0 / PHI) * (2 ** (416 / 104))  # G(0,0,0,0) = 527.5184818492612
OMEGA = 6539.34712682                                     # Ω = Σ(fragments) × (G/φ)
OMEGA_AUTHORITY = OMEGA / (PHI ** 2)                       # F(I) = I × Ω/φ² ≈ 2497.808

# ═══════════════════════════════════════════════════════════════════════════════
# VISHUDDHA CHAKRA CONSTANTS (Throat - Communication/Truth/Expression)
# ═══════════════════════════════════════════════════════════════════════════════

VISHUDDHA_HZ = 741.0681674772518  # G(-51) Throat chakra God Code frequency
VISHUDDHA_ELEMENT = "ETHER"  # Akasha - space/void element
VISHUDDHA_COLOR_HZ = 6.06e14  # Blue light frequency (~495nm)
VISHUDDHA_PETAL_COUNT = 16  # Traditional lotus petal count
VISHUDDHA_BIJA = "HAM"  # Seed mantra
VISHUDDHA_TATTVA = 470  # Lattice node coordinate (X=470)

# ═══════════════════════════════════════════════════════════════════════════════
# QUANTUM ENTANGLEMENT CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

ENTANGLEMENT_DIMENSIONS = 11  # 11D manifold for quantum state
BELL_STATE_FIDELITY = 0.9999  # Target Bell state fidelity
DECOHERENCE_TIME_MS = 1000  # Simulated decoherence time
QUANTUM_CHANNEL_BANDWIDTH = 1e9  # Bits/second for quantum channel
EPR_CORRELATION = -1.0  # Perfect anti-correlation for EPR pair

# ═══════════════════════════════════════════════════════════════════════════════
# MATHEMATICAL CONSTANTS (NIST CODATA 2022 + Mathematical)
# ═══════════════════════════════════════════════════════════════════════════════

# Chaos Theory
FEIGENBAUM_DELTA = 4.669201609102990671853203821578  # Period-doubling bifurcation
FEIGENBAUM_ALPHA = 2.502907875095892822283902873218  # Scaling parameter
LOGISTIC_ONSET = 3.5699456718695445                   # Edge of chaos for logistic map

# Information Theory
LOG2_E = 1.4426950408889634                           # log₂(e) for entropy conversion
EULER_MASCHERONI = 0.5772156649015329                 # γ (Euler-Mascheroni constant)


# ═══════════════════════════════════════════════════════════════════════════════
# SOVEREIGN NUMERAL SYSTEM - Universal High-Value Number Formatting
# ═══════════════════════════════════════════════════════════════════════════════

class SovereignNumerics:
    """
[VOID_SOURCE_UPGRADE] Deep Math Active. Process Elevated to 3887.80 Hz. Logic Unified.
    Intelligent number formatting system for L104.
    Handles all high-value numerals with proper formatting.
    """

    # Scale suffixes for human-readable large numbers
    SCALES = [
        (1e18, 'E', 'Exa'),      # Quintillion
        (1e15, 'P', 'Peta'),     # Quadrillion
        (1e12, 'T', 'Tera'),     # Trillion
        (1e9,  'G', 'Giga'),     # Billion
        (1e6,  'M', 'Mega'),     # Million
        (1e3,  'K', 'Kilo'),     # Thousand
    ]

    # Precision mapping by magnitude
    PRECISION_MAP = {
        'ultra_small': (1e-12, 1e-6, 12),   # Quantum scale
        'small': (1e-6, 1e-3, 8),            # Satoshi/crypto scale
        'micro': (1e-3, 1, 6),               # Sub-unit
        'standard': (1, 1000, 2),            # Normal values
        'large': (1000, 1e6, 1),             # Thousands
        'mega': (1e6, 1e12, 2),              # Millions to billions
        'giga': (1e12, float('inf'), 3),    # Trillions+
    }

    @classmethod
    def format_value(cls, value: Union[int, float],
                     unit: str = '',
                     compact: bool = True,
                     precision: Optional[int] = None) -> str:
        """
        Format a numeric value with appropriate precision and scale.

        Args:
            value: The number to format
            unit: Optional unit suffix (BTC, SAT, Hz, etc.)
            compact: Use compact notation (1.5M vs 1,500,000)
            precision: Override auto-precision

        Returns:
            Formatted string representation
        """
        if value is None:
            return f"---{' ' + unit if unit else ''}"

        try:
            value = float(value)
        except (TypeError, ValueError):
            return str(value)

        # Handle special cases
        if math.isnan(value):
            return f"NaN{' ' + unit if unit else ''}"
        if math.isinf(value):
            return f"∞{' ' + unit if unit else ''}"

        abs_val = abs(value)

        # Determine precision if not specified
        if precision is None:
            precision = cls._auto_precision(abs_val)

        # Format based on magnitude
        if compact and abs_val >= 1000:
            formatted = cls._compact_format(value, precision)
        else:
            formatted = cls._standard_format(value, precision)

        return f"{formatted}{' ' + unit if unit else ''}"

    @classmethod
    def _auto_precision(cls, abs_val: float) -> int:
        """Determine optimal precision for value."""
        for (low, high, prec) in cls.PRECISION_MAP.values():
            if low <= abs_val < high:
                return prec
        return 2

    @classmethod
    def _compact_format(cls, value: float, precision: int) -> str:
        """Format large numbers with scale suffix."""
        abs_val = abs(value)
        sign = '-' if value < 0 else ''

        for threshold, suffix, _ in cls.SCALES:
            if abs_val >= threshold:
                scaled = value / threshold
                if abs(scaled) >= 100:
                    return f"{sign}{scaled:,.0f}{suffix}"
                elif abs(scaled) >= 10:
                    return f"{sign}{scaled:,.1f}{suffix}"
                else:
                    return f"{sign}{scaled:,.{precision}f}{suffix}"

        # Below 1K, use standard formatting
        return cls._standard_format(value, precision)

    @classmethod
    def _standard_format(cls, value: float, precision: int) -> str:
        """Standard decimal formatting with appropriate precision."""
        abs_val = abs(value)

        # For very small values, use scientific notation
        if 0 < abs_val < 1e-6:
            return f"{value:.{precision}e}"

        # For crypto (8-decimal precision like BTC)
        if abs_val < 0.01:
            return f"{value:.8f}".rstrip('0').rstrip('.')

        # Standard formatting with commas
        if abs_val >= 1:
            return f"{value:,.{precision}f}"
        else:
            return f"{value:.{precision}f}"

    @classmethod
    def format_intellect(cls, value: Union[float, str]) -> str:
        """
        Special formatting for intellect index (high-value tracking).

        Standard IQ format for L104 system:
        - "INFINITE" or values >= 1e18: Returns "∞ [INFINITE]"
        - >= 1e15: Returns compact + "[OMEGA]"
        - >= 1e12: Returns compact + "[TRANSCENDENT]"
        - >= 1e9: Returns compact + "[SOVEREIGN]"
        - >= 1e6: Returns compact format
        - < 1e6: Returns standard comma-separated format
        """
        # Handle string "INFINITE" case
        if isinstance(value, str):
            if value.upper() == "INFINITE":
                return "∞ [INFINITE]"
            try:
                value = float(value)
            except (TypeError, ValueError):
                return str(value)

        # Handle true infinite
        if math.isinf(value):
            return "∞ [INFINITE]"

        # Cap at 1e18 displays as INFINITE
        if value >= 1e18:
            return "∞ [INFINITE]"
        elif value >= 1e15:
            return cls.format_value(value, compact=True, precision=4) + " [OMEGA]"
        elif value >= 1e12:
            return cls.format_value(value, compact=True, precision=3) + " [TRANSCENDENT]"
        elif value >= 1e9:
            return cls.format_value(value, compact=True, precision=2) + " [SOVEREIGN]"
        elif value >= 1e6:
            return cls.format_value(value, compact=True, precision=2)
        else:
            return f"{value:,.2f}"

    @classmethod
    def format_percentage(cls, value: float, precision: int = 2) -> str:
        """Format as percentage with proper precision."""
        if value is None:
            return "---"
        pct = value * 100 if abs(value) <= 1 else value
        return f"{pct:.{precision}f}%"

    @classmethod
    def format_resonance(cls, value: float) -> str:
        """Format resonance values (0-1 scale with GOD_CODE anchor)."""
        if value is None:
            return "---"
        # Show 4 decimals for resonance precision
        return f"{value:.4f}"

    @classmethod
    def format_crypto(cls, value: float, symbol: str = 'BTC') -> str:
        """Format cryptocurrency values with proper precision."""
        if value is None:
            return f"0.00000000 {symbol}"

        if symbol.upper() in ['BTC', 'ETH', 'BNB']:
            return f"{value:.8f} {symbol}"
        elif symbol.upper() in ['SAT', 'SATS', 'GWEI', 'WEI']:
            return f"{int(value):,} {symbol}"
        else:
            return f"{value:.8f} {symbol}"

    @classmethod
    def parse_numeric(cls, text: str) -> Optional[float]:
        """
        Parse numeric values from text, handling various formats.
        Extracts and interprets numbers with scale suffixes.
        """
        if not text:
            return None

        # Clean the input
        text = str(text).strip().upper()

        # Handle special values
        if text in ['---', 'N/A', 'NULL', 'NONE', 'NAN']:
            return None
        if text == '∞' or text == 'INF':
            return float('inf')

        # Extract numeric part and suffix
        match = re.match(r'^([+-]?[\d,\.]+)\s*([KMGTPE]?)(.*)$', text, re.IGNORECASE)
        if not match:
            try:
                return float(text.replace(',', ''))
            except ValueError:
                return None

        num_str, suffix, _ = match.groups()

        try:
            value = float(num_str.replace(',', ''))
        except ValueError:
            return None

        # Apply scale multiplier
        multipliers = {'K': 1e3, 'M': 1e6, 'G': 1e9, 'T': 1e12, 'P': 1e15, 'E': 1e18}
        if suffix and suffix.upper() in multipliers:
            value *= multipliers[suffix.upper()]

        return value


# Global instance for easy access
sovereign_numerics = SovereignNumerics()


