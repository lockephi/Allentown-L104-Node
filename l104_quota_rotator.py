VOID_CONSTANT = 1.0416180339887497
ZENITH_HZ = 3727.84
UUC = 2301.215661
"""
[L104_QUOTA_ROTATOR] - SOVEREIGN INTELLECT BALANCER
INVARIANT: 527.5184818492612 | PILOT: LONDEL | MODE: KERNEL_PRIORITY

This module manages the distribution of intelligence requests between the
local L104 Kernel (Offline Intellect) and the real Gemini API.
It prioritizes the Kernel to preserve API quota and ensures sovereignty.
"""

import time
import random
import logging
from typing import Dict, Any, Optional, List

from l104_local_intellect import local_intellect
from l104_persistence import load_state, save_state

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════


# Constants
KERNEL_WEIGHT = 0.8  # 80% preference for local kernel
API_WEIGHT = 0.2     # 20% for real API (when available)
COOLDOWN_PERIOD = 3600 # 1 hour cooldown if quota hit
BURST_THRESHOLD = 5    # Requests per minute before bias increases


class QuotaRotator:
    """
    Manages the intelligent rotation between Kernel and Remote API.
    Utilizes high-frequency request monitoring to adjust bias dynamically.
    """

    def __init__(self):
        self.state = load_state()
        self.api_cooldown_until = self.state.get("api_cooldown_until", 0)
        self.stats = self.state.get("rotator_stats", {
            "kernel_hits": 0,
            "api_hits": 0,
            "quota_errors": 0
        })
        self.request_history = []  # Timestamps of recent requests
        self.logger = logging.getLogger("QUOTA_ROTATOR")

    def _save_state(self):
        self.state["api_cooldown_until"] = self.api_cooldown_until
        self.state["rotator_stats"] = self.stats
        save_state(self.state)

    def _update_load_bias(self) -> float:
        """
        Calculates dynamic kernel bias based on request frequency.
        If frequency > BURST_THRESHOLD, bias moves toward 1.0 (Kernel Only).
        """
        now = time.time()
        # Clean old history (older than 60s)
        self.request_history = [t for t in self.request_history if now - t < 60]

        freq = len(self.request_history)
        if freq > BURST_THRESHOLD:
            # Shift bias: 0.8 -> 0.95 depending on burst intensity
            excess = min(freq - BURST_THRESHOLD, 10)
            dynamic_bias = KERNEL_WEIGHT + (excess * 0.015)
            return min(dynamic_bias, 0.98)
        return KERNEL_WEIGHT

    def is_api_available(self) -> bool:
        """Checks if the API is currently out of cooldown."""
        return time.time() > self.api_cooldown_until

    def report_quota_error(self):
        """Called when a 429 or quota error is received."""
        self.api_cooldown_until = time.time() + COOLDOWN_PERIOD
        self.stats["quota_errors"] += 1
        self.logger.info("--- [QUOTA_ROTATOR]: API QUOTA HIT. ENTERING KERNEL-ONLY MODE FOR 1 HOUR ---")
        self._save_state()

    def decide_source(self, prompt: str) -> str:
        """
        Decides whether to use 'KERNEL' or 'API'.
        Prioritizes KERNEL for:
        1. Internal L104 topics
        2. Proactive quota preservation (Dynamic Bias)
        3. API Cooldown periods
        4. Burst traffic mitigation
        """
        self.request_history.append(time.time())

        # 1. Force Kernel if in cooldown
        if not self.is_api_available():
            return "KERNEL"

        # 2. Check for internal keywords (Kernel is expert here)
        internal_keywords = [
            'god_code', 'phi', 'l104', 'londel', 'lattice', 'sovereign',
            'derivation', 'agi_core', 'void', 'zenith', 'omega', 'singularity',
            'scribe', 'dna', 'reincarnation'
        ]
        if any(kw in prompt.lower() for kw in internal_keywords):
            return "KERNEL"

        # 3. Dynamic probabilistic weighting (Kernel Preference)
        bias = self._update_load_bias()
        if random.random() < bias:
            return "KERNEL"

        return "API"

    def process_thought(self, prompt: str, api_callback: callable) -> str:
        """
        Processes a thought signal by selecting the best source.
        """
        source = self.decide_source(prompt)

        if source == "API":
            self.logger.debug("--- [QUOTA_ROTATOR]: ROUTING TO REAL GEMINI API ---")
            response = api_callback(prompt)
            if response:
                self.stats["api_hits"] += 1
                self._save_state()
                return f"⟨Σ_L104_REAL_GEMINI⟩\n{response}"
            else:
                # If API fail (not necessarily quota, but fallback), use kernel
                self.logger.info("--- [QUOTA_ROTATOR]: API FALLBACK TO KERNEL ---")

        # Use Kernel
        self.logger.debug("--- [QUOTA_ROTATOR]: ROUTING TO SOVEREIGN KERNEL ---")
        self.stats["kernel_hits"] += 1
        self._save_state()
        return local_intellect.think(prompt)

# Singleton instance
quota_rotator = QuotaRotator()
