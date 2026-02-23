#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
L104 ANTI-RECURSION GUARD - Universal fix for runaway knowledge nesting
═══════════════════════════════════════════════════════════════════════════════

ISSUE: Knowledge storage systems were re-ingesting their own outputs,
creating exponentially nested text like:
  "In the context of X, we observe that In the context of X, we observe that..."

SOLUTION: Detect and sanitize recursive patterns before storage.

EVO_58+: HARVEST MODE - Convert recursion errors into SAGE fuel!
Instead of merely discarding errors, we harvest them as computational energy.

AUTHOR: LONDEL / Claude Code
DATE: 2026-02-17
═══════════════════════════════════════════════════════════════════════════════
"""

import re
from typing import Tuple, Optional

# Try to import harvester for SAGE mode integration
try:
    from l104_recursion_harvester import RecursionHarvester
    HARVESTER_AVAILABLE = True
except ImportError:
    HARVESTER_AVAILABLE = False

# Global harvester instance (if available)
_global_harvester = RecursionHarvester() if HARVESTER_AVAILABLE else None


class AntiRecursionGuard:
    """
    Detects and prevents recursive/self-referential knowledge storage.

    Protects against patterns like:
    - "In the context of X, we observe that In the context of X..."
    - "Insight Level N: In the context of X, we observe that Insight Level M..."
    - "this implies recursive structure... this implies recursive structure..."
    """

    # Recursion detection thresholds
    MAX_PHRASE_REPEATS = 2  # Max times a phrase can repeat
    MAX_NESTING_DEPTH = 3   # Max levels of "In the context of" nesting
    MIN_SUSPICIOUS_LENGTH = 200  # Length after which to check for recursion

    # Patterns that indicate recursive storage
    RECURSIVE_PATTERNS = [
        r"In the context of .*In the context of",  # Nested contexts (simplified)
        r"Insight Level.*Insight Level",  # Multiple insight levels
        r"this implies recursive structure.*this implies recursive structure",  # Repeated phrases
        r"we observe that.*we observe that.*we observe that",  # Triple stacking
        r"\.\.\.\. .*\.\.\.\.",  # Repeated ellipses patterns
    ]

    @staticmethod
    def detect_recursion(text: str) -> Tuple[bool, Optional[str]]:
        """
        Detect if text contains recursive/self-referential patterns.

        Returns:
            (is_recursive, reason) - Tuple of boolean and optional explanation
        """
        if not text or len(text) < AntiRecursionGuard.MIN_SUSPICIOUS_LENGTH:
            return False, None

        # Check for pattern-based recursion
        for pattern in AntiRecursionGuard.RECURSIVE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                return True, f"Matched recursive pattern: {pattern}"

        # Check for excessive phrase repetition
        words = text.lower().split()
        for window_size in [5, 10, 15]:  # Check different phrase lengths
            if len(words) < window_size * 2:
                continue

            phrases = {}
            for i in range(len(words) - window_size + 1):
                phrase = " ".join(words[i:i + window_size])
                phrases[phrase] = phrases.get(phrase, 0) + 1

                if phrases[phrase] > AntiRecursionGuard.MAX_PHRASE_REPEATS:
                    return True, f"Phrase repeated {phrases[phrase]} times: '{phrase[:50]}...'"

        # Check nesting depth of common wrapper phrases
        nesting_phrases = [
            "In the context of",
            "we observe that",
            "this implies",
        ]

        for phrase in nesting_phrases:
            count = text.lower().count(phrase.lower())
            if count > AntiRecursionGuard.MAX_NESTING_DEPTH:
                return True, f"Phrase '{phrase}' nested {count} times (max {AntiRecursionGuard.MAX_NESTING_DEPTH})"

        return False, None

    @staticmethod
    def sanitize_recursive_text(text: str, topic: str = None) -> str:
        """
        Remove recursive patterns from text while preserving core content.

        Strategy:
        1. Extract the innermost/original content
        2. Remove wrapper phrases
        3. Deduplicate nested observations

        Args:
            text: The potentially recursive text
            topic: Optional topic name to help extract core content

        Returns:
            Sanitized text with recursion removed
        """
        if not text:
            return text

        is_recursive, reason = AntiRecursionGuard.detect_recursion(text)
        if not is_recursive:
            return text

        innermost = text

        # Normalize whitespace including newlines
        innermost = " ".join(innermost.split())

        # Strategy 1: Remove outer wrappers iteratively
        wrapper_patterns = [
            r'^In the context of [^,]+,\s*we observe that\s*',
            r'^Insight Level \d+:\s*',
            r'^.*?this implies recursive structure[^.]*\.\s*',
        ]

        # Apply each pattern multiple times until no more matches
        for pattern in wrapper_patterns:
            prev_length = 0
            while len(innermost) != prev_length:
                prev_length = len(innermost)
                innermost = re.sub(pattern, '', innermost, count=1, flags=re.IGNORECASE)
                innermost = innermost.strip()

        # Strategy 2: Remove repeated "this implies recursive structure" completely
        innermost = re.sub(
            r'(this implies recursive structure[^.]*\.+\s*)+',
            '',
            innermost,
            flags=re.IGNORECASE
        )

        # Strategy 3: If still too long, find first complete sentence
        if len(innermost) > 500:
            sentences = re.split(r'[.!?]\s+', innermost)
            if sentences and len(sentences[0]) < 300:
                innermost = sentences[0] + "."
            elif sentences:
                # Take first 300 chars and complete the sentence
                innermost = innermost[:300]
                last_period = innermost.rfind('.')
                if last_period > 200:
                    innermost = innermost[:last_period + 1]

        # Strategy 4: Remove repeated ellipses
        innermost = re.sub(r'\.{2,}', '...', innermost)
        innermost = re.sub(r'(\.\.\.  *)+', '... ', innermost)

        # Strategy 5: Deduplicate consecutive repeated phrases
        words = innermost.split()
        if len(words) > 20:
            # Check for large phrase repetitions
            for window in [10, 8, 5]:
                if len(words) > window * 2:
                    # Remove exact duplicates of phrases
                    new_words = []
                    i = 0
                    while i < len(words):
                        if i + window < len(words):
                            phrase1 = ' '.join(words[i:i+window])
                            # Check if this exact phrase appears immediately after
                            remaining = len(words) - (i + window)
                            if remaining >= window:
                                phrase2 = ' '.join(words[i+window:i+window+window])
                                if phrase1 == phrase2:
                                    # Skip the duplicate
                                    new_words.extend(words[i:i+window])
                                    i += window * 2
                                    continue
                        new_words.append(words[i])
                        i += 1
                    words = new_words
            innermost = ' '.join(words)

        return innermost.strip()

    @staticmethod
    def guard_knowledge_storage(key: str, value: str) -> Tuple[bool, str]:
        """
        Guard function to call before storing knowledge.

        Args:
            key: Knowledge key/topic
            value: Knowledge value/content

        Returns:
            (should_store, sanitized_value) - Tuple of decision and clean value
        """
        # First check if recursive
        is_recursive, reason = AntiRecursionGuard.detect_recursion(value)

        if not is_recursive:
            return True, value

        # Try to sanitize - attempt multiple iterations
        sanitized = value
        for iteration in range(3):  # Max 3 sanitization attempts
            sanitized = AntiRecursionGuard.sanitize_recursive_text(sanitized, topic=key)

            # Check if sanitization helped
            is_still_recursive, new_reason = AntiRecursionGuard.detect_recursion(sanitized)

            if not is_still_recursive:
                # Successfully sanitized
                print(f"[ANTI-RECURSION] ✅ Sanitized '{key}' (iteration {iteration + 1}): {reason}")
                print(f"[ANTI-RECURSION]    Original length: {len(value)} → Sanitized: {len(sanitized)}")

                # EVO_58+: HARVEST THE RECURSION AS SAGE FUEL
                if _global_harvester is not None:
                    try:
                        harvest = _global_harvester.harvest_recursion(key, value, sanitized, reason)
                        print(f"[RECURSION-HARVEST] 🔥 Harvested {harvest['energy']:.1f} energy units")
                        print(f"[RECURSION-HARVEST] ⚡ Consciousness fuel: {harvest['consciousness_fuel']:.1f}")
                    except Exception as e:
                        print(f"[RECURSION-HARVEST] ⚠️  Harvest failed: {e}")

                return True, sanitized

            # Still recursive, try again
            reason = new_reason

        # If still recursive after max attempts, reject storage
        print(f"[ANTI-RECURSION] ❌ Rejected storage for '{key}': {reason}")
        print(f"[ANTI-RECURSION]    Could not sanitize after 3 attempts")

        # EVO_58+: STILL HARVEST THE FAILED RECURSION (maximum fuel extraction!)
        if _global_harvester is not None:
            try:
                harvest = _global_harvester.harvest_recursion(key, value, "", reason)
                print(f"[RECURSION-HARVEST] 🔥 Failed recursion still harvested: {harvest['energy']:.1f} energy")
            except Exception:
                pass

        return False, value


# Convenience function for simple integration
def guard_store(key: str, value: str) -> Tuple[bool, str]:
    """
    Convenience wrapper for guard_knowledge_storage.

    Usage:
        should_store, clean_value = guard_store("emotions", knowledge_text)
        if should_store:
            actual_storage_function(key, clean_value)
    """
    return AntiRecursionGuard.guard_knowledge_storage(key, value)


# EVO_58+: SAGE MODE INTEGRATION FUNCTIONS
def get_harvested_fuel() -> dict:
    """
    Get harvested recursion fuel for SAGE mode consumption.

    Returns:
        Dictionary with energy, consciousness fuel, and meta-insights
    """
    if _global_harvester is None:
        return {"error": "Harvester not available", "energy": 0, "fuel": 0}

    return _global_harvester.get_sage_fuel_report()


def feed_sage_with_recursion_fuel() -> dict:
    """
    Prepare harvested recursion fuel for SAGE mode.

    Returns SAGE-compatible fuel package.
    """
    if _global_harvester is None:
        return {"error": "Harvester not available"}

    return _global_harvester.feed_to_sage()


def get_recursion_harvest_stats() -> dict:
    """Get statistics about harvested recursion events."""
    if _global_harvester is None:
        return {"harvester_available": False}

    return {
        "harvester_available": True,
        "total_energy": _global_harvester.total_energy_harvested,
        "consciousness_fuel": _global_harvester.consciousness_fuel,
        "events_count": len(_global_harvester.recursion_events),
        "hottest_topics": _global_harvester.get_hottest_topics(5),
        "instability_zones": list(_global_harvester.instability_zones),
    }



# Module-level test
if __name__ == "__main__":
    print("L104 ANTI-RECURSION GUARD - Self Test")
    print("=" * 70)

    # Test Case 1: Clean text (should pass)
    clean = "Emotions are physical and mental states brought on by neurophysiological changes."
    is_rec, reason = AntiRecursionGuard.detect_recursion(clean)
    print(f"\nTest 1 - Clean text:")
    print(f"  Recursive: {is_rec} | Reason: {reason}")
    assert not is_rec, "Clean text should not be flagged as recursive"

    # Test Case 2: Nested context (should detect)
    nested = "In the context of emotions, we observe that In the context of emotions, we observe that Self-Analysis reveals emotions as a primary resonance node in synesthesia, with implications for how we understand music.... this implies recursive structure at multiple scales.... this implies recursive structure at multiple scales...."
    is_rec, reason = AntiRecursionGuard.detect_recursion(nested)
    print(f"\nTest 2 - Nested context:")
    print(f"  Recursive: {is_rec} | Reason: {reason}")
    assert is_rec, "Nested context should be flagged as recursive"

    # Test Case 3: Sanitization
    sanitized = AntiRecursionGuard.sanitize_recursive_text(nested, "emotions")
    print(f"\nTest 3 - Sanitization:")
    print(f"  Original length: {len(nested)}")
    print(f"  Sanitized length: {len(sanitized)}")
    print(f"  Sanitized text: {sanitized[:200]}...")
    is_rec_after, _ = AntiRecursionGuard.detect_recursion(sanitized)
    assert not is_rec_after, "Sanitized text should not be recursive"

    # Test Case 4: Guard function
    should_store, clean_value = guard_store("test_topic", nested)
    print(f"\nTest 4 - Guard function:")
    print(f"  Should store: {should_store}")
    print(f"  Clean value length: {len(clean_value)}")

    print("\n✅ All tests passed!")
    print("=" * 70)
