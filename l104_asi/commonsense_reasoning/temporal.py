"""Layer 4: Temporal Reasoning Engine — Before/After, Sequences, Duration."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .constants import PHI, GOD_CODE, VOID_CONSTANT

logger = logging.getLogger(__name__)


@dataclass
class TemporalSequence:
    """An ordered sequence of steps in a process."""
    name: str
    domain: str
    steps: List[str]
    keywords: List[str] = field(default_factory=list)


class TemporalReasoningEngine:
    """Temporal logic for commonsense reasoning.

    Handles before/after, sequences, durations, and process-order
    questions common in ARC science benchmarks.
    """

    def __init__(self):
        self.sequences: List[TemporalSequence] = []
        self._duration_facts: Dict[str, str] = {}
        self._built = False

    def build(self):
        """Build temporal knowledge base."""
        if self._built:
            return
        self._build_process_sequences()
        self._build_duration_facts()
        self._built = True

    def _build_process_sequences(self):
        """Define ordered process sequences from science domains."""
        self.sequences = [
            # Water cycle
            TemporalSequence("water_cycle", "earth_science",
                             ["evaporation", "condensation", "cloud_formation",
                              "precipitation", "collection", "runoff"],
                             ["water", "cycle", "rain", "evaporate", "cloud"]),
            # Rock cycle
            TemporalSequence("rock_cycle", "earth_science",
                             ["weathering", "erosion", "deposition", "compaction",
                              "cementation", "sedimentary_rock"],
                             ["rock", "cycle", "sediment", "weather", "erode"]),
            TemporalSequence("igneous_formation", "earth_science",
                             ["magma_rises", "eruption_or_intrusion", "cooling", "crystallization",
                              "igneous_rock"],
                             ["magma", "lava", "cool", "igneous", "volcano"]),
            TemporalSequence("metamorphic_formation", "earth_science",
                             ["existing_rock", "heat_and_pressure", "mineral_change",
                              "metamorphic_rock"],
                             ["heat", "pressure", "metamorphic"]),
            # Scientific method
            TemporalSequence("scientific_method", "science",
                             ["observation", "question", "hypothesis", "experiment",
                              "data_collection", "analysis", "conclusion"],
                             ["scientific", "method", "experiment", "hypothesis"]),
            # Photosynthesis
            TemporalSequence("photosynthesis", "biology",
                             ["light_absorbed_by_chlorophyll", "water_split",
                              "carbon_dioxide_fixed", "glucose_produced", "oxygen_released"],
                             ["photosynthesis", "chlorophyll", "light", "glucose"]),
            # Cellular respiration
            TemporalSequence("cellular_respiration", "biology",
                             ["glucose_enters_cell", "glycolysis", "krebs_cycle",
                              "electron_transport_chain", "ATP_produced", "CO2_released"],
                             ["respiration", "ATP", "glucose", "energy", "mitochondria"]),
            # Metamorphosis (complete)
            TemporalSequence("complete_metamorphosis", "biology",
                             ["egg", "larva", "pupa", "adult"],
                             ["metamorphosis", "butterfly", "caterpillar", "larva", "pupa"]),
            # Metamorphosis (incomplete)
            TemporalSequence("incomplete_metamorphosis", "biology",
                             ["egg", "nymph", "adult"],
                             ["metamorphosis", "grasshopper", "nymph", "incomplete"]),
            # Plant life cycle
            TemporalSequence("plant_life_cycle", "biology",
                             ["seed", "germination", "seedling", "mature_plant",
                              "flowering", "pollination", "seed_production"],
                             ["plant", "seed", "grow", "flower", "germinate"]),
            # Star life cycle
            TemporalSequence("star_life_cycle", "astronomy",
                             ["nebula", "protostar", "main_sequence_star", "red_giant",
                              "planetary_nebula_or_supernova", "white_dwarf_or_neutron_star"],
                             ["star", "nebula", "red_giant", "supernova", "white_dwarf"]),
            # Food chain energy flow
            TemporalSequence("energy_flow", "biology",
                             ["sun", "producer", "primary_consumer", "secondary_consumer",
                              "tertiary_consumer", "decomposer"],
                             ["food", "chain", "energy", "producer", "consumer"]),
            # Digestion
            TemporalSequence("digestion", "biology",
                             ["mouth_chewing", "esophagus", "stomach_acid", "small_intestine_absorption",
                              "large_intestine_water_absorption", "waste_elimination"],
                             ["digest", "stomach", "intestine", "food", "absorb"]),
            # Earth formation
            TemporalSequence("earth_day_cycle", "earth_science",
                             ["sunrise", "morning", "noon", "afternoon", "sunset",
                              "evening", "night", "midnight"],
                             ["day", "night", "sunrise", "sunset", "noon"]),
            # Seasons (Northern Hemisphere)
            TemporalSequence("seasons_cycle", "earth_science",
                             ["spring", "summer", "autumn", "winter"],
                             ["season", "spring", "summer", "fall", "autumn", "winter"]),
            # Moon phases
            TemporalSequence("moon_phases", "earth_science",
                             ["new_moon", "waxing_crescent", "first_quarter", "waxing_gibbous",
                              "full_moon", "waning_gibbous", "third_quarter", "waning_crescent"],
                             ["moon", "phase", "crescent", "quarter", "full", "waning", "waxing"]),
            # Erosion process
            TemporalSequence("erosion_process", "earth_science",
                             ["weathering_breaks_rock", "loose_material_formed",
                              "transport_by_water_wind_ice", "deposition_in_new_location"],
                             ["erosion", "weather", "transport", "deposit"]),
            # Electricity flow
            TemporalSequence("circuit_operation", "physics",
                             ["energy_source_provides_voltage", "current_flows_through_conductor",
                              "load_converts_energy", "current_returns_to_source"],
                             ["circuit", "current", "battery", "wire", "electricity"]),
        ]

    def _build_duration_facts(self):
        """Build duration comparison facts."""
        self._duration_facts = {
            "earth_rotation": "24 hours (1 day)",
            "earth_revolution": "365.25 days (1 year)",
            "moon_orbit": "27.3 days",
            "moon_phases_cycle": "29.5 days",
            "light_year": "9.461 trillion km",
            "speed_of_light": "3 × 10⁸ m/s",
            "speed_of_sound_air": "343 m/s",
            "human_gestation": "9 months",
            "butterfly_metamorphosis": "2-4 weeks as chrysalis",
            "fossil_formation": "millions of years",
            "rock_cycle": "millions to billions of years",
            "ice_age_cycle": "~100,000 years",
            "continental_drift": "centimeters per year",
        }

    def query_sequence(self, text: str, top_k: int = 3) -> List[Tuple[TemporalSequence, float]]:
        """Find temporal sequences relevant to a question."""
        text_lower = text.lower()
        scored = []
        for seq in self.sequences:
            score = 0.0
            # Keyword matching
            for kw in seq.keywords:
                if kw in text_lower:
                    score += 0.3
            # Step name matching
            for step in seq.steps:
                step_clean = step.lower().replace('_', ' ')
                if step_clean in text_lower:
                    score += 0.4
                # Word-level overlap
                step_words = set(step_clean.split())
                text_words = set(text_lower.split())
                overlap = len(step_words & text_words)
                if overlap > 0:
                    score += overlap * 0.15
            if score > 0.1:
                scored.append((seq, round(score, 4)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def resolve_order(self, step_a: str, step_b: str) -> Optional[str]:
        """Determine if step_a comes before or after step_b in any known sequence."""
        a_lower = step_a.lower().replace(' ', '_')
        b_lower = step_b.lower().replace(' ', '_')
        for seq in self.sequences:
            steps_lower = [s.lower() for s in seq.steps]
            idx_a = None
            idx_b = None
            for i, s in enumerate(steps_lower):
                if a_lower in s or s in a_lower:
                    idx_a = i
                if b_lower in s or s in b_lower:
                    idx_b = i
            if idx_a is not None and idx_b is not None:
                if idx_a < idx_b:
                    return "before"
                elif idx_a > idx_b:
                    return "after"
                else:
                    return "same"
        return None

    def score_choice_temporal(self, question: str, choice: str) -> float:
        """Score a choice based on temporal reasoning.

        Handles questions like:
        - 'What happens first in...'
        - 'What is the correct order...'
        - 'What comes after...'
        - 'Which step is before...'
        """
        q_lower = question.lower()
        c_lower = choice.lower()
        score = 0.0

        # Find relevant sequences
        seq_matches = self.query_sequence(q_lower, top_k=3)
        if not seq_matches:
            return 0.0

        # Detect temporal question type
        is_first = any(w in q_lower for w in ['first', 'begin', 'start', 'initial'])
        is_last = any(w in q_lower for w in ['last', 'final', 'end', 'result'])
        is_before = any(w in q_lower for w in ['before', 'prior', 'precede'])
        is_after = any(w in q_lower for w in ['after', 'next', 'follow', 'then'])
        is_order = any(w in q_lower for w in ['order', 'sequence', 'steps', 'stage'])

        for seq, seq_score in seq_matches:
            steps_lower = [s.lower().replace('_', ' ') for s in seq.steps]

            # Find which step the choice matches
            choice_idx = None
            for i, step in enumerate(steps_lower):
                if c_lower in step or step in c_lower:
                    choice_idx = i
                    break
                # Word overlap
                c_words = set(c_lower.split())
                s_words = set(step.split())
                if len(c_words & s_words) >= max(1, len(c_words) // 2):
                    choice_idx = i
                    break

            if choice_idx is None:
                continue

            n_steps = len(steps_lower)
            # Score based on temporal position
            if is_first and choice_idx == 0:
                score += 0.6 * seq_score
            elif is_first and choice_idx <= 1:
                score += 0.3 * seq_score
            elif is_last and choice_idx == n_steps - 1:
                score += 0.6 * seq_score
            elif is_last and choice_idx >= n_steps - 2:
                score += 0.3 * seq_score

            # Before/after: find what the question references
            if is_before or is_after:
                for j, step in enumerate(steps_lower):
                    if step in q_lower and j != choice_idx:
                        if is_before and choice_idx < j:
                            score += 0.5 * seq_score
                        elif is_after and choice_idx > j:
                            score += 0.5 * seq_score
                        break

            # General sequence relevance
            if is_order and choice_idx is not None:
                score += 0.2 * seq_score

            # Any match to a step in a relevant sequence
            score += 0.1 * seq_score

        return score

    def get_status(self) -> Dict[str, Any]:
        return {
            'sequences': len(self.sequences),
            'duration_facts': len(self._duration_facts),
            'built': self._built,
        }

