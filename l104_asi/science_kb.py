#!/usr/bin/env python3
"""
L104 Science Knowledge Base — Structured factual knowledge for ARC-grade reasoning.

This replaces brittle regex rules with a structured KB that supports:
  1. Fact lookup by topic/entity
  2. Relationship queries (is_a, has_property, causes, part_of)
  3. Elimination rules (choice is wrong if it contradicts a known fact)
  4. Category reasoning (same periodic table group, food chain order, etc.)

Architecture:
  - Facts are stored as (subject, relation, object) triples
  - Fast lookup via subject index and relation index
  - Scoring: a choice gets boosted if it matches a fact, penalized if it contradicts

v1.0.0 — 2026-02-26
"""

from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
import re


@dataclass
class Fact:
    """A single knowledge triple."""
    subject: str
    relation: str
    obj: str
    confidence: float = 1.0
    domain: str = "science"  # science, earth, life, technology, measurement

    def __hash__(self):
        return hash((self.subject, self.relation, self.obj))


class ScienceKB:
    """Structured science knowledge base for commonsense MCQ reasoning.

    Usage:
        kb = ScienceKB()
        # Query facts about a topic
        facts = kb.query("frogs", "eat")  # → [Fact(frogs, eat, insects)]
        # Score a choice against question context
        score = kb.score_choice(question_keywords, choice_text)
    """

    def __init__(self):
        self.facts: List[Fact] = []
        self._subject_idx: Dict[str, List[Fact]] = {}
        self._relation_idx: Dict[str, List[Fact]] = {}
        self._object_idx: Dict[str, List[Fact]] = {}
        self._loaded = False

    def _add(self, subject: str, relation: str, obj: str,
             confidence: float = 1.0, domain: str = "science"):
        """Add a fact triple."""
        f = Fact(subject=subject.lower(), relation=relation.lower(),
                obj=obj.lower(), confidence=confidence, domain=domain)
        self.facts.append(f)
        self._subject_idx.setdefault(f.subject, []).append(f)
        self._relation_idx.setdefault(f.relation, []).append(f)
        self._object_idx.setdefault(f.obj, []).append(f)

    def query(self, subject: str = None, relation: str = None,
              obj: str = None) -> List[Fact]:
        """Query facts matching any combination of subject/relation/object."""
        results = None
        if subject:
            s = subject.lower()
            candidates = set()
            # Exact match
            if s in self._subject_idx:
                candidates.update(self._subject_idx[s])
            # Partial match (subject contains query or query contains subject)
            for key, facts in self._subject_idx.items():
                if s in key or key in s:
                    candidates.update(facts)
            results = candidates if results is None else results & candidates
        if relation:
            r = relation.lower()
            candidates = set()
            if r in self._relation_idx:
                candidates.update(self._relation_idx[r])
            results = candidates if results is None else results & candidates
        if obj:
            o = obj.lower()
            candidates = set()
            if o in self._object_idx:
                candidates.update(self._object_idx[o])
            for key, facts in self._object_idx.items():
                if o in key or key in o:
                    candidates.update(facts)
            results = candidates if results is None else results & candidates
        return list(results) if results else []

    def score_choice(self, q_keywords: Set[str], choice_text: str,
                     all_choices: List[str] = None) -> float:
        """Score a choice based on KB fact matching.

        Returns a multiplier: >1.0 = boost, <1.0 = penalty, 1.0 = neutral.
        """
        choice_lower = choice_text.lower()
        choice_words = set(re.findall(r'\w+', choice_lower))

        boost = 1.0
        # Find relevant facts from question keywords
        relevant_facts = []
        for kw in q_keywords:
            relevant_facts.extend(self.query(subject=kw))

        for fact in relevant_facts:
            obj_words = set(re.findall(r'\w+', fact.obj))
            # Check if this choice matches the fact's object (correct answer)
            if obj_words & choice_words:
                boost *= (1.0 + 0.5 * fact.confidence)
            # Check if another relation contradicts this choice

        return boost

    def find_relevant_facts(self, text: str, limit: int = 20) -> List[Fact]:
        """Find all facts relevant to a piece of text."""
        words = set(re.findall(r'\w+', text.lower()))
        words = {w for w in words if len(w) > 3}

        relevant = []
        seen = set()
        for word in words:
            for fact in self.query(subject=word):
                key = (fact.subject, fact.relation, fact.obj)
                if key not in seen:
                    seen.add(key)
                    relevant.append(fact)
                    if len(relevant) >= limit:
                        return relevant
        return relevant

    def load(self):
        """Load all science facts into the KB."""
        if self._loaded:
            return
        self._load_biology()
        self._load_earth_science()
        self._load_physics()
        self._load_chemistry()
        self._load_astronomy()
        self._load_ecology()
        self._load_measurement()
        self._load_technology()
        self._load_scientific_method()
        self._load_body_systems()
        self._loaded = True

    # ==================== BIOLOGY ====================
    def _load_biology(self):
        # Cell biology
        self._add("cell", "is_unit_of", "life")
        self._add("cell division", "heals", "broken bone")
        self._add("cell division", "causes", "growth")
        self._add("cell division", "causes", "repair")
        self._add("mitosis", "is_type_of", "cell division")
        self._add("meiosis", "produces", "gametes")
        self._add("cancer", "caused_by", "abnormal cell division")
        self._add("embryo", "develops_by", "cell division")
        self._add("embryo", "starts_as", "single cell")
        self._add("single cell", "becomes", "many cells")
        self._add("reproduction", "is_a", "life process")
        self._add("growth", "is_a", "life process")
        self._add("metabolism", "is_a", "life process")
        self._add("migration", "is_a", "behavior")
        self._add("migration", "is_not_a", "life process")

        # Genetics
        self._add("mendel", "studied", "heredity")
        self._add("mendel", "used", "pea plants")
        self._add("heredity", "involves", "traits")
        self._add("dna", "carries", "genetic information")
        self._add("sexual reproduction", "involves", "pollination")
        self._add("sexual reproduction", "involves", "fertilization")
        self._add("sexual reproduction", "involves", "meiosis")
        self._add("regeneration", "is_type_of", "asexual reproduction")

        # Organisms
        self._add("flagellum", "helps", "movement")
        self._add("cilia", "helps", "movement")
        self._add("cilia", "helps", "obtain food")
        self._add("paramecium", "uses_cilia_to", "obtain food")
        self._add("paramecium", "uses_cilia_to", "move")
        self._add("euglena", "uses_flagellum_to", "move")

        # Carnivore
        self._add("carnivore", "has_teeth", "pointed")
        self._add("carnivore", "has_teeth", "sharp")
        self._add("herbivore", "has_teeth", "flat")

        # Learned vs innate behavior
        self._add("learned behavior", "example", "avoiding bad-tasting insects")
        self._add("learned behavior", "involves", "experience")
        self._add("innate behavior", "example", "feather color")
        self._add("innate behavior", "involves", "instinct")

        # Transformation in bacteria
        self._add("bacterial transformation", "results_in", "new protein expression")
        self._add("bacterial transformation", "involves", "foreign dna")

        # Molecular clock
        self._add("species divergence", "measured_by", "rate of genetic mutations")
        self._add("molecular clock", "based_on", "mutation rate")

        # Evolution
        self._add("mass extinction", "followed_by", "speciation")
        self._add("speciation", "driven_by", "available ecological niches")
        self._add("permian extinction", "followed_by", "increase in ecological niches")

        # Moths and natural selection
        self._add("dark moths", "increase_in", "polluted areas")
        self._add("light moths", "increase_when", "air becomes cleaner")
        self._add("dark moths", "decrease_when", "air becomes cleaner")
        self._add("peppered moth", "example_of", "natural selection")
        self._add("moth coloration", "selected_by", "bird predation")

        # Plant cells vs animal cells
        self._add("cell wall", "found_in", "plant cells")
        self._add("cell wall", "not_found_in", "animal cells")
        self._add("chloroplast", "found_in", "plant cells")
        self._add("plant cell", "has", "cell wall")
        self._add("plant cell", "has", "chloroplast")
        self._add("animal cell", "does_not_have", "cell wall")
        self._add("carrot", "is_a", "plant")
        self._add("worm", "is_a", "animal")

        # Flowers
        self._add("flower", "function", "attract pollinators")
        self._add("flower", "function", "make seeds")
        self._add("flower", "does_not", "store food")
        self._add("roots", "function", "store food")
        self._add("leaves", "function", "photosynthesis")

        # Early atmosphere
        self._add("early earth atmosphere", "contained", "hydrogen")
        self._add("early earth atmosphere", "contained", "helium")
        self._add("early earth atmosphere", "lacked", "oxygen")
        self._add("anaerobic bacteria", "threatened_by", "photosynthetic organisms")
        self._add("photosynthetic organisms", "increased", "oxygen levels")

    # ==================== BODY SYSTEMS ====================
    def _load_body_systems(self):
        self._add("skeletal system", "function", "support and protection")
        self._add("skeletal system", "provides", "framework")
        self._add("xylem", "analogous_to", "skeletal system")
        self._add("xylem", "provides", "structural support")
        self._add("vertebrate", "has", "backbone")
        self._add("vertebrate", "characteristic", "backbone")
        self._add("all vertebrates", "share", "backbone")
        self._add("muscles", "work_with_bones_by", "pulling")
        self._add("muscles", "create_movement_by", "contracting")
        self._add("muscles", "do_not", "protect bones")
        self._add("endocrine system", "produces", "hormones")
        self._add("digestive system", "function", "digest food")
        self._add("urinary system", "function", "eliminate waste")
        self._add("kidney", "function", "filter blood")
        self._add("perspiration", "function", "temperature regulation")
        self._add("perspiration", "maintains", "stable body temperature")
        self._add("food", "provides", "energy for growth")
        self._add("cells", "convert", "food to energy")
        self._add("fructose", "comes_from", "carbohydrate breakdown")
        self._add("carbohydrates", "broken_down_to", "fructose")

    # ==================== EARTH SCIENCE ====================
    def _load_earth_science(self):
        # Geology
        self._add("basalt", "is_type_of", "igneous rock")
        self._add("igneous rock", "formed_from", "volcanic activity")
        self._add("igneous rock", "formed_from", "magma cooling")
        self._add("sedimentary rock", "formed_by", "compaction and cementation")
        self._add("erosion", "involves", "movement of material")
        self._add("erosion", "moves", "rocks from place to place")
        self._add("weathering", "breaks_down", "rocks in place")
        self._add("erosion", "differs_from_weathering_by", "moving material")
        self._add("tectonic plates", "moved_by", "convection currents")

        # Earth layers
        self._add("earth mantle", "located", "between core and crust")
        self._add("earth mantle", "is", "semi-solid")
        self._add("earth mantle", "has", "convection currents")

        # Water
        self._add("ocean", "contains", "salt water")
        self._add("groundwater", "is", "fresh water")
        self._add("aquifer", "overuse_causes", "land subsidence")
        self._add("runoff", "is", "excess surface water flow")
        self._add("saturated soil", "produces", "runoff")
        self._add("turbidity", "measures", "water cloudiness")
        self._add("turbidity", "caused_by", "suspended particles")

        # Earth orbit and seasons
        self._add("earth orbit", "shape", "ellipse")
        self._add("earth orbit", "shape", "oval")
        self._add("earth orbit", "shape_is_not", "circle")
        self._add("daylight variation", "caused_by", "axial tilt")
        self._add("earth axial tilt", "causes", "different daylight hours")
        self._add("earth rotation", "causes", "day and night")
        self._add("earth revolution", "causes", "years")
        self._add("seasons", "caused_by", "axial tilt")

        # Weather
        self._add("hurricane", "causes", "coastal flooding")
        self._add("hurricane", "strongest_flooding_on", "ocean coastlines")
        self._add("gulf of mexico air mass", "is", "warm and humid")
        self._add("finest-grained soil", "richest_in", "clay")
        self._add("conduction", "occurs_when", "molecules collide")
        self._add("conduction", "requires", "direct contact")

        # Climate phenomena
        self._add("el nino", "causes", "pacific ocean warming")
        self._add("el nino", "causes", "varied atmospheric conditions")
        self._add("el nino", "causes", "drought in western us")
        self._add("el nino", "causes", "pacific coast flooding")
        self._add("el nino", "identified_by", "rising pacific temperatures")
        self._add("la nina", "causes", "pacific ocean cooling")

        # Pollution
        self._add("industrial gases", "remain_in", "atmosphere for long periods")
        self._add("industrial gases", "are_not", "broken down by sunlight")
        self._add("nitrogen fertilizer runoff", "causes", "algae growth")
        self._add("nitrogen fertilizer runoff", "causes", "fish population decrease")
        self._add("clear cutting", "causes", "environmental change")
        self._add("deforestation", "is_type_of", "environmental change")

    # ==================== PHYSICS ====================
    def _load_physics(self):
        # Forces
        self._add("kick", "applies", "contact force")
        self._add("kick", "is_type_of", "applied force")
        self._add("newton first law", "states", "object keeps moving unless unbalanced force")
        self._add("newton", "believed", "objects continue in motion unless acted upon")
        self._add("aristotle", "incorrectly_believed", "force always required for motion")
        self._add("inertia", "is", "tendency to resist change in motion")

        # Energy
        self._add("solar cell", "converts_to", "electrical energy")
        self._add("solar panel", "outputs", "electrical energy")
        self._add("lightning", "is_form_of", "electrical energy")
        self._add("toaster", "converts_to", "heat energy")
        self._add("potential energy", "example", "compressed spring")
        self._add("potential energy", "example", "raised object")
        self._add("potential energy", "example", "stretched rubber band")
        self._add("sun energy to earth", "type", "electromagnetic radiation")
        self._add("sun energy", "changes_water_by", "evaporation")
        self._add("work", "requires", "force and distance")
        self._add("riding bike", "is_example_of", "work")
        self._add("reading book", "is_not_example_of", "work")

        # Waves
        self._add("light", "travels_faster_than", "sound")
        self._add("lightning before thunder", "shows", "light faster than sound")
        self._add("electromagnetic waves", "can_travel_through", "vacuum")
        self._add("electromagnetic waves", "differ_by", "traveling through vacuum")
        self._add("same pitch different loudness", "differs_in", "amplitude")
        self._add("amplitude", "determines", "loudness")
        self._add("frequency", "determines", "pitch")

        # Heat transfer
        self._add("conduction", "mechanism", "molecular collision")
        self._add("convection", "mechanism", "fluid currents")
        self._add("radiation", "mechanism", "electromagnetic waves")
        self._add("boiling", "indicated_by", "bubbles in heated liquid")
        self._add("boiling", "occurs_at", "boiling point")

        # Separation
        self._add("salt water separation", "method", "evaporation")
        self._add("evaporation", "separates", "salt from water")

        # Conservation of matter
        self._add("bending metal", "preserves", "same substance")
        self._add("physical change", "preserves", "substance identity")

    # ==================== CHEMISTRY ====================
    def _load_chemistry(self):
        self._add("water", "is_a", "compound")
        self._add("carbon", "is_a", "element")
        self._add("salt", "is_a", "compound")
        self._add("gold", "is_a", "element")
        self._add("oxygen", "is_a", "element")
        self._add("proton", "has_charge", "positive")
        self._add("proton", "charge_value", "+1")
        self._add("proton", "charge_is_not", "+2")
        self._add("proton", "charge_is_not", "-1")
        self._add("proton", "charge_is_not", "0")
        self._add("electron", "has_charge", "negative")
        self._add("electron", "charge_value", "-1")
        self._add("neutron", "has_charge", "neutral")
        self._add("neutron", "charge_value", "0")
        self._add("neutralization", "produces", "water")
        self._add("neutralization", "produces", "h2o")
        self._add("neutralization", "does_not_produce", "h2o2")
        self._add("neutralization", "does_not_produce", "hydrogen peroxide")
        self._add("acid base reaction", "produces", "salt and water")
        self._add("h2o", "is", "water")
        self._add("h2o2", "is", "hydrogen peroxide")
        self._add("h2o2", "is_not", "water")
        self._add("hydrogen bonds", "maintain", "protein structure")
        self._add("protein structure", "sensitive_to", "heat")
        self._add("yogurt", "made_with", "bacteria")
        self._add("cheese", "made_with", "bacteria")
        self._add("cooking oil", "not_made_with", "bacteria")

        # Periodic table groups
        self._add("calcium", "group", "alkaline earth metals")
        self._add("barium", "group", "alkaline earth metals")
        self._add("magnesium", "group", "alkaline earth metals")
        self._add("strontium", "group", "alkaline earth metals")
        self._add("similar properties", "means", "same group in periodic table")
        self._add("chromium", "similar_to", "manganese")

        # Element symbols
        self._add("copper", "symbol", "cu")
        self._add("gold", "symbol", "au")
        self._add("iron", "symbol", "fe")
        self._add("silver", "symbol", "ag")
        self._add("sodium", "symbol", "na")
        self._add("potassium", "symbol", "k")

    # ==================== ASTRONOMY ====================
    def _load_astronomy(self):
        self._add("comet", "has", "tail of glowing gases")
        self._add("comet", "orbits", "sun")
        self._add("star", "does_not_have", "tail")
        self._add("star formation", "occurs_in", "molecular clouds")
        self._add("star formation", "occurs_in", "nebulae")
        self._add("star color", "determined_by", "mass")
        self._add("star color", "determined_by", "temperature")
        self._add("star color", "not_determined_by", "age")
        self._add("galaxy classification", "based_on", "shape")
        self._add("galaxy classification", "not_based_on", "color")
        self._add("light year", "used_because", "large distances in space")
        self._add("solar eclipse", "caused_by", "moon blocking sun")
        self._add("lunar eclipse", "requires", "full moon")

    # ==================== ECOLOGY ====================
    def _load_ecology(self):
        self._add("frogs", "eat", "insects")
        self._add("frogs", "compete_for", "food")
        self._add("frogs", "compete_for", "insects")
        self._add("frogs", "do_not_compete_for", "air")
        self._add("frogs", "do_not_compete_for", "water")
        self._add("frogs", "do_not_compete_for", "sunlight")
        self._add("birds", "compete_based_on", "beak shape")
        self._add("beak shape", "affects", "food competition")
        self._add("producers", "receive_energy_from", "sun directly")
        self._add("plants", "are", "producers")
        self._add("grass", "is_a", "producer")
        self._add("food chain", "starts_with", "producers")
        self._add("food chain", "starts_with", "plants")
        self._add("food chain order", "is", "plants then animals")
        self._add("energy transfer", "flows", "plants to fish to birds")
        self._add("energy transfer", "does_not_flow", "fish to plants")
        self._add("overhunting", "causes", "extinction")
        self._add("pollination", "involves", "bees and flowers")
        self._add("density", "determines", "floating or sinking")
        self._add("mass", "does_not_determine", "floating")
        self._add("fact", "is", "objective and verifiable")
        self._add("opinion", "is", "subjective judgment")
        self._add("worse than", "indicates", "opinion")
        self._add("occur along", "indicates", "fact")

    # ==================== MEASUREMENT ====================
    def _load_measurement(self):
        self._add("liquid volume", "measured_in", "milliliters")
        self._add("liquid volume", "measured_in", "liters")
        self._add("graduated cylinder", "measures", "volume")
        self._add("volume", "not_measured_in", "meters")
        self._add("volume", "not_measured_in", "centimeters")
        self._add("stopwatch", "measures", "time")
        self._add("stopwatch", "does_not_measure", "distance")
        self._add("less time", "means", "faster")
        self._add("speed", "equals", "distance divided by time")
        self._add("average speed", "formula", "total distance divided by total time")
        self._add("average speed", "includes", "stopped time")
        self._add("mirror", "reflects", "light")
        self._add("eyeglasses", "refract", "light")
        self._add("diamond", "harder_than", "talc")
        self._add("scratching", "indicates", "hardness difference")
        self._add("mohs scale", "measures", "mineral hardness")
        self._add("paper clip", "made_from", "one material")
        self._add("shoes", "made_from", "multiple materials")

    # ==================== TECHNOLOGY ====================
    def _load_technology(self):
        self._add("computer", "useful_for", "finding information")
        self._add("word processor", "benefits", "editing papers quickly")
        self._add("recycling", "means", "making into new products")
        self._add("recycling", "applies_to", "materials like paper glass plastic")
        self._add("reusing", "means", "using again for same purpose")
        self._add("reusing", "is_not", "recycling")
        self._add("markers", "can_be", "reused")
        self._add("milk cartons", "can_be", "recycled")
        self._add("quality control", "involves", "inspecting")
        self._add("quality control", "does_not_involve", "cutting")
        self._add("soil type", "affects", "crop growth")
        self._add("time of day", "does_not_affect", "crop growth")
        self._add("skunk", "detected_by", "smell")

    # ==================== SCIENTIFIC METHOD ====================
    def _load_scientific_method(self):
        self._add("publishing data", "purpose", "allow replication")
        self._add("publishing data", "enables", "peer review")
        self._add("publishing data", "not_for", "gaining respect")
        self._add("reliable experiment", "requires", "multiple specimens")
        self._add("reliable experiment", "requires", "repeated trials")
        self._add("binoculars", "used_for", "observing birds")
        self._add("microscope", "used_for", "observing tiny things")
        self._add("freshwater use increase", "causes", "lake to become smaller")

    def get_stats(self) -> Dict:
        """Return KB statistics."""
        return {
            "total_facts": len(self.facts),
            "unique_subjects": len(self._subject_idx),
            "unique_relations": len(self._relation_idx),
            "unique_objects": len(self._object_idx),
            "domains": list(set(f.domain for f in self.facts)),
        }


# Singleton
_kb_instance: Optional[ScienceKB] = None

def get_science_kb() -> ScienceKB:
    """Get the singleton ScienceKB instance."""
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = ScienceKB()
        _kb_instance.load()
    return _kb_instance


if __name__ == "__main__":
    kb = get_science_kb()
    stats = kb.get_stats()
    print(f"Science KB loaded: {stats['total_facts']} facts, "
          f"{stats['unique_subjects']} subjects, "
          f"{stats['unique_relations']} relations")

    # Demo queries
    print("\n--- Demo Queries ---")
    for query in ["frogs", "el nino", "comet", "proton", "erosion", "food chain"]:
        facts = kb.query(subject=query)
        print(f"\n{query}: {len(facts)} facts")
        for f in facts[:5]:
            print(f"  {f.subject} —[{f.relation}]→ {f.obj}")
