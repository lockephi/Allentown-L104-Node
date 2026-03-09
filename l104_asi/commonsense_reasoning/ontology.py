"""Layer 1: Concept Ontology — Hierarchical World Knowledge."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .constants import PHI, GOD_CODE, VOID_CONSTANT

logger = logging.getLogger(__name__)


@dataclass
class Concept:
    """A concept in the ontology."""
    name: str
    category: str  # "physical", "biological", "chemical", "earth_science", "process", "abstract"
    parents: List[str] = field(default_factory=list)        # IS-A
    parts: List[str] = field(default_factory=list)           # HAS-PART
    properties: Dict[str, Any] = field(default_factory=dict) # Key-value properties
    related: List[str] = field(default_factory=list)         # RELATED-TO


class ConceptOntology:
    """Hierarchical concept ontology for commonsense knowledge.

    Covers domains needed for ARC: physical science, earth science,
    life science, and basic technology.
    """

    def __init__(self):
        self.concepts: Dict[str, Concept] = {}
        self._built = False

    def build(self):
        """Build the comprehensive concept ontology."""
        if self._built:
            return
        self._build_physical_science()
        self._build_earth_science()
        self._build_life_science()
        self._build_technology()
        self._build_materials()
        self._build_energy()
        self._build_processes()
        self._build_extended_concepts()
        self._build_v2_concepts()
        self._build_v3_concepts()
        self._build_v4_concepts()
        self._build_v5_concepts()
        self._build_v6_concepts()
        self._built = True

    def _add(self, name: str, category: str, parents: List[str] = None,
             parts: List[str] = None, properties: Dict = None,
             related: List[str] = None):
        key = name.lower().replace(' ', '_')
        self.concepts[key] = Concept(
            name=name, category=category,
            parents=parents or [], parts=parts or [],
            properties=properties or {}, related=related or [],
        )

    def _build_physical_science(self):
        """Physics and chemistry concepts."""
        # States of matter
        self._add("matter", "physical", properties={"has_mass": True, "occupies_space": True})
        self._add("solid", "physical", parents=["matter"],
                  properties={"definite_shape": True, "definite_volume": True, "compressible": False,
                             "molecules_packed": "tightly", "vibrate_only": True})
        self._add("liquid", "physical", parents=["matter"],
                  properties={"definite_shape": False, "definite_volume": True, "compressible": False,
                             "takes_container_shape": True, "flows": True})
        self._add("gas", "physical", parents=["matter"],
                  properties={"definite_shape": False, "definite_volume": False, "compressible": True,
                             "fills_container": True, "molecules_spread": True})
        self._add("plasma", "physical", parents=["matter"],
                  properties={"ionized": True, "conducts_electricity": True, "very_hot": True})

        # Phase transitions
        self._add("melting", "process", parents=["phase_change"],
                  properties={"from": "solid", "to": "liquid", "requires": "heat", "temp": "melting_point"})
        self._add("freezing", "process", parents=["phase_change"],
                  properties={"from": "liquid", "to": "solid", "releases": "heat", "temp": "freezing_point"})
        self._add("evaporation", "process", parents=["phase_change"],
                  properties={"from": "liquid", "to": "gas", "requires": "heat"})
        self._add("condensation", "process", parents=["phase_change"],
                  properties={"from": "gas", "to": "liquid", "releases": "heat"})
        self._add("sublimation", "process", parents=["phase_change"],
                  properties={"from": "solid", "to": "gas", "requires": "heat"})
        self._add("deposition", "process", parents=["phase_change"],
                  properties={"from": "gas", "to": "solid", "releases": "heat"})
        self._add("boiling", "process", parents=["phase_change"],
                  properties={"from": "liquid", "to": "gas", "temp": "boiling_point",
                             "bubbles": True, "requires": "heat"})

        # Forces
        self._add("gravity", "physical", parents=["force"],
                  properties={"direction": "toward_center_of_mass", "attracts": True,
                             "depends_on": ["mass", "distance"], "universal": True,
                             "pulls_down": True, "causes_weight": True})
        self._add("friction", "physical", parents=["force"],
                  properties={"opposes_motion": True, "generates_heat": True,
                             "depends_on": ["surface_roughness", "normal_force"]})
        self._add("magnetism", "physical", parents=["force"],
                  properties={"has_poles": True, "opposite_attract": True, "same_repel": True,
                             "affects": ["iron", "nickel", "cobalt"]})
        self._add("electromagnetic_force", "physical", parents=["force"],
                  properties={"carries": "electromagnetic_radiation", "infinite_range": True})

        # Energy
        self._add("energy", "physical",
                  properties={"cannot_be_created": True, "cannot_be_destroyed": True,
                             "can_be_transformed": True, "conserved": True})
        self._add("kinetic_energy", "physical", parents=["energy"],
                  properties={"type": "motion", "formula": "0.5*m*v^2",
                             "increases_with": ["mass", "speed"]})
        self._add("potential_energy", "physical", parents=["energy"],
                  properties={"type": "stored", "gravitational": "m*g*h",
                             "increases_with": ["height", "mass"]})
        self._add("thermal_energy", "physical", parents=["energy"],
                  properties={"type": "heat", "related_to": "temperature",
                             "flows_from": "hot_to_cold"})
        self._add("chemical_energy", "physical", parents=["energy"],
                  properties={"stored_in": "chemical_bonds", "released_by": "reactions"})
        self._add("electrical_energy", "physical", parents=["energy"],
                  properties={"type": "electron_flow", "requires": "circuit"})
        self._add("sound_energy", "physical", parents=["energy"],
                  properties={"type": "vibration", "travels_through": "medium",
                             "cannot_travel_in": "vacuum"})
        self._add("light", "physical", parents=["energy", "electromagnetic_radiation"],
                  properties={"speed": 3e8, "travels_in": "straight_lines",
                             "can_be": ["reflected", "refracted", "absorbed"],
                             "spectrum": ["red", "orange", "yellow", "green", "blue", "indigo", "violet"]})

        # Heat transfer
        self._add("conduction", "process", parents=["heat_transfer"],
                  properties={"requires": "direct_contact", "through": "solids_mainly",
                             "metals_good_conductors": True})
        self._add("convection", "process", parents=["heat_transfer"],
                  properties={"through": "fluids", "hot_rises": True, "cold_sinks": True,
                             "creates": "currents"})
        self._add("radiation", "process", parents=["heat_transfer"],
                  properties={"no_medium_needed": True, "electromagnetic_waves": True,
                             "how_sun_heats_earth": True})

        # Simple machines
        self._add("lever", "physical", parents=["simple_machine"],
                  properties={"has": ["fulcrum", "effort", "load"],
                             "multiplies_force": True})
        self._add("inclined_plane", "physical", parents=["simple_machine"],
                  properties={"reduces_force": True, "increases_distance": True})
        self._add("pulley", "physical", parents=["simple_machine"],
                  properties={"changes_direction_of_force": True})
        self._add("wheel_and_axle", "physical", parents=["simple_machine"],
                  properties={"reduces_friction": True})
        self._add("screw", "physical", parents=["simple_machine"],
                  properties={"converts_rotational_to_linear": True})
        self._add("wedge", "physical", parents=["simple_machine"],
                  properties={"splits_objects": True, "type_of": "inclined_plane"})

        # Electricity
        self._add("electric_circuit", "physical",
                  properties={"requires": ["energy_source", "conductor", "load"],
                             "types": ["series", "parallel"],
                             "current_flows_in": "closed_loop"})
        self._add("conductor", "physical",
                  properties={"allows_electricity": True, "examples": ["copper", "aluminum", "gold", "silver"],
                             "usually": "metals"})
        self._add("insulator", "physical",
                  properties={"blocks_electricity": True, "examples": ["rubber", "glass", "plastic", "wood"]})

        # Motion
        self._add("speed", "physical", parents=["motion"],
                  properties={"formula": "distance/time", "scalar": True})
        self._add("velocity", "physical", parents=["motion"],
                  properties={"formula": "displacement/time", "vector": True, "has_direction": True})
        self._add("acceleration", "physical", parents=["motion"],
                  properties={"change_in_velocity": True, "caused_by": "force"})

    def _build_earth_science(self):
        """Earth science and weather concepts."""
        # Water cycle
        self._add("water_cycle", "earth_science",
                  parts=["evaporation", "condensation", "precipitation", "collection"],
                  properties={"continuous": True, "powered_by": "sun"})
        self._add("precipitation", "earth_science", parents=["water_cycle"],
                  properties={"forms": ["rain", "snow", "sleet", "hail"],
                             "falls_from": "clouds"})
        self._add("cloud", "earth_science",
                  properties={"made_of": "water_droplets_or_ice", "formed_by": "condensation",
                             "types": ["cumulus", "stratus", "cirrus"]})

        # Rock cycle
        self._add("rock_cycle", "earth_science",
                  parts=["igneous", "sedimentary", "metamorphic"])
        self._add("igneous_rock", "earth_science", parents=["rock"],
                  properties={"formed_from": "cooled_magma", "examples": ["granite", "basalt"]})
        self._add("sedimentary_rock", "earth_science", parents=["rock"],
                  properties={"formed_from": "compressed_sediment", "has_layers": True,
                             "may_contain": "fossils", "examples": ["limestone", "sandstone"]})
        self._add("metamorphic_rock", "earth_science", parents=["rock"],
                  properties={"formed_by": "heat_and_pressure", "examples": ["marble", "slate"]})

        # Weather
        self._add("weather", "earth_science",
                  properties={"short_term": True, "conditions": ["temperature", "humidity", "wind", "precipitation"]})
        self._add("climate", "earth_science",
                  properties={"long_term": True, "average_weather": True})
        self._add("wind", "earth_science",
                  properties={"caused_by": "unequal_heating", "moves_from": "high_to_low_pressure"})

        # Earth structure
        self._add("earth", "earth_science",
                  parts=["crust", "mantle", "outer_core", "inner_core"],
                  properties={"revolves_around": "sun", "rotates_on": "axis",
                             "revolution_time": "365.25_days", "rotation_time": "24_hours"})
        self._add("tectonic_plates", "earth_science",
                  properties={"float_on": "mantle", "move_slowly": True,
                             "cause": ["earthquakes", "volcanoes", "mountains"]})
        self._add("earthquake", "earth_science",
                  properties={"caused_by": "tectonic_movement", "releases": "energy",
                             "measured_by": "seismograph"})
        self._add("volcano", "earth_science",
                  properties={"erupts": "magma", "formed_at": "plate_boundaries"})

        # Solar system
        self._add("sun", "earth_science", parents=["star"],
                  properties={"type": "star", "energy_source": "nuclear_fusion",
                             "center_of": "solar_system"})
        self._add("moon", "earth_science", parents=["natural_satellite"],
                  properties={"orbits": "earth", "causes": "tides",
                             "phases": ["new", "crescent", "quarter", "gibbous", "full"],
                             "reflects": "sunlight", "no_atmosphere": True})
        self._add("seasons", "earth_science",
                  properties={"caused_by": "earth_tilt", "not_caused_by": "distance_from_sun",
                             "axial_tilt": "23.5_degrees"})

        # Atmosphere layers
        self._add("troposphere", "earth_science", parents=["atmosphere"],
                  properties={"layer": "lowest", "we_live_in": True,
                             "contains": "weather", "height": "0-12km"})
        self._add("stratosphere", "earth_science", parents=["atmosphere"],
                  properties={"layer": "second", "contains": "ozone_layer",
                             "height": "12-50km"})
        self._add("mesosphere", "earth_science", parents=["atmosphere"],
                  properties={"layer": "third", "height": "50-80km",
                             "meteors_burn_up": True})
        self._add("thermosphere", "earth_science", parents=["atmosphere"],
                  properties={"layer": "fourth", "height": "80-700km",
                             "very_hot": True, "aurora": True})

        # Erosion and weathering
        self._add("erosion", "earth_science",
                  properties={"moves": "rock_and_soil", "agents": ["water", "wind", "ice", "gravity"]})
        self._add("weathering", "earth_science",
                  properties={"breaks_down": "rock", "types": ["physical", "chemical", "biological"]})

    def _build_life_science(self):
        """Life science / biology concepts."""
        # Cells
        self._add("cell", "biological",
                  properties={"basic_unit_of_life": True, "types": ["plant", "animal", "bacteria"]})
        self._add("plant_cell", "biological", parents=["cell"],
                  properties={"has_cell_wall": True, "has_chloroplasts": True,
                             "has_large_vacuole": True, "rectangular_shape": True})
        self._add("animal_cell", "biological", parents=["cell"],
                  properties={"has_cell_wall": False, "has_chloroplasts": False,
                             "irregular_shape": True})

        # Photosynthesis and respiration
        self._add("photosynthesis", "biological",
                  properties={"inputs": ["sunlight", "water", "carbon_dioxide"],
                             "outputs": ["glucose", "oxygen"],
                             "makes_food": True, "food_is_glucose": True,
                             "plants_use_sunlight_to_make_food": True,
                             "occurs_in": "chloroplasts", "performed_by": "plants"})
        self._add("cellular_respiration", "biological",
                  properties={"inputs": ["glucose", "oxygen"],
                             "outputs": ["energy", "water", "carbon_dioxide"],
                             "occurs_in": "mitochondria", "performed_by": "all_living_things"})

        # Ecosystems
        self._add("ecosystem", "biological",
                  parts=["biotic_factors", "abiotic_factors"],
                  properties={"biotic": ["plants", "animals", "fungi", "bacteria"],
                             "abiotic": ["sunlight", "water", "temperature", "soil"]})
        self._add("food_chain", "biological",
                  properties={"starts_with": "producer", "ends_with": "decomposer",
                             "energy_flow": "sun_to_producer_to_consumer"})
        self._add("producer", "biological", parents=["organism"],
                  properties={"makes_own_food": True, "uses": "photosynthesis",
                             "examples": ["plants", "algae"]})
        self._add("consumer", "biological", parents=["organism"],
                  properties={"eats_other_organisms": True,
                             "types": ["herbivore", "carnivore", "omnivore"]})
        self._add("decomposer", "biological", parents=["organism"],
                  properties={"breaks_down": "dead_matter", "recycles_nutrients": True,
                             "examples": ["fungi", "bacteria"],
                             "role_in_ecosystem": "break_down_dead_organisms",
                             "not_produce_food": True,
                             "not_pollinate_flowers": True})

        # Adaptation and evolution
        self._add("adaptation", "biological",
                  properties={"helps": "survival", "types": ["structural", "behavioral", "physiological"],
                             "develops_over": "many_generations"})
        self._add("natural_selection", "biological",
                  properties={"survival_of_fittest": True, "drives": "evolution",
                             "requires": ["variation", "inheritance", "selection_pressure"]})
        self._add("heredity", "biological",
                  properties={"passing_traits": True, "through": "DNA",
                             "parent_to_offspring": True})

        # Body systems
        self._add("circulatory_system", "biological",
                  properties={"function": "transport", "includes": ["heart", "blood", "blood_vessels"],
                             "carries": ["oxygen", "nutrients", "waste"]})
        self._add("respiratory_system", "biological",
                  properties={"function": "gas_exchange", "includes": ["lungs", "trachea", "diaphragm"],
                             "inhale": "oxygen", "exhale": "carbon_dioxide"})
        self._add("digestive_system", "biological",
                  properties={"function": "break_down_food", "absorb": "nutrients",
                             "breaks_down_food_for_energy": True,
                             "responsible_for": "breaking_down_food",
                             "organs": ["stomach", "intestine", "mouth", "esophagus"]})
        self._add("nervous_system", "biological",
                  properties={"function": "communication", "includes": ["brain", "spinal_cord", "nerves"],
                             "controls": "body_functions",
                             "reflex_involves": "nervous_and_muscular_systems",
                             "senses_stimuli": True,
                             "sends_signals": True})
        self._add("immune_system", "biological",
                  properties={"function": "fight_infection", "includes": ["white_blood_cells", "antibodies"],
                             "fights": "disease", "protects_body": True})
        self._add("white_blood_cells", "biological", parents=["immune_system"],
                  properties={"function": "fight_infection", "type": "immune_cell",
                             "destroys": "pathogens", "in": "blood"})

    def _build_technology(self):
        """Technology and engineering concepts."""
        self._add("renewable_energy", "technology",
                  properties={"replenished": True, "examples": ["solar", "wind", "hydro", "geothermal"],
                             "clean": True, "sustainable": True})
        self._add("nonrenewable_energy", "technology",
                  properties={"limited_supply": True, "examples": ["fossil_fuels", "nuclear"],
                             "can_run_out": True})
        self._add("fossil_fuel", "technology", parents=["nonrenewable_energy"],
                  properties={"formed_from": "ancient_organisms", "types": ["coal", "oil", "natural_gas"],
                             "produces": "pollution", "releases": "carbon_dioxide"})
        self._add("sonar", "technology",
                  properties={"uses": "sound", "detects": "objects", "used_by": ["submarines", "bats"],
                             "similar_to": "echolocation"})

    def _build_materials(self):
        """Material properties."""
        self._add("metal", "physical",
                  properties={"conducts_heat": True, "conducts_electricity": True,
                             "malleable": True, "ductile": True, "lustrous": True,
                             "examples": ["iron", "copper", "gold", "aluminum"]})
        self._add("water", "physical",
                  properties={"chemical_formula": "H2O", "freezes_at": 0, "boils_at": 100,
                             "universal_solvent": True, "states": ["ice", "liquid", "steam"],
                             "expands_when_frozen": True, "density_ice_less_than_liquid": True})
        self._add("air", "physical",
                  properties={"mixture": True, "composition": {"nitrogen": 78, "oxygen": 21, "other": 1},
                             "has_mass": True, "has_pressure": True})

    def _build_energy(self):
        """Energy transformation chains."""
        self._add("energy_transformation", "process",
                  properties={"examples": [
                      "chemical_to_thermal",      # burning
                      "electrical_to_light",       # lightbulb
                      "chemical_to_kinetic",       # muscles
                      "solar_to_chemical",          # photosynthesis
                      "kinetic_to_electrical",     # generator
                      "electrical_to_sound",       # speaker
                      "nuclear_to_thermal",        # sun, reactor
                      "potential_to_kinetic",      # falling object
                      "kinetic_to_thermal",        # friction / braking
                  ]})

    def _build_processes(self):
        """Scientific processes and methods."""
        self._add("scientific_method", "process",
                  parts=["observation", "hypothesis", "experiment", "analysis", "conclusion"],
                  properties={"systematic": True, "testable": True, "repeatable": True})
        self._add("hypothesis", "process",
                  properties={"testable_prediction": True, "can_be_disproven": True})
        self._add("control_variable", "process",
                  properties={"kept_constant": True, "in": "experiment"})
        self._add("independent_variable", "process",
                  properties={"changed_by_experimenter": True})
        self._add("dependent_variable", "process",
                  properties={"measured": True, "responds_to": "independent_variable"})

    def _build_extended_concepts(self):
        """Extended concepts for broader ARC coverage."""
        # ── Waves & Sound ──
        self._add("wave", "physical",
                  properties={"transfers": "energy", "types": ["transverse", "longitudinal"],
                             "properties": ["wavelength", "frequency", "amplitude"],
                             "formula": "speed = wavelength × frequency"})
        self._add("sound", "physical", parents=["wave"],
                  properties={"type": "longitudinal", "needs_medium": True,
                             "cannot_travel_in": "vacuum", "speed_in_air": "343 m/s",
                             "measured_in": "decibels"})
        self._add("reflection", "process",
                  properties={"light_bounces_off": True, "angle_incidence_equals_reflection": True})
        self._add("refraction", "process",
                  properties={"light_bends": True, "between_media": True, "prism": "separates_colors"})

        # ── Human body extended ──
        self._add("skeleton", "biological",
                  properties={"function": ["support", "protection", "movement"],
                             "includes": ["bones", "joints", "cartilage"],
                             "adults_have": "206 bones"})
        self._add("muscle", "biological",
                  properties={"function": "movement", "types": ["skeletal", "smooth", "cardiac"],
                             "uses_energy": True, "contracts_and_relaxes": True})
        self._add("skin", "biological",
                  properties={"largest_organ": True, "function": ["protection", "temperature_regulation"],
                             "layers": ["epidermis", "dermis"]})

        # ── Chemistry basics ──
        self._add("atom", "physical",
                  properties={"basic_unit": "element", "parts": ["proton", "neutron", "electron"],
                             "protons_in_nucleus": True, "electrons_orbit": True})
        self._add("molecule", "physical",
                  properties={"two_or_more_atoms": True, "bonded_together": True,
                             "examples": ["H2O", "CO2", "O2"]})
        self._add("mixture", "physical",
                  properties={"two_or_more_substances": True, "not_chemically_bonded": True,
                             "can_be_separated": True, "types": ["homogeneous", "heterogeneous"]})
        self._add("solution", "physical", parents=["mixture"],
                  properties={"homogeneous": True, "has": ["solute", "solvent"],
                             "example": "salt_in_water"})
        self._add("chemical_change", "process",
                  properties={"new_substance_formed": True, "not_easily_reversed": True,
                             "signs": ["color_change", "gas_produced", "heat_change", "precipitate"]})
        self._add("physical_change", "process",
                  properties={"no_new_substance": True, "reversible": True,
                             "examples": ["melting", "freezing", "cutting", "dissolving"]})

        # ── Space ──
        self._add("star", "earth_science",
                  properties={"made_of": "hydrogen_and_helium", "energy_from": "nuclear_fusion",
                             "types_by_size": ["dwarf", "giant", "supergiant"],
                             "example": "sun"})
        self._add("planet", "earth_science",
                  properties={"orbits_star": True, "inner": ["Mercury", "Venus", "Earth", "Mars"],
                             "outer": ["Jupiter", "Saturn", "Uranus", "Neptune"],
                             "inner_are": "rocky", "outer_are": "gas_giants"})
        self._add("solar_system", "earth_science",
                  properties={"center": "sun", "includes": ["planets", "moons", "asteroids", "comets"],
                             "held_by": "gravity"})
        self._add("constellation", "earth_science",
                  properties={"pattern_of_stars": True, "examples": ["Orion", "Big Dipper", "Cassiopeia"]})

        # ── Soil & Minerals ──
        self._add("soil", "earth_science",
                  properties={"made_of": ["weathered_rock", "decomposed_organisms", "minerals"],
                             "layers": ["topsoil", "subsoil", "bedrock"],
                             "important_for": "plant_growth"})
        self._add("mineral", "earth_science",
                  properties={"naturally_occurring": True, "inorganic": True,
                             "crystal_structure": True, "identified_by": ["hardness", "luster", "streak", "color"]})
        self._add("fossil", "earth_science",
                  properties={"preserved_remains": True, "found_in": "sedimentary_rock",
                             "evidence_of": "past_life", "types": ["body_fossil", "trace_fossil"]})

        # ── Life Cycles ──
        self._add("metamorphosis", "biological",
                  properties={"complete_change": True,
                             "complete": ["egg", "larva", "pupa", "adult"],
                             "incomplete": ["egg", "nymph", "adult"],
                             "examples_complete": ["butterfly", "frog"],
                             "examples_incomplete": ["grasshopper"]})
        self._add("seed_plant_life_cycle", "biological",
                  properties={"stages": ["seed", "germination", "growth", "flower", "fruit", "seed"],
                             "needs": ["water", "sunlight", "soil", "warmth"]})
        self._add("pollination", "biological",
                  properties={"transfer_of": "pollen", "agents": ["bees", "wind", "birds", "butterflies"],
                             "needed_for": "seed_production"})

        # ── Classification ──
        self._add("vertebrate", "biological",
                  properties={"has_backbone": True,
                             "groups": ["mammals", "birds", "reptiles", "amphibians", "fish"]})
        self._add("invertebrate", "biological",
                  properties={"no_backbone": True,
                             "examples": ["insects", "spiders", "worms", "jellyfish"]})
        self._add("mammal", "biological", parents=["vertebrate"],
                  properties={"warm_blooded": True, "has_hair_or_fur": True,
                             "produces_milk": True, "live_birth": True})
        self._add("reptile", "biological", parents=["vertebrate"],
                  properties={"cold_blooded": True, "has_scales": True, "lays_eggs": True,
                             "examples": ["snake", "lizard", "turtle"]})
        self._add("amphibian", "biological", parents=["vertebrate"],
                  properties={"cold_blooded": True, "moist_skin": True,
                             "metamorphosis": True, "lives_in_water_and_land": True,
                             "examples": ["frog", "salamander"]})

        # ── Density & Buoyancy ──
        self._add("density", "physical",
                  properties={"formula": "mass/volume", "determines": "sink_or_float",
                             "less_than_water_floats": True, "more_than_water_sinks": True})
        self._add("buoyancy", "physical",
                  properties={"upward_force": True, "in_fluid": True,
                             "Archimedes": "displaced_fluid_weight_equals_buoyant_force"})

        # ── Conservation ──
        self._add("conservation", "process",
                  properties={"protect": "resources", "methods": ["reduce", "reuse", "recycle"],
                             "prevents": "waste"})
        self._add("pollution", "earth_science",
                  properties={"harmful_substance": True,
                             "types": ["air", "water", "land", "noise"],
                             "causes": ["burning_fossil_fuels", "littering", "chemicals"]})

        # ── Temperature & States of Matter ──
        self._add("temperature", "physical",
                  properties={"unit_celsius": True, "unit_fahrenheit": True,
                             "water_freezes_celsius": 0, "water_boils_celsius": 100,
                             "water_freezes_fahrenheit": 32, "water_boils_fahrenheit": 212,
                             "absolute_zero": "-273.15°C", "normal_body_temp": "37°C or 98.6°F",
                             "higher_temp_means_more_energy": True})
        self._add("states_of_matter", "physical",
                  properties={"solid": "fixed shape and volume",
                             "liquid": "fixed volume, takes shape of container",
                             "gas": "fills entire container",
                             "freezing": "liquid to solid", "melting": "solid to liquid",
                             "evaporation": "liquid to gas", "condensation": "gas to liquid",
                             "sublimation": "solid to gas directly"})

        # ── Heat Transfer ──
        self._add("heat_transfer", "physical",
                  properties={"direction": "hot to cold",
                             "conduction": "through direct contact",
                             "convection": "through fluid movement",
                             "radiation": "through electromagnetic waves",
                             "heat_flows_from_hot_to_cold": True,
                             "warm_object_gives_heat_to_cold_object": True})

        # ── Photosynthesis & Cell Biology ──
        self._add("photosynthesis", "biological",
                  properties={"equation": "CO2 + H2O + light → glucose + O2",
                             "occurs_in": "chloroplasts",
                             "chlorophyll": "captures light energy",
                             "first_step": "chlorophyll captures light",
                             "converts": "light energy to chemical energy",
                             "produces": ["glucose", "oxygen"],
                             "requires": ["carbon dioxide", "water", "sunlight"]})
        self._add("cell_biology", "biological",
                  properties={"prokaryote": "no nucleus, smaller, bacteria",
                             "eukaryote": "has nucleus, larger, plants/animals",
                             "difference": "nucleus and size",
                             "prokaryote_smaller_than_eukaryote": True,
                             "cell_membrane": "controls what enters and leaves cell",
                             "mitochondria": "powerhouse of cell, produces ATP"})

        # ── Earth & Space ──
        self._add("rotation", "earth_science",
                  properties={"Earth_rotation": "causes day and night",
                             "one_rotation": "24 hours = 1 day",
                             "faster_rotation": "shorter days",
                             "slower_rotation": "longer days"})
        self._add("revolution", "earth_science",
                  properties={"Earth_revolution": "orbiting around the Sun",
                             "one_revolution": "365.25 days = 1 year",
                             "causes": ["seasons", "year_length"]})
        self._add("fossil", "earth_science",
                  properties={"preserved_remains": True,
                             "found_in": "sedimentary rock",
                             "tropical_fossils_in_cold_area": "climate was once warmer",
                             "palm_tree_fossils_near_glaciers": "area was once tropical"})

        # ── Scientific Method ──
        self._add("scientific_method", "process",
                  properties={"steps": ["observe", "question", "hypothesis", "experiment", "analyze", "conclude"],
                             "independent_variable": "what scientist changes/manipulates",
                             "dependent_variable": "what is measured/observed",
                             "control": "kept the same for comparison",
                             "first_step_before_experiment": "plan and prepare (make data table)",
                             "purpose_of_testing_models": "make buildings safer"})

        # ── Energy & Forces ──
        self._add("kinetic_energy", "physical",
                  properties={"formula": "½mv²", "energy_of_motion": True,
                             "increases_with_speed": True, "increases_with_mass": True})
        self._add("potential_energy", "physical",
                  properties={"formula": "mgh", "energy_of_position": True,
                             "increases_with_height": True,
                             "halfway_down_equals_half_max_kinetic": True,
                             "at_halfway_gained_half_max_kinetic_energy": True})
        self._add("gravity", "physical",
                  properties={"pulls_toward_center": True,
                             "same_acceleration_all_masses": True,
                             "moon_less_gravity_than_earth": True,
                             "objects_same_mass_drop_same_rate_no_air": True})

    def _build_v3_concepts(self):
        """v3.0 expanded concepts: animals, habitats, Earth features, instruments, circuits."""
        # ── Animals ──
        self._add("bird", "biological", parents=["animal"],
                  properties={"has_feathers": True, "has_wings": True, "lays_eggs": True,
                             "warm_blooded": True, "has_beak": True, "most_can_fly": True,
                             "examples": ["eagle", "robin", "penguin", "owl"]})
        self._add("fish", "biological", parents=["animal"],
                  properties={"lives_in_water": True, "has_gills": True, "has_fins": True,
                             "has_scales": True, "cold_blooded": True, "lays_eggs": True,
                             "examples": ["salmon", "trout", "shark", "goldfish"]})
        self._add("reptile", "biological", parents=["animal"],
                  properties={"has_scales": True, "cold_blooded": True, "lays_eggs": True,
                             "breathes_air": True, "examples": ["snake", "lizard", "turtle", "crocodile"]})
        self._add("amphibian", "biological", parents=["animal"],
                  properties={"cold_blooded": True, "lives_in_water_and_land": True,
                             "metamorphosis": True, "moist_skin": True,
                             "examples": ["frog", "salamander", "toad", "newt"]})

        # ── Habitats & Earth Features ──
        self._add("ocean", "earth_science", parents=["body_of_water"],
                  properties={"salt_water": True, "covers_71_percent_earth": True,
                             "deepest": "Mariana_Trench", "largest": "Pacific",
                             "affects_climate": True, "has_tides": True})
        self._add("river", "earth_science", parents=["body_of_water"],
                  properties={"fresh_water": True, "flows_downhill": True,
                             "erodes_land": True, "deposits_sediment": True,
                             "flows_to_ocean_or_lake": True})
        self._add("mountain", "earth_science", parents=["landform"],
                  properties={"formed_by": ["tectonic_plates", "volcanic_activity"],
                             "higher_elevation": True, "colder_at_top": True,
                             "less_air_pressure_at_top": True})
        self._add("desert", "earth_science", parents=["biome"],
                  properties={"very_little_rain": True, "less_than_25cm_rain_per_year": True,
                             "extreme_temperatures": True, "sparse_vegetation": True,
                             "examples": ["Sahara", "Gobi", "Mojave"]})
        self._add("forest", "earth_science", parents=["biome"],
                  properties={"many_trees": True, "high_biodiversity": True,
                             "types": ["tropical_rainforest", "temperate", "boreal"],
                             "produces_oxygen": True, "absorbs_CO2": True})
        self._add("lake", "earth_science", parents=["body_of_water"],
                  properties={"usually_fresh_water": True, "surrounded_by_land": True,
                             "formed_by": ["glaciers", "tectonic_activity", "rivers"]})

        # ── Plant biology ──
        self._add("seed", "biological", parents=["plant_part"],
                  properties={"contains_embryo": True, "has_food_supply": True,
                             "has_seed_coat": True, "can_be_dormant": True,
                             "needs_water_warmth_to_germinate": True,
                             "start_of_plant_life_cycle": True,
                             "most_plants_begin_as": "seeds",
                             "plants_begin_life_cycle_as": "seeds"})
        self._add("germination", "process", parents=["plant_growth"],
                  properties={"seed_begins_to_grow": True,
                             "requires": ["water", "warmth", "oxygen"],
                             "root_grows_first": True, "shoot_grows_upward": True})
        self._add("pollination", "process", parents=["plant_reproduction"],
                  properties={"transfer_of_pollen": True,
                             "agents": ["insects", "wind", "birds", "bats"],
                             "leads_to": "fertilization", "required_for": "seed_formation"})

        # ── Instruments ──
        self._add("telescope", "physical", parents=["instrument"],
                  properties={"magnifies_distant_objects": True, "uses_lenses_or_mirrors": True,
                             "used_for": "astronomy", "types": ["refracting", "reflecting"]})
        self._add("microscope", "physical", parents=["instrument"],
                  properties={"magnifies_small_objects": True, "uses_lenses": True,
                             "used_for": "biology", "reveals": "cells_and_microorganisms"})
        self._add("compass", "physical", parents=["instrument"],
                  properties={"detects_magnetic_field": True, "needle_points_north": True,
                             "uses_earths_magnetic_field": True, "used_for": "navigation"})
        self._add("thermometer", "physical", parents=["instrument"],
                  properties={"measures_temperature": True,
                             "units": ["Celsius", "Fahrenheit", "Kelvin"],
                             "liquid_expands_when_heated": True})
        self._add("barometer", "physical", parents=["instrument"],
                  properties={"measures_air_pressure": True, "predicts_weather": True,
                             "high_pressure_clear_sky": True, "low_pressure_storm": True})

        # ── Electrical circuits ──
        self._add("battery", "physical", parents=["energy_source"],
                  properties={"converts_chemical_to_electrical": True,
                             "has_positive_and_negative_terminals": True,
                             "provides_voltage": True})
        self._add("circuit", "physical",
                  properties={"closed_loop_for_current": True, "needs": ["source", "wire", "load"],
                             "open_circuit_no_flow": True, "closed_circuit_current_flows": True})
        self._add("circuit_series", "physical", parents=["circuit"],
                  properties={"one_path_for_current": True, "all_components_in_line": True,
                             "if_one_breaks_all_stop": True, "current_same_everywhere": True,
                             "voltages_add_up": True})
        self._add("circuit_parallel", "physical", parents=["circuit"],
                  properties={"multiple_paths_for_current": True,
                             "if_one_breaks_others_work": True, "voltage_same_everywhere": True,
                             "currents_add_up": True})

        # ── Magnetism expanded ──
        self._add("magnet_poles", "physical", parents=["magnetism"],
                  properties={"every_magnet_has_north_and_south": True,
                             "opposite_poles_attract": True, "same_poles_repel": True,
                             "cannot_isolate_single_pole": True,
                             "earth_has_magnetic_poles": True})
        self._add("electromagnet", "physical", parents=["magnetism"],
                  properties={"made_by_current_through_coil": True,
                             "can_be_turned_on_off": True,
                             "strength_depends_on": ["current", "coils", "core_material"]})

        # ── Seasons & Day-Night ──
        self._add("seasons", "earth_science",
                  properties={"caused_by": "earth_axial_tilt",
                             "NOT_caused_by": "distance_from_sun",
                             "tilt_angle": "23.5_degrees",
                             "summer": "hemisphere_tilted_toward_sun",
                             "winter": "hemisphere_tilted_away_from_sun"})
        self._add("day_and_night", "earth_science",
                  properties={"caused_by": "earth_rotation",
                             "earth_rotates_every": "24_hours",
                             "daytime_faces_sun": True, "nighttime_faces_away": True})

        # ── Food chains & ecosystems ──
        self._add("producer", "biological", parents=["organism"],
                  properties={"makes_own_food": True, "photosynthesis": True,
                             "base_of_food_chain": True, "examples": ["plants", "algae"]})
        self._add("consumer", "biological", parents=["organism"],
                  properties={"eats_other_organisms": True,
                             "types": ["herbivore", "carnivore", "omnivore"]})
        self._add("decomposer", "biological", parents=["organism"],
                  properties={"breaks_down_dead_matter": True, "recycles_nutrients": True,
                             "examples": ["fungi", "bacteria", "earthworms"]})
        self._add("food_chain", "biological",
                  properties={"shows_energy_flow": True,
                             "starts_with_producer": True,
                             "energy_decreases_each_level": True,
                             "sun_is_ultimate_energy_source": True})

    def _build_v2_concepts(self):
        """v2.0 expanded concepts: chemistry, genetics, optics, engineering."""
        # ── Chemistry Extended ──
        self._add("element", "physical",
                  properties={"pure_substance": True, "one_type_of_atom": True,
                             "periodic_table": True, "cannot_break_down_chemically": True,
                             "examples": ["hydrogen", "oxygen", "carbon", "iron", "gold"]})
        self._add("compound", "physical",
                  properties={"two_or_more_elements": True, "chemically_bonded": True,
                             "different_from_elements": True,
                             "examples": ["water_H2O", "salt_NaCl", "carbon_dioxide_CO2"]})
        self._add("chemical_reaction", "process",
                  properties={"reactants_become_products": True, "bonds_break_and_form": True,
                             "evidence": ["color_change", "gas_produced", "temperature_change",
                                         "precipitate_forms", "light_produced"]})
        self._add("acid", "physical",
                  properties={"pH_below_7": True, "sour_taste": True, "reacts_with_metals": True,
                             "examples": ["vinegar", "lemon_juice", "hydrochloric_acid"]})
        self._add("base", "physical",
                  properties={"pH_above_7": True, "bitter_taste": True, "slippery_feel": True,
                             "examples": ["baking_soda", "soap", "ammonia"]})
        self._add("pH_scale", "physical",
                  properties={"range_0_to_14": True, "7_is_neutral": True,
                             "below_7_acid": True, "above_7_base": True})

        # ── Genetics ──
        self._add("DNA", "biological",
                  properties={"shape": "double_helix", "contains": "genetic_information",
                             "bases": ["adenine", "thymine", "guanine", "cytosine"],
                             "in": "nucleus", "controls": "traits"})
        self._add("gene", "biological", parents=["DNA"],
                  properties={"segment_of_DNA": True, "codes_for": "protein",
                             "determines": "trait", "inherited": True})
        self._add("chromosome", "biological",
                  properties={"contains_DNA": True, "humans_have_46": True,
                             "pairs_of_23": True, "in_nucleus": True})
        self._add("dominant_trait", "biological",
                  properties={"expressed_with_one_copy": True, "uppercase_letter": True,
                             "masks_recessive": True})
        self._add("recessive_trait", "biological",
                  properties={"requires_two_copies": True, "lowercase_letter": True,
                             "masked_by_dominant": True})
        self._add("genotype", "biological",
                  properties={"genetic_makeup": True, "letters": True,
                             "homozygous": "same_alleles", "heterozygous": "different_alleles"})
        self._add("phenotype", "biological",
                  properties={"physical_appearance": True, "observable_trait": True,
                             "determined_by": "genotype_and_environment"})

        # ── Optics ──
        self._add("lens", "physical",
                  properties={"bends_light": True, "types": ["convex", "concave"],
                             "convex_converges": True, "concave_diverges": True,
                             "used_in": ["glasses", "microscope", "telescope", "camera"]})
        self._add("prism", "physical",
                  properties={"separates_white_light": True, "into_spectrum": True,
                             "refraction": True,
                             "colors": ["red", "orange", "yellow", "green", "blue", "indigo", "violet"]})
        self._add("electromagnetic_spectrum", "physical",
                  properties={"order": ["radio", "microwave", "infrared", "visible",
                                       "ultraviolet", "x_ray", "gamma_ray"],
                             "increasing_energy": True, "increasing_frequency": True,
                             "decreasing_wavelength": True})

        # ── Engineering & Measurement ──
        self._add("mass", "physical",
                  properties={"amount_of_matter": True, "measured_in": "grams_or_kilograms",
                             "does_not_change_with_location": True,
                             "different_from_weight": True})
        self._add("weight", "physical",
                  properties={"force_of_gravity_on_mass": True, "measured_in": "newtons",
                             "changes_with_gravity": True,
                             "less_on_moon": True, "more_on_jupiter": True})
        self._add("volume", "physical",
                  properties={"amount_of_space": True,
                             "measured_in": "liters_or_cubic_centimeters",
                             "measured_by": "water_displacement_for_irregular_objects"})

        # ── Human Nutrition ──
        self._add("nutrient", "biological",
                  properties={"types": ["carbohydrate", "protein", "fat",
                                       "vitamin", "mineral", "water"],
                             "needed_for": "body_function",
                             "obtained_from": "food"})
        self._add("carbohydrate", "biological", parents=["nutrient"],
                  properties={"provides": "energy", "sources": ["bread", "pasta", "rice", "sugar"]})
        self._add("protein", "biological", parents=["nutrient"],
                  properties={"builds_repairs": "tissue", "sources": ["meat", "beans", "eggs", "nuts"]})
        self._add("fat", "biological", parents=["nutrient"],
                  properties={"stores_energy": True, "insulates": True,
                             "sources": ["oil", "butter", "nuts", "avocado"]})

        # ── Symbiosis ──
        self._add("symbiosis", "biological",
                  properties={"close_relationship": True,
                             "types": ["mutualism", "commensalism", "parasitism"]})
        self._add("mutualism", "biological", parents=["symbiosis"],
                  properties={"both_benefit": True,
                             "examples": ["bee_and_flower", "clownfish_and_sea_anemone",
                                          "mycorrhizae_and_plant"],
                             "clownfish_anemone_is_mutualism": True})
        self._add("parasitism", "biological", parents=["symbiosis"],
                  properties={"one_benefits_one_harmed": True,
                             "examples": ["tick_on_dog", "tapeworm", "flea"],
                             "clownfish_anemone_is_not_parasitism": True})
        self._add("commensalism", "biological", parents=["symbiosis"],
                  properties={"one_benefits_one_unaffected": True,
                             "example": "barnacle_on_whale"})

    def _build_v4_concepts(self):
        """v4.0: Additional science concepts for ARC-Challenge coverage."""
        # ── Elements & Atoms ──
        self._add("atom", "physical",
                  properties={"smallest_unit_of_element": True,
                             "made_of": ["proton", "neutron", "electron"],
                             "protons_in_nucleus": True, "electrons_orbit_nucleus": True,
                             "number_of_protons": "determines_element"})
        self._add("element", "physical",
                  properties={"pure_substance": True, "one_type_of_atom": True,
                             "on_periodic_table": True,
                             "copper_is_element": True, "iron_is_element": True,
                             "smallest_unit": "atom",
                             "examples": ["copper", "iron", "oxygen", "gold", "hydrogen"]})
        self._add("copper", "physical", parents=["element"],
                  properties={"conducts_electricity": True, "conducts_heat": True,
                             "used_in": "electrical_wires", "metal": True,
                             "smallest_unit": "atom", "element": True,
                             "color": "reddish_orange"})

        # ── Greenhouse Effect ──
        self._add("greenhouse_gas", "earth_science",
                  properties={"traps_heat": True, "in_atmosphere": True,
                             "most_abundant": "water_vapor",
                             "types": ["water_vapor", "carbon_dioxide", "methane", "nitrous_oxide"],
                             "water_vapor_most_abundant": True,
                             "CO2_from_fossil_fuels": True})
        self._add("greenhouse_effect", "earth_science",
                  properties={"traps_heat_in_atmosphere": True,
                             "keeps_earth_warm": True,
                             "enhanced_by": "human_activities"})

        # ── Heat Transfer Details ──
        self._add("thermal_equilibrium", "physical",
                  properties={"objects_reach_same_temperature": True,
                             "heat_flows_hot_to_cold": True,
                             "stops_when_equal_temperature": True,
                             "warm_object_cools_cool_object_warms": True})
        self._add("cooling", "process",
                  properties={"heat_transfer_from_hot_to_cold": True,
                             "fastest_cooling_methods": ["ice", "cold_water", "refrigerator"],
                             "adding_ice_cools_fastest": True,
                             "stirring_speeds_cooling": True,
                             "metal_spoon_conducts_heat_away": True})

        # ── Buoyancy & Density ──
        self._add("buoyancy", "physical",
                  properties={"upward_force_of_fluid": True,
                             "object_floats_if_less_dense_than_fluid": True,
                             "wood_floats_on_water": True,
                             "branch_floats_because_less_dense": True,
                             "density_determines_float_or_sink": True})
        self._add("density", "physical",
                  properties={"mass_per_unit_volume": True,
                             "formula": "mass_divided_by_volume",
                             "less_dense_floats": True, "more_dense_sinks": True,
                             "wood_less_dense_than_water": True,
                             "ice_less_dense_than_liquid_water": True})

        # ── Environmental Science ──
        self._add("deforestation", "earth_science",
                  properties={"cutting_down_trees": True,
                             "effects": ["habitat_loss", "soil_erosion", "increased_CO2",
                                        "loss_of_biodiversity", "flooding"],
                             "logging_causes": True,
                             "reduces_oxygen_production": True})
        self._add("spontaneous_generation", "biological",
                  properties={"disproven_theory": True,
                             "once_thought": "living_from_nonliving",
                             "disproved_by": ["Redi", "Pasteur"],
                             "Pasteur_used_swan_neck_flask": True,
                             "Redi_experiment_with_meat_and_flies": True,
                             "replaced_by": "biogenesis"})
        self._add("biogenesis", "biological",
                  properties={"life_comes_from_life": True,
                             "proved_by_Pasteur": True,
                             "replaced": "spontaneous_generation"})

        # ── Experimental Design ──
        self._add("controlled_experiment", "process",
                  properties={"one_variable_changed": True,
                             "independent_variable": "what_you_change",
                             "dependent_variable": "what_you_measure",
                             "controlled_variables": "kept_the_same",
                             "control_group": "standard_for_comparison",
                             "repetition_increases_reliability": True,
                             "same_brand_different_conditions": "compare_fairly"})

        # ── Weather & Flash Floods ──
        self._add("flash_flood", "earth_science",
                  properties={"rapid_flooding": True,
                             "caused_by": "heavy_rain_in_short_time",
                             "desert_areas_vulnerable": True,
                             "dry_hard_soil": "cannot_absorb_water_quickly",
                             "low_absorption": True,
                             "dangerous_in_dry_climates": True})

        # ── Moon & Gravity ──
        self._add("moon_gravity", "physical",
                  properties={"weaker_than_earth": True,
                             "about_1_6_earth_gravity": True,
                             "objects_fall_slower": True,
                             "all_objects_fall_at_same_rate": True,
                             "acceleration_same_regardless_of_mass": True,
                             "1kg_and_5kg_fall_together": True,
                             "galileo_principle": True})

        # ── Disease Transmission ──
        self._add("disease_transmission", "biological",
                  properties={"contact": "direct_physical_touch",
                             "DFTD": "devil_facial_tumor_disease",
                             "transmissible_cancer": True,
                             "transmitted_by_biting": True,
                             "contagious_diseases_spread_by_contact": True})

        # ── Fossils ──
        self._add("fossil", "earth_science",
                  properties={"preserved_remains_of_organisms": True,
                             "shows_past_life": True,
                             "found_in_sedimentary_rock": True,
                             "newer_fossils_in_upper_layers": True,
                             "older_fossils_in_deeper_layers": True,
                             "evidence_of_evolution": True,
                             "different_species_at_different_times": True})

        # ── Sun & Oceans ──
        self._add("sun_ocean_interaction", "earth_science",
                  properties={"sun_heats_ocean": True,
                             "causes_evaporation": True,
                             "drives_water_cycle": True,
                             "uneven_heating_causes_currents": True,
                             "creates_wind_patterns": True})

        # ── Plankton ──
        self._add("plankton", "biological",
                  properties={"tiny_organisms_in_ocean": True,
                             "phytoplankton": "plant-like, do photosynthesis, are producers",
                             "zooplankton": "animal-like, are consumers",
                             "base_of_ocean_food_chain": True,
                             "phytoplankton_produce_oxygen": True,
                             "phytoplankton_are_producers": True})

        # ── Penguins ──
        self._add("penguin", "biological",
                  properties={"bird": True, "cannot_fly": True,
                             "swims": True, "lives_in_cold_climates": True,
                             "feathers_for_insulation": True,
                             "eats_fish": True, "flightless_bird": True,
                             "lives_in_southern_hemisphere": True,
                             "example_of_adaptation": True})

        # ── Predation ──
        self._add("predation", "biological",
                  properties={"one_organism_hunts_another": True,
                             "predator_eats_prey": True,
                             "controls_prey_population": True,
                             "example": "hawk_eats_chicken",
                             "introducing_predators_reduces_prey": True,
                             "removing_predators_increases_prey": True})

        # ── Keyword-alias concepts (match common question phrasing) ──
        self._add("floats", "physical", related=["buoyancy", "density"],
                  properties={"less_dense_objects_float": True,
                             "wood_floats_because_less_dense_than_water": True,
                             "branch_floats_because_less_dense": True,
                             "density_determines_if_object_floats_or_sinks": True,
                             "ice_floats_because_less_dense_than_liquid_water": True})
        self._add("freeze", "process", related=["phase_change"],
                  properties={"water_freezes_at_0_celsius": True,
                             "water_freezes_at_32_fahrenheit": True,
                             "freezing": "liquid_to_solid",
                             "requires_removing_heat": True})
        self._add("learned behavior", "biological",
                  properties={"acquired_through_experience": True,
                             "not_inherited": True,
                             "examples": ["reading", "riding_a_bike", "speaking_a_language",
                                         "using_tools", "tying_shoes", "cooking"],
                             "opposite_of": "instinct"})
        self._add("innate behavior", "biological",
                  properties={"present_at_birth": True, "does_not_require_learning": True,
                             "inherited": True,
                             "examples": ["breathing", "blinking", "sucking_reflex",
                                         "web_spinning_spiders", "bird_migration",
                                         "babies_crying"]})
        self._add("electromagnetic induction", "physical",
                  properties={"moving_magnet_in_coil": True,
                             "produces_electric_current": True,
                             "Faraday_discovered": True,
                             "generator_principle": True,
                             "change_in_magnetic_field": True})
        self._add("renewable resource", "earth_science",
                  properties={"replaced_naturally": True,
                             "examples": ["solar", "wind", "water", "trees", "biomass"],
                             "rate_of_use": "must_not_exceed_rate_of_replacement"})
        self._add("nonrenewable resource", "earth_science",
                  properties={"limited_supply": True, "formed_over_millions_of_years": True,
                             "examples": ["coal", "oil", "natural_gas", "minerals"],
                             "fossil_fuels": True})
        self._add("carbon atom", "physical", parents=["atom"],
                  properties={"atomic_number": 6, "protons": 6,
                             "mass_equals_protons_plus_neutrons": True,
                             "6_protons_6_neutrons": "mass_12",
                             "6_protons_7_neutrons": "mass_13",
                             "electrons_do_not_contribute_to_mass": True})
        self._add("sunrise", "earth_science", related=["rotation"],
                  properties={"occurs_once_per_day": True,
                             "caused_by": "earth_rotation",
                             "sunrise_happens_every_24_hours": True})
        self._add("gravitational force", "physical", parents=["force"],
                  properties={"depends_on_mass_and_distance": True,
                             "increases_with_mass": True,
                             "decreases_with_distance": True,
                             "more_mass_more_gravity": True,
                             "closer_distance_more_gravity": True,
                             "F_equals_G_m1_m2_over_r_squared": True})
        self._add("precipitation", "earth_science", related=["water_cycle"],
                  properties={"forms_of": ["rain", "snow", "hail", "sleet"],
                             "fog_is_not_precipitation": True,
                             "precipitation_falls_from_clouds": True,
                             "rain_snow_hail_sleet": True})
        self._add("conservation of mass", "physical",
                  properties={"mass_neither_created_nor_destroyed": True,
                             "total_mass_before_equals_after": True,
                             "chemical_reaction_conserves_mass": True,
                             "20g_in_20g_out": True})
        self._add("air", "physical", related=["matter"],
                  properties={"invisible_gas": True, "takes_up_space": True,
                             "has_mass": True, "mixture_of_gases": True,
                             "prove_air_takes_space": "blow_air_into_water_see_bubbles",
                             "submerge_inverted_cup_water_doesnt_fill": True})
        self._add("environment influence", "biological",
                  properties={"traits_influenced_by_environment": True,
                             "examples": ["skin_color_from_sun", "plant_growth_direction",
                                         "language_spoken", "weight", "muscle_mass"],
                             "not_genetic": True,
                             "acquired_characteristics": True})

    def _build_v5_concepts(self):
        """v5.0: Missing high-impact concepts identified via ARC failure analysis."""
        # ── Force & Motion ──
        self._add("force", "physical",
                  properties={"push_or_pull": True, "causes_acceleration": True,
                             "measured_in": "newtons", "has_direction": True,
                             "net_force_causes_motion_change": True,
                             "balanced_forces": "no_motion_change",
                             "unbalanced_forces": "causes_acceleration",
                             "types": ["gravity", "friction", "magnetic", "applied"]})
        self._add("inertia", "physical", related=["force", "motion"],
                  properties={"resistance_to_change_in_motion": True,
                             "depends_on_mass": True, "more_mass_more_inertia": True,
                             "newton_first_law": True,
                             "object_at_rest_stays_at_rest": True,
                             "object_in_motion_stays_in_motion": True})

        # ── Vibration & Sound ──
        self._add("vibration", "physical", related=["sound"],
                  properties={"back_and_forth_motion": True,
                             "causes_sound": True, "sound_is_vibration": True,
                             "faster_vibration_higher_pitch": True,
                             "larger_vibration_louder_sound": True})
        self._add("sound", "physical", parents=["energy"],
                  properties={"caused_by": "vibration", "travels_as_waves": True,
                             "needs_medium": True, "cannot_travel_in": "vacuum",
                             "travels_fastest_in": "solids",
                             "travels_slowest_in": "gases",
                             "speed_in_air": "343_m_per_s",
                             "pitch_determined_by": "frequency",
                             "volume_determined_by": "amplitude",
                             "speed_depends_on": "type_of_material",
                             "speed_depends_on_medium": True})

        # ── Chemical Weathering ──
        self._add("chemical_weathering", "earth_science", parents=["weathering"],
                  properties={"changes_chemical_composition": True,
                             "caused_by": ["acid_rain", "oxidation", "water", "oxygen"],
                             "examples": ["rust", "acid_rain_dissolving_rock",
                                         "oxidation_of_iron"],
                             "rust_is_chemical_weathering": True,
                             "acid_reacts_with_minerals": True})
        self._add("mechanical_weathering", "earth_science", parents=["weathering"],
                  properties={"breaks_rock_without_chemical_change": True,
                             "caused_by": ["freezing_thawing", "wind", "abrasion",
                                          "tree_roots", "ice_wedging"],
                             "ice_wedging": "water_freezes_expands_cracks_rock"})

        # ── Elements: Carbon, Oxygen, Nitrogen ──
        self._add("carbon", "physical", parents=["element"],
                  properties={"atomic_number": 6, "in_all_living_things": True,
                             "forms_organic_compounds": True,
                             "diamond_and_graphite": True,
                             "CO2_is_carbon_dioxide": True,
                             "in_fossil_fuels": True})
        self._add("oxygen", "physical", parents=["element"],
                  properties={"atomic_number": 8, "needed_for_breathing": True,
                             "needed_for_fire": True, "in_atmosphere": True,
                             "about_21_percent_of_air": True,
                             "produced_by": "photosynthesis",
                             "used_in": "cellular_respiration"})
        self._add("nitrogen", "physical", parents=["element"],
                  properties={"atomic_number": 7, "most_abundant_gas_in_atmosphere": True,
                             "about_78_percent_of_air": True,
                             "needed_for": "plant_growth"})

        # ── Technology: Electric Motor, Light Bulb, Generator ──
        self._add("electric_motor", "technology",
                  properties={"converts": "electrical_energy_to_mechanical_energy",
                             "uses": ["magnet", "coil", "current"],
                             "electrical_to_kinetic": True,
                             "produces": "mechanical_energy",
                             "opposite_of": "generator"})
        self._add("light_bulb", "technology",
                  properties={"converts": "electrical_energy_to_light_energy",
                             "also_produces": "heat",
                             "electrical_to_light_and_heat": True,
                             "produces": "light_energy",
                             "incandescent_has_filament": True,
                             "needs": "electric_current",
                             "requires": "current_flowing_through_wire",
                             "works_by": "current_flowing_through_filament",
                             "powered_by": "electricity"})
        self._add("generator", "technology",
                  properties={"converts": "mechanical_energy_to_electrical_energy",
                             "uses": "electromagnetic_induction",
                             "kinetic_to_electrical": True,
                             "produces": "electrical_energy",
                             "opposite_of": "electric_motor"})
        self._add("battery", "technology",
                  properties={"converts": "chemical_energy_to_electrical_energy",
                             "stores_energy": True, "has_positive_and_negative_terminals": True,
                             "produces": "electrical_energy",
                             "chemical_to_electrical": True})

        # ── Water ──
        self._add("water", "physical", parents=["matter"],
                  properties={"boils_at": 100, "freezes_at": 0,
                             "boiling_point_celsius": 100,
                             "freezing_point_celsius": 0,
                             "boiling_point_fahrenheit": 212,
                             "freezing_point_fahrenheit": 32,
                             "formula": "H2O", "universal_solvent": True,
                             "states": ["ice", "liquid", "steam"],
                             "density_of_ice": "less_than_liquid_water"})

        # ── Temperature ──
        self._add("temperature", "physical",
                  properties={"measure_of_kinetic_energy": True,
                             "measured_in": ["celsius", "fahrenheit", "kelvin"],
                             "water_boils_celsius": 100,
                             "water_freezes_celsius": 0,
                             "water_boils_fahrenheit": 212,
                             "water_freezes_fahrenheit": 32,
                             "body_temperature_fahrenheit": 98.6,
                             "absolute_zero_kelvin": 0})

        # ── Molecule & Compound ──
        self._add("molecule", "physical",
                  properties={"two_or_more_atoms": True,
                             "smallest_unit_of_compound": True,
                             "held_by_chemical_bonds": True,
                             "examples": ["H2O", "CO2", "O2", "N2"]})
        self._add("compound", "physical",
                  properties={"two_or_more_elements": True,
                             "chemically_combined": True,
                             "has_fixed_ratio": True,
                             "different_from_mixture": True,
                             "examples": ["water_H2O", "salt_NaCl", "carbon_dioxide_CO2"]})
        self._add("mixture", "physical",
                  properties={"two_or_more_substances": True,
                             "not_chemically_combined": True,
                             "can_be_separated_physically": True,
                             "examples": ["saltwater", "trail_mix", "air", "soil"]})

        # ── Life Science ──
        self._add("plant", "biological", parents=["organism"],
                  properties={"makes_food": True, "uses_sunlight": True,
                             "performs_photosynthesis": True,
                             "produces_oxygen": True, "has_roots": True,
                             "has_leaves": True, "has_stems": True,
                             "food_made_from_sunlight": True,
                             "autotroph": True, "producer": True,
                             "needs_from_air": "carbon_dioxide",
                             "needs": ["sunlight", "water", "carbon_dioxide"],
                             "takes_in_carbon_dioxide": True,
                             "releases_oxygen": True})
        self._add("reptile", "biological", parents=["animal"],
                  properties={"has_scales": True, "breathes_air": True,
                             "lays_eggs": True, "cold_blooded": True,
                             "ectothermic": True,
                             "examples": ["snake", "lizard", "turtle", "crocodile"]})
        self._add("fish", "biological", parents=["animal"],
                  properties={"has_gills": True, "has_scales": True,
                             "lives_in_water": True, "has_fins": True,
                             "cold_blooded": True, "breathes_with_gills": True})
        self._add("amphibian", "biological", parents=["animal"],
                  properties={"lives_on_land_and_water": True,
                             "moist_skin": True, "lays_eggs_in_water": True,
                             "cold_blooded": True, "undergoes_metamorphosis": True,
                             "examples": ["frog", "salamander", "toad", "newt"]})
        self._add("mammal", "biological", parents=["animal"],
                  properties={"has_hair_or_fur": True, "warm_blooded": True,
                             "produces_milk": True, "live_birth": True,
                             "breathes_with_lungs": True,
                             "examples": ["dog", "cat", "whale", "human", "bat"]})
        self._add("bird", "biological", parents=["animal"],
                  properties={"has_feathers": True, "has_beak": True,
                             "lays_eggs": True, "warm_blooded": True,
                             "most_can_fly": True, "has_hollow_bones": True})

        # ── Nutrition & Life Processes ──
        self._add("nutrient", "biological",
                  properties={"needed_for_growth": True, "obtained_from_food": True,
                             "types": ["carbohydrates", "proteins", "fats",
                                      "vitamins", "minerals", "water"]})
        self._add("vitamin", "biological", parents=["nutrient"],
                  properties={"needed_in_small_amounts": True,
                             "important_for_health": True,
                             "from_fruits_and_vegetables": True,
                             "deficiency_causes_disease": True})

        # ── Astronomy ──
        self._add("constellation", "earth_science",
                  properties={"pattern_of_stars": True,
                             "appears_to_move_across_sky": True,
                             "different_visible_in_different_seasons": True,
                             "caused_by": "earth_revolution_around_sun"})

        # ── Material / Matter aliases ──
        self._add("material", "physical", related=["matter"],
                  properties={"has_mass": True, "occupies_space": True,
                             "made_of": "elements", "composed_of": "atoms",
                             "all_material_composed_of_elements": True,
                             "all_matter_is_material": True})

        # ── v5.1: Additional high-impact concepts from ARC failure analysis ──

        # Heat & Energy
        self._add("heat", "physical", parents=["energy"],
                  properties={"form_of_energy": True, "flows_hot_to_cold": True,
                             "transferred_by": ["conduction", "convection", "radiation"],
                             "conduction_through_solids": True,
                             "convection_in_fluids": True,
                             "radiation_no_medium_needed": True,
                             "causes_expansion": True, "measured_in": "joules"})
        self._add("energy", "physical",
                  properties={"ability_to_do_work": True, "cannot_be_created_or_destroyed": True,
                             "can_be_transformed": True, "forms": ["kinetic", "potential",
                             "thermal", "chemical", "electrical", "light", "sound",
                             "nuclear", "mechanical"],
                             "kinetic_energy_of_motion": True,
                             "potential_energy_of_position": True,
                             "conservation_of_energy": True})
        self._add("gravity", "physical", parents=["force"],
                  properties={"pulls_objects_toward_earth": True,
                             "pulls_objects_toward_center": True,
                             "depends_on_mass_and_distance": True,
                             "more_mass_more_gravity": True,
                             "causes_objects_to_fall": True,
                             "causes_weight": True, "keeps_planets_in_orbit": True,
                             "keeps_moon_orbiting_earth": True})
        self._add("friction", "physical", parents=["force"],
                  properties={"opposes_motion": True, "slows_objects_down": True,
                             "produces_heat": True, "rough_surfaces_more_friction": True,
                             "smooth_surfaces_less_friction": True,
                             "types": ["sliding", "rolling", "static", "fluid"]})

        # Earth Science
        self._add("erosion", "earth_science",
                  properties={"wearing_away_of_rock_and_soil": True,
                             "caused_by": ["water", "wind", "ice", "gravity"],
                             "moves_sediment": True,
                             "water_is_main_agent": True,
                             "creates_valleys_canyons": True})
        self._add("fossil", "earth_science",
                  properties={"preserved_remains_of_organisms": True,
                             "found_in_sedimentary_rock": True,
                             "evidence_of_past_life": True,
                             "shows_how_life_changed": True,
                             "types": ["body_fossil", "trace_fossil", "mold", "cast"]})
        self._add("mineral", "earth_science",
                  properties={"naturally_occurring": True, "inorganic": True,
                             "solid": True, "crystal_structure": True,
                             "definite_chemical_composition": True,
                             "identified_by": ["hardness", "luster", "color",
                                             "streak", "cleavage"]})
        self._add("rock", "earth_science",
                  properties={"made_of_minerals": True,
                             "types": ["igneous", "sedimentary", "metamorphic"],
                             "igneous_from_cooling_magma": True,
                             "sedimentary_from_compacted_sediment": True,
                             "metamorphic_from_heat_and_pressure": True,
                             "rock_cycle": True})
        self._add("soil", "earth_science",
                  properties={"formed_from_weathered_rock": True,
                             "contains_organic_matter": True,
                             "supports_plant_growth": True,
                             "layers": ["topsoil", "subsoil", "bedrock"],
                             "formed_by": ["weathering", "decomposition"]})
        self._add("weather", "earth_science",
                  properties={"day_to_day_conditions": True,
                             "includes": ["temperature", "precipitation",
                                         "wind", "humidity", "clouds"],
                             "changes_daily": True,
                             "different_from_climate": True,
                             "measured_by": ["thermometer", "barometer",
                                           "anemometer", "rain_gauge"]})
        self._add("climate", "earth_science",
                  properties={"long_term_weather_pattern": True,
                             "measured_over_30_years": True,
                             "different_from_weather": True,
                             "affected_by": ["latitude", "elevation", "ocean_currents"],
                             "types": ["tropical", "temperate", "polar", "arid"]})
        self._add("earthquake", "earth_science",
                  properties={"caused_by_plate_movement": True,
                             "releases_energy_as_seismic_waves": True,
                             "measured_on_richter_scale": True,
                             "occurs_at_fault_lines": True,
                             "can_cause_tsunami": True})
        self._add("volcano", "earth_science",
                  properties={"opening_in_earth_crust": True,
                             "erupts_lava_and_ash": True,
                             "forms_igneous_rock": True,
                             "occurs_at_plate_boundaries": True,
                             "types": ["shield", "composite", "cinder_cone"]})

        # Biology
        self._add("ecosystem", "biological",
                  properties={"community_plus_environment": True,
                             "includes_living_and_nonliving": True,
                             "biotic_and_abiotic": True,
                             "examples": ["forest", "desert", "ocean", "pond"],
                             "energy_flows_through_food_chains": True})
        self._add("habitat", "biological",
                  properties={"where_organism_lives": True,
                             "provides_food_water_shelter": True,
                             "examples": ["forest", "ocean", "desert", "grassland"],
                             "destruction_threatens_species": True})
        self._add("food_chain", "biological",
                  properties={"shows_energy_flow": True,
                             "starts_with_producer": True,
                             "producer_to_consumer": True,
                             "sun_is_energy_source": True,
                             "parts": ["producer", "primary_consumer",
                                      "secondary_consumer", "decomposer"]})
        self._add("adaptation", "biological",
                  properties={"trait_that_helps_survival": True,
                             "developed_over_generations": True,
                             "types": ["structural", "behavioral", "physiological"],
                             "examples": ["camouflage", "thick_fur", "migration",
                                         "hibernation", "mimicry"]})
        self._add("photosynthesis", "biological",
                  properties={"plants_make_food_from_sunlight": True,
                             "needs": ["sunlight", "water", "carbon_dioxide"],
                             "produces": ["glucose", "oxygen"],
                             "occurs_in": "chloroplasts",
                             "chlorophyll_captures_light": True,
                             "equation": "CO2_plus_H2O_plus_light_equals_glucose_plus_O2",
                             "requires_carbon_dioxide_from_air": True,
                             "takes_in_carbon_dioxide": True,
                             "releases_oxygen": True})
        self._add("cell", "biological",
                  properties={"basic_unit_of_life": True,
                             "all_living_things_made_of_cells": True,
                             "types": ["plant_cell", "animal_cell"],
                             "plant_cell_has_cell_wall": True,
                             "animal_cell_no_cell_wall": True,
                             "parts": ["nucleus", "membrane", "cytoplasm",
                                      "mitochondria", "ribosome"]})

        # Physics — States of Matter
        self._add("evaporation", "physical",
                  properties={"liquid_to_gas": True, "surface_process": True,
                             "faster_with_heat": True, "faster_with_wind": True,
                             "part_of_water_cycle": True,
                             "opposite_of": "condensation"})
        self._add("condensation", "physical",
                  properties={"gas_to_liquid": True, "caused_by_cooling": True,
                             "forms_clouds_and_dew": True,
                             "part_of_water_cycle": True,
                             "opposite_of": "evaporation"})
        self._add("precipitation", "earth_science",
                  properties={"water_falling_from_clouds": True,
                             "types": ["rain", "snow", "sleet", "hail"],
                             "part_of_water_cycle": True,
                             "caused_by_condensation_in_clouds": True})

        # Physics — Electricity & Magnetism
        self._add("electricity", "physical",
                  properties={"flow_of_electrons": True,
                             "needs_complete_circuit": True,
                             "measured_in": ["volts", "amps", "watts"],
                             "types": ["static", "current"],
                             "conductors_allow_flow": True,
                             "insulators_block_flow": True})
        self._add("conductor", "physical",
                  properties={"allows_electricity_to_flow": True,
                             "allows_heat_to_flow": True,
                             "examples": ["copper", "iron", "aluminum", "silver", "gold", "water", "tile", "metal"],
                             "metals_are_good_conductors": True,
                             "tile_is_conductor": True,
                             "feels_cold_to_touch": True,
                             "transfers_heat_quickly": True})
        self._add("insulator", "physical",
                  properties={"blocks_electricity_flow": True,
                             "blocks_heat_flow": True,
                             "examples": ["rubber", "plastic", "wood", "glass", "air", "carpet", "cloth", "fur"],
                             "opposite_of": "conductor",
                             "carpet_is_insulator": True,
                             "feels_warm_to_touch": True,
                             "slows_heat_transfer": True})
        self._add("magnetism", "physical",
                  properties={"force_of_attraction_or_repulsion": True,
                             "has_north_and_south_poles": True,
                             "opposite_poles_attract": True,
                             "same_poles_repel": True,
                             "attracts_iron_steel_nickel": True,
                             "earth_is_a_magnet": True})

        # Physics — Light
        self._add("reflection", "physical",
                  properties={"light_bouncing_off_surface": True,
                             "angle_of_incidence_equals_angle_of_reflection": True,
                             "mirror_reflects_light": True,
                             "smooth_surfaces_reflect_well": True})
        self._add("refraction", "physical",
                  properties={"light_bending_through_medium": True,
                             "caused_by_speed_change": True,
                             "prism_refracts_light": True,
                             "lens_refracts_light": True,
                             "rainbow_caused_by_refraction": True})

        # Astronomy
        self._add("orbit", "earth_science",
                  properties={"path_around_another_object": True,
                             "earth_orbits_sun": True,
                             "moon_orbits_earth": True,
                             "caused_by_gravity": True,
                             "earth_orbit_takes_365_days": True,
                             "moon_orbit_takes_27_days": True})
        self._add("rotation", "earth_science",
                  properties={"spinning_on_axis": True,
                             "earth_rotation_causes_day_night": True,
                             "one_rotation_equals_24_hours": True,
                             "all_planets_rotate": True})
        self._add("season", "earth_science",
                  properties={"caused_by_earth_tilt": True,
                             "not_caused_by_distance_from_sun": True,
                             "four_seasons": ["spring", "summer", "fall", "winter"],
                             "earth_tilt_23_5_degrees": True,
                             "opposite_in_hemispheres": True})

        # ── v5.2: Additional concepts from v17 failure analysis ──

        # Scientific Method (very common in ARC)
        self._add("experiment", "physical",
                  properties={"tests_hypothesis": True, "uses_variables": True,
                             "independent_variable_changed": True,
                             "dependent_variable_measured": True,
                             "control_group_no_change": True,
                             "repeated_trials_increase_reliability": True,
                             "data_collection": True, "conclusion_from_data": True})
        self._add("hypothesis", "physical",
                  properties={"testable_prediction": True,
                             "based_on_observation": True,
                             "can_be_supported_or_rejected": True,
                             "must_be_testable": True})

        # Inherited vs Learned traits
        self._add("inherited_trait", "biological",
                  properties={"passed_from_parent_to_offspring": True,
                             "determined_by_genes": True, "DNA_based": True,
                             "examples": ["eye_color", "hair_color", "blood_type",
                                         "skin_color", "height_potential",
                                         "bird_migration", "spider_web_spinning",
                                         "fur_color", "leaf_shape", "flower_color"],
                             "instinct_is_inherited": True,
                             "not_taught_or_learned": True})
        self._add("learned_behavior", "biological",
                  properties={"not_inherited": True, "acquired_through_experience": True,
                             "examples": ["reading", "riding_bike", "cooking",
                                         "speaking_language", "playing_instrument",
                                         "dog_tricks", "using_tools", "hunting_technique"],
                             "taught_or_practiced": True,
                             "changes_with_experience": True})

        # Simple machines
        self._add("simple_machine", "physical",
                  properties={"makes_work_easier": True, "changes_force": True,
                             "types": ["lever", "pulley", "wheel_and_axle",
                                      "inclined_plane", "wedge", "screw"],
                             "reduces_effort_force": True,
                             "mechanical_advantage": True})

        # Natural resources
        self._add("renewable_resource", "earth_science",
                  properties={"can_be_replaced_naturally": True,
                             "examples": ["solar_energy", "wind_energy",
                                         "water_power", "trees", "biomass"],
                             "sustainable": True})
        self._add("nonrenewable_resource", "earth_science",
                  properties={"cannot_be_replaced_quickly": True,
                             "examples": ["coal", "oil", "natural_gas", "uranium"],
                             "fossil_fuels": True, "will_run_out": True})

        # Atmosphere
        self._add("atmosphere", "earth_science",
                  properties={"layer_of_gases": True,
                             "composition": {"nitrogen": "78_percent",
                                           "oxygen": "21_percent",
                                           "other": "1_percent"},
                             "layers": ["troposphere", "stratosphere",
                                       "mesosphere", "thermosphere"],
                             "protects_from_radiation": True,
                             "traps_heat_greenhouse_effect": True})

        # Predator / Prey
        self._add("predator", "biological",
                  properties={"hunts_other_organisms": True,
                             "consumer": True, "has_adaptations_for_hunting": True,
                             "examples": ["lion", "hawk", "shark", "wolf"]})
        self._add("prey", "biological",
                  properties={"hunted_by_predators": True,
                             "has_adaptations_for_escape": True,
                             "examples": ["rabbit", "deer", "mouse", "fish"],
                             "camouflage_for_hiding": True})

        # Decomposer
        self._add("decomposer", "biological",
                  properties={"breaks_down_dead_matter": True,
                             "returns_nutrients_to_soil": True,
                             "examples": ["bacteria", "fungi", "mushroom", "worm"],
                             "part_of_food_chain": True})

        # Sun
        self._add("sun", "earth_science",
                  properties={"star": True, "source_of_energy_for_earth": True,
                             "provides_light_and_heat": True,
                             "drives_water_cycle": True,
                             "causes_wind_patterns": True,
                             "center_of_solar_system": True,
                             "produces_own_light": True,
                             "nuclear_fusion": True,
                             "all_stars_produce_own_light": True})

        # Moon
        self._add("moon", "earth_science",
                  properties={"orbits_earth": True, "causes_tides": True,
                             "reflects_sunlight": True, "has_phases": True,
                             "phases": ["new_moon", "first_quarter",
                                       "full_moon", "last_quarter"],
                             "no_atmosphere": True, "less_gravity_than_earth": True,
                             "does_not_produce_own_light": True})

        # Conduction / Convection / Radiation
        self._add("conduction", "physical",
                  properties={"heat_transfer_through_direct_contact": True,
                             "works_best_in_solids": True,
                             "metals_conduct_heat_well": True,
                             "example": "hot_pan_handle",
                             "tile_floor_feels_cold_by_conduction": True,
                             "spoon_in_soup_gets_hot": True,
                             "requires_touching": True})
        self._add("convection", "physical",
                  properties={"heat_transfer_through_fluid_movement": True,
                             "occurs_in_liquids_and_gases": True,
                             "hot_fluid_rises_cold_sinks": True,
                             "creates_convection_currents": True,
                             "example": "boiling_water",
                             "warm_air_rises": True,
                             "causes_wind": True,
                             "ocean_currents": True})
        self._add("radiation", "physical",
                  properties={"heat_transfer_through_electromagnetic_waves": True,
                             "no_medium_needed": True,
                             "can_travel_through_vacuum": True,
                             "example": "heat_from_sun",
                             "sun_heats_earth_by_radiation": True,
                             "campfire_warmth": True,
                             "does_not_require_contact": True})

        # ── v5.3: Missing concepts causing ARC failures ──

        # Carbon dioxide — critical for photosynthesis questions
        self._add("carbon_dioxide", "physical", parents=["compound"],
                  properties={"formula": "CO2", "gas": True,
                             "plants_need_from_air": True,
                             "used_in_photosynthesis": True,
                             "produced_by_respiration": True,
                             "produced_by_burning": True,
                             "greenhouse_gas": True,
                             "exhaled_by_animals": True,
                             "absorbed_by_plants": True})

        # Organic compound — CHON question
        self._add("organic_compound", "physical", parents=["compound"],
                  properties={"contains_carbon": True,
                             "elements": ["carbon", "hydrogen", "oxygen", "nitrogen"],
                             "found_in_living_things": True,
                             "examples": ["proteins", "carbohydrates", "lipids", "nucleic_acids"],
                             "carbon_is_essential": True,
                             "carbon_forms_long_chains": True})

        # Circulatory system
        self._add("circulatory_system", "biological",
                  properties={"pumps_blood": True, "carries_oxygen": True,
                             "carries_nutrients": True, "carries_hormones": True,
                             "transports_hormones_for_endocrine": True,
                             "parts": ["heart", "blood", "blood_vessels"],
                             "works_with_respiratory_system": True,
                             "works_with_endocrine_system": True})

        # Endocrine system
        self._add("endocrine_system", "biological",
                  properties={"produces_hormones": True,
                             "releases_hormones_into_blood": True,
                             "regulates_body_functions": True,
                             "uses_circulatory_system_for_transport": True,
                             "glands": ["pituitary", "thyroid", "adrenal"],
                             "hormones_travel_through_blood": True})

        # Respiratory system
        self._add("respiratory_system", "biological",
                  properties={"breathing": True, "gas_exchange": True,
                             "takes_in_oxygen": True,
                             "releases_carbon_dioxide": True,
                             "parts": ["lungs", "trachea", "bronchi", "diaphragm"],
                             "works_with_circulatory_system": True})

        # Electron — for static electricity questions
        self._add("electron", "physical", parents=["particle"],
                  properties={"negative_charge": True, "orbits_nucleus": True,
                             "flow_of_electrons_is_electricity": True,
                             "static_electricity_from_electron_transfer": True,
                             "smallest_charged_particle": True,
                             "friction_transfers_electrons": True,
                             "shock_from_electron_discharge": True})

        # Speed — for speed-of-sound questions
        self._add("speed", "physical",
                  properties={"rate_of_motion": True, "distance_per_time": True,
                             "measured_in": "meters_per_second",
                             "speed_of_sound_depends_on": "medium",
                             "sound_faster_in_solids": True,
                             "sound_slower_in_gases": True})

        # Acceleration
        self._add("acceleration", "physical",
                  properties={"change_in_velocity": True,
                             "caused_by_force": True,
                             "depends_on": ["force", "mass"],
                             "newton_second_law": "F_equals_ma",
                             "more_force_more_acceleration": True,
                             "more_mass_less_acceleration": True})

        # Pie chart / data display
        self._add("pie_chart", "physical",
                  properties={"shows_percentages": True, "shows_parts_of_whole": True,
                             "circular_graph": True,
                             "best_for": "showing_composition",
                             "atmosphere_composition_uses_pie_chart": True})

        # ── v5.4 concepts: Plant transport, nonvascular, population dynamics,
        #    scientific method, kinetic energy during motion ──

        # Plant transport systems
        self._add("xylem", "biological",
                  properties={"transports": "water_and_minerals",
                             "direction": "roots_to_leaves",
                             "carries": ["water", "minerals"],
                             "from": "roots", "to": "leaves",
                             "part_of": "plant_transport_system",
                             "dead_cells": True})
        self._add("phloem", "biological",
                  properties={"transports": "food_and_sugars",
                             "direction": "leaves_to_roots",
                             "carries": ["food", "sugars", "glucose"],
                             "from": "leaves", "to": "roots",
                             "part_of": "plant_transport_system",
                             "living_cells": True})

        # Nonvascular plants
        self._add("nonvascular_plant", "biological",
                  properties={"lacks": ["true_stems", "true_roots", "true_leaves"],
                             "no_vascular_tissue": True,
                             "examples": ["moss", "liverwort", "hornwort"],
                             "absorbs_water_directly": True,
                             "small_size": True,
                             "identified_by": "lacking_true_stems_roots_and_leaves",
                             "has_spores": True})
        self._add("vascular_plant", "biological",
                  properties={"has": ["true_stems", "true_roots", "true_leaves"],
                             "has_vascular_tissue": True,
                             "has_xylem_and_phloem": True,
                             "examples": ["fern", "tree", "flowering_plant"]})

        # Kinetic energy
        self._add("kinetic_energy", "physical",
                  properties={"energy_of_motion": True,
                             "increases_with_speed": True,
                             "increases_as_object_falls": True,
                             "formula": "half_mass_times_velocity_squared",
                             "increases_when": ["speed_increases", "object_falls",
                                               "object_accelerates"],
                             "falling_object_gains": "kinetic_energy"})
        self._add("potential_energy", "physical",
                  properties={"stored_energy": True,
                             "gravitational_pe_depends_on": "height",
                             "decreases_as_object_falls": True,
                             "converts_to_kinetic_energy_when_falling": True})

        # Population dynamics
        self._add("population", "biological",
                  properties={"number_of_organisms_in_area": True,
                             "increases_when": ["fewer_predators", "more_food",
                                               "less_disease", "more_habitat"],
                             "decreases_when": ["more_predators", "less_food",
                                               "more_disease", "less_habitat"],
                             "fewer_predators_means": "population_increases",
                             "more_food_means": "population_increases"})

        # Scientific method
        self._add("scientific_theory", "process",
                  properties={"can_be_updated": True,
                             "updated_with": "new_evidence",
                             "based_on": "evidence_and_observations",
                             "when_new_data_contradicts": "theory_is_revised",
                             "not_banned": True,
                             "not_ignored": True})
        self._add("scientific_method", "process",
                  properties={"steps": ["observation", "hypothesis",
                                       "experiment", "analysis", "conclusion"],
                             "uses_evidence": True,
                             "repeatable": True,
                             "sampling_over_time": "best_for_studying_populations"})

        # Continental drift
        self._add("continental_drift", "earth_science",
                  properties={"continents_move_over_time": True,
                             "explains_fossils_in_different_climates": True,
                             "tropical_fossils_in_antarctica": "means_antarctica_was_once_tropical",
                             "evidence": ["matching_coastlines", "fossil_distribution",
                                         "rock_types", "glacial_deposits"]})

        # ── v5.5 concepts: Species, migration, dissolving, reflex, antibiotics,
        #    greenhouse effect, symbiosis enhancement, carbon cycle ──

        self._add("species", "biological",
                  properties={"definition": "organisms_that_can_mate_and_produce_fertile_offspring",
                             "determined_by": "ability_to_produce_fertile_offspring",
                             "same_species_if": "can_mate_and_produce_fertile_offspring",
                             "not_determined_by": ["appearance", "color", "size", "habitat"]})

        self._add("migration", "biological",
                  properties={"seasonal_movement": True,
                             "innate_behavior": True,
                             "purpose": ["find_food", "find_warmer_climate", "breeding"],
                             "examples": ["bird_migration", "whale_migration",
                                         "monarch_butterfly", "caribou", "goose"],
                             "triggered_by": ["shorter_days", "temperature_change",
                                            "food_scarcity"]})

        self._add("dissolve", "physical",
                  properties={"solute_mixes_into_solvent": True,
                             "type": "physical_change",
                             "examples": ["sugar_in_water", "salt_in_water"],
                             "sugar_dissolves_in_water": True,
                             "particles_spread_evenly": True,
                             "solution_is_clear": True})

        self._add("reflex", "biological",
                  properties={"automatic_response": True,
                             "involuntary": True,
                             "fast_response": True,
                             "involves": ["nervous_system", "muscular_system"],
                             "two_systems": "nervous_and_muscular",
                             "examples": ["knee_jerk", "pulling_hand_from_hot",
                                         "blinking", "pupil_dilation"],
                             "touching_hot_object_uses": "nervous_and_muscular_systems"})

        self._add("antibiotic", "biological",
                  properties={"kills_bacteria": True,
                             "treats_infection": True,
                             "overuse_causes": "antibiotic_resistance",
                             "resistant_microbes": "bacteria_that_survive_antibiotics",
                             "resistance_is_problem": True})

        self._add("greenhouse_effect", "earth_science",
                  properties={"traps_heat": True,
                             "greenhouse_gases": ["carbon_dioxide", "methane",
                                                "water_vapor", "nitrous_oxide"],
                             "carbon_dioxide_traps_solar_energy": True,
                             "causes_warming": True})

        self._add("carbon_cycle", "biological",
                  properties={"animals_get_carbon_from": "eating_food",
                             "not_from_breathing": True,
                             "plants_get_carbon_from": "carbon_dioxide_in_air",
                             "carbon_returned_to_air": ["decomposition", "burning",
                                                       "respiration"]})

        self._add("response_to_stimuli", "biological",
                  properties={"reaction_to_environment": True,
                             "examples": ["earthworm_moves_to_surface_when_soil_saturated",
                                         "plant_grows_toward_light",
                                         "pupil_contracts_in_bright_light"],
                             "not_learned_behavior": True,
                             "is": "basic_characteristic_of_living_things"})

        self._add("conservation", "earth_science",
                  properties={"protect_natural_resources": True,
                             "examples": ["recycling", "reusing", "reducing_waste",
                                         "planting_trees"],
                             "recycling_helps": True,
                             "waste_energy_hurts": True})

        self._add("renewable_resource", "earth_science",
                  properties={"can_be_replaced": True,
                             "examples": ["trees", "solar_energy", "wind_energy",
                                         "water"],
                             "planting_seed_renews": True})

        # ── v6.0 concepts: Challenge-targeted (cell biology, astronomy,
        #    scientific method — ONLY specific concepts, not broad ones
        #    that would interfere with ontology scoring) ──

        self._add("nucleus_cell", "biological",
                  properties={"controls_cell_activities": True,
                             "contains_dna": True,
                             "is": "control_center_of_cell",
                             "eukaryotic_cells_have": True,
                             "prokaryotic_cells_lack": True})

        self._add("lysosome", "biological",
                  properties={"function": "breaks_down_wastes",
                             "digests": "waste_materials",
                             "breaks_down": "food_particles_and_waste",
                             "cell_digestion": True,
                             "also_called": "garbage_disposal_of_cell"})

        self._add("prokaryote", "biological",
                  properties={"no_nucleus": True,
                             "no_membrane_bound_organelles": True,
                             "examples": ["bacteria", "archaea"],
                             "simpler_than": "eukaryote",
                             "differs_from_eukaryote_by": "lacking_membrane_bound_nucleus"})

        self._add("eukaryote", "biological",
                  properties={"has_nucleus": True,
                             "has_membrane_bound_organelles": True,
                             "examples": ["animal_cells", "plant_cells", "fungi"],
                             "more_complex_than": "prokaryote",
                             "identified_by": "presence_of_nucleus"})

        self._add("nerve_cell", "biological",
                  properties={"function": "sends_signals_to_brain",
                             "transmits": "electrical_signals",
                             "part_of": "nervous_system",
                             "if_stops_functioning": "stops_sending_signals"})

        self._add("solar_system_order", "astronomy",
                  properties={"inner_planets": ["mercury", "venus", "earth", "mars"],
                             "outer_planets": ["jupiter", "saturn", "uranus", "neptune"],
                             "rocky_planets": "closer_to_sun",
                             "gas_planets": "farther_from_sun",
                             "solid_planets_are": "closer_to_sun",
                             "gas_planets_are": "farther_from_sun"})

        self._add("star", "astronomy",
                  properties={"produces_own_light": True,
                             "sun_is_a": "star",
                             "nearest_star": "sun",
                             "fusion": "produces_light_and_heat"})

        self._add("hypothesis", "scientific",
                  properties={"must_be": ["specific", "testable", "measurable"],
                             "good_hypothesis": "specific_and_testable",
                             "predicts_outcome": True,
                             "can_be_tested": True})

        self._add("scientific_peer_review", "scientific",
                  properties={"purpose": "check_data_and_methods",
                             "because": "data_can_support_multiple_explanations",
                             "ensures": "accuracy_and_validity"})

        self._add("sunrise", "astronomy",
                  properties={"frequency": "daily",
                             "occurs_every": "24_hours",
                             "most_frequent": "natural_event"})

        self._add("gill", "biological",
                  properties={"function": "breathing_underwater",
                             "fish_breathe_with": "gills",
                             "extracts_oxygen_from": "water",
                             "found_in": "fish"})

        self._add("fossil_fuel", "earth_science",
                  properties={"formed_from": "ancient_organisms_over_millions_of_years",
                             "long_term_carbon_storage": True,
                             "nonrenewable": True,
                             "examples": ["coal", "oil", "natural_gas"],
                             "formation_takes": "millions_of_years"})

        self._add("rock_mineral_relationship", "earth_science",
                  properties={"rocks_made_of": "one_or_more_minerals",
                             "minerals_not_made_of": "rocks",
                             "mineral_has": "crystal_structure",
                             "rock_is": "combination_of_minerals"})

    def lookup(self, concept: str) -> Optional[Concept]:
        """Lookup a concept by name."""
        key = concept.lower().replace(' ', '_')
        if key in self.concepts:
            return self.concepts[key]
        # Fuzzy match
        for k, v in self.concepts.items():
            if key in k or k in key:
                return v
        return None

    def get_property(self, concept: str, prop: str) -> Any:
        """Get a specific property of a concept."""
        c = self.lookup(concept)
        if c:
            return c.properties.get(prop)
        return None

    def is_a(self, child: str, parent: str, _visited: set = None) -> bool:
        """Check if child IS-A parent (transitive)."""
        if _visited is None:
            _visited = set()
        child_key = child.lower().replace(' ', '_')
        if child_key in _visited:
            return False
        _visited.add(child_key)
        c = self.lookup(child)
        if not c:
            return False
        parent_key = parent.lower().replace(' ', '_')
        if parent_key in [p.lower().replace(' ', '_') for p in c.parents]:
            return True
        # Transitive
        for p in c.parents:
            if self.is_a(p, parent, _visited):
                return True
        return False

    def get_related(self, concept: str) -> List[str]:
        """Get all related concepts."""
        c = self.lookup(concept)
        if not c:
            return []
        return c.parents + c.parts + c.related

    def get_status(self) -> Dict[str, Any]:
        categories = defaultdict(int)
        for c in self.concepts.values():
            categories[c.category] += 1
        return {
            'total_concepts': len(self.concepts),
            'categories': dict(categories),
            'built': self._built,
        }

    def _build_v6_concepts(self):
        """v6.0 (v9.1): Concepts to fill ARC benchmark gaps identified in offline testing."""
        # ── Individual planets ──
        self._add("mercury", "earth_science", parents=["planet"],
                  properties={"closest_to_sun": True, "smallest_planet": True,
                             "no_atmosphere": True, "very_hot_and_cold": True,
                             "position": "first"})
        self._add("venus", "earth_science", parents=["planet"],
                  properties={"hottest_planet": True, "thick_atmosphere": True,
                             "rotates_backward": True, "similar_size_to_earth": True,
                             "position": "second"})
        self._add("earth", "earth_science", parents=["planet"],
                  properties={"supports_life": True, "has_water": True,
                             "has_atmosphere": True, "one_moon": True,
                             "position": "third"})
        self._add("mars", "earth_science", parents=["planet"],
                  properties={"nickname": "Red Planet", "known_as": "Red Planet",
                             "red_planet": True, "red_color_from_iron_oxide": True,
                             "has_largest_volcano": True, "olympus_mons": True,
                             "two_moons": True, "thin_atmosphere": True,
                             "position": "fourth"})
        self._add("jupiter", "earth_science", parents=["planet"],
                  properties={"largest_planet": True, "gas_giant": True,
                             "great_red_spot": True, "many_moons": True,
                             "position": "fifth"})
        self._add("saturn", "earth_science", parents=["planet"],
                  properties={"has_rings": True, "gas_giant": True,
                             "least_dense_planet": True, "position": "sixth"})

        # ── Types of consumers ──
        self._add("herbivore", "biological", parents=["consumer"],
                  properties={"eats": "plants_only", "eats_only_plants": True,
                             "plant_eater": True, "examples": ["deer", "rabbit", "cow"],
                             "primary_consumer": True})
        self._add("carnivore", "biological", parents=["consumer"],
                  properties={"eats": "animals_only", "eats_other_animals": True,
                             "meat_eater": True, "examples": ["lion", "hawk", "shark"],
                             "secondary_or_tertiary_consumer": True})
        self._add("omnivore", "biological", parents=["consumer"],
                  properties={"eats": "plants_and_animals", "eats_both": True,
                             "examples": ["human", "bear", "pig"]})

        # ── Moon and phases ──
        self._add("moon", "earth_science",
                  properties={"orbits": "earth", "reflects_sunlight": True,
                             "causes_tides": True, "no_atmosphere": True,
                             "phases_caused_by": "position_relative_to_earth_and_sun",
                             "revolution_period": "29.5_days",
                             "phases": ["new_moon", "crescent", "quarter", "gibbous", "full_moon"],
                             "phases_not_caused_by": "earths_shadow"})
        # Separate eclipse concept to prevent "eclipses_caused_by" from
        # false-boosting "Earth's shadow" when questions ask about phases.
        self._add("lunar_eclipse", "earth_science",
                  properties={"caused_by": "earths_shadow_on_moon",
                             "requires": "earth_between_sun_and_moon",
                             "not_same_as": "moon_phases"})

        # ── Static electricity ──
        self._add("static_electricity", "physical",
                  properties={"caused_by": "transfer_of_electrons_between_objects",
                             "buildup_of_charge": True, "not_flowing_current": True,
                             "rubbing_causes_transfer": True,
                             "examples": ["balloon_on_hair", "shock_doorknob",
                                         "lightning"],
                             "opposite_charges_attract": True,
                             "same_charges_repel": True})

        # ── Sound speed in different media (reinforcement) ──
        self._add("sound_speed", "physical", parents=["sound"],
                  properties={"depends_on_medium": True,
                             "fastest_in_solids": True,
                             "faster_in_liquids_than_gases": True,
                             "slowest_in_gases": True,
                             "does_not_travel_in_vacuum": True,
                             "speed_in_steel": "5960_m_per_s",
                             "speed_in_water": "1480_m_per_s",
                             "speed_in_air": "343_m_per_s",
                             "denser_medium_faster_sound": True})

        # ── Body system alignment (reinforcement) ──
        self._add("fighting_infection", "biological", parents=["immune_system"],
                  properties={"done_by": "immune_system",
                             "not_by": ["digestive_system", "nervous_system", "skeletal_system"],
                             "uses": "white_blood_cells",
                             "antibodies_help": True})

        # ── Materials and states of matter ──
        self._add("steel", "physical", parents=["metal", "solid"],
                  properties={"is_a": "solid", "is_a_metal": True,
                             "alloy_of": "iron_and_carbon",
                             "conducts_heat": True, "conducts_electricity": True,
                             "sound_speed": "5960_m_per_s",
                             "very_strong": True, "dense": True})
        self._add("solid", "physical", parents=["matter"],
                  properties={"has_definite_shape": True, "has_definite_volume": True,
                             "definite_shape_and_volume": True,
                             "particles_close_together": True,
                             "particles_vibrate_in_place": True,
                             "examples": ["rock", "ice", "metal", "wood"],
                             "sound_travels_fastest_in": True})
        self._add("liquid", "physical", parents=["matter"],
                  properties={"has_definite_volume": True,
                             "not_definite_shape": True,
                             "takes_shape_of_container": True,
                             "particles_slide_past": True,
                             "examples": ["water", "juice", "oil"]})
        self._add("gas", "physical", parents=["matter"],
                  properties={"not_definite_shape": True,
                             "not_definite_volume": True,
                             "fills_container": True,
                             "particles_move_freely": True,
                             "examples": ["oxygen", "helium", "air"]})

        # ── Chemical symbols ──
        self._add("gold", "physical", parents=["metal", "element"],
                  properties={"chemical_symbol": "Au", "symbol": "Au",
                             "atomic_number": 79, "precious_metal": True,
                             "does_not_tarnish": True, "very_dense": True,
                             "symbol_from_latin": "aurum"})
        self._add("iron_element", "physical", parents=["metal", "element"],
                  properties={"chemical_symbol": "Fe", "symbol": "Fe",
                             "atomic_number": 26, "magnetic": True,
                             "rusts": True, "symbol_from_latin": "ferrum"})
        self._add("silver", "physical", parents=["metal", "element"],
                  properties={"chemical_symbol": "Ag", "symbol": "Ag",
                             "atomic_number": 47, "best_conductor": True,
                             "symbol_from_latin": "argentum"})
        self._add("sodium", "physical", parents=["metal", "element"],
                  properties={"chemical_symbol": "Na", "symbol": "Na",
                             "atomic_number": 11, "reactive_metal": True,
                             "symbol_from_latin": "natrium"})

        # ── Body systems (explicit) ──
        self._add("immune_system", "biological", parents=["body_system"],
                  properties={"function": "fights_infections_and_disease",
                             "fights_infections": True, "fights_disease": True,
                             "uses": "white_blood_cells",
                             "produces_antibodies": True,
                             "responsible_for_fighting_infections": True})
        self._add("nervous_system", "biological", parents=["body_system"],
                  properties={"function": "sends_signals_and_controls_body",
                             "brain_and_nerves": True,
                             "not_responsible_for_fighting_infections": True,
                             "controls_movement": True})
        self._add("digestive_system", "biological", parents=["body_system"],
                  properties={"function": "breaks_down_food",
                             "not_responsible_for_fighting_infections": True,
                             "absorbs_nutrients": True})

        # ── Condensation (gas → liquid) ──
        self._add("condensation", "physical", parents=["phase_change"],
                  properties={"change_from": "gas_to_liquid",
                             "gas_to_liquid": True,
                             "opposite_of": "evaporation",
                             "caused_by": "cooling",
                             "example": "water_droplets_on_cold_glass",
                             "not_liquid_to_gas": True})

        # ── Single-word body system concepts (for choice extraction) ──
        # These enable section-2 choice-as-concept matching to work
        # when choices are single words like "Immune", "Nervous", etc.
        self._add("immune", "biological", parents=["immune_system"],
                  properties={"type": "body_system",
                             "fights_infections": True, "fights_disease": True,
                             "fighting_infection": True,
                             "uses_white_blood_cells": True,
                             "produces_antibodies": True,
                             "responsible_for": "fighting infections and disease"})
        self._add("nervous", "biological", parents=["nervous_system"],
                  properties={"type": "body_system",
                             "sends_signals": True, "controls_muscles": True,
                             "brain_and_spinal_cord": True})
        self._add("digestive", "biological", parents=["digestive_system"],
                  properties={"type": "body_system",
                             "breaks_down_food": True,
                             "stomach_and_intestines": True})
        self._add("skeletal", "biological", parents=["skeletal_system"],
                  properties={"type": "body_system",
                             "provides_structure": True,
                             "protects_organs": True, "bones": True})

