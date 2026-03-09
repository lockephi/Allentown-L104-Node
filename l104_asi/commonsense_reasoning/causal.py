"""Layer 2: Causal Reasoning Engine — If-Then Cause-Effect Chains."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .constants import PHI, GOD_CODE, VOID_CONSTANT

logger = logging.getLogger(__name__)


@dataclass
class CausalRule:
    """A cause-effect rule."""
    condition: str
    effect: str
    domain: str
    confidence: float = 0.9
    keywords: List[str] = field(default_factory=list)


class CausalReasoningEngine:
    """If-then causal rules for commonsense reasoning.

    Contains hundreds of everyday causal rules covering physics,
    biology, earth science, and daily life.
    """

    def __init__(self):
        self.rules: List[CausalRule] = []
        self._built = False

    def build(self):
        """Build the causal rule base."""
        if self._built:
            return

        # ── Physics Rules ──
        self._add_rules([
            ("object is dropped", "falls due to gravity", "physics", ["drop", "fall", "gravity"]),
            ("heat is added to solid", "solid melts to liquid", "physics", ["heat", "melt", "solid"]),
            ("heat is added to liquid", "liquid evaporates to gas", "physics", ["heat", "evaporate", "boil"]),
            ("heat is removed from liquid", "liquid freezes to solid", "physics", ["cool", "freeze", "cold"]),
            ("heat is removed from gas", "gas condenses to liquid", "physics", ["cool", "condense"]),
            ("metal is heated", "metal expands", "physics", ["heat", "expand", "metal"]),
            ("metal is cooled", "metal contracts", "physics", ["cool", "contract", "metal"]),
            ("water is cooled below 0°C", "water freezes to ice", "physics", ["water", "freeze", "ice"]),
            ("water is heated above 100°C", "water boils to steam", "physics", ["water", "boil", "steam"]),
            ("ice is warmer than 0°C", "ice melts to water", "physics", ["ice", "melt", "warm"]),
            ("force is applied to object", "object accelerates", "physics", ["force", "accelerate", "push"]),
            ("no force acts on moving object", "object continues moving (Newton's 1st)", "physics", ["inertia", "motion"]),
            ("friction acts on moving object", "object slows down", "physics", ["friction", "slow"]),
            ("surfaces rub together", "heat is produced by friction", "physics", ["rub", "friction", "heat"]),
            ("light hits opaque object", "shadow is formed", "physics", ["light", "shadow", "opaque"]),
            ("light enters denser medium", "light bends (refracts)", "physics", ["light", "bend", "refract"]),
            ("light hits mirror", "light reflects", "physics", ["mirror", "reflect"]),
            ("electric circuit is complete", "current flows", "physics", ["circuit", "current", "electricity"]),
            ("electric circuit is broken", "current stops", "physics", ["circuit", "break", "switch"]),
            ("magnet approaches iron", "iron is attracted", "physics", ["magnet", "iron", "attract"]),
            ("north poles of magnets meet", "magnets repel", "physics", ["north", "repel", "magnet"]),
            ("opposite poles of magnets meet", "magnets attract", "physics", ["opposite", "attract", "magnet"]),
            ("sound travels through vacuum", "no sound is heard", "physics", ["sound", "vacuum"]),
            ("speed of sound depends on medium", "sound speed depends on type of material it travels through", "physics", ["speed", "sound", "depends", "material", "medium", "type"]),
            ("object vibrates", "sound is produced", "physics", ["vibrate", "sound"]),
            ("hot air rises", "convection current forms", "physics", ["hot", "rise", "convection"]),
            ("balloon is inflated", "volume increases, pressure increases", "physics", ["balloon", "inflate"]),
        ])

        # ── Earth Science Rules ──
        self._add_rules([
            ("sun heats ocean", "water evaporates (water cycle begins)", "earth_science", ["sun", "ocean", "evaporate"]),
            ("water vapor rises and cools", "clouds form by condensation", "earth_science", ["vapor", "cool", "cloud"]),
            ("clouds become saturated", "precipitation falls", "earth_science", ["cloud", "rain", "snow", "precipitation"]),
            ("earth rotates on axis", "day and night cycle occurs", "earth_science", ["rotate", "day", "night"]),
            ("earth revolves around sun", "seasons change over a year", "earth_science", ["revolve", "season", "year"]),
            ("earth axis is tilted", "different seasons in hemispheres", "earth_science", ["tilt", "season"]),
            ("moon orbits earth", "tides occur", "earth_science", ["moon", "tide"]),
            ("moon phases change", "the position of the moon relative to earth and sun determines its illuminated phase", "earth_science", ["moon", "phase", "phases", "position", "relative", "earth", "sun"]),
            ("earth shadow falls on moon", "a lunar eclipse not moon phases occurs from earth shadow", "earth_science", ["eclipse", "lunar", "shadow", "earth", "moon"]),
            ("tectonic plates move apart", "rift valley or mid-ocean ridge forms", "earth_science", ["plate", "apart"]),
            ("tectonic plates collide", "mountains or earthquakes occur", "earth_science", ["plate", "collide", "mountain"]),
            ("rock is exposed to wind and water", "erosion occurs", "earth_science", ["wind", "water", "erosion"]),
            ("sediment is compressed over time", "sedimentary rock forms", "earth_science", ["sediment", "compress", "rock"]),
            ("rock is heated and pressurized", "metamorphic rock forms", "earth_science", ["heat", "pressure", "metamorphic"]),
            ("volcano erupts", "lava cools to form igneous rock", "earth_science", ["volcano", "lava", "igneous"]),
            ("greenhouse gases increase", "global temperature rises", "earth_science", ["greenhouse", "temperature", "warming"]),
            ("carbon dioxide traps solar energy", "carbon dioxide is a greenhouse gas that traps heat", "earth_science", ["carbon", "dioxide", "greenhouse", "trap", "solar", "energy", "heat"]),
            ("deforestation occurs", "habitat is lost, CO2 increases", "earth_science", ["deforestation", "habitat"]),
            ("fossil fuels are burned", "CO2 is released", "earth_science", ["fossil", "burn", "CO2"]),
        ])

        # ── Life Science Rules ──
        self._add_rules([
            ("plant receives sunlight and water", "photosynthesis produces glucose and oxygen", "biology", ["plant", "sunlight", "photosynthesis"]),
            ("animal eats food", "cellular respiration produces energy", "biology", ["eat", "energy", "respiration"]),
            ("decomposers role in ecosystem", "decomposers break down dead organisms and recycle nutrients not produce food", "biology", ["decomposer", "decomposers", "dead", "break", "recycle", "nutrient", "ecosystem", "role"]),
            # v5.5: species definition
            ("two organisms are same species", "same species organisms can mate and produce fertile offspring", "biology", ["species", "same", "mate", "fertile", "offspring", "reproduce"]),
            # v5.5: migration
            ("animal migrates in winter", "migration is an innate behavior for finding food or warmth", "biology", ["migrate", "winter", "food", "warm", "goose", "bird", "innate"]),
            # v5.5: dissolving
            ("sugar is mixed into water", "sugar dissolves in water forming a solution", "physics", ["sugar", "water", "dissolve", "mix", "solution"]),
            # v5.5: reflex
            ("person touches hot object", "nervous system and muscular system work together for reflex", "biology", ["touch", "hot", "reflex", "nervous", "muscular", "hand", "pull"]),
            # v5.5: plant life cycle start
            ("most plants begin life cycle", "most plants begin their life cycle as seeds", "biology", ["plant", "begin", "life", "cycle", "seed"]),
            # v5.5: carbon in animals
            ("animals get carbon for bodies", "animals get carbon by eating food", "biology", ["animal", "carbon", "eat", "food", "body"]),
            # v5.5: symbiosis
            ("clownfish and sea anemone relationship", "clownfish and sea anemone have mutualism both benefit", "biology", ["clownfish", "anemone", "mutualism", "benefit"]),
            # v5.5: earthworm response
            ("earthworm moves when soil saturated", "earthworm moving to surface is response to stimuli", "biology", ["earthworm", "soil", "saturated", "surface", "stimuli", "response"]),
            # v5.5: seasons
            ("earth orbiting sun causes", "earth orbiting the sun causes changing of the seasons", "earth_science", ["earth", "orbit", "sun", "season", "change"]),
            ("day and night cycle", "earth rotating on its axis causes day and night", "earth_science", ["day", "night", "earth", "rotate", "axis", "rotation"]),
            # v5.5: conservation
            ("conserve earth natural resources", "recycling helps conserve earth natural resources", "earth_science", ["conserve", "resource", "recycle", "natural"]),
            # v5.5: renewing resource
            ("planting a seed renews resource", "planting a seed from a tree helps renew a resource", "earth_science", ["seed", "plant", "tree", "renew", "resource"]),
            ("organism is well-adapted", "organism survives and reproduces", "biology", ["adapted", "survive"]),
            ("population exceeds resources", "competition increases", "biology", ["population", "resource", "competition"]),
            ("predator population increases", "prey population decreases", "biology", ["predator", "prey"]),
            ("fewer predators in area", "prey population increases because fewer predators means less predation", "biology", ["fewer", "predator", "population", "increase", "rabbit"]),
            ("prey population decreases", "predator population decreases (later)", "biology", ["prey", "predator"]),
            ("environment changes", "natural selection favors adapted organisms", "biology", ["environment", "selection", "adapt"]),
            ("DNA is inherited", "offspring resemble parents", "biology", ["DNA", "inherit", "offspring", "trait"]),
            ("mutation occurs in DNA", "new trait may appear", "biology", ["mutation", "trait"]),
            ("decomposer breaks down dead matter", "nutrients return to soil", "biology", ["decompose", "nutrient", "soil"]),
            ("habitat is destroyed", "species may go extinct", "biology", ["habitat", "destroy", "extinct"]),
            ("plant lacks sunlight", "photosynthesis cannot occur, plant dies", "biology", ["no sunlight", "plant", "die"]),
            ("animal lacks water", "dehydration occurs", "biology", ["water", "dehydrate"]),
            # v5.4: Plant transport
            ("xylem in plants", "xylem carries water and minerals from roots to leaves", "biology", ["xylem", "water", "mineral", "root", "leaves", "plant"]),
            ("phloem in plants", "phloem carries food and sugars from leaves to roots", "biology", ["phloem", "food", "sugar", "leaves", "root", "plant"]),
            ("nonvascular plant identified", "nonvascular plants lack true stems roots and leaves", "biology", ["nonvascular", "stem", "root", "leaves", "moss", "lack"]),
            # v5.4: Continental drift
            ("tropical fossils found in Antarctica", "millions of years ago Antarctica was in a different warmer location", "geology", ["tropical", "fossil", "antarctica", "warm", "continent", "million"]),
            # v5.4: Coral reef + climate change
            ("global temperatures increase in ocean", "organisms in shallow warm habitats like coral reefs are most affected", "biology", ["temperature", "global", "coral", "reef", "shallow", "warm", "affected"]),
        ])

        # ── Technology & Daily Life Rules ──
        self._add_rules([
            ("switch is turned on", "light turns on (circuit completes)", "technology", ["switch", "light", "on"]),
            ("electric current flows through light bulb filament", "light bulb gives off light and heat", "technology", ["current", "light", "bulb", "wire", "filament"]),
            ("battery runs out", "device stops working", "technology", ["battery", "stop"]),
            ("object is placed in water", "buoyancy force acts upward", "physics", ["water", "float", "buoyancy"]),
            ("dense object is placed in water", "object sinks", "physics", ["heavy", "dense", "sink"]),
            ("less dense object is placed in water", "object floats", "physics", ["light", "less dense", "float"]),
            ("metal spoon is put in hot soup", "spoon gets hot (conduction)", "physics", ["spoon", "hot", "conduction"]),
            ("metal object near magnet", "iron nail is attracted to magnet", "physics", ["magnet", "iron", "nail", "attract"]),
            ("water freezes", "water expands increasing in volume", "physics", ["water", "freeze", "expand"]),
            ("moon gravity pulls on ocean", "tides occur on earth caused by moon gravity", "earth_science", ["moon", "tide", "gravity"]),
            ("we live in troposphere", "troposphere is the layer of atmosphere we live in", "earth_science", ["troposphere", "atmosphere", "live"]),
            ("white blood cells detect pathogen", "white blood cells fight infection and disease", "biology", ["white", "blood", "cell", "infection", "fight"]),
            ("bat uses echolocation", "bat uses sound to navigate similar to submarine sonar", "technology", ["bat", "sound", "sonar", "submarine"]),
            ("flat surface at angle", "inclined plane is a simple machine that reduces force", "physics", ["inclined", "plane", "angle", "flat", "surface"]),
            ("ice at room temperature", "ice melts at room temperature", "physics", ["ice", "room", "temperature", "melt"]),
            ("plants need sunlight", "plants make food through photosynthesis using sunlight", "biology", ["plant", "food", "sunlight", "photosynthesis"]),
            # Direct definition rules
            ("gravity is force", "gravity is the force that pulls objects toward earth", "physics", ["gravity", "force", "pull", "earth", "toward"]),
            ("moon gravity", "moon gravity causes the tides on earth", "earth_science", ["moon", "gravity", "tide", "cause", "earth"]),
            ("inclined plane definition", "an inclined plane is a flat surface at an angle", "physics", ["inclined", "plane", "flat", "surface", "angle"]),
        ])

        # ── Extended Physics Rules ──
        self._add_rules([
            ("object is in motion", "object has kinetic energy", "physics", ["motion", "kinetic", "energy", "moving"]),
            ("object is above ground", "object has potential energy", "physics", ["height", "potential", "energy"]),
            ("energy cannot be created", "energy is conserved and transforms between forms", "physics", ["conserve", "energy", "transform"]),
            ("temperature increases", "molecules move faster", "physics", ["temperature", "molecule", "speed", "faster"]),
            ("wire is coiled around nail with current", "electromagnet is created", "physics", ["wire", "coil", "electromagnet", "current"]),
            ("object spins on axis", "rotation occurs", "physics", ["spin", "rotate", "axis"]),
            ("pulley is used", "direction of force changes and effort is reduced", "physics", ["pulley", "force", "machine"]),
            ("lever is used", "force is multiplied by mechanical advantage", "physics", ["lever", "fulcrum", "force"]),
            ("wheel and axle is used", "force is multiplied or direction changed", "physics", ["wheel", "axle", "machine"]),
            ("wedge is used", "object is split apart by concentrating force", "physics", ["wedge", "split", "force"]),
            ("screw is used", "rotational force converts to linear force", "physics", ["screw", "force", "turn"]),
            ("work is done on object", "force moves object over a distance", "physics", ["work", "force", "distance"]),
            ("power is applied", "work is done per unit time", "physics", ["power", "work", "time"]),
            ("pendulum swings", "energy converts between potential and kinetic", "physics", ["pendulum", "swing", "energy"]),
            ("white light hits prism", "light separates into spectrum of colors", "physics", ["prism", "light", "color", "spectrum", "rainbow"]),
        ])

        # ── Extended Earth Science Rules ──
        self._add_rules([
            ("minerals have crystal structure", "minerals are naturally occurring solid with crystal structure", "earth_science", ["mineral", "crystal", "natural"]),
            ("weathering breaks down rock at surface", "rock fragments become sediment through weathering", "earth_science", ["weather", "break", "rock", "surface"]),
            ("rivers carry sediment", "sediment deposits form in rivers and deltas", "earth_science", ["river", "sediment", "deposit"]),
            ("unequal heating of earth surface", "wind patterns form from unequal heating", "earth_science", ["wind", "heat", "surface", "unequal"]),
            ("warm air meets cold air", "weather fronts produce storms and precipitation", "earth_science", ["warm", "cold", "front", "storm"]),
            ("earth has magnetic field", "compass needle points north due to magnetic field", "earth_science", ["compass", "north", "magnetic"]),
            ("fossils are found in rock", "fossils show what organisms lived in the past", "earth_science", ["fossil", "organism", "past", "evidence"]),
            ("layers of rock are stacked", "lower rock layers are older (law of superposition)", "earth_science", ["layer", "rock", "older", "superposition"]),
            ("soil is formed", "soil forms from weathered rock and decomposed organisms", "earth_science", ["soil", "form", "rock", "decompose"]),
            ("water cycle continues", "water evaporates then condenses then precipitates", "earth_science", ["water", "cycle", "evaporate", "condense", "precipitate"]),
            ("ocean currents flow", "ocean currents distribute heat around the globe", "earth_science", ["ocean", "current", "heat", "flow"]),
            ("earthquake occurs", "seismic waves travel through earth from the focus", "earth_science", ["earthquake", "seismic", "wave"]),
            ("earth revolves 365 days", "one year is the time earth takes to orbit the sun", "earth_science", ["year", "orbit", "sun", "365"]),
            ("earth rotates 24 hours", "one day is the time earth takes to rotate once", "earth_science", ["day", "rotate", "24", "hours"]),
        ])

        # ── Extended Life Science Rules ──
        self._add_rules([
            ("animal has thick fur", "thick fur insulates and keeps animal warm", "biology", ["fur", "warm", "insulate", "cold"]),
            ("plant has deep roots", "deep roots help plant reach water in dry conditions", "biology", ["root", "water", "deep", "dry"]),
            ("caterpillar becomes butterfly", "metamorphosis is a complete change in body form", "biology", ["metamorphosis", "caterpillar", "butterfly", "change"]),
            ("organism lives in desert", "desert adaptations include conserving water", "biology", ["desert", "water", "conserve", "adapt"]),
            ("organism lives in arctic", "arctic adaptations include thick insulation", "biology", ["arctic", "cold", "insulate", "adapt"]),
            ("food chain is disrupted", "other populations in ecosystem are affected", "biology", ["food", "chain", "population", "affect"]),
            ("cells divide", "organisms grow by cell division (mitosis)", "biology", ["cell", "divide", "grow", "mitosis"]),
            ("sexual reproduction occurs", "offspring have genetic variation from two parents", "biology", ["sexual", "reproduction", "genetic", "variation", "parent"]),
            ("asexual reproduction occurs", "offspring are genetically identical to parent", "biology", ["asexual", "clone", "identical"]),
            ("chloroplasts absorb light", "chloroplasts in plants capture light for photosynthesis", "biology", ["chloroplast", "light", "green"]),
            ("mitochondria produce ATP", "mitochondria are powerhouse of the cell", "biology", ["mitochondria", "energy", "ATP", "cell"]),
            ("organism has camouflage", "camouflage helps organism hide from predators", "biology", ["camouflage", "hide", "predator", "color"]),
            ("seeds are dispersed", "wind, water, or animals spread seeds to new locations", "biology", ["seed", "spread", "disperse", "wind"]),
            ("pollination occurs", "pollen transfers from stamen to pistil for reproduction", "biology", ["pollen", "flower", "bee", "pollinate"]),
            ("blood carries oxygen", "blood delivers oxygen to cells throughout the body", "biology", ["blood", "oxygen", "cell", "carry"]),
            ("vaccine is administered", "immune system produces antibodies without causing disease", "biology", ["vaccine", "immune", "antibody"]),
            ("antibiotic is taken", "antibiotic kills or stops growth of bacteria", "biology", ["antibiotic", "bacteria", "infection"]),
            ("trait is dominant", "dominant trait appears even with one copy of the allele", "biology", ["dominant", "trait", "allele", "gene"]),
            ("trait is recessive", "recessive trait only appears with two copies", "biology", ["recessive", "trait", "allele", "two"]),
        ])

        # ── Simple Machines & Technology ──
        self._add_rules([
            ("renewable resource is used", "renewable resources can be replenished (solar, wind)", "technology", ["renewable", "solar", "wind", "replenish"]),
            ("nonrenewable resource is used", "nonrenewable resources will eventually run out (fossil fuels)", "technology", ["nonrenewable", "fossil", "fuel", "oil", "coal"]),
            ("telescope is used", "telescope magnifies distant objects in space", "technology", ["telescope", "distant", "space", "magnify"]),
            ("microscope is used", "microscope magnifies tiny objects not visible to naked eye", "technology", ["microscope", "tiny", "magnify", "cell"]),
            ("thermometer is used", "thermometer measures temperature", "technology", ["thermometer", "temperature", "measure"]),
            ("recycling occurs", "recycling reduces waste and conserves resources", "technology", ["recycle", "waste", "conserve", "resource"]),
        ])

        # ── Heat & Temperature ──
        self._add_rules([
            ("heat flows between objects", "heat always flows from hot to cold objects", "physics", ["heat", "flow", "hot", "cold", "warm", "cool", "transfer"]),
            ("object is heated to 100 celsius", "water boils at 100 degrees celsius (212 F)", "physics", ["boil", "100", "celsius", "water", "steam"]),
            ("object is cooled to 0 celsius", "water freezes at 0 degrees celsius (32 F)", "physics", ["freeze", "0", "celsius", "water", "ice", "32"]),
            ("ice is added to hot liquid", "heat transfers from hot liquid to ice causing ice to melt", "physics", ["ice", "hot", "melt", "tea", "coffee", "warm", "cold"]),
            ("temperature rises during day", "sun heats the surface causing temperature to increase", "physics", ["temperature", "rise", "sun", "warm", "day", "morning", "afternoon"]),
            ("temperature drops at night", "lack of sunlight causes temperature to decrease", "physics", ["temperature", "drop", "night", "cool", "evening"]),
            ("metal is heated", "metal expands when heated", "physics", ["metal", "heat", "expand"]),
            ("metal is cooled", "metal contracts when cooled", "physics", ["metal", "cool", "contract", "shrink"]),
        ])

        # ── States of Matter ──
        self._add_rules([
            ("solid is heated", "solid can melt into liquid when heated enough", "physics", ["solid", "heat", "melt", "liquid"]),
            ("liquid is heated", "liquid can evaporate into gas when heated enough", "physics", ["liquid", "heat", "evaporate", "gas", "boil"]),
            ("gas is cooled", "gas can condense into liquid when cooled enough", "physics", ["gas", "cool", "condense", "liquid"]),
            ("liquid is cooled", "liquid can freeze into solid when cooled enough", "physics", ["liquid", "cool", "freeze", "solid", "ice"]),
            ("water evaporates", "water changes from liquid to gas gaining energy", "physics", ["evaporate", "water", "gas", "energy"]),
            ("water condenses", "water changes from gas to liquid releasing energy", "physics", ["condense", "water", "liquid", "cloud", "dew"]),
            # v3.0 additions for Q02
            ("process changes liquid to gas", "evaporation is the process that changes liquid to gas", "physics", ["process", "liquid", "gas", "evaporation", "change"]),
            ("water changes from liquid to gas", "the process of evaporation turns liquid water into gas", "physics", ["water", "liquid", "gas", "evaporation", "change", "process"]),
        ])

        # ── Rotation & Astronomy ──
        self._add_rules([
            ("planet rotates faster", "faster rotation means shorter days", "astronomy", ["rotate", "faster", "day", "shorter", "spin"]),
            ("planet rotates slower", "slower rotation means longer days", "astronomy", ["rotate", "slower", "day", "longer", "spin"]),
            ("earth rotates on axis", "one full rotation of earth takes about 24 hours making one day", "astronomy", ["earth", "rotate", "axis", "day", "24"]),
            ("earth revolves around sun", "one full revolution takes about 365 days making one year", "astronomy", ["earth", "revolve", "sun", "year", "365", "orbit"]),
            ("moon revolves around earth", "moon takes about 27 days to orbit earth", "astronomy", ["moon", "orbit", "earth", "27"]),
            ("planet is tilted on axis", "axial tilt causes seasons", "astronomy", ["tilt", "axis", "season", "summer", "winter"]),
            # v3.0 additions for Q15
            ("day and night occur on earth", "earth rotation on its axis causes the day and night cycle", "astronomy", ["day", "night", "earth", "rotation", "axis", "cause", "cycle"]),
        ])

        # ── Photosynthesis & Plants ──
        self._add_rules([
            ("plant performs photosynthesis", "chlorophyll in leaves captures light energy as the first step", "biology", ["photosynthesis", "chlorophyll", "light", "capture", "first", "leaf"]),
            ("light hits chlorophyll", "chlorophyll absorbs light energy to start photosynthesis", "biology", ["chlorophyll", "light", "absorb", "green", "leaf"]),
            ("plant absorbs carbon dioxide", "plants take in CO2 through stomata for photosynthesis", "biology", ["carbon", "dioxide", "co2", "stomata", "plant"]),
            ("plant releases oxygen", "oxygen is a byproduct of photosynthesis", "biology", ["oxygen", "photosynthesis", "release", "byproduct"]),
            ("plant needs water for photosynthesis", "roots absorb water which is used in photosynthesis", "biology", ["water", "root", "absorb", "photosynthesis"]),
            # v3.0 additions for Q13
            ("plants produce gas during photosynthesis", "plants produce oxygen gas during photosynthesis", "biology", ["plant", "produce", "gas", "oxygen", "photosynthesis"]),
            ("photosynthesis gas output", "the gas produced by photosynthesis is oxygen", "biology", ["photosynthesis", "gas", "produce", "output", "oxygen"]),
        ])

        # ── Fossils & Earth History ──
        self._add_rules([
            ("tropical fossil found in cold area", "area must have once had a tropical climate", "geology", ["tropical", "fossil", "cold", "climate", "warm", "once"]),
            ("marine fossil found on mountain", "area was once underwater", "geology", ["marine", "fossil", "mountain", "sea", "ocean", "underwater"]),
            ("fossil fuels are burned", "burning fossil fuels releases carbon dioxide", "geology", ["fossil", "fuel", "burn", "carbon", "dioxide", "co2"]),
            ("layers of rock are observed", "deeper layers are generally older", "geology", ["layer", "rock", "deep", "old", "sediment"]),
        ])

        # ── Energy & Motion ──
        self._add_rules([
            ("object falls from height", "potential energy converts to kinetic energy as object falls", "physics", ["fall", "potential", "kinetic", "energy", "height"]),
            ("object is at halfway point of fall", "at halfway point kinetic energy equals half of maximum", "physics", ["halfway", "half", "kinetic", "energy", "fall"]),
            ("all objects fall in vacuum", "all objects fall at same rate regardless of mass in vacuum", "physics", ["fall", "vacuum", "mass", "same", "rate", "gravity"]),
            ("force is applied to object", "object accelerates in direction of net force", "physics", ["force", "accelerate", "push", "pull", "newton"]),
            ("friction acts on moving object", "friction slows down moving objects converting kinetic energy to heat", "physics", ["friction", "slow", "heat", "energy", "moving"]),
            # v3.0 additions for Q08, Q09
            ("friction acts on speed of moving object", "friction decreases the speed of a moving object", "physics", ["friction", "speed", "decrease", "slow", "moving", "object"]),
            ("ball rolls down hill", "rolling down a hill converts potential energy to kinetic energy", "physics", ["ball", "roll", "hill", "potential", "kinetic", "energy", "down", "convert"]),
            ("object moves downhill", "potential energy converts to kinetic energy going downhill", "physics", ["downhill", "potential", "kinetic", "energy", "convert", "height"]),
            ("heat flows between objects", "heat always flows from warmer object to cooler object", "physics", ["heat", "flow", "warm", "cool", "cold", "direction"]),
        ])

        # ── Scientific Method ──
        self._add_rules([
            ("experiment is planned", "first step is to plan and prepare before conducting experiment", "science", ["experiment", "plan", "prepare", "first", "step", "method"]),
            ("hypothesis is formed", "hypothesis is a testable prediction before experimenting", "science", ["hypothesis", "predict", "test", "before"]),
            ("variable is changed", "the changed variable is the independent variable", "science", ["variable", "change", "independent", "manipulate"]),
            ("variable is measured", "the measured variable is the dependent variable", "science", ["variable", "measure", "dependent", "result", "outcome"]),
            ("experiment is repeated", "repeating experiments improves reliability of results", "science", ["repeat", "reliable", "consistent", "trial"]),
            # v5.4: Scientific process rules
            ("new data contradicts old theory", "scientists use new information to update the old theory", "science", ["new", "data", "theory", "update", "revise", "information"]),
            ("old theory has wrong data", "scientists revise the theory using new evidence", "science", ["theory", "wrong", "revise", "evidence", "update", "new"]),
            ("study fish population in lake", "sampling fish populations over time is best method", "science", ["fish", "population", "sample", "study", "lake", "time"]),
        ])

        # ── Weather & Water Cycle ──
        self._add_rules([
            ("sun heats ocean water", "water evaporates from ocean surface into atmosphere", "weather", ["sun", "heat", "ocean", "evaporate", "water"]),
            ("warm air rises", "warm air rises and cools forming clouds", "weather", ["warm", "air", "rise", "cool", "cloud"]),
            ("water vapor cools", "water vapor condenses into water droplets forming clouds", "weather", ["vapor", "cool", "condense", "cloud", "droplet"]),
            ("clouds release precipitation", "rain or snow falls when water droplets get heavy enough", "weather", ["cloud", "rain", "snow", "precipitation", "fall"]),
            # v3.0 additions for Q18
            ("warm air rises in atmosphere", "convection causes warm air to rise in the atmosphere", "weather", ["warm", "air", "rise", "convection", "atmosphere", "cause"]),
            ("hot air rises", "convection is the process by which warm or hot air rises", "weather", ["hot", "warm", "air", "rise", "convection"]),
        ])

        # ══════════════════════════════════════════════════════════════
        # v19: Additional causal rules for weak ARC themes
        # Target: temperature/heat (29.1%), water cycle (32.0%),
        #         force/motion (33.3%), life science (34.0%)
        # ══════════════════════════════════════════════════════════════

        # ── Heat Transfer Mechanisms ──
        self._add_rules([
            ("heat conducts through solid", "conduction transfers heat through direct contact between solids", "physics", ["conduction", "contact", "solid", "heat", "transfer", "touch"]),
            ("tile floor feels cold", "tile is a good conductor that transfers heat away from skin quickly", "physics", ["tile", "cold", "conductor", "heat", "floor"]),
            ("carpet floor feels warm", "carpet is a good insulator that slows heat transfer from skin", "physics", ["carpet", "warm", "insulator", "heat", "floor"]),
            ("metal conducts heat", "metals are good conductors of heat and feel cold to touch", "physics", ["metal", "conductor", "heat", "cold", "spoon", "iron", "copper"]),
            ("wood insulates heat", "wood plastic rubber are good insulators that slow heat transfer", "physics", ["wood", "plastic", "rubber", "insulator", "heat", "slow"]),
            ("radiation transfers heat", "radiation transfers heat through electromagnetic waves without contact", "physics", ["radiation", "wave", "heat", "sun", "infrared", "transfer"]),
            ("convection transfers heat", "convection transfers heat through fluid movement liquid or gas", "physics", ["convection", "fluid", "liquid", "gas", "heat", "rise"]),
            ("sunny day heats surface", "sunshine increases surface temperature during the day", "physics", ["sunny", "day", "temperature", "increase", "warm", "morning", "afternoon"]),
            ("glass of hot water cools", "hot water transfers heat to surrounding cooler air", "physics", ["hot", "water", "cool", "heat", "transfer", "room", "glass"]),
        ])

        # ── Gravity & Force specifics ──
        self._add_rules([
            ("objects fall on moon", "all objects fall at same rate on moon regardless of mass", "physics", ["moon", "fall", "same", "rate", "mass", "drop", "gravity"]),
            ("gravity depends on mass and distance", "gravitational force increases with mass and decreases with distance", "physics", ["gravity", "mass", "distance", "increase", "decrease", "closer"]),
            ("weight changes on moon", "objects weigh less on moon because moon has less gravity", "physics", ["weight", "moon", "less", "gravity", "lighter"]),
            ("mass stays same everywhere", "mass does not change regardless of location", "physics", ["mass", "same", "change", "moon", "earth"]),
            ("acceleration is force divided by mass", "heavier objects need more force to accelerate", "physics", ["mass", "force", "accelerate", "heavy", "light", "newton"]),
            ("speed is distance over time", "speed equals distance divided by time traveled", "physics", ["speed", "distance", "time", "fast", "slow", "kilometer", "hour"]),
        ])

        # ── Life Science: Behavior & Traits ──
        self._add_rules([
            ("behavior is learned", "learned behaviors are acquired through experience not born with", "biology", ["learned", "behavior", "experience", "taught", "practice", "training"]),
            ("behavior is inherited", "inherited behaviors are present from birth encoded in genes", "biology", ["inherited", "instinct", "born", "gene", "innate", "reflex"]),
            ("bird migrates", "migration is an inherited behavior birds are born knowing", "biology", ["migrate", "bird", "inherited", "instinct", "winter", "south"]),
            ("dog learns tricks", "learning tricks is a learned behavior from training", "biology", ["learn", "trick", "dog", "train", "taught"]),
            ("spider spins web", "web spinning is an inherited instinct not learned", "biology", ["spider", "web", "instinct", "inherited", "born"]),
            ("organism decomposes dead matter", "decomposers break down dead organisms returning nutrients to soil", "biology", ["decompose", "dead", "nutrient", "soil", "bacteria", "fungi", "mushroom"]),
        ])

        # ── Astronomy: Light & Cycles ──
        self._add_rules([
            ("stars produce light", "stars including the sun produce their own light through fusion", "astronomy", ["star", "light", "produce", "own", "sun", "shine", "glow"]),
            ("planets reflect light", "planets do not produce their own light they reflect sunlight", "astronomy", ["planet", "reflect", "light", "sun", "moon"]),
            ("earth rotates once", "earth rotates once every 24 hours causing one day and night cycle", "astronomy", ["rotate", "once", "day", "24", "hours", "daily"]),
            ("earth revolves once", "earth revolves around sun once every 365 days making one year", "astronomy", ["revolve", "once", "year", "365", "orbit", "annual"]),
        ])

        # ── Water Cycle Details ──
        self._add_rules([
            ("most evaporation from ocean", "most evaporation happens in oceans because they have largest water surface", "weather", ["evaporation", "ocean", "most", "surface", "area", "largest"]),
            ("rain forest deforestation", "deforestation reduces transpiration and affects global water cycle", "weather", ["deforestation", "rain", "forest", "water", "cycle", "global"]),
            ("wintry mix forecast", "wintry mix means both rain and snow or sleet will fall", "weather", ["wintry", "mix", "rain", "snow", "sleet", "freeze"]),
            ("nutrient runoff into ocean", "excess nutrients cause algae blooms that deplete oxygen", "biology", ["nutrient", "runoff", "algae", "bloom", "oxygen", "deplete", "gulf"]),
        ])

        # ── Matter & Decomposition ──
        self._add_rules([
            ("organic material decomposes", "organic materials like food and paper decompose faster than synthetic", "biology", ["decompose", "organic", "food", "paper", "fast", "time"]),
            ("plastic takes long to decompose", "plastic and glass take very long to decompose in nature", "biology", ["plastic", "glass", "decompose", "long", "time", "slow", "years"]),
            ("metal rusts", "iron and steel rust when exposed to water and oxygen", "physics", ["rust", "iron", "steel", "water", "oxygen", "corrode"]),
        ])

        # ── v6.0: Challenge-targeted causal rules ──
        self._add_rules([
            # Heat transfer direction
            ("heat transfer between objects", "heat always transfers from the hotter object to the colder object", "physics", ["heat", "transfer", "hot", "cold", "flow", "warm", "cool", "direction"]),
            ("ice placed in warm drink", "heat flows from the warm drink to the ice causing ice to melt", "physics", ["ice", "tea", "drink", "warm", "hot", "melt", "heat", "flow", "cold"]),
            # Phase changes
            ("liquid placed in freezer", "liquid freezes and becomes a solid in the freezer", "physics", ["freezer", "freeze", "liquid", "solid", "ice", "juice", "tray"]),
            ("magma cools at surface", "cooled magma becomes solid igneous rock", "geology", ["magma", "lava", "cool", "solid", "rock", "igneous"]),
            # Chemical vs physical change
            ("dough bakes in oven", "baking dough is a chemical change producing new substance", "physics", ["bake", "oven", "chemical", "change", "dough", "bread", "cake"]),
            ("metal rusts over time", "rusting is a chemical change creating iron oxide", "physics", ["rust", "chemical", "change", "iron", "oxide", "corrode"]),
            ("garbage can rusting", "can rusting is a chemical change not physical change", "physics", ["rust", "garbage", "can", "chemical"]),
            # Cell biology
            ("prokaryotic cell identified", "prokaryotic cells lack a membrane-bound nucleus", "biology", ["prokaryot", "nucleus", "membrane", "cell", "bacteria"]),
            ("eukaryotic cell identified", "eukaryotic cells have a membrane-bound nucleus", "biology", ["eukaryot", "nucleus", "membrane", "cell"]),
            ("lysosome function in cell", "lysosomes break down waste materials inside the cell", "biology", ["lysosome", "waste", "break", "down", "cell", "digest"]),
            ("nerve cell function", "nerve cells send electrical signals to the brain", "biology", ["nerve", "cell", "signal", "brain", "electrical", "neuron"]),
            ("plant and animal cells compared", "both share cell membrane nucleus and mitochondria but only plants have chloroplasts", "biology", ["plant", "animal", "cell", "membrane", "nucleus", "mitochondria", "chloroplast"]),
            # Solar system
            ("inner planets of solar system", "rocky solid planets mercury venus earth mars are closer to the sun", "astronomy", ["inner", "planet", "rocky", "solid", "sun", "closer", "mercury", "venus", "earth", "mars"]),
            ("outer planets of solar system", "gas giant planets jupiter saturn uranus neptune are farther from sun", "astronomy", ["outer", "planet", "gas", "giant", "farther", "jupiter", "saturn"]),
            # Buoyancy
            ("wood floats on water", "wood floats because it is buoyant meaning less dense than water", "physics", ["wood", "float", "water", "buoyant", "density", "less", "dense"]),
            # Air composition
            ("air composition", "air is a mixture of gases primarily nitrogen and oxygen", "physics", ["air", "mixture", "gas", "nitrogen", "oxygen", "composition"]),
            # Fact vs opinion
            ("fact about organisms", "facts are observable measurable statements not opinions", "science", ["fact", "observable", "measurable", "opinion", "true"]),
            # Decomposition
            ("organic material decomposes fastest", "organic materials like cut grass apple core leaf decompose faster than metal or plastic", "biology", ["decompose", "organic", "grass", "apple", "leaf", "faster", "metal", "plastic"]),
            # Supply/demand
            ("fewer trees harvested", "fewer trees means fewer boards and higher lumber prices", "economics", ["tree", "board", "price", "higher", "fewer", "lumber", "mill"]),
            # Plankton
            ("plankton take energy from sun", "some plankton photosynthesize using sun energy and release oxygen", "biology", ["plankton", "sun", "energy", "oxygen", "photosynthesis", "release"]),
            # Stars and light
            ("sun is a star", "the sun is the nearest star and produces its own light through nuclear fusion", "astronomy", ["sun", "star", "light", "produce", "nearest", "fusion"]),
            ("planets reflect light", "planets and moons do not produce their own light they reflect sunlight", "astronomy", ["planet", "moon", "reflect", "light", "sun"]),
            # Sunrise frequency
            ("sunrise occurs daily", "sunrise happens every day making it the most frequent regular natural event", "astronomy", ["sunrise", "daily", "frequent", "day", "morning"]),
            # Fossils and climate
            ("tropical fossils in cold area", "tropical plant fossils found near glaciers means the climate was once tropical", "geology", ["tropical", "fossil", "palm", "glacier", "climate", "once", "warm", "cold"]),
            # Building safety
            ("testing building designs", "engineers test building designs to learn how to make buildings safer", "technology", ["engineer", "building", "design", "test", "safe", "safer", "earthquake"]),
            # Conductors and insulators
            ("tile floor feels cold", "tile conducts heat away from your feet faster making it feel cold", "physics", ["tile", "floor", "cold", "conduct", "heat", "feet"]),
            ("carpet feels warm", "carpet insulates and slows heat transfer from your feet", "physics", ["carpet", "warm", "insulate", "heat", "feet"]),
            # Condensation
            ("cold glass in warm room", "water vapor in warm air condenses on the cold glass surface", "physics", ["cold", "glass", "warm", "condense", "water", "vapor", "drops"]),
            # Fish gills
            ("fish gas exchange", "fish use gills to extract oxygen from water for breathing", "biology", ["fish", "gill", "oxygen", "water", "breathe"]),
            # Genetics
            ("sexual reproduction advantage", "sexual reproduction combines traits from two parents creating genetic variation", "biology", ["sexual", "reproduction", "trait", "two", "parent", "combine", "variation", "genetic"]),
            # Parasitism
            ("tapeworm in dog", "tapeworm and dog relationship is parasitism where tapeworm benefits and dog is harmed", "biology", ["tapeworm", "dog", "parasit", "harm", "benefit"]),
            # Mass conservation
            ("mass in chemical reaction", "total mass stays the same in a chemical reaction because matter cannot be created or destroyed", "physics", ["mass", "reaction", "same", "conservation", "matter", "created", "destroyed"]),

            # ── v6.1 Causal Rules (Easy/Challenge failure patterns) ──
            # Seasons
            ("earth seasons cause", "earth's seasons are caused by the tilt of the axis during revolution around the sun", "astronomy", ["season", "revolution", "tilt", "axis", "sun", "orbit"]),
            # Body systems
            ("oxygen delivery body systems", "circulatory and respiratory systems work together to deliver oxygen to cells", "biology", ["circulatory", "respiratory", "oxygen", "cell", "deliver"]),
            ("nervous system muscles", "the nervous system sends signals to muscle tissue to control movement", "biology", ["nervous", "system", "muscle", "signal", "contract", "move"]),
            ("digestive system food", "the digestive system breaks down food into nutrients the body can use for energy", "biology", ["digestive", "food", "break", "nutrient", "energy", "absorb"]),
            # Reptile identification
            ("scales and lungs animal", "an animal with scales that breathes with lungs only is a reptile", "biology", ["scales", "lungs", "reptile", "breathe"]),
            # Carbon in life
            ("carbon essential life", "carbon is essential to life because it can bond in many ways with itself and other elements", "biology", ["carbon", "essential", "life", "bond", "many", "ways"]),
            # Climate vs weather
            ("climate change long term", "climate is a long-term pattern of weather measured over decades while weather is daily", "earth_science", ["climate", "long", "term", "average", "annual", "weather", "daily"]),
            # Metamorphic rocks
            ("metamorphic rock formation", "metamorphic rocks form when existing rocks are changed by extreme heat and pressure", "geology", ["metamorphic", "heat", "pressure", "extreme", "form", "change"]),
            # Sedimentary from erosion
            ("lava to sedimentary", "when volcanic rock weathers and erodes into fragments that compact it becomes sedimentary rock", "geology", ["lava", "igneous", "weather", "erosion", "sedimentary", "fragment", "compact"]),
            # Active volcanoes location
            ("active volcano location", "active volcanoes are most commonly found where tectonic plates meet at plate boundaries", "geology", ["volcano", "active", "tectonic", "plate", "boundary", "meet", "ring"]),
            # Energy types in waves
            ("energy travel waves", "light energy and sound energy both travel in waves", "physics", ["light", "sound", "energy", "wave", "travel"]),
            # Gravity on other planets
            ("weight mass different planet", "on a planet with less gravity an object weighs less but has the same mass", "physics", ["weight", "mass", "gravity", "less", "planet", "mars", "moon", "same"]),
            # Defense against predators
            ("animal defense predator", "animals defend against predators using strong odors camouflage mimicry venom or sharp spines", "biology", ["defend", "predator", "odor", "camouflage", "mimic", "venom", "spine"]),
            # Vaccination
            ("prevent pandemic vaccination", "vaccination is the best way to prevent flu from becoming a pandemic by building immunity", "biology", ["vaccine", "vaccination", "pandemic", "prevent", "flu", "immunity", "immunize"]),
            # Daylight and axis tilt
            ("daylight hours change", "the number of daylight hours changes throughout the year because earth tilts on its axis", "astronomy", ["daylight", "hours", "tilt", "axis", "earth", "season"]),
            # Chemical property
            ("chemical property definition", "a chemical property describes how a substance reacts with other substances like flammability or reactivity", "chemistry", ["chemical", "property", "react", "flammable", "reactivity", "acid"]),
            # Plant cell unique features
            ("plant cell vs animal cell", "plant cells have cell walls and chloroplasts that animal cells do not have", "biology", ["plant", "cell", "wall", "chloroplast", "cellulose", "animal"]),

            # ── v6.2 Causal Rules ──
            # Moon rises daily
            ("moon rises daily", "the moon rises once per day due to earth rotation", "astronomy", ["moon", "rise", "daily", "day", "rotation", "earth"]),
            # Kinetic and potential energy
            ("kinetic potential energy swap", "as an object moves downward kinetic energy increases while potential energy decreases", "physics", ["kinetic", "potential", "energy", "down", "increase", "decrease", "swing", "slide"]),
            # Stream deposition
            ("stream velocity deposition", "when stream velocity decreases sediment is deposited because water can no longer carry particles", "earth_science", ["stream", "velocity", "decrease", "deposit", "sediment", "settle", "particle"]),
            # Weight vs mass
            ("gravity affects weight not mass", "gravity affects weight but not mass so objects weigh less on smaller planets but mass stays the same", "physics", ["gravity", "weight", "mass", "planet", "less", "same"]),
            # Photosynthesis function
            ("photosynthetic cells function", "photosynthetic cells convert sunlight energy into food energy through photosynthesis", "biology", ["photosynthesis", "light", "food", "energy", "convert", "sunlight", "sugar"]),
            # Solar system structure
            ("inner planets solid outer gas", "in the solar system the solid rocky planets are closer to the sun and gas giants are farther", "astronomy", ["solid", "rocky", "inner", "planet", "gas", "giant", "outer", "sun", "closer"]),
            # Sun produces light
            ("sun only light producer", "in our solar system only the sun produces its own light while planets and moons reflect sunlight", "astronomy", ["sun", "light", "produce", "reflect", "planet", "moon"]),
            # Refraction
            ("eyeglasses refract light", "eyeglasses work by refracting bending light to correct vision", "physics", ["eyeglasses", "refract", "bend", "light", "lens", "prism"]),
            # Acid base neutralization
            ("acid base neutralization", "when an acid like HCl reacts with a base like NaOH it produces salt NaCl and water H2O", "chemistry", ["acid", "base", "HCl", "NaOH", "NaCl", "salt", "water", "neutralization"]),
            # Conservation of resources
            ("repair conserves resources", "repairing broken items is the best way to conserve natural resources instead of buying new", "environment", ["repair", "conserve", "resource", "reuse", "recycle", "natural"]),
            # DNA base pairing
            ("cytosine pairs with guanine", "in DNA cytosine always pairs with guanine and adenine pairs with thymine", "biology", ["cytosine", "guanine", "adenine", "thymine", "base", "pair", "DNA", "complementary"]),
            # Dominant trait
            ("dominant trait always expressed", "when both parents carry a dominant trait all offspring will express that dominant trait", "genetics", ["dominant", "trait", "express", "always", "offspring", "parent", "round", "seed"]),
            # Troposphere density
            ("troposphere greatest density", "the troposphere is the lowest atmospheric layer and has the greatest density of air", "earth_science", ["troposphere", "density", "atmosphere", "layer", "lowest", "densest", "weather"]),
            # Not inherited traits
            ("hair style not inherited", "hair style scars knowledge and skills are not inherited they are acquired through environment", "genetics", ["hair", "style", "scar", "not", "inherit", "environment", "acquired"]),
            # Earthquake boundary volcanism
            ("earthquake boundary volcanism", "regions where earthquakes originate are often volcanic because both occur at tectonic plate boundaries", "geology", ["earthquake", "volcano", "boundary", "tectonic", "plate", "region"]),
            # Cold air flows downhill
            ("cold air flows downhill", "cold air at mountain tops flows down to valleys because cold air is denser and moves toward lower pressure", "earth_science", ["cold", "air", "mountain", "valley", "flow", "dense", "pressure", "downhill"]),
        ])

        self._built = True

    def _add_rules(self, rules_data: List[Tuple[str, str, str, List[str]]]):
        for condition, effect, domain, keywords in rules_data:
            self.rules.append(CausalRule(
                condition=condition, effect=effect,
                domain=domain, keywords=keywords,
            ))

    # Shared stem cache for query matching
    _STEM_RE = re.compile(
        r'(ation|tion|sion|ing|ment|ness|ity|ous|ive|able|ible|ful|less|ical|ence|ance|ate|ise|ize|ly|ed|er|es|al|en|s)$')

    @staticmethod
    def _stem_q(word: str) -> str:
        if len(word) <= 4:
            return word
        prev = word
        for _ in range(2):
            stemmed = CausalReasoningEngine._STEM_RE.sub('', prev) or prev[:4]
            if len(stemmed) > 3 and stemmed.endswith('e'):
                stemmed = stemmed[:-1]
            if stemmed == prev or len(stemmed) <= 4:
                break
            prev = stemmed
        return prev if len(prev) >= 3 else word[:4]

    def query(self, text: str, top_k: int = 5) -> List[Tuple[CausalRule, float]]:
        """Find causal rules relevant to a question.

        v2.0: Uses re.findall tokenization + suffix stemming for
        morphological matching (evaporates↔evaporate, heated↔heat).
        """
        text_lower = text.lower()
        text_words = set(re.findall(r'\w+', text_lower))
        text_stems = {self._stem_q(w) for w in text_words if len(w) > 2}
        scored = []
        for rule in self.rules:
            score = 0.0
            for kw in rule.keywords:
                # Multi-word keywords: substring match
                # Single-word: exact or stem match
                if ' ' in kw:
                    if kw in text_lower:
                        score += 0.3
                elif kw in text_words:
                    score += 0.3
                elif self._stem_q(kw) in text_stems:
                    score += 0.2  # Slightly lower for stem-only match
            # Condition/effect text overlap (word + stem)
            cond_words = set(re.findall(r'\w+', rule.condition.lower()))
            cond_stems = {self._stem_q(w) for w in cond_words if len(w) > 2}
            eff_words = set(re.findall(r'\w+', rule.effect.lower()))
            eff_stems = {self._stem_q(w) for w in eff_words if len(w) > 2}
            cond_exact = len(cond_words & text_words)
            cond_stem_extra = max(len(cond_stems & text_stems) - cond_exact, 0)
            cond_overlap = (cond_exact + cond_stem_extra * 0.7) / max(len(cond_words), 1)
            eff_exact = len(eff_words & text_words)
            eff_stem_extra = max(len(eff_stems & text_stems) - eff_exact, 0)
            eff_overlap = (eff_exact + eff_stem_extra * 0.7) / max(len(eff_words), 1)
            score += cond_overlap * 0.4 + eff_overlap * 0.3
            if score > 0.1:
                scored.append((rule, round(score, 4)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


