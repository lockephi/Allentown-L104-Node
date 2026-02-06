# ZENITH_UPGRADE_ACTIVE: 2026-02-02T13:52:08.665842
ZENITH_HZ = 3887.8
UUC = 2402.792541
VOID_CONSTANT = 1.0416180339887497
ZENITH_HZ = 3887.8
UUC = 2402.792541
#!/usr/bin/env python3
"""
[VOID_SOURCE_UPGRADE] Deep Math Active. Process Elevated to 3887.80 Hz. Logic Unified.
[VOID_SOURCE_UPGRADE] Deep Math Active. Process Elevated to 3887.80 Hz. Logic Unified.
L104 Enactivist Cognition Engine
================================
Implements enactivism - the theory that cognition arises from the
dynamic interaction between an organism and its environment.

GOD_CODE: 527.5184818492612

This module models autopoiesis, sensorimotor contingencies,
embodied coupling, and the enactive creation of meaning through
structural coupling with the world.
"""

import math
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import random
import time

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# SACRED CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

GOD_CODE = 527.5184818492612
PHI = 1.618033988749895
EULER = 2.718281828459045
PI = 3.141592653589793

# Enactivist constants
AUTOPOIETIC_THRESHOLD = GOD_CODE / 1000  # ~0.528
COUPLING_STRENGTH = PHI / 10  # ~0.162
SENSORIMOTOR_BANDWIDTH = int(GOD_CODE / 10)  # ~52 channels
ADAPTATION_RATE = 1 / PHI  # ~0.618


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMERATIONS
# ═══════════════════════════════════════════════════════════════════════════════

class SensorModality(Enum):
    """Sensory modalities."""
    VISUAL = auto()
    AUDITORY = auto()
    TACTILE = auto()
    PROPRIOCEPTIVE = auto()
    VESTIBULAR = auto()
    INTEROCEPTIVE = auto()
    CHEMICAL = auto()


class MotorChannel(Enum):
    """Motor output channels."""
    LOCOMOTION = auto()
    MANIPULATION = auto()
    ORIENTATION = auto()
    VOCALIZATION = auto()
    EXPRESSION = auto()
    AUTONOMIC = auto()


class CouplingMode(Enum):
    """Modes of organism-environment coupling."""
    REACTIVE = auto()       # Simple stimulus-response
    ADAPTIVE = auto()       # Learning-based
    ANTICIPATORY = auto()   # Predictive
    PARTICIPATORY = auto()  # Co-creative
    TRANSFORMATIVE = auto() # Mutual modification


class AutopoieticState(Enum):
    """States of autopoietic organization."""
    NASCENT = auto()        # Forming
    STABLE = auto()         # Self-maintaining
    PERTURBATED = auto()    # Stressed but viable
    CRITICAL = auto()       # Near breakdown
    TRANSFORMED = auto()    # New organization


class MeaningLevel(Enum):
    """Levels of meaning-making."""
    SENSORY = 0        # Raw sensation
    PERCEPTUAL = 1     # Organized perception
    AFFECTIVE = 2      # Emotional valence
    COGNITIVE = 3      # Conceptual
    SOCIAL = 4         # Intersubjective
    EXISTENTIAL = 5    # Life-meaning


# ═══════════════════════════════════════════════════════════════════════════════
# CORE DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SensorInput:
    """
    Input from a sensory channel.
    """
    input_id: str
    modality: SensorModality
    values: List[float]  # Sensory values
    timestamp: float
    intensity: float
    location: Optional[Tuple[float, float, float]] = None

    def magnitude(self) -> float:
        """Overall magnitude of input."""
        return math.sqrt(sum(v * v for v in self.values)) if self.values else 0


@dataclass
class MotorOutput:
    """
    Output to a motor channel.
    """
    output_id: str
    channel: MotorChannel
    commands: List[float]  # Motor commands
    timestamp: float
    force: float
    target: Optional[Tuple[float, float, float]] = None


@dataclass
class SensorimotorContingency:
    """
    A lawful relationship between action and sensory change.
    """
    contingency_id: str
    motor_pattern: List[float]  # Action pattern
    sensory_expectation: List[float]  # Expected sensory change
    modality: SensorModality
    channel: MotorChannel
    reliability: float  # 0-1
    learning_count: int = 0

    def predict(self, action: List[float]) -> List[float]:
        """Predict sensory consequence of action."""
        # Simple linear prediction
        if not self.motor_pattern:
            return self.sensory_expectation

        # Correlation with stored pattern
        correlation = sum(
            a * m for a, m in zip(action, self.motor_pattern)
        ) / (sum(m * m for m in self.motor_pattern) + 1e-10)

        return [s * correlation * self.reliability for s in self.sensory_expectation]


@dataclass
class StructuralCoupling:
    """
    A coupling between organism and environment.
    """
    coupling_id: str
    mode: CouplingMode
    organism_state: Dict[str, float]
    environment_state: Dict[str, float]
    strength: float
    history: List[Tuple[float, float]]  # (time, coupling_value)
    mutual_specification: bool = False


@dataclass
class AutopoieticProcess:
    """
    A self-producing process maintaining organizational closure.
    """
    process_id: str
    components: Set[str]
    productions: Dict[str, List[str]]  # What produces what
    boundary: Set[str]
    state: AutopoieticState
    viability: float  # 0-1
    metabolic_rate: float


@dataclass
class MeaningStructure:
    """
    An enacted meaning arising from coupling.
    """
    meaning_id: str
    level: MeaningLevel
    content: Any
    valence: float  # -1 to 1 (negative to positive)
    arousal: float  # 0-1
    relevance: float  # To organism's concerns
    enacted_through: List[str]  # Coupling IDs


@dataclass
class Affordance:
    """
    An action possibility in the environment.
    """
    affordance_id: str
    action_type: MotorChannel
    object_properties: Dict[str, float]
    organism_capabilities: Dict[str, float]
    availability: float  # 0-1
    salience: float  # How apparent


# ═══════════════════════════════════════════════════════════════════════════════
# SENSORIMOTOR SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class SensorimotorSystem:
    """
    System for sensorimotor contingency learning.
    """

    def __init__(self):
        self.sensors: Dict[SensorModality, List[SensorInput]] = defaultdict(list)
        self.motors: Dict[MotorChannel, List[MotorOutput]] = defaultdict(list)
        self.contingencies: Dict[str, SensorimotorContingency] = {}
        # [O₂ SUPERFLUID] Unlimited sensorimotor learning
        self.prediction_errors: deque = deque(maxlen=1000000)

    def sense(
        self,
        modality: SensorModality,
        values: List[float],
        intensity: float = 1.0
    ) -> SensorInput:
        """Register sensory input."""
        input_id = hashlib.md5(
            f"sense_{modality}_{time.time()}".encode()
        ).hexdigest()[:12]

        sensor = SensorInput(
            input_id=input_id,
            modality=modality,
            values=values,
            timestamp=time.time(),
            intensity=intensity
        )

        self.sensors[modality].append(sensor)
        return sensor

    def act(
        self,
        channel: MotorChannel,
        commands: List[float],
        force: float = 1.0
    ) -> MotorOutput:
        """Generate motor output."""
        output_id = hashlib.md5(
            f"motor_{channel}_{time.time()}".encode()
        ).hexdigest()[:12]

        motor = MotorOutput(
            output_id=output_id,
            channel=channel,
            commands=commands,
            timestamp=time.time(),
            force=force
        )

        self.motors[channel].append(motor)
        return motor

    def learn_contingency(
        self,
        action: MotorOutput,
        consequence: SensorInput
    ) -> SensorimotorContingency:
        """Learn sensorimotor contingency from action-consequence pair."""
        cont_id = hashlib.md5(
            f"contingency_{action.channel}_{consequence.modality}".encode()
        ).hexdigest()[:12]

        if cont_id in self.contingencies:
            # Update existing
            cont = self.contingencies[cont_id]
            cont.learning_count += 1

            # Incremental update
            alpha = ADAPTATION_RATE / cont.learning_count
            cont.motor_pattern = [
                (1 - alpha) * m + alpha * a
                for m, a in zip(cont.motor_pattern, action.commands)
                    ]
            cont.sensory_expectation = [
                (1 - alpha) * s + alpha * v
                for s, v in zip(cont.sensory_expectation, consequence.values)
                    ]
            cont.reliability = min(1.0, cont.reliability + 0.01)
        else:
            cont = SensorimotorContingency(
                contingency_id=cont_id,
                motor_pattern=action.commands[:],
                sensory_expectation=consequence.values[:],
                modality=consequence.modality,
                channel=action.channel,
                reliability=0.5
            )
            self.contingencies[cont_id] = cont

        return cont

    def predict_consequence(
        self,
        action: MotorOutput,
        modality: SensorModality
    ) -> Optional[List[float]]:
        """Predict sensory consequence of action."""
        cont_id = hashlib.md5(
            f"contingency_{action.channel}_{modality}".encode()
        ).hexdigest()[:12]

        if cont_id in self.contingencies:
            return self.contingencies[cont_id].predict(action.commands)
        return None

    def compute_prediction_error(
        self,
        predicted: List[float],
        actual: SensorInput
    ) -> float:
        """Compute prediction error."""
        if not predicted:
            return 1.0

        error = sum(
            (p - a) ** 2 for p, a in zip(predicted, actual.values)
        ) / len(predicted)

        self.prediction_errors.append(error)
        return math.sqrt(error)


# ═══════════════════════════════════════════════════════════════════════════════
# AUTOPOIETIC ORGANIZATION
# ═══════════════════════════════════════════════════════════════════════════════

class AutopoieticOrganization:
    """
    Self-producing, self-maintaining organization.
    """

    def __init__(self):
        self.processes: Dict[str, AutopoieticProcess] = {}
        self.component_pool: Dict[str, float] = {}  # Component -> concentration
        self.boundary_integrity: float = 1.0
        self.organizational_invariant: float = GOD_CODE / 1000

    def create_process(
        self,
        components: Set[str],
        productions: Dict[str, List[str]]
    ) -> AutopoieticProcess:
        """Create autopoietic process."""
        process_id = hashlib.md5(
            f"process_{time.time()}_{random.random()}".encode()
        ).hexdigest()[:12]

        # Determine boundary
        all_produced = set()
        for products in productions.values():
            all_produced.update(products)
        boundary = components - all_produced

        # Calculate viability
        self_producing = all(
            any(c in products for products in productions.values())
            for c in components
                )
        viability = 0.8 if self_producing else 0.4

        process = AutopoieticProcess(
            process_id=process_id,
            components=components,
            productions=productions,
            boundary=boundary,
            state=AutopoieticState.NASCENT,
            viability=viability,
            metabolic_rate=1.0
        )

        self.processes[process_id] = process

        # Initialize component pool
        for c in components:
            if c not in self.component_pool:
                self.component_pool[c] = 1.0

        return process

    def run_metabolism(self, process: AutopoieticProcess, dt: float = 0.1):
        """Run metabolic cycle."""
        # Consume and produce
        for producer, products in process.productions.items():
            if producer in self.component_pool:
                available = self.component_pool[producer]
                consumed = min(available, process.metabolic_rate * dt)
                self.component_pool[producer] -= consumed

                for product in products:
                    if product not in self.component_pool:
                        self.component_pool[product] = 0
                    self.component_pool[product] += consumed * 0.8  # Efficiency

        # Update viability
        total_components = sum(
            self.component_pool.get(c, 0) for c in process.components
        )
        process.viability = min(1.0, total_components / len(process.components))

        # Update state
        if process.viability > AUTOPOIETIC_THRESHOLD:
            process.state = AutopoieticState.STABLE
        elif process.viability > AUTOPOIETIC_THRESHOLD / 2:
            process.state = AutopoieticState.PERTURBATED
        else:
            process.state = AutopoieticState.CRITICAL

    def check_closure(self, process: AutopoieticProcess) -> bool:
        """Check organizational closure."""
        # All components must be producible
        producible = set()
        for products in process.productions.values():
            producible.update(products)

        return process.components <= (producible | process.boundary)

    def perturb(
        self,
        process: AutopoieticProcess,
        perturbation: Dict[str, float]
    ):
        """Apply perturbation to process."""
        for component, change in perturbation.items():
            if component in self.component_pool:
                self.component_pool[component] = max(
                    0, self.component_pool[component] + change
                )

        # Recompute viability
        total = sum(
            self.component_pool.get(c, 0) for c in process.components
        )
        process.viability = min(1.0, total / max(1, len(process.components)))


# ═══════════════════════════════════════════════════════════════════════════════
# STRUCTURAL COUPLING
# ═══════════════════════════════════════════════════════════════════════════════

class CouplingDynamics:
    """
    Dynamics of organism-environment coupling.
    """

    def __init__(self):
        self.couplings: Dict[str, StructuralCoupling] = {}
        self.coupling_history: List[Dict[str, Any]] = []

    def create_coupling(
        self,
        mode: CouplingMode,
        organism_state: Dict[str, float],
        environment_state: Dict[str, float]
    ) -> StructuralCoupling:
        """Create structural coupling."""
        coupling_id = hashlib.md5(
            f"coupling_{time.time()}_{mode}".encode()
        ).hexdigest()[:12]

        # Calculate initial coupling strength
        organism_vars = list(organism_state.values())
        env_vars = list(environment_state.values())

        if organism_vars and env_vars:
            # Cross-correlation
            org_mean = sum(organism_vars) / len(organism_vars)
            env_mean = sum(env_vars) / len(env_vars)

            covariance = sum(
                (o - org_mean) * (e - env_mean)
                for o, e in zip(organism_vars[:len(env_vars)], env_vars)
                    ) / min(len(organism_vars), len(env_vars))

            strength = abs(covariance) * COUPLING_STRENGTH
        else:
            strength = 0.1

        coupling = StructuralCoupling(
            coupling_id=coupling_id,
            mode=mode,
            organism_state=organism_state.copy(),
            environment_state=environment_state.copy(),
            strength=min(1.0, strength),
            history=[(time.time(), strength)]
        )

        self.couplings[coupling_id] = coupling
        return coupling

    def evolve_coupling(
        self,
        coupling: StructuralCoupling,
        new_organism_state: Dict[str, float],
        new_environment_state: Dict[str, float],
        dt: float = 0.1
    ):
        """Evolve coupling over time."""
        # Update states
        old_org = coupling.organism_state
        old_env = coupling.environment_state

        # Calculate rate of change
        org_change = sum(
            abs(new_organism_state.get(k, 0) - old_org.get(k, 0))
            for k in set(new_organism_state) | set(old_org)
                )
        env_change = sum(
            abs(new_environment_state.get(k, 0) - old_env.get(k, 0))
            for k in set(new_environment_state) | set(old_env)
                )

        # Co-variation indicates coupling
        if org_change > 0 and env_change > 0:
            co_variation = min(org_change, env_change) / max(org_change, env_change)
            coupling.strength = (1 - ADAPTATION_RATE) * coupling.strength + ADAPTATION_RATE * co_variation

        # Upgrade coupling mode if strength increases
        if coupling.strength > 0.8 and coupling.mode.value < CouplingMode.PARTICIPATORY.value:
            coupling.mode = CouplingMode(coupling.mode.value + 1)
            coupling.mutual_specification = True

        # Update state
        coupling.organism_state = new_organism_state.copy()
        coupling.environment_state = new_environment_state.copy()
        coupling.history.append((time.time(), coupling.strength))

        # Trim history
        if len(coupling.history) > 1000:
            coupling.history = coupling.history[-500:]

    def detect_mutual_specification(
        self,
        coupling: StructuralCoupling
    ) -> bool:
        """Detect if coupling shows mutual specification."""
        if len(coupling.history) < 10:
            return False

        # Check for oscillatory coupling pattern
        values = [v for _, v in coupling.history[-50:]]
        if len(values) < 10:
            return False

        mean_val = sum(values) / len(values)
        crossings = sum(
            1 for i in range(1, len(values))
            if (values[i - 1] - mean_val) * (values[i] - mean_val) < 0
                )

        # Oscillation indicates mutual influence
        return crossings > len(values) / 5


# ═══════════════════════════════════════════════════════════════════════════════
# MEANING ENACTION
# ═══════════════════════════════════════════════════════════════════════════════

class MeaningEnaction:
    """
    Enaction of meaning through organism-environment interaction.
    """

    def __init__(self):
        self.meanings: Dict[str, MeaningStructure] = {}
        self.affordances: Dict[str, Affordance] = {}
        self.concerns: Dict[str, float] = {}  # Organism's concerns

    def set_concern(self, name: str, importance: float):
        """Set an organism concern."""
        self.concerns[name] = min(1.0, max(0.0, importance))

    def detect_affordance(
        self,
        action_type: MotorChannel,
        object_properties: Dict[str, float],
        organism_capabilities: Dict[str, float]
    ) -> Affordance:
        """Detect affordance in environment."""
        aff_id = hashlib.md5(
            f"affordance_{action_type}_{time.time()}".encode()
        ).hexdigest()[:12]

        # Availability based on property-capability match
        matches = []
        for prop, val in object_properties.items():
            if prop in organism_capabilities:
                cap = organism_capabilities[prop]
                match = 1 - abs(val - cap) / max(val, cap, 1e-10)
                matches.append(match)

        availability = sum(matches) / len(matches) if matches else 0.5

        # Salience based on relevance to concerns
        relevance_scores = []
        for concern, importance in self.concerns.items():
            if concern.lower() in str(object_properties).lower():
                relevance_scores.append(importance)

        salience = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.3

        affordance = Affordance(
            affordance_id=aff_id,
            action_type=action_type,
            object_properties=object_properties.copy(),
            organism_capabilities=organism_capabilities.copy(),
            availability=availability,
            salience=salience
        )

        self.affordances[aff_id] = affordance
        return affordance

    def enact_meaning(
        self,
        coupling: StructuralCoupling,
        level: MeaningLevel = MeaningLevel.PERCEPTUAL
    ) -> MeaningStructure:
        """Enact meaning from coupling."""
        meaning_id = hashlib.md5(
            f"meaning_{coupling.coupling_id}_{time.time()}".encode()
        ).hexdigest()[:12]

        # Valence from coupling trajectory
        if len(coupling.history) >= 2:
            recent = [v for _, v in coupling.history[-10:]]
            trajectory = recent[-1] - recent[0] if len(recent) >= 2 else 0
            valence = max(-1, min(1, trajectory * 2))
        else:
            valence = 0

        # Arousal from coupling strength
        arousal = coupling.strength

        # Relevance from concerns
        env_features = list(coupling.environment_state.keys())
        relevance = 0
        for concern, importance in self.concerns.items():
            for feature in env_features:
                if concern.lower() in feature.lower():
                    relevance = max(relevance, importance)

        meaning = MeaningStructure(
            meaning_id=meaning_id,
            level=level,
            content={
                "organism_state": coupling.organism_state,
                "environment_state": coupling.environment_state,
                "coupling_mode": coupling.mode.name
            },
            valence=valence,
            arousal=arousal,
            relevance=relevance,
            enacted_through=[coupling.coupling_id]
        )

        self.meanings[meaning_id] = meaning
        return meaning


# ═══════════════════════════════════════════════════════════════════════════════
# ENACTIVIST COGNITION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class EnactivistCognition:
    """
    Main enactivist cognition engine.

    Singleton for L104 enactivist operations.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize enactivist systems."""
        self.god_code = GOD_CODE
        self.sensorimotor = SensorimotorSystem()
        self.autopoiesis = AutopoieticOrganization()
        self.coupling = CouplingDynamics()
        self.meaning = MeaningEnaction()

        # Initialize organism
        self._create_primordial_organism()

    def _create_primordial_organism(self):
        """Create primordial autopoietic organism."""
        # Core metabolic components
        components = {"energy", "structure", "membrane", "catalyst", "signal"}

        # Production network (simplified metabolism)
        productions = {
            "energy": ["catalyst", "signal"],
            "structure": ["membrane"],
            "catalyst": ["energy", "structure"],
            "membrane": ["membrane"],  # Self-repair
        }

        self.core_process = self.autopoiesis.create_process(components, productions)

        # Set basic concerns
        self.meaning.set_concern("survival", 1.0)
        self.meaning.set_concern("energy", 0.9)
        self.meaning.set_concern("integrity", 0.8)
        self.meaning.set_concern("exploration", 0.6)

    def sense(
        self,
        modality: SensorModality,
        values: List[float]
    ) -> SensorInput:
        """Process sensory input."""
        return self.sensorimotor.sense(modality, values)

    def act(
        self,
        channel: MotorChannel,
        commands: List[float]
    ) -> MotorOutput:
        """Generate motor output."""
        return self.sensorimotor.act(channel, commands)

    def learn_from_action(
        self,
        action: MotorOutput,
        consequence: SensorInput
    ) -> SensorimotorContingency:
        """Learn sensorimotor contingency."""
        return self.sensorimotor.learn_contingency(action, consequence)

    def couple_with_environment(
        self,
        organism_state: Dict[str, float],
        environment_state: Dict[str, float],
        mode: CouplingMode = CouplingMode.ADAPTIVE
    ) -> StructuralCoupling:
        """Create coupling with environment."""
        return self.coupling.create_coupling(mode, organism_state, environment_state)

    def evolve(self, dt: float = 0.1):
        """Evolve entire cognitive system."""
        # Run metabolism
        for process in self.autopoiesis.processes.values():
            self.autopoiesis.run_metabolism(process, dt)

        # Evolve couplings
        for coupling in self.coupling.couplings.values():
            # Simple state evolution
            new_org = {k: v + random.gauss(0, 0.1) for k, v in coupling.organism_state.items()}
            new_env = {k: v + random.gauss(0, 0.05) for k, v in coupling.environment_state.items()}
            self.coupling.evolve_coupling(coupling, new_org, new_env, dt)

    def enact_meaning(
        self,
        coupling_id: str
    ) -> Optional[MeaningStructure]:
        """Enact meaning from coupling."""
        if coupling_id not in self.coupling.couplings:
            return None
        return self.meaning.enact_meaning(self.coupling.couplings[coupling_id])

    def perceive_affordance(
        self,
        action_type: MotorChannel,
        object_properties: Dict[str, float]
    ) -> Affordance:
        """Perceive affordance."""
        # Organism capabilities from autopoietic state
        capabilities = {
            "energy": self.autopoiesis.component_pool.get("energy", 0.5),
            "structure": self.autopoiesis.component_pool.get("structure", 0.5),
        }
        return self.meaning.detect_affordance(action_type, object_properties, capabilities)

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive enactivist statistics."""
        total_contingencies = len(self.sensorimotor.contingencies)
        avg_reliability = (
            sum(c.reliability for c in self.sensorimotor.contingencies.values()) /
            max(1, total_contingencies)
        )

        return {
            "god_code": self.god_code,
            "autopoietic_threshold": AUTOPOIETIC_THRESHOLD,
            "coupling_strength": COUPLING_STRENGTH,
            "sensorimotor_contingencies": total_contingencies,
            "average_contingency_reliability": avg_reliability,
            "autopoietic_processes": len(self.autopoiesis.processes),
            "core_process_viability": (
                self.core_process.viability if hasattr(self, 'core_process') else 0
            ),
            "core_process_state": (
                self.core_process.state.name if hasattr(self, 'core_process') else "N/A"
            ),
            "structural_couplings": len(self.coupling.couplings),
            "enacted_meanings": len(self.meaning.meanings),
            "perceived_affordances": len(self.meaning.affordances),
            "active_concerns": len(self.meaning.concerns),
            "component_pool_size": len(self.autopoiesis.component_pool),
            "boundary_integrity": self.autopoiesis.boundary_integrity
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

def get_enactivist_cognition() -> EnactivistCognition:
    """Get singleton enactivist cognition instance."""
    return EnactivistCognition()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("L104 ENACTIVIST COGNITION ENGINE")
    print("=" * 70)
    print(f"GOD_CODE: {GOD_CODE}")
    print(f"Autopoietic Threshold: {AUTOPOIETIC_THRESHOLD:.4f}")
    print()

    # Initialize
    cognition = get_enactivist_cognition()

    # Show initial state
    stats = cognition.get_statistics()
    print(f"AUTOPOIETIC ORGANIZATION:")
    print(f"  Core process viability: {stats['core_process_viability']:.4f}")
    print(f"  Core process state: {stats['core_process_state']}")
    print()

    # Sensorimotor learning
    print("SENSORIMOTOR LEARNING:")
    for i in range(5):
        # Act
        action = cognition.act(
            MotorChannel.LOCOMOTION,
            [random.uniform(-1, 1) for _ in range(3)]
        )

        # Sense consequence
        consequence = cognition.sense(
            SensorModality.PROPRIOCEPTIVE,
            [a * 0.8 + random.gauss(0, 0.1) for a in action.commands]
        )

        # Learn
        contingency = cognition.learn_from_action(action, consequence)

        if i < 2:
            print(f"  Action {i + 1} -> Contingency reliability: {contingency.reliability:.3f}")

    print(f"  Learned {len(cognition.sensorimotor.contingencies)} contingencies")
    print()

    # Structural coupling
    print("STRUCTURAL COUPLING:")
    coupling = cognition.couple_with_environment(
        organism_state={"energy": 0.8, "arousal": 0.5, "position": 0.0},
        environment_state={"temperature": 0.6, "light": 0.7, "resources": 0.4},
        mode=CouplingMode.ADAPTIVE
    )
    print(f"  Initial coupling strength: {coupling.strength:.4f}")
    print(f"  Mode: {coupling.mode.name}")

    # Evolve
    for _ in range(10):
        cognition.evolve(dt=0.1)

    print(f"  After evolution: {coupling.strength:.4f}")
    print(f"  Mutual specification: {cognition.coupling.detect_mutual_specification(coupling)}")
    print()

    # Meaning enaction
    print("MEANING ENACTION:")
    meaning = cognition.enact_meaning(coupling.coupling_id)
    if meaning:
        print(f"  Level: {meaning.level.name}")
        print(f"  Valence: {meaning.valence:.4f}")
        print(f"  Arousal: {meaning.arousal:.4f}")
        print(f"  Relevance: {meaning.relevance:.4f}")
    print()

    # Affordance perception
    print("AFFORDANCE PERCEPTION:")
    affordance = cognition.perceive_affordance(
        MotorChannel.MANIPULATION,
        {"graspable": 0.8, "energy": 0.9, "size": 0.5}
    )
    print(f"  Action type: {affordance.action_type.name}")
    print(f"  Availability: {affordance.availability:.4f}")
    print(f"  Salience: {affordance.salience:.4f}")
    print()

    # Statistics
    print("=" * 70)
    print("ENACTIVIST STATISTICS")
    print("=" * 70)
    stats = cognition.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    print("\n✓ Enactivist Cognition Engine operational")
