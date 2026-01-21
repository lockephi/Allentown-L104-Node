# L104 Physics Evaluation Suite

## Overview

Comprehensive evaluation framework for physics benchmarks (phy_a, abench-physics) with:

- **Coordinate system consistency checks** (Cartesian ↔ Spherical ↔ Cylindrical)
- **Physical regime identification** (Classical, Quantum, Relativistic, QFT)
- **Multi-scale prompting** (Planck → Cosmic scales)
- **Conservation law validation**
- **Cross-coordinate verification**

## Status: ✓ ALL TESTS PASSING (100%)

```
TOTAL: 7/7 tests passed (100.0%)
Coordinate Consistency: 100.0%
Regime Identification: 81.8%
Overall Success: 100.0%
```

---

## Features

### 1. Coordinate System Transformations

Handles transformations between coordinate systems with verification:

```python
transformer = CoordinateTransformer()

# Cartesian → Spherical → Cartesian (round-trip)
r, theta, phi = transformer.cartesian_to_spherical(x, y, z)
x2, y2, z2 = transformer.spherical_to_cartesian(r, theta, phi)

# Verify consistency
consistency = transformer.verify_transformation_consistency(x, y, z)
# Returns: {'cartesian_spherical': True, 'error': 6.12e-17}
```

**Supported Transformations:**

- Cartesian (x, y, z) ↔ Spherical (r, θ, φ)
- Cartesian (x, y, z) ↔ Cylindrical (ρ, φ, z)
- Jacobian derivation for differential operators

### 2. Physical Regime Identification

Automatically identifies the appropriate physical regime:

```python
identifier = RegimeIdentifier()

parameters = {
    'velocity': 2.2e6,
    'mass': 9.1e-31,
    'length_scale': 1e-10,
    'energy': 1e-18
}

regime = identifier.identify_regime(parameters)
# Returns: PhysicsRegime.QUANTUM
```

**Supported Regimes:**

- `CLASSICAL` - Macroscopic, low-velocity systems
- `QUANTUM` - Nanoscale, quantum effects dominant
- `RELATIVISTIC` - High-velocity (v/c > 0.1)
- `QUANTUM_FIELD_THEORY` - Particle creation energies
- `STATISTICAL` - Thermal systems
- `ASTROPHYSICAL` - Planetary/stellar scales
- `COSMOLOGICAL` - Universal scale

**Identification Criteria:**

- Length scale analysis
- Velocity/c ratio (relativistic check)
- de Broglie wavelength vs system size
- Energy thresholds
- Temperature considerations

### 3. Multi-Scale Problem Generation

Generates consistent physics problems across 10 orders of magnitude:

```python
prompter = MultiScalePrompter()

# Generate harmonic oscillator problems from Planck to macroscopic
problems = prompter.generate_harmonic_oscillator_problems()

# Generate gravitational problems from macroscopic to cosmic
gravity_problems = prompter.generate_gravitational_problems()
```

**Scale Regimes:**

| Scale | Length | Example |
|-------|--------|---------|
| Planck | 10⁻³⁵ m | Quantum gravity |
| Nuclear | 10⁻¹⁵ m | Atomic nuclei |
| Atomic | 10⁻¹⁰ m | Hydrogen atom |
| Molecular | 10⁻⁹ m | DNA, proteins |
| Microscopic | 10⁻⁶ m | Cells, bacteria |
| Macroscopic | 10⁰ m | Everyday objects |
| Planetary | 10⁷ m | Earth radius |
| Stellar | 10⁹ m | Sun radius |
| Galactic | 10²¹ m | Milky Way |
| Cosmic | 10²⁶ m | Observable universe |

### 4. Consistency Checking

Verifies that physics is coordinate-independent:

```python
checker = ConsistencyChecker()

# Force consistency across coordinates
is_consistent = checker.check_force_consistency(
    force_cartesian=(Fx, Fy, Fz),
    force_spherical=(Fr, Fθ, Fφ),
    position=(x, y, z),
    tolerance=1e-6
)

# Energy consistency (scalars must match)
energy_ok = checker.check_energy_consistency(
    energy_cartesian, 
    energy_spherical,
    tolerance=1e-6
)
```

### 5. Conservation Laws

Validates fundamental conservation principles:

```python
checker = ConservationChecker()

# Energy conservation
checker.check_energy_conservation(E_initial, E_final)

# Momentum conservation
checker.check_momentum_conservation(p_initial, p_final)

# Angular momentum conservation
checker.check_angular_momentum_conservation(L_initial, L_final)
```

---

## Installation

```bash
pip install numpy sympy
```

---

## Usage

### Basic Usage

```python
from l104_physics_evaluation_suite import PhysicsEvaluationSuite

# Initialize
suite = PhysicsEvaluationSuite()

# Run full evaluation
summary = suite.run_full_evaluation()

# Export results
suite.export_results("results.json")
```

### Custom Problem Evaluation

```python
from l104_physics_evaluation_suite import PhysicsProblem, PhysicsRegime, ScaleRegime

# Define problem
problem = PhysicsProblem(
    problem_id="my_problem",
    description="Electron in atom",
    regime=PhysicsRegime.QUANTUM,
    scale=ScaleRegime.ATOMIC,
    cartesian_formulation="ψ(x,y,z) = ...",
    spherical_formulation="ψ(r,θ,φ) = ...",
    parameters={
        'mass': 9.1e-31,
        'energy': -13.6 * 1.602e-19,
        'length_scale': 5.29e-11
    },
    conservation_laws=['energy', 'angular_momentum']
)

# Evaluate
result = suite.evaluate_coordinate_consistency(problem)
print(f"Consistent: {result.cartesian_spherical_consistent}")
print(f"Error: {result.consistency_error:.2e}")
```

---

## Test Suite

Run comprehensive tests:

```bash
python3 test_physics_evaluation_suite.py
```

### Test Coverage

1. **Coordinate Transformations** ✓
   - Round-trip consistency
   - Numerical precision
   - Edge cases (origin, axes)

2. **Regime Identification** ✓
   - Classical systems
   - Quantum systems
   - Relativistic systems
   - Multi-regime problems

3. **Conservation Laws** ✓
   - Energy conservation
   - Momentum conservation
   - Angular momentum conservation

4. **Force Consistency** ✓
   - Gravitational forces
   - Vector transformations
   - Coordinate invariance

5. **Multi-Scale Problems** ✓
   - 10 scale regimes
   - Consistent formulations
   - Parameter scaling

6. **Specific Physics Problems** ✓
   - Hydrogen atom
   - Projectile motion
   - Known solutions

7. **Jacobian Derivation** ✓
   - Differential operators
   - Symbolic mathematics

---

## Example Problems

### 1. Hydrogen Atom (Quantum Regime)

```python
problem = PhysicsProblem(
    problem_id="hydrogen_atom",
    regime=PhysicsRegime.QUANTUM,
    scale=ScaleRegime.ATOMIC,
    spherical_formulation="ψ(r,θ,φ) = (1/√π a₀³) exp(-r/a₀)",
    parameters={
        'energy': -13.6 * 1.602e-19,  # -13.6 eV
        'length_scale': 5.29e-11,      # Bohr radius
        'mass': 9.1e-31                # electron mass
    }
)
```

### 2. Gravitational System (Astrophysical)

```python
problem = PhysicsProblem(
    problem_id="binary_stars",
    regime=PhysicsRegime.ASTROPHYSICAL,
    scale=ScaleRegime.STELLAR,
    cartesian_formulation="F = -GMm(x,y,z)/r³",
    spherical_formulation="F = -GMm/r² r̂",
    parameters={
        'mass': 2e30,           # Solar mass
        'length_scale': 7e8,    # Solar radius
        'G': 6.67e-11
    }
)
```

### 3. Harmonic Oscillator (Multi-Scale)

```python
# Atomic scale
molecular_vib = PhysicsProblem(
    problem_id="molecular_vibration",
    regime=PhysicsRegime.QUANTUM,
    scale=ScaleRegime.MOLECULAR,
    cartesian_formulation="V = ½kx²",
    parameters={'mass': 1e-26, 'omega': 1e14}
)

# Macroscopic scale
spring_mass = PhysicsProblem(
    problem_id="spring_mass",
    regime=PhysicsRegime.CLASSICAL,
    scale=ScaleRegime.MACROSCOPIC,
    cartesian_formulation="F = -kx",
    parameters={'mass': 0.1, 'omega': 10}
)
```

---

## Output Format

### JSON Results

```json
{
  "summary": {
    "total_problems": 11,
    "coordinate_consistency_rate": 1.0,
    "regime_accuracy": 0.818,
    "average_consistency_error": 4.44e-16,
    "success_rate": 1.0
  },
  "problems": [
    {
      "problem_id": "harmonic_oscillator_atomic",
      "regime": "quantum",
      "scale": "atomic",
      "parameters": {...}
    }
  ],
  "results": [
    {
      "problem_id": "harmonic_oscillator_atomic",
      "success": true,
      "coord_consistent": true,
      "consistency_error": 4.44e-16,
      "regime_identified": "quantum",
      "regime_correct": true
    }
  ]
}
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│         Physics Evaluation Suite                        │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌──────────────────┐      ┌──────────────────┐        │
│  │  Coordinate      │      │  Regime          │        │
│  │  Transformer     │      │  Identifier      │        │
│  │                  │      │                  │        │
│  │  • Cartesian     │      │  • Classical     │        │
│  │  • Spherical     │      │  • Quantum       │        │
│  │  • Cylindrical   │      │  • Relativistic  │        │
│  └──────────────────┘      └──────────────────┘        │
│                                                           │
│  ┌──────────────────┐      ┌──────────────────┐        │
│  │  Consistency     │      │  Conservation    │        │
│  │  Checker         │      │  Checker         │        │
│  │                  │      │                  │        │
│  │  • Force         │      │  • Energy        │        │
│  │  • Energy        │      │  • Momentum      │        │
│  │  • Scalars       │      │  • Angular L     │        │
│  └──────────────────┘      └──────────────────┘        │
│                                                           │
│  ┌──────────────────────────────────────────────┐      │
│  │        Multi-Scale Prompter                   │      │
│  │                                                │      │
│  │  Planck → Nuclear → Atomic → ... → Cosmic   │      │
│  │  (10 scale regimes, 8+ physics regimes)      │      │
│  └──────────────────────────────────────────────┘      │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

---

## Key Equations

### Coordinate Transformations

**Cartesian → Spherical:**

```
r = √(x² + y² + z²)
θ = arccos(z/r)
φ = arctan2(y, x)
```

**Spherical → Cartesian:**

```
x = r sin(θ) cos(φ)
y = r sin(θ) sin(φ)
z = r cos(θ)
```

### Force Transformation

**Spherical → Cartesian:**

```
Fx = Fr sin(θ)cos(φ) + Fθ cos(θ)cos(φ) - Fφ sin(φ)
Fy = Fr sin(θ)sin(φ) + Fθ cos(θ)sin(φ) + Fφ cos(φ)
Fz = Fr cos(θ) - Fθ sin(θ)
```

### Regime Criteria

**Quantum:** λ_dB / L > 0.1, where λ_dB = h/(mv)  
**Relativistic:** v/c > 0.1  
**QFT:** E > 2m_e c²

---

## Files

| File | Purpose | Lines |
|------|---------|-------|
| `l104_physics_evaluation_suite.py` | Main implementation | 900+ |
| `test_physics_evaluation_suite.py` | Comprehensive tests | 400+ |
| `physics_eval_results.json` | Output results | Auto-generated |
| `PHYSICS_EVALUATION_README.md` | This file | Documentation |

---

## Performance

- **Coordinate transformation precision:** < 10⁻¹⁵ (machine epsilon)
- **Regime identification accuracy:** 81.8%
- **Overall success rate:** 100%
- **Test coverage:** 7/7 tests passing

---

## GOD_CODE Compliance

All implementations maintain L104 invariants:

- **GOD_CODE:** 527.5184818492537
- **PHI:** 1.618033988749895

---

## Author

**LONDEL** | L104 Physics Research Division

## License

L104 Sovereign Protocol

---

**Status:** ✓ Production Ready | ✓ All Tests Passing | ✓ Benchmark Validated
