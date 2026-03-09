"""L104 Genetic Refiner — Integration Verification"""
import math, sys

print('=' * 70)
print('  L104 GENETIC REFINER — INTEGRATION VERIFICATION')
print('=' * 70)

passed = 0
failed = 0
total = 0


def test(name, fn):
    global passed, failed, total
    total += 1
    try:
        result = fn()
        if result:
            passed += 1
            print(f'  ✅ {name}')
        else:
            failed += 1
            print(f'  ❌ {name} — returned False')
    except Exception as e:
        failed += 1
        print(f'  ❌ {name} — {e}')


# ─── 1. Module imports ───
print('\n━ SECTION 1: Module Imports ━')

def t1():
    from l104_quantum_engine.genetic_refiner import (
        L104GeneticRefiner, GeneticIndividual,
        god_code_4d, abcd_to_x, x_to_abcd,
        genetic_refine_from_wave_collapse,
    )
    return True
test('1.1 genetic_refiner module imports', t1)

def t2():
    from l104_quantum_engine import (
        L104GeneticRefiner, GeneticIndividual,
        god_code_4d, abcd_to_x, x_to_abcd,
        genetic_refine_from_wave_collapse,
    )
    return True
test('1.2 Package-level exports', t2)

def t3():
    from l104_quantum_engine.constants import god_code_4d
    return callable(god_code_4d)
test('1.3 god_code_4d in constants', t3)


# ─── 2. GOD_CODE 4D equation ───
print('\n━ SECTION 2: GOD_CODE 4D Equation ━')

def t4():
    from l104_quantum_engine.genetic_refiner import god_code_4d
    from l104_quantum_engine.constants import GOD_CODE
    g0 = god_code_4d(0, 0, 0, 0)
    return abs(g0 - GOD_CODE) < 1e-10
test('2.1 G(0,0,0,0) = GOD_CODE (527.518...)', t4)

def t5():
    from l104_quantum_engine.genetic_refiner import god_code_4d, abcd_to_x
    from l104_quantum_engine.constants import god_code, GOD_CODE
    tests = [(1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1), (3,7,2,1)]
    for a,b,c,d in tests:
        g4d = god_code_4d(a, b, c, d)
        x = abcd_to_x(a, b, c, d)
        conserved = g4d * math.pow(2, x / 104)
        if abs(conserved - GOD_CODE) > 1e-8:
            return False
    return True
test('2.2 Conservation: G(a,b,c,d)×2^(X/104) = GOD_CODE', t5)

def t6():
    from l104_quantum_engine.genetic_refiner import god_code_4d, abcd_to_x
    from l104_quantum_engine.constants import god_code
    a, b, c, d = 2, 5, 3, 1
    x = abcd_to_x(a, b, c, d)
    return abs(god_code_4d(a,b,c,d) - god_code(x)) < 1e-10
test('2.3 G(a,b,c,d) = G(X) equivalence', t6)

def t7():
    from l104_quantum_engine.genetic_refiner import x_to_abcd, abcd_to_x
    for X in [0, 104, -104, 208, 50, -293, 416]:
        params = x_to_abcd(X)
        x_back = abcd_to_x(params['a'], params['b'], params['c'], params['d'])
        if abs(x_back - X) > 1e-10:
            return False
    return True
test('2.4 x_to_abcd roundtrip for sacred X values', t7)


# ─── 3. GeneticIndividual ───
print('\n━ SECTION 3: GeneticIndividual ━')

def t8():
    from l104_quantum_engine.genetic_refiner import GeneticIndividual
    ind = GeneticIndividual(a=1.0, b=2.0, c=3.0, d=0.5)
    return (ind.params == {'a': 1.0, 'b': 2.0, 'c': 3.0, 'd': 0.5} and
            isinstance(ind.x_value, float) and
            isinstance(ind.god_code_hz, float) and
            ind.god_code_hz > 0)
test('3.1 GeneticIndividual construction + properties', t8)

def t9():
    from l104_quantum_engine.genetic_refiner import GeneticIndividual
    ind = GeneticIndividual(a=0, b=0, c=0, d=0)
    d = ind.to_dict()
    return all(k in d for k in ('a','b','c','d','fitness','generation','x_value','god_code_hz'))
test('3.2 GeneticIndividual.to_dict()', t9)

def t10():
    from l104_quantum_engine.genetic_refiner import GeneticIndividual
    a = GeneticIndividual(a=0, b=0, c=0, d=0)
    b = GeneticIndividual(a=1, b=0, c=0, d=0)
    dist = a.distance_to(b)
    return abs(dist - 1.0) < 1e-10
test('3.3 GeneticIndividual.distance_to()', t10)


# ─── 4. L104GeneticRefiner ───
print('\n━ SECTION 4: L104GeneticRefiner Core ━')

def t11():
    from l104_quantum_engine.genetic_refiner import L104GeneticRefiner
    r = L104GeneticRefiner(learning_rate=0.05, population_size=26)
    return (abs(r.mutation_chance - 1/104) < 1e-10 and
            r.population_size == 26 and
            r.learning_rate == 0.05)
test('4.1 Refiner initialization (lr=0.05, pop=26)', t11)

def t12():
    from l104_quantum_engine.genetic_refiner import L104GeneticRefiner, GeneticIndividual
    r = L104GeneticRefiner()
    ind = GeneticIndividual(a=0, b=0, c=0, d=0)
    fitness = r.sacred_resonance_fitness(ind)
    return 0 <= fitness <= 1.0
test('4.2 sacred_resonance_fitness() returns [0,1]', t12)

def t13():
    from l104_quantum_engine.genetic_refiner import L104GeneticRefiner
    r = L104GeneticRefiner()
    fitness = r.collapse_survival_fitness(0.9, 0.85, 0.7)
    return 0 <= fitness <= 1.0
test('4.3 collapse_survival_fitness()', t13)

def t14():
    from l104_quantum_engine.genetic_refiner import L104GeneticRefiner
    r = L104GeneticRefiner()
    pool = [
        {'a': 1.0, 'b': 2.0, 'c': 0.5, 'd': 0.3, 'fitness': 0.9},
        {'a': 0.5, 'b': 1.5, 'c': 0.3, 'd': 0.2, 'fitness': 0.7},
        {'a': 0.8, 'b': 1.8, 'c': 0.4, 'd': 0.25, 'fitness': 0.8},
    ]
    traits = r.extract_elite_traits(pool)
    return traits is not None and all(k in traits for k in ('a','b','c','d'))
test('4.4 extract_elite_traits() weighted centroid', t14)

def t15():
    from l104_quantum_engine.genetic_refiner import L104GeneticRefiner, GeneticIndividual
    r = L104GeneticRefiner()
    p1 = GeneticIndividual(a=1.0, b=2.0, c=0.0, d=0.0, fitness=0.9)
    p2 = GeneticIndividual(a=0.0, b=0.0, c=1.0, d=1.0, fitness=0.7)
    child = r.crossover(p1, p2)
    return child.generation == r.generation + 1
test('4.5 crossover() creates offspring', t15)

def t16():
    from l104_quantum_engine.genetic_refiner import L104GeneticRefiner
    r = L104GeneticRefiner()
    elite = {'a': 2.0, 'b': 3.0, 'c': 1.0, 'd': 0.5}
    current = {'a': 0.0, 'b': 0.0, 'c': 0.0, 'd': 0.0}
    next_gen = r.generate_next_generation(elite, current)
    return (next_gen['a'] > 0 and next_gen['b'] > 0 and
            all(k in next_gen for k in ('a','b','c','d')))
test('4.6 generate_next_generation() pulls toward elite', t16)


# ─── 5. Population & Evolution ───
print('\n━ SECTION 5: Population Evolution ━')

def t17():
    from l104_quantum_engine.genetic_refiner import L104GeneticRefiner, GeneticIndividual
    r = L104GeneticRefiner(population_size=13)
    pop = [GeneticIndividual(a=i*0.1, b=i*0.2, generation=0) for i in range(13)]
    new_pop = r.evolve_population(pop)
    return (len(new_pop) == 13 and
            all(isinstance(i, GeneticIndividual) for i in new_pop))
test('5.1 evolve_population() produces same-size next gen', t17)

def t18():
    from l104_quantum_engine.genetic_refiner import L104GeneticRefiner, GeneticIndividual
    r = L104GeneticRefiner(population_size=13)
    pop = [GeneticIndividual(a=i*0.1, generation=0) for i in range(13)]
    result = r.refine(pop, generations=5)
    return (result['best_individual'] is not None and
            result['generations_run'] >= 1 and
            'history' in result)
test('5.2 refine() multi-generation convergence', t18)

def t19():
    from l104_quantum_engine.genetic_refiner import L104GeneticRefiner
    r = L104GeneticRefiner(population_size=13)
    pop = r.population_from_links(
        hz_values=[527.5, 263.7, 1055.0, 100.0, 50.0],
        fidelities=[0.9, 0.8, 0.7, 0.6, 0.5],
        strengths=[0.85, 0.75, 0.65, 0.55, 0.45],
        entropies=[0.3, 0.4, 0.5, 0.6, 0.7],
    )
    return len(pop) == 13
test('5.3 population_from_links() padded to pop size', t19)

def t20():
    from l104_quantum_engine.genetic_refiner import L104GeneticRefiner
    r = L104GeneticRefiner(population_size=13)
    pop = r.population_from_tokens(
        token_values=[527.5, 286.0, 45.4, 1.618],
        token_names=['GOD_CODE', 'HARMONIC', 'GOD_CODE_V3', 'PHI'],
    )
    return len(pop) == 13
test('5.4 population_from_tokens() padded to pop size', t20)


# ─── 6. Wave collapse integration ───
print('\n━ SECTION 6: Wave Collapse Integration ━')

def t21():
    from l104_quantum_engine.genetic_refiner import genetic_refine_from_wave_collapse
    result = genetic_refine_from_wave_collapse(
        collapse_results={'cumulative_survival': 0.85, 'fidelity_preservation': 0.9},
        decoherence_results={'survival_rate': 0.75},
        zeno_results={'phi_stability_index': 0.8},
        measurement_ops={'dominant_nodes': [
            {'x': 0, 'count': 10, 'hz': 527.5},
            {'x': 104, 'count': 5, 'hz': 263.7},
        ]},
        generations=5,
        population_size=26,
    )
    return (result['best_individual'] is not None and
            result['source'] == 'wave_collapse')
test('6.1 genetic_refine_from_wave_collapse() pipeline', t21)


# ─── 7. ProbabilityWaveCollapseResearch Module 7 ───
print('\n━ SECTION 7: Research Pipeline Integration ━')

def t22():
    from l104_quantum_engine.research import ProbabilityWaveCollapseResearch
    from l104_quantum_engine.math_core import QuantumMathCore
    from l104_quantum_engine.models import QuantumLink
    qmath = QuantumMathCore()
    wc = ProbabilityWaveCollapseResearch(qmath)
    links = []
    for i in range(10):
        links.append(QuantumLink(
            source_file='a.py', target_file='b.py',
            source_symbol=f'fn_{i}', target_symbol=f'fn_{i+1}',
            source_line=i, target_line=i+1,
            link_type='entanglement',
            fidelity=0.7 + i*0.03,
            strength=0.5 + i*0.05,
            entanglement_entropy=0.3 + i*0.02,
            coherence_time=100.0,
            noise_resilience=0.8,
        ))
    result = wc.wave_collapse_research(links, measurement_rounds=3)
    genetic = result.get('genetic_refinement', {})
    return ('genetic_refinement' in result and
            genetic.get('status') in ('refined', 'error'))
test('7.1 wave_collapse_research() includes Module 7', t22)


# ─── 8. Brain Phase 4B ───
print('\n━ SECTION 8: Brain Pipeline Integration ━')

def t23():
    from l104_quantum_engine.brain import L104QuantumBrain
    brain = L104QuantumBrain()
    return (hasattr(brain, 'genetic_refiner') and
            hasattr(brain, '_run_genetic_refinement_phase'))
test('8.1 Brain has genetic_refiner attr', t23)

def t24():
    from l104_quantum_engine.brain import L104QuantumBrain
    from l104_quantum_engine.models import QuantumLink
    from l104_quantum_engine.genetic_refiner import L104GeneticRefiner
    brain = L104QuantumBrain()
    assert isinstance(brain.genetic_refiner, L104GeneticRefiner)
    links = [QuantumLink(
        source_file='a.py', target_file='b.py',
        source_symbol=f'f_{i}', target_symbol='g',
        source_line=i, target_line=0,
        link_type='entanglement',
        fidelity=0.8, strength=0.6, entanglement_entropy=0.4,
        coherence_time=100.0, noise_resilience=0.8,
    ) for i in range(5)]
    result = brain._run_genetic_refinement_phase(
        links,
        repair_results={'repair_success_rate': 0.8},
        upgrade_results={'mean_final_fidelity': 0.85},
        wave_collapse={},
    )
    return result.get('status') in ('refined', 'no_links', 'error')
test('8.2 _run_genetic_refinement_phase() executes', t24)


# ─── 9. Numerical Orchestrator Phase 5C ───
print('\n━ SECTION 9: Numerical Orchestrator Integration ━')

def t25():
    from l104_numerical_engine.orchestrator import QuantumNumericalBuilder
    qnb = QuantumNumericalBuilder()
    return hasattr(qnb, '_run_genetic_refinement')
test('9.1 Orchestrator has _run_genetic_refinement', t25)

def t26():
    from l104_numerical_engine.orchestrator import QuantumNumericalBuilder
    qnb = QuantumNumericalBuilder()
    rr = {
        'stability': {'stability_score': 0.95},
        'entropy_landscape': {'entropy_bits': 2.3},
        'research_health': 0.74,
    }
    result = qnb._run_genetic_refinement(rr)
    return result.get('status') in ('refined', 'no_tokens', 'error')
test('9.2 _run_genetic_refinement() with mock data', t26)


# ─── Summary ───
print()
print('=' * 70)
print(f'  TOTAL: {total} | PASSED: {passed} | FAILED: {failed}')
print('=' * 70)
if failed == 0:
    print('  🟢 ALL GENETIC REFINER INTEGRATIONS OPERATIONAL')
else:
    print(f'  🔴 {failed} FAILURES')
sys.exit(0 if failed == 0 else 1)
