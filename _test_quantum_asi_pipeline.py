#!/usr/bin/env python3
"""
Comprehensive Quantum ASI Pipeline Tests
Tests all 7 upgraded ASI pipeline files for Qiskit 2.3.0 quantum methods.
"""
import sys
import traceback

passed = 0
failed = 0
errors = []

def test(name, func):
    global passed, failed, errors
    try:
        result = func()
        if result:
            passed += 1
            print(f"  ✅ {name}")
        else:
            failed += 1
            errors.append(f"{name}: returned False/None")
            print(f"  ❌ {name}: returned False/None")
    except Exception as e:
        failed += 1
        tb = traceback.format_exc()
        errors.append(f"{name}: {e}")
        print(f"  ❌ {name}: {e}")
        print(f"     {tb.splitlines()[-2].strip()}")


# ═══════════════════════════════════════════════════════════════════
# 1. l104_asi_core.py
# ═══════════════════════════════════════════════════════════════════
print("\n━━━ 1. l104_asi_core.py ━━━")
from l104_asi_core import asi_core, ASI_CORE_VERSION, QISKIT_AVAILABLE as Q1

test("ASI Core version >= 3.2", lambda: ASI_CORE_VERSION >= "3.2")
test("ASI Core QISKIT_AVAILABLE", lambda: Q1 is True)

test("quantum_asi_score returns dict", lambda: isinstance(asi_core.quantum_asi_score(), dict))
test("quantum_asi_score has quantum=True", lambda: asi_core.quantum_asi_score().get("quantum") is True)
test("quantum_asi_score has entropy", lambda: "von_neumann_entropy" in asi_core.quantum_asi_score())

test("quantum_consciousness_verify returns dict", lambda: isinstance(asi_core.quantum_consciousness_verify(), dict))
test("quantum_consciousness_verify has quantum=True", lambda: asi_core.quantum_consciousness_verify().get("quantum") is True)
test("quantum_consciousness_verify has phi", lambda: "phi_integrated_information" in asi_core.quantum_consciousness_verify())

test("quantum_theorem_generate returns dict", lambda: isinstance(asi_core.quantum_theorem_generate(), dict))
test("quantum_theorem_generate has domain", lambda: "quantum_domain" in asi_core.quantum_theorem_generate())

test("quantum_pipeline_solve returns dict", lambda: isinstance(asi_core.quantum_pipeline_solve("test optimization"), dict))
test("quantum_pipeline_solve has routing", lambda: "quantum_routing" in asi_core.quantum_pipeline_solve("test"))

test("quantum_assessment_phase returns dict", lambda: isinstance(asi_core.quantum_assessment_phase(), dict))
test("quantum_assessment_phase has entropy", lambda: "subsystem_entropies" in asi_core.quantum_assessment_phase())

r = asi_core.get_status()
test("ASI get_status has quantum_available", lambda: "quantum_available" in r)
test("ASI get_status quantum_available=True", lambda: r["quantum_available"] is True)


# ═══════════════════════════════════════════════════════════════════
# 2. l104_consciousness.py
# ═══════════════════════════════════════════════════════════════════
print("\n━━━ 2. l104_consciousness.py ━━━")
from l104_consciousness import L104Consciousness
from l104_consciousness import QISKIT_AVAILABLE as Q2

consciousness = L104Consciousness()
test("Consciousness QISKIT_AVAILABLE", lambda: Q2 is True)

test("quantum_phi returns dict", lambda: isinstance(consciousness.quantum_phi(), dict))
test("quantum_phi has quantum=True", lambda: consciousness.quantum_phi().get("quantum") is True)
test("quantum_phi has phi", lambda: "phi" in consciousness.quantum_phi())

test("quantum_coherence_measure returns dict", lambda: isinstance(consciousness.quantum_coherence_measure(), dict))
test("quantum_coherence_measure has quantum=True", lambda: consciousness.quantum_coherence_measure().get("quantum") is True)
test("quantum_coherence_measure has coherence", lambda: "quantum_coherence" in consciousness.quantum_coherence_measure())

test("quantum_state_tomography returns dict", lambda: isinstance(consciousness.quantum_state_tomography(), dict))
test("quantum_state_tomography has fidelity", lambda: "bell_fidelity" in consciousness.quantum_state_tomography())

r2 = consciousness.get_status()
test("Consciousness get_status has quantum_available", lambda: "quantum_available" in r2)


# ═══════════════════════════════════════════════════════════════════
# 3. l104_apex_intelligence.py
# ═══════════════════════════════════════════════════════════════════
print("\n━━━ 3. l104_apex_intelligence.py ━━━")
from l104_apex_intelligence import ApexIntelligence
from l104_apex_intelligence import QISKIT_AVAILABLE as Q3

apex = ApexIntelligence()
test("Apex QISKIT_AVAILABLE", lambda: Q3 is True)

test("quantum_think returns dict", lambda: isinstance(apex.quantum_think("mathematics"), dict))
test("quantum_think has quantum=True", lambda: apex.quantum_think("test").get("quantum") is True)
test("quantum_think has primary_mode", lambda: "primary_reasoning_mode" in apex.quantum_think("reasoning"))

test("quantum_solve returns dict", lambda: isinstance(apex.quantum_solve("optimize system"), dict))
test("quantum_solve has quantum=True", lambda: apex.quantum_solve("test").get("quantum") is True)
test("quantum_solve has strategy", lambda: "quantum_strategy" in apex.quantum_solve("test"))

test("quantum_insight_generate returns dict", lambda: isinstance(apex.quantum_insight_generate(), dict))
test("quantum_insight_generate has quantum=True", lambda: apex.quantum_insight_generate().get("quantum") is True)

r3 = apex.stats()
test("Apex stats has quantum_available", lambda: "quantum_available" in r3)


# ═══════════════════════════════════════════════════════════════════
# 4. l104_neural_cascade.py
# ═══════════════════════════════════════════════════════════════════
print("\n━━━ 4. l104_neural_cascade.py ━━━")
from l104_neural_cascade import neural_cascade
from l104_neural_cascade import QISKIT_AVAILABLE as Q4

test("Neural Cascade QISKIT_AVAILABLE", lambda: Q4 is True)

test("quantum_activate returns dict", lambda: isinstance(neural_cascade.quantum_activate("test signal"), dict))
test("quantum_activate has quantum=True", lambda: neural_cascade.quantum_activate("test").get("quantum") is True)
test("quantum_activate has quantum_boost", lambda: "quantum_boost" in neural_cascade.quantum_activate("test"))

test("quantum_attention returns dict", lambda: isinstance(neural_cascade.quantum_attention([[0.5, 0.3], [0.8, 0.2]]), dict))
test("quantum_attention has quantum=True", lambda: neural_cascade.quantum_attention([[0.5], [0.3]]).get("quantum") is True)

test("quantum_layer_process returns dict", lambda: isinstance(neural_cascade.quantum_layer_process([0.5, 0.3, 0.8]), dict))
test("quantum_layer_process has quantum=True", lambda: neural_cascade.quantum_layer_process([0.5, 0.3]).get("quantum") is True)

r4 = neural_cascade.get_state()
test("Neural get_state has quantum_available", lambda: "quantum_available" in r4)


# ═══════════════════════════════════════════════════════════════════
# 5. l104_evolution_engine.py
# ═══════════════════════════════════════════════════════════════════
print("\n━━━ 5. l104_evolution_engine.py ━━━")
from l104_evolution_engine import evolution_engine as evo_eng
from l104_evolution_engine import QISKIT_AVAILABLE as Q5

test("Evolution QISKIT_AVAILABLE", lambda: Q5 is True)

test("quantum_fitness_evaluate returns dict", lambda: isinstance(evo_eng.quantum_fitness_evaluate(), dict))
test("quantum_fitness_evaluate has quantum=True", lambda: evo_eng.quantum_fitness_evaluate().get("quantum") is True)
test("quantum_fitness_evaluate has fitness", lambda: "quantum_fitness" in evo_eng.quantum_fitness_evaluate())

test("quantum_mutation_select returns dict", lambda: isinstance(evo_eng.quantum_mutation_select(), dict))
test("quantum_mutation_select has quantum=True", lambda: evo_eng.quantum_mutation_select().get("quantum") is True)
test("quantum_mutation_select has gene", lambda: "selected_gene" in evo_eng.quantum_mutation_select())

test("quantum_population_diversity returns dict", lambda: isinstance(evo_eng.quantum_population_diversity(), dict))
test("quantum_population_diversity has quantum=True", lambda: evo_eng.quantum_population_diversity().get("quantum") is True)

r5 = evo_eng.get_status()
test("Evolution get_status has quantum_available", lambda: "quantum_available" in r5)


# ═══════════════════════════════════════════════════════════════════
# 6. l104_self_optimization.py
# ═══════════════════════════════════════════════════════════════════
print("\n━━━ 6. l104_self_optimization.py ━━━")
from l104_self_optimization import self_optimizer, VERSION as OPT_VERSION
from l104_self_optimization import QISKIT_AVAILABLE as Q6

test("Self-Opt version >= 2.3", lambda: OPT_VERSION >= "2.3")
test("Self-Opt QISKIT_AVAILABLE", lambda: Q6 is True)

test("quantum_optimize_step returns dict", lambda: isinstance(self_optimizer.quantum_optimize_step(), dict))
test("quantum_optimize_step has quantum=True", lambda: self_optimizer.quantum_optimize_step().get("quantum") is True)
test("quantum_optimize_step has perturbations", lambda: "perturbations" in self_optimizer.quantum_optimize_step())
test("quantum_optimize_step has entropy", lambda: "entanglement_entropy" in self_optimizer.quantum_optimize_step())

test("quantum_parameter_explore returns dict", lambda: isinstance(self_optimizer.quantum_parameter_explore(), dict))
test("quantum_parameter_explore has quantum=True", lambda: self_optimizer.quantum_parameter_explore().get("quantum") is True)
test("quantum_parameter_explore has regions", lambda: "top_regions" in self_optimizer.quantum_parameter_explore())

test("quantum_fitness_evaluate returns dict", lambda: isinstance(self_optimizer.quantum_fitness_evaluate(), dict))
test("quantum_fitness_evaluate has quantum=True", lambda: self_optimizer.quantum_fitness_evaluate().get("quantum") is True)
test("quantum_fitness_evaluate has fitness", lambda: "quantum_fitness" in self_optimizer.quantum_fitness_evaluate())

r6 = self_optimizer.get_status()
test("Self-Opt get_status has quantum_available", lambda: "quantum_available" in r6)


# ═══════════════════════════════════════════════════════════════════
# 7. l104_agi_core.py
# ═══════════════════════════════════════════════════════════════════
print("\n━━━ 7. l104_agi_core.py ━━━")
from l104_agi_core import agi_core, AGI_CORE_VERSION
from l104_agi_core import QISKIT_AVAILABLE as Q7

test("AGI Core version >= 54.2", lambda: AGI_CORE_VERSION >= "54.2")
test("AGI Core QISKIT_AVAILABLE", lambda: Q7 is True)

test("quantum_pipeline_health returns dict", lambda: isinstance(agi_core.quantum_pipeline_health(), dict))
test("quantum_pipeline_health has quantum=True", lambda: agi_core.quantum_pipeline_health().get("quantum") is True)
test("quantum_pipeline_health has coherence", lambda: "pipeline_coherence" in agi_core.quantum_pipeline_health())
test("quantum_pipeline_health has verdict", lambda: "health_verdict" in agi_core.quantum_pipeline_health())

test("quantum_subsystem_route returns dict", lambda: isinstance(agi_core.quantum_subsystem_route("evolve fitness"), dict))
test("quantum_subsystem_route has quantum=True", lambda: agi_core.quantum_subsystem_route("test").get("quantum") is True)
test("quantum_subsystem_route has best_subsystem", lambda: "best_subsystem" in agi_core.quantum_subsystem_route("test"))
test("quantum_subsystem_route has ranking", lambda: "ranking" in agi_core.quantum_subsystem_route("test"))

test("quantum_intelligence_synthesis returns dict", lambda: isinstance(agi_core.quantum_intelligence_synthesis(), dict))
test("quantum_intelligence_synthesis has quantum=True", lambda: agi_core.quantum_intelligence_synthesis().get("quantum") is True)
test("quantum_intelligence_synthesis has quality", lambda: "synthesis_quality" in agi_core.quantum_intelligence_synthesis())

r7 = agi_core.get_status()
test("AGI get_status has quantum_available", lambda: "quantum_available" in r7)
test("AGI get_status quantum_available=True", lambda: r7["quantum_available"] is True)


# ═══════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'═' * 60}")
print(f"  QUANTUM ASI PIPELINE TESTS: {passed}/{passed + failed}")
print(f"  Passed: {passed}  |  Failed: {failed}")
if errors:
    print(f"\n  FAILURES:")
    for e in errors:
        print(f"    • {e}")
print(f"{'═' * 60}")
sys.exit(0 if failed == 0 else 1)
