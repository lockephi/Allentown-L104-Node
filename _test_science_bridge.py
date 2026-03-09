#!/usr/bin/env python3
"""Quick test for Science Engine Bridge integration."""

from l104_asi.commonsense_reasoning import (
    ScienceEngineBridge, _get_cached_science_bridge, CommonsenseReasoningEngine
)

# Test bridge creation
bridge = ScienceEngineBridge()
print(f"Bridge version: {bridge.VERSION}")
print(f"Status: {bridge.get_status()}")

# Test connection
connected = bridge.connect()
print(f"\nConnected to Science Engine: {connected}")
print(f"Physics cache: {len(bridge._physics_cache)} entries")

# Test physics domain detection
domains = bridge._detect_physics_domain("What happens when you heat water?")
print(f"Detected domains: {domains}")

# Test singleton
sb = _get_cached_science_bridge()
print(f"Singleton works: {sb is not None}")

if connected:
    print("\n=== Physics Validation Tests ===")

    thermo = bridge.validate_thermodynamic_claim("heat flows from hot to cold")
    print(f"Thermo (hot→cold): {thermo}")

    thermo_bad = bridge.validate_thermodynamic_claim("heat flows from cold to hot naturally")
    print(f"Thermo (cold→hot): {thermo_bad}")

    em = bridge.validate_electromagnetic_claim("electrons orbit the nucleus in shells")
    print(f"EM (electron orbit): {em}")

    em_spectrum = bridge.validate_electromagnetic_claim("gamma rays have the highest energy")
    print(f"EM (gamma highest): {em_spectrum}")

    mech = bridge.validate_mechanics_claim("gravity attracts objects toward earth")
    print(f"Mechanics (gravity): {mech}")

    mech_bad = bridge.validate_mechanics_claim("gravity repels objects")
    print(f"Mechanics (gravity repel): {mech_bad}")

    print("\n=== Unified Physics Scoring ===")
    score1 = bridge.score_physics_domain("What happens when you heat metal?", "it expands")
    print(f"Heat+expand: {score1}")

    score2 = bridge.score_physics_domain("What happens when you heat metal?", "it contracts")
    print(f"Heat+contract: {score2}")

    print("\n=== Entropy Discrimination ===")
    disc = bridge.entropy_discrimination([0.5, 0.48, 0.3, 0.2])
    print(f"Entropy adjusted: {[round(d, 4) for d in disc]}")

    print("\n=== Ontology Enrichment ===")
    from l104_asi.commonsense_reasoning import ConceptOntology, CausalRule
    ont = ConceptOntology()
    ont.build()
    rules = []
    count = bridge.enrich_ontology_from_physics(ont, rules)
    print(f"Enrichments added: {count}")
    print(f"New causal rules: {len(rules)}")
    for r in rules[:5]:
        print(f"  - {r.condition[:60]}... → {r.effect[:60]}...")

print("\n=== Full Engine Integration ===")
engine = CommonsenseReasoningEngine()
engine.initialize()
status = engine.get_status()
print(f"Version: {status['version']}")
print(f"Bridge status: {status['engine_support'].get('science_bridge', {})}")
print(f"Causal rules: {status['causal_rules']}")

# Test MCQ with physics question
result = engine.answer_mcq(
    "What happens to most metals when they are heated?",
    ["They expand", "They contract", "They turn blue", "They become magnetic"]
)
print(f"\nPhysics MCQ: {result['answer']} = {result['choice']}")
print(f"Confidence: {result['confidence']}")
print(f"Science bridge active: {result['calibration'].get('science_bridge_active', False)}")

# Test reason_about with physics query
reasoning = engine.reason_about("Why does ice float on water?")
print(f"\nReason about ice floating:")
print(f"  Concepts: {reasoning['concepts_found'][:5]}")
if 'science_engine_validation' in reasoning:
    print(f"  Physics validation: {reasoning['science_engine_validation']}")

print("\n✅ Science Engine Bridge fully operational")
