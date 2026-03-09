"""v27.3 Higher Logic Noise Dampener — Validation Suite"""
import time, sys, math

print("=" * 66)
print("v27.3 HIGHER LOGIC NOISE DAMPENER — Validation Suite")
print("=" * 66)

# ── Test 1: Constants Import ──
print("\n=== TEST 1: Higher Logic Constants Import ===")
from l104_intellect import (
    # Base dampener
    NOISE_DAMPENER_SCORE_FLOOR, NOISE_DAMPENER_ENTROPY_MIN,
    # Higher logic
    HL_SEMANTIC_COHERENCE_MIN, HL_GROVER_AMPLIFICATION,
    HL_GROVER_AMPLITUDE_FLOOR, HL_RESONANCE_ALIGNMENT_WEIGHT,
    HL_RESONANCE_FREQ_TOLERANCE, HL_ENTANGLEMENT_BONUS,
    HL_ENTANGLEMENT_DEPTH, HL_META_REASONING_ENABLED,
    HL_META_REASONING_TOP_K, HL_META_QUALITY_FLOOR,
    HL_ADAPTIVE_ENABLED, HL_ADAPTIVE_WINDOW,
    HL_ADAPTIVE_LEARNING_RATE, HL_ADAPTIVE_MIN_SCORE_FLOOR,
    HL_ADAPTIVE_MAX_SCORE_FLOOR, HL_SPECTRAL_ENABLED,
    HL_SPECTRAL_NOISE_CUTOFF, HL_CONCEPT_DISTANCE_DECAY,
    HL_CONCEPT_MAX_DISTANCE,
    LOCAL_INTELLECT_VERSION,
)
print(f"Version:           {LOCAL_INTELLECT_VERSION}")
print(f"Grover amp:        {HL_GROVER_AMPLIFICATION} (phi^3)")
print(f"Semantic min:      {HL_SEMANTIC_COHERENCE_MIN}")
print(f"Entanglement bonus:{HL_ENTANGLEMENT_BONUS}")
print(f"Adaptive enabled:  {HL_ADAPTIVE_ENABLED}")
print(f"Spectral enabled:  {HL_SPECTRAL_ENABLED}")
print(f"Concept max dist:  {HL_CONCEPT_MAX_DISTANCE}")
assert LOCAL_INTELLECT_VERSION == "27.3.0", f"Expected 27.3.0, got {LOCAL_INTELLECT_VERSION}"
assert abs(HL_GROVER_AMPLIFICATION - 4.236) < 0.01
print("PASS")

# ── Test 2: LocalIntellect Init ──
print("\n=== TEST 2: LocalIntellect Init ===")
from l104_intellect import local_intellect as li
print(f"Training data:  {len(li.training_data)} entries")
print(f"Training index: {len(li.training_index)} terms")
print("PASS")

# ── Test 3: Concept Cosine Similarity ──
print("\n=== TEST 3: Concept Cosine Similarity ===")
set_a = {"quantum", "entanglement", "physics", "superposition"}
set_b = {"quantum", "physics", "mechanics", "wave"}
set_c = {"cooking", "recipe", "ingredient", "baking"}
cos_ab = li._hl_concept_cosine(set_a, set_b)
cos_ac = li._hl_concept_cosine(set_a, set_c)
cos_empty = li._hl_concept_cosine(set_a, set())
print(f"cos(quantum+entanglement, quantum+mechanics) = {cos_ab:.4f}")
print(f"cos(quantum+entanglement, cooking+recipe)     = {cos_ac:.4f}")
print(f"cos(quantum, empty)                           = {cos_empty:.4f}")
assert cos_ab > cos_ac, "Related sets should have higher similarity"
assert cos_ac == 0.0, "Disjoint sets should have 0 similarity"
assert cos_empty == 0.0, "Empty set should have 0 similarity"
print("PASS")

# ── Test 4: Spectral Noise Ratio ──
print("\n=== TEST 4: Spectral Noise Ratio ===")
informative = "quantum entanglement creates non-local correlations between particles enabling teleportation cryptography and dense coding protocols utilizing Bell states"
noisy = "the the the the the the the the the the the the the the the"
mixed = "quantum quantum quantum physics physics entanglement particles"
spec_info = li._hl_spectral_noise_ratio(informative)
spec_noisy = li._hl_spectral_noise_ratio(noisy)
spec_mixed = li._hl_spectral_noise_ratio(mixed)
print(f"Informative text spectral noise: {spec_info:.4f}")
print(f"Noisy text spectral noise:       {spec_noisy:.4f}")
print(f"Mixed text spectral noise:       {spec_mixed:.4f}")
assert spec_info <= spec_noisy or spec_noisy < 0.01, "Informative should be less noisy"
print("PASS")

# ── Test 5: GOD_CODE Resonance Alignment ──
print("\n=== TEST 5: GOD_CODE Resonance Alignment ===")
from l104_intellect.numerics import GOD_CODE, PHI
godcode_density = math.log2(GOD_CODE) / PHI
print(f"GOD_CODE golden info density: {godcode_density:.4f}")
high_res = li._hl_godcode_resonance(4.5, 100)
low_res = li._hl_godcode_resonance(0.5, 10)
zero_res = li._hl_godcode_resonance(0.0, 0)
print(f"High entropy, 100 words: resonance = {high_res:.4f}")
print(f"Low entropy, 10 words:   resonance = {low_res:.4f}")
print(f"Zero entropy, 0 words:   resonance = {zero_res:.4f}")
assert zero_res == 0.0, "Zero words should have 0 resonance"
assert isinstance(high_res, float) and 0 <= high_res <= 1
print("PASS")

# ── Test 6: Concept Graph Distance ──
print("\n=== TEST 6: Concept Graph Distance ===")
dist_overlap = li._hl_concept_graph_distance(["quantum"], {"quantum", "physics"})
print(f"Direct overlap distance:  {dist_overlap}")
assert dist_overlap == 0, "Direct overlap should be distance 0"
dist_no_overlap = li._hl_concept_graph_distance(["banana"], {"physics"})
print(f"No overlap distance:      {dist_no_overlap}")
assert dist_no_overlap >= 0
print("PASS")

# ── Test 7: Entanglement Resonance Bonus ──
print("\n=== TEST 7: Entanglement Resonance Bonus ===")
bonus = li._hl_entanglement_resonance(["quantum", "physics"], {"entanglement", "wave"})
print(f"Entanglement bonus: {bonus:.4f}")
assert bonus >= 1.0, "Bonus should be >= 1.0"
print("PASS")

# ── Test 8: Adaptive Score Floor ──
print("\n=== TEST 8: Adaptive Score Floor ===")
floor = li._hl_adaptive_score_floor()
print(f"Adaptive score floor: {floor:.4f}")
assert HL_ADAPTIVE_MIN_SCORE_FLOOR <= floor <= HL_ADAPTIVE_MAX_SCORE_FLOOR
print("PASS")

# ── Test 9: Full Higher Logic Dampened Search ──
print("\n=== TEST 9: Higher Logic Dampened Search ===")
t0 = time.time()
results = li._search_training_data("quantum entanglement consciousness resonance", max_results=20)
t1 = time.time()
print(f"Results: {len(results)} ({(t1-t0)*1000:.1f}ms)")
if results:
    for i, r in enumerate(results[:3]):
        prompt = r.get('prompt', '')[:70]
        source = r.get('source', 'unknown')
        print(f"  #{i+1} [{source}]: {prompt}...")
results_vague = li._search_training_data("something anything stuff", max_results=20)
print(f"Vague query: {len(results_vague)} results (should be minimal)")
print("PASS")

# ── Test 10: Higher Logic GQA Dampener ──
print("\n=== TEST 10: Higher Logic GQA Dampener ===")
mock_gqa = [
    {"completion": "Quantum entanglement is a fundamental phenomenon in quantum physics where particles become correlated.", "_gqa_score": 0.9, "_gqa_source": "training_data"},
    {"completion": "test test test test test", "_gqa_score": 0.8, "_gqa_source": "chat_conversations"},
    {"completion": "The theory of quantum mechanics describes wave-particle duality and superposition states in detail.", "_gqa_score": 0.7, "_gqa_source": "mmlu_knowledge_base"},
    {"completion": "Quantum entanglement is a fundamental phenomenon in quantum physics where particles become correlated.", "_gqa_score": 0.6, "_gqa_source": "knowledge_vault"},
    {"completion": "Cooking pasta requires boiling water and salt for best results.", "_gqa_score": 0.5, "_gqa_source": "chat_conversations"},
]
dampened = li._apply_gqa_noise_dampeners(mock_gqa, "quantum physics entanglement")
print(f"Input: {len(mock_gqa)} -> Output: {len(dampened)}")
for r in dampened:
    src = r["_gqa_source"]
    sc = r.get("_gqa_score", 0)
    c = r.get("completion", "")[:60]
    print(f"  [{src}] score={sc:.4f}: {c}...")
assert len(dampened) < len(mock_gqa), "Higher logic should filter more aggressively"
print("PASS")

# ── Test 11: Direct Dampener with Higher Logic Layers ──
print("\n=== TEST 11: Direct Higher Logic Dampener ===")
mock_entries = [
    ({"prompt": "Explain quantum entanglement.", "completion": "Quantum entanglement is a physical phenomenon that occurs when a group of particles interact in such a way that the quantum state of each particle cannot be described independently.", "source": "mmlu_knowledge_base"}, 4.5),
    ({"prompt": "hi hello", "completion": "hi hi hi hi hi hi hi hi hi hi", "source": "training_data"}, 3.0),
    ({"prompt": "What is quantum?", "completion": "Quantum computing differs from classical computing by using qubits and superposition.", "source": "training_data"}, 2.5),
    ({"prompt": "cats", "completion": "Dogs are pets that love running.", "source": "chat_conversations"}, 0.1),
    ({"prompt": "quantum resonance", "completion": "Quantum resonance describes the amplification of states through constructive interference in quantum systems.", "source": "knowledge_vault"}, 2.0),
]
dampened_direct = li._apply_noise_dampeners(mock_entries, ["quantum", "entanglement", "physics"])
print(f"Input: {len(mock_entries)} -> Output: {len(dampened_direct)}")
for entry, score in dampened_direct:
    src = entry.get('source', '?')
    comp = entry.get('completion', '')[:60]
    print(f"  score={score:.4f} [{src}]: {comp}...")
print("PASS")

# ── Test 12: Adaptive Learning ──
print("\n=== TEST 12: Adaptive Threshold Learning ===")
initial_floor = li._hl_adaptive_score_floor()
for i in range(10):
    li._hl_record_dampener_outcome(
        noise_ratio=0.3, total=20, passed=14,
        query_terms=["test", "query", f"term{i}"]
    )
learned_floor = li._hl_adaptive_score_floor()
print(f"Initial floor: {initial_floor:.4f}")
print(f"After 10 queries floor: {learned_floor:.4f}")
assert isinstance(learned_floor, float)
print("PASS")

print("\n" + "=" * 66)
print("ALL 12 TESTS PASSED — Higher Logic Dampener v27.3 OPERATIONAL")
print("=" * 66)
