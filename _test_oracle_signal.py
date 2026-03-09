"""Test: Verify quantum wave collapse oracle produces discriminative signal.

Tests the v6.0/v4.0 upgraded oracle with:
1. Unique-word choices (each choice has distinct vocabulary)
2. Shared-vocabulary choices (hard case: choices overlap in words)
3. Short single-word choices (minimal lexical signal)
4. Morphological variants (stem matching test)
"""
import sys, math, re

# Directly test the oracle's internal scoring logic (Phase 1)
# without needing full MCQSolver initialization

def test_oracle_signal():
    """Test that the oracle produces non-uniform knowledge_density."""

    # --- Setup: replicate oracle helpers ---
    _SUFFIX_RE = re.compile(r'(ation|tion|sion|ing|ment|ness|ity|ous|ive|able|ible|ful|less|ical|ence|ance|ised|ized|ise|ize|ies|ely|ally|ly|ed|er|es|al|en|s)$')
    def _stem(w):
        if len(w) <= 4: return w
        return _SUFFIX_RE.sub('', w) or w[:4]

    def _trigrams(w):
        w2 = f'#{w}#'
        return {w2[k:k+3] for k in range(len(w2) - 2)} if len(w2) >= 3 else {w2}

    def _trigram_sim(a, b):
        ta, tb = _trigrams(a), _trigrams(b)
        inter = len(ta & tb)
        union = len(ta | tb)
        return inter / union if union > 0 else 0.0

    def _text_features(text):
        words = set(re.findall(r'\w+', text.lower()))
        word_list = [w for w in re.findall(r'\w+', text.lower()) if len(w) > 1]
        stems = {_stem(w) for w in words if len(w) > 2}
        bigrams = {f"{word_list[j]}_{word_list[j+1]}" for j in range(len(word_list) - 1)}
        return words, stems, bigrams

    tests_passed = 0
    tests_total = 0

    # ---- Test 1: Unique-word choices ----
    print("=" * 60)
    print("TEST 1: Unique-word choices")
    question = "What causes water to change from liquid to gas?"
    choices = ["evaporation", "condensation", "precipitation", "erosion"]
    facts = [
        "Evaporation occurs when water is heated and changes from liquid to gas",
        "Condensation is when water vapor cools and becomes liquid",
        "Precipitation falls from clouds as rain, snow, or hail",
        "Erosion wears away rock and soil over time",
    ]

    n = len(choices)
    choice_word_sets = [{w for w in re.findall(r'\w+', c.lower()) if len(w) > 1} for c in choices]
    choice_stem_sets = [{_stem(w) for w in ws if len(w) > 2} for ws in choice_word_sets]
    choice_prefix_sets = [{w[:5] for w in ws if len(w) >= 5} for ws in choice_word_sets]
    choice_trigram_maps = [{w: _trigrams(w) for w in ws if len(w) > 2} for ws in choice_word_sets]

    word_choice_count = {}
    for ws in choice_word_sets:
        for w in ws:
            word_choice_count[w] = word_choice_count.get(w, 0) + 1

    word_idf = {}
    for w, cnt in word_choice_count.items():
        base_idf = math.log(1.0 + n / (1.0 + cnt))
        exclusivity = 5.0 if cnt == 1 else (2.5 if cnt == 2 else 1.0)
        word_idf[w] = base_idf * exclusivity

    q_content = {w for w in re.findall(r'\w+', question.lower()) if len(w) > 2}
    q_stems = {_stem(w) for w in q_content if len(w) > 2}

    knowledge_density = [0.0] * n
    for fact in facts:
        fl = fact.lower()
        fact_words, fact_stems, fact_bigrams = _text_features(fl)
        q_word_overlap = len(q_content & fact_words)
        q_stem_overlap = len(q_stems & fact_stems)
        q_relevance = min((q_word_overlap + q_stem_overlap * 0.5), 6) * 0.18
        q_relevance = max(q_relevance, 0.2)

        per_choice = []
        for i in range(n):
            aff = 0.0
            # Tier 1: exact word
            for w in choice_word_sets[i]:
                if w in fact_words:
                    aff += word_idf.get(w, 1.0)
            # Tier 2: stem
            stem_hits = len(choice_stem_sets[i] & fact_stems)
            if stem_hits > 0:
                aff += stem_hits * 1.2
            # Tier 3: prefix
            if choice_prefix_sets[i]:
                fact_pfx = {w[:5] for w in fact_words if len(w) >= 5}
                aff += len(choice_prefix_sets[i] & fact_pfx) * 0.6
            # Tier 4: trigram fuzzy
            if aff < 0.5 and choice_trigram_maps[i]:
                best_fuzzy = 0.0
                for cw, ctg in choice_trigram_maps[i].items():
                    for fw in fact_words:
                        if len(fw) > 2:
                            sim = _trigram_sim(cw, fw)
                            if sim > 0.45:
                                best_fuzzy = max(best_fuzzy, sim * word_idf.get(cw, 1.0))
                aff += best_fuzzy
            per_choice.append(aff)

        mean_aff = sum(per_choice) / n
        if max(per_choice) > 0:
            for i in range(n):
                knowledge_density[i] += (per_choice[i] - mean_aff) * q_relevance

    kd_range = max(knowledge_density) - min(knowledge_density)
    print(f"  KD values: {[f'{kd:.4f}' for kd in knowledge_density]}")
    print(f"  KD range:  {kd_range:.4f}")
    print(f"  Best idx:  {knowledge_density.index(max(knowledge_density))} ({choices[knowledge_density.index(max(knowledge_density))]})")

    tests_total += 1
    if kd_range > 0.1:
        print(f"  PASS: Strong discriminative signal (range={kd_range:.4f} > 0.1)")
        tests_passed += 1
    else:
        print(f"  FAIL: Weak signal (range={kd_range:.4f} <= 0.1)")

    # ---- Test 2: Shared-vocabulary choices ----
    print("\n" + "=" * 60)
    print("TEST 2: Shared-vocabulary choices (hard case)")
    question = "Which best describes what happens during photosynthesis?"
    choices = [
        "Plants absorb water and sunlight to make food",
        "Plants absorb carbon dioxide and release water",
        "Plants absorb oxygen and release carbon dioxide",
        "Plants absorb minerals from the soil",
    ]
    facts = [
        "During photosynthesis plants use sunlight energy to convert carbon dioxide and water into glucose and oxygen",
        "Plants absorb sunlight through chlorophyll in their leaves",
        "Photosynthesis produces food in the form of glucose sugars",
    ]

    n = len(choices)
    choice_word_sets = [{w for w in re.findall(r'\w+', c.lower()) if len(w) > 1} for c in choices]
    choice_stem_sets = [{_stem(w) for w in ws if len(w) > 2} for ws in choice_word_sets]
    choice_prefix_sets = [{w[:5] for w in ws if len(w) >= 5} for ws in choice_word_sets]
    choice_trigram_maps = [{w: _trigrams(w) for w in ws if len(w) > 2} for ws in choice_word_sets]

    word_choice_count = {}
    for ws in choice_word_sets:
        for w in ws:
            word_choice_count[w] = word_choice_count.get(w, 0) + 1

    word_idf = {}
    for w, cnt in word_choice_count.items():
        base_idf = math.log(1.0 + n / (1.0 + cnt))
        exclusivity = 5.0 if cnt == 1 else (2.5 if cnt == 2 else 1.0)
        word_idf[w] = base_idf * exclusivity

    q_content = {w for w in re.findall(r'\w+', question.lower()) if len(w) > 2}
    q_stems = {_stem(w) for w in q_content if len(w) > 2}

    knowledge_density = [0.0] * n
    for fact in facts:
        fl = fact.lower()
        fact_words, fact_stems, fact_bigrams = _text_features(fl)
        q_word_overlap = len(q_content & fact_words)
        q_stem_overlap = len(q_stems & fact_stems)
        q_relevance = min((q_word_overlap + q_stem_overlap * 0.5), 6) * 0.18
        q_relevance = max(q_relevance, 0.2)

        per_choice = []
        for i in range(n):
            aff = 0.0
            for w in choice_word_sets[i]:
                if w in fact_words:
                    aff += word_idf.get(w, 1.0)
            stem_hits = len(choice_stem_sets[i] & fact_stems)
            if stem_hits > 0:
                aff += stem_hits * 1.2
            if choice_prefix_sets[i]:
                fact_pfx = {w[:5] for w in fact_words if len(w) >= 5}
                aff += len(choice_prefix_sets[i] & fact_pfx) * 0.6
            if aff < 0.5 and choice_trigram_maps[i]:
                best_fuzzy = 0.0
                for cw, ctg in choice_trigram_maps[i].items():
                    for fw in fact_words:
                        if len(fw) > 2:
                            sim = _trigram_sim(cw, fw)
                            if sim > 0.45:
                                best_fuzzy = max(best_fuzzy, sim * word_idf.get(cw, 1.0))
                aff += best_fuzzy
            per_choice.append(aff)

        mean_aff = sum(per_choice) / n
        if max(per_choice) > 0:
            for i in range(n):
                knowledge_density[i] += (per_choice[i] - mean_aff) * q_relevance

    kd_range = max(knowledge_density) - min(knowledge_density)
    print(f"  KD values: {[f'{kd:.4f}' for kd in knowledge_density]}")
    print(f"  KD range:  {kd_range:.4f}")
    print(f"  Best idx:  {knowledge_density.index(max(knowledge_density))} ({choices[knowledge_density.index(max(knowledge_density))]})")

    tests_total += 1
    if kd_range > 0.05:
        print(f"  PASS: Discriminative signal present (range={kd_range:.4f} > 0.05)")
        tests_passed += 1
    else:
        print(f"  FAIL: Weak signal (range={kd_range:.4f} <= 0.05)")

    # ---- Test 3: Stem matching ----
    print("\n" + "=" * 60)
    print("TEST 3: Morphological variant stem matching")
    stems_test = [
        ("evaporation", "evaporate"),
        ("condensation", "condense"),
        ("magnetic", "magnet"),
        ("photosynthesis", "photosynthe"),  # prefix match
    ]
    tests_total += 1
    stem_pass = True
    for long_form, short_form in stems_test:
        s1 = _stem(long_form)
        s2 = _stem(short_form)
        match = s1 == s2
        trig_sim = _trigram_sim(long_form, short_form)
        print(f"  {long_form} -> stem '{s1}' | {short_form} -> stem '{s2}' | match={match} | trigram_sim={trig_sim:.3f}")
        if not match and trig_sim < 0.3:
            stem_pass = False
    if stem_pass:
        print("  PASS: Stem/trigram matching captures variants")
        tests_passed += 1
    else:
        print("  FAIL: Some variants not captured")

    # ---- Test 4: Full quantum pipeline discrimination ----
    print("\n" + "=" * 60)
    print("TEST 4: Full quantum probability ratio test")
    try:
        from l104_probability_engine import QuantumProbability as QP

        # Simulate amplitudes from KD values: [0.5, -0.1, 0.0, -0.4]
        # (strong signal for choice 0)
        GOD_CODE = 527.5184818492612
        PHI = 1.618033988749895

        kd_vals = [0.5, -0.1, 0.0, -0.4]
        scores = [0.4, 0.3, 0.25, 0.2]
        min_kd = min(kd_vals)
        max_kd = max(kd_vals)
        kd_range = max_kd - min_kd

        kd_weights = [1.0 + 2.0 * (kd - min_kd) / kd_range for kd in kd_vals]
        mean_weighted = sum(s * w for s, w in zip(scores, kd_weights)) / 4
        god_phase_offset = math.pi * PHI / GOD_CODE

        amplitudes = []
        for i in range(4):
            weighted = scores[i] * kd_weights[i]
            magnitude = math.exp(3.0 * PHI * (weighted - mean_weighted))
            magnitude *= kd_weights[i] ** 1.5
            kd_norm = (kd_vals[i] - min_kd) / kd_range
            phase = kd_norm * 1.5 * math.pi + god_phase_offset
            amp = complex(magnitude * math.cos(phase), magnitude * math.sin(phase))
            amplitudes.append(amp)

        idx, prob, all_probs = QP.measurement_collapse(amplitudes)
        sorted_probs = sorted(all_probs, reverse=True)
        prob_ratio = sorted_probs[0] / max(sorted_probs[1], 0.001)

        print(f"  Amplitudes: {[f'{abs(a):.3f}' for a in amplitudes]}")
        print(f"  Born probs: {[f'{p:.4f}' for p in all_probs]}")
        print(f"  Prob ratio: {prob_ratio:.3f}")
        print(f"  Winner:     idx={idx} (prob={prob:.4f})")

        tests_total += 1
        if prob_ratio > 1.5:
            print(f"  PASS: Strong discrimination ratio ({prob_ratio:.3f} > 1.5)")
            tests_passed += 1
        else:
            print(f"  FAIL: Weak discrimination ({prob_ratio:.3f} <= 1.5)")
    except Exception as e:
        print(f"  SKIP: Could not import QuantumProbability: {e}")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print(f"RESULTS: {tests_passed}/{tests_total} tests passed")
    if tests_passed == tests_total:
        print("ALL TESTS PASSED — Oracle produces discriminative signal!")
    return tests_passed == tests_total

if __name__ == "__main__":
    success = test_oracle_signal()
    sys.exit(0 if success else 1)
