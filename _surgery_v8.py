#!/usr/bin/env python3
"""Surgery: Remove fact table entries + phrase patterns from commonsense_reasoning.py.
Replaces hardcoded answers with empty table — algorithmic scoring added separately.
"""
import re

path = 'l104_asi/commonsense_reasoning.py'
with open(path) as f:
    lines = f.readlines()

print(f"Original file: {len(lines)} lines")

# ── 1. Find _build_fact_table method boundaries ──
ft_start = None
ft_end = None
for i, line in enumerate(lines):
    if '    def _build_fact_table(self)' in line:
        ft_start = i
    if ft_start is not None and i > ft_start + 3:
        if re.match(r'    def \w+\(self', line):
            ft_end = i
            break

if ft_start is None or ft_end is None:
    print("ERROR: Could not find _build_fact_table method boundaries")
    exit(1)

print(f"_build_fact_table: lines {ft_start+1}-{ft_end} ({ft_end - ft_start} lines)")

# Replace method with empty return
new_method = [
    '    def _build_fact_table(self) -> List[Tuple[List[str], List[str], float]]:\n',
    '        """Fact table — v8.0: emptied. Algorithmic scoring replaces hardcoded answers."""\n',
    '        return []\n',
    '\n',
]

lines = lines[:ft_start] + new_method + lines[ft_end:]
print(f"After fact table removal: {len(lines)} lines")

# ── 2. Find and remove phrase-pattern matching section ──
# Look for the "_phrase_patterns = [" block
pp_start = None
pp_end = None
for i, line in enumerate(lines):
    if '        _phrase_patterns = [' in line:
        # Go back to find the comment header
        j = i - 1
        while j >= 0 and lines[j].strip().startswith('#'):
            j -= 1
        pp_start = j + 1  # First comment line
        break

if pp_start is not None:
    # Find end: line after the "cs['score'] -= pboost * 0.5" closing
    for i in range(pp_start, len(lines)):
        if "pboost * 0.5" in lines[i]:
            pp_end = i + 1
            break

    if pp_end is not None:
        print(f"Phrase patterns: lines {pp_start+1}-{pp_end} ({pp_end - pp_start} lines)")
        lines = lines[:pp_start] + lines[pp_end:]
        print(f"After phrase pattern removal: {len(lines)} lines")
    else:
        print("WARNING: Could not find end of phrase patterns")
else:
    print("WARNING: Could not find phrase patterns section")

# ── 3. Find and remove Direct Fact Table Matching + Dominance sections ──
ftm_start = None
ftm_end = None
for i, line in enumerate(lines):
    if '# ── Direct Fact Table Matching' in line:
        ftm_start = i
        break

if ftm_start is not None:
    # Find end: the line starting "# ── Quantum probability" or "# ── Quantum entanglement"
    for i in range(ftm_start + 1, len(lines)):
        if ('# ── Quantum probability' in lines[i] or
            '# ── Quantum entanglement' in lines[i] or
            '# ── Algorithmic Pattern' in lines[i]):
            ftm_end = i
            break

    if ftm_end is not None:
        print(f"Fact table matching + dominance: lines {ftm_start+1}-{ftm_end} ({ftm_end - ftm_start} lines)")

        # Replace with algorithmic pattern scoring
        new_section = [
            '        # ══════════════════════════════════════════════════════════════\n',
            '        # ALGORITHMIC PATTERN SCORING (v8.0)\n',
            '        # Replaces hardcoded fact table with general-purpose algorithms:\n',
            '        # question-type classification, answer-type validation,\n',
            '        # keyword exclusivity, and cross-choice contrastive scoring.\n',
            '        # ══════════════════════════════════════════════════════════════\n',
            '\n',
            '        # ── 1. Question-type classification via structural patterns ──\n',
            "        _q_type = 'general'\n",
            "        if re.search(r'\\bwhat\\s+(?:cause|happen|result|occur|lead|produce)', q_lower):\n",
            "            _q_type = 'cause_effect'\n",
            "        elif re.search(r'\\bwhat\\s+(?:is|are)\\s+(?:the\\s+)?(?:smallest|largest|biggest|most|least|best|main|primary|greatest)', q_lower):\n",
            "            _q_type = 'superlative'\n",
            "        elif re.search(r'\\bwhat\\s+(?:temperature|speed|distance|mass|weight|volume|force|pressure|length)\\b', q_lower):\n",
            "            _q_type = 'measurement'\n",
            "        elif re.search(r'\\bwhich\\s+(?:type|kind|process|device|tool|system|part|structure|form|method)\\b', q_lower):\n",
            "            _q_type = 'classification'\n",
            "        elif re.search(r'\\bwhat\\s+(?:is|are)\\b', q_lower):\n",
            "            _q_type = 'definition'\n",
            "        elif re.search(r'\\bhow\\s+(?:does|do|is|are|can|would|many|much)\\b', q_lower):\n",
            "            _q_type = 'mechanism'\n",
            "        elif re.search(r'\\bwhy\\b', q_lower):\n",
            "            _q_type = 'explanation'\n",
            "        elif re.search(r'\\bwhich\\s+(?:of\\s+the\\s+following\\s+)?(?:is|are|best|most|would|could)\\b', q_lower):\n",
            "            _q_type = 'selection'\n",
            '\n',
            '        # ── 2. Answer-type morphological validation ──\n',
            '        # Score choices by morphological fit to question type.\n',
            '        _type_patterns = {\n',
            "            'cause_effect': [r'(?:tion|ment|ing|sis|ence|ance)\\b'],\n",
            "            'mechanism': [r'(?:tion|ing|sis|ment)\\b'],\n",
            "            'measurement': [r'\\d', r'[°]', r'(?:meter|gram|liter|second|newton|joule|watt|celsius|fahrenheit)\\b'],\n",
            '        }\n',
            '        if _q_type in _type_patterns:\n',
            '            for cs in choice_scores:\n',
            "                _c_text = cs['choice']\n",
            '                _type_hits = sum(1 for p in _type_patterns[_q_type]\n',
            '                                if re.search(p, _c_text, re.IGNORECASE))\n',
            '                if _type_hits > 0:\n',
            "                    cs['score'] += min(_type_hits, 2) * 0.08\n",
            '\n',
            '        # ── 3. Keyword exclusivity scoring ──\n',
            '        # Words unique to one choice that also appear in the question\n',
            '        # or concept properties are strongly discriminative.\n',
            '        all_cw = [set(re.findall(r\'\\w+\', cs[\'choice\'].lower())) for cs in choice_scores]\n',
            '        _word_choice_map = {}\n',
            '        for _idx, _cw_set in enumerate(all_cw):\n',
            '            for w in _cw_set:\n',
            '                if len(w) > 3:\n',
            '                    _word_choice_map.setdefault(w, []).append(_idx)\n',
            '\n',
            '        # Exclusive question-word matches\n',
            '        for w in q_words_set:\n',
            '            if len(w) > 3 and w in _word_choice_map and len(_word_choice_map[w]) == 1:\n',
            "                choice_scores[_word_choice_map[w][0]]['score'] += 0.20\n",
            '\n',
            '        # Exclusive concept-property matches\n',
            '        _concept_vocab = set()\n',
            '        for _ck in concepts:\n',
            '            _cc = self.ontology.concepts.get(_ck)\n',
            '            if _cc:\n',
            "                _concept_vocab.update(re.findall(r'\\w+', str(_cc.properties).lower()))\n",
            '        for w in _concept_vocab:\n',
            '            if len(w) > 5 and w in _word_choice_map and len(_word_choice_map[w]) == 1:\n',
            '                if w in q_words_set or self._stem_sc(w) in q_stems_set:\n',
            "                    choice_scores[_word_choice_map[w][0]]['score'] += 0.12\n",
            '\n',
            '        # ── 4. Cross-choice contrastive scoring ──\n',
            '        # Unique words per choice that match question constraints\n',
            '        # are the most discriminative signal.\n',
            '        if len(choice_scores) >= 2:\n',
            '            _q_constraint = {w for w in q_words_set if len(w) > 4}\n',
            '            for i in range(len(choice_scores)):\n',
            '                _others = set().union(*(all_cw[j] for j in range(len(all_cw)) if j != i))\n',
            '                _unique_i = all_cw[i] - _others\n',
            '                _cm = len(_unique_i & _q_constraint)\n',
            '                if _cm > 0:\n',
            "                    choice_scores[i]['score'] += min(_cm, 2) * 0.12\n",
            '                _ucv = len(_unique_i & _concept_vocab)\n',
            '                if _ucv > 0:\n',
            "                    choice_scores[i]['score'] += min(_ucv, 2) * 0.06\n",
            '\n',
            '        # ── 5. Score compression ──\n',
            '        # Prevent any single choice from dominating via accumulated\n',
            '        # concept-overlap bonuses. Log-compress extreme outliers.\n',
            '        import math as _sc_math\n',
            "        _raw_vals = [cs['score'] for cs in choice_scores]\n",
            '        _mean_r = sum(_raw_vals) / max(len(_raw_vals), 1)\n',
            '        _std_r = (_sc_math.fsum((s - _mean_r)**2 for s in _raw_vals) / max(len(_raw_vals), 1)) ** 0.5\n',
            '        if _std_r > 0.5 and max(_raw_vals) > _mean_r + 3 * _std_r:\n',
            '            for cs in choice_scores:\n',
            "                if cs['score'] > 1.0:\n",
            "                    cs['score'] = 1.0 + _sc_math.log(cs['score'])\n",
            '\n',
        ]

        lines = lines[:ftm_start] + new_section + lines[ftm_end:]
        print(f"Replaced fact table matching with algorithmic scoring")
    else:
        print("WARNING: Could not find end of fact table matching section")
else:
    print("WARNING: Could not find fact table matching section")

# Write result
with open(path, 'w') as f:
    f.writelines(lines)

print(f"\nFinal file: {len(lines)} lines")
print("Surgery complete!")
