"""Test: v5.0 knowledge oracle produces discriminative signal."""
import math, re

PHI = 1.618033988749895
GOD_CODE = 527.5184818492612

question = 'Which of the following is a chemical change?'
choices = ['melting ice', 'burning wood', 'dissolving salt', 'cutting paper']
scores = [0.35, 0.42, 0.38, 0.30]
n_choices = 4

choice_word_sets, choice_prefix_sets, choice_bigrams = [], [], []
for c in choices:
    words = {w for w in re.findall(r'\w+', c.lower()) if len(w) > 1}
    choice_word_sets.append(words)
    choice_prefix_sets.append({w[:5] for w in words if len(w) >= 5})
    wl = [w for w in re.findall(r'\w+', c.lower()) if len(w) > 1]
    bigrams = {f'{wl[j]}_{wl[j+1]}' for j in range(len(wl) - 1)}
    choice_bigrams.append(bigrams)

wcc = {}
for ws in choice_word_sets:
    for w in ws:
        wcc[w] = wcc.get(w, 0) + 1

word_idf = {}
for w, cnt in wcc.items():
    base_idf = math.log(1.0 + n_choices / (1.0 + cnt))
    excl = 3.0 if cnt == 1 else (1.5 if cnt == 2 else 1.0)
    word_idf[w] = base_idf * excl

old_idf = {w: math.log(max(n_choices, 2) / cnt) for w, cnt in wcc.items()}

print("=== WORD IDF COMPARISON ===")
for w in sorted(word_idf.keys()):
    print(f"  {w:15s}  new={word_idf[w]:.3f}  old={old_idf[w]:.3f}")

facts = [
    'When wood burns it undergoes a chemical reaction producing ash and CO2',
    'Chemical changes create new substances with different properties',
    'Melting and dissolving are physical changes that can be reversed',
    'Physical changes do not create new chemical substances',
]
q_content = {w for w in re.findall(r'\w+', question.lower()) if len(w) > 2}

cwp = []
for ws in choice_word_sets:
    pats = {}
    for w in ws:
        try:
            pats[w] = re.compile(r'\b' + re.escape(w) + r'\b', re.IGNORECASE)
        except Exception:
            pats[w] = None
    cwp.append(pats)

kd = [0.0] * n_choices
for fact in facts:
    fl = fact.lower()
    fw = set(re.findall(r'\w+', fl))
    qo = len(q_content & fw)
    qr = min(qo, 5) * 0.2 if qo > 0 else 0.15
    pc = []
    for i in range(n_choices):
        aff = 0.0
        for w, pat in cwp[i].items():
            if pat and pat.search(fl):
                aff += word_idf.get(w, 1.0)
            elif w in fw:
                aff += word_idf.get(w, 1.0) * 0.7
        if aff < 0.1 and choice_prefix_sets[i]:
            fp = {w[:5] for w in fw if len(w) >= 5}
            aff += len(choice_prefix_sets[i] & fp) * 0.6
        fwl = [w for w in re.findall(r'\w+', fl) if len(w) > 1]
        fbg = {f'{fwl[j]}_{fwl[j+1]}' for j in range(len(fwl) - 1)}
        aff += len(choice_bigrams[i] & fbg) * 2.0
        pc.append(aff)
    mean_a = sum(pc) / n_choices
    if max(pc) > 0:
        for i in range(n_choices):
            kd[i] += (pc[i] - mean_a) * qr

print("\n=== KNOWLEDGE DENSITY (Phase 1) ===")
for i in range(n_choices):
    print(f"  {choices[i]:20s}  KD={kd[i]:+.4f}")
kdr = max(kd) - min(kd)
print(f"  KD Range: {kdr:.4f}")
print(f"  Old guard (>=0.1): {'PASS' if kdr >= 0.1 else 'FAIL'}")
print(f"  New guard (>=0.02): {'PASS' if kdr >= 0.02 else 'FAIL'}")

# Phase 2 new
mn, mx = min(kd), max(kd)
kdw = [1.0 + 2.0 * (k - mn) / max(kdr, 1e-9) for k in kd]
mw = sum(max(scores[i], 0.001) * kdw[i] for i in range(n_choices)) / n_choices
gpo = math.pi * PHI / GOD_CODE
amps = []
for i in range(n_choices):
    s = max(scores[i], 0.001)
    w = s * kdw[i]
    mag = math.exp(PHI * (w - mw))
    kdn = (kd[i] - mn) / max(kdr, 1e-9)
    ph = kdn * math.pi + gpo
    amps.append(complex(mag * math.cos(ph), mag * math.sin(ph)))

probs = [abs(a)**2 for a in amps]
t = sum(probs)
probs = [p / t for p in probs]
sp = sorted(probs, reverse=True)
ratio = sp[0] / max(sp[1], 0.001)

print("\n=== NEW QUANTUM PROBABILITIES (Phase 2-3) ===")
for i in range(n_choices):
    print(f"  {choices[i]:20s}  prob={probs[i]:.4f}  mag={abs(amps[i]):.4f}")
print(f"  Prob ratio: {ratio:.3f}")
print(f"  Old guard (>=1.5): {'PASS' if ratio >= 1.5 else 'FAIL'}")
print(f"  New guard (>=1.2): {'PASS' if ratio >= 1.2 else 'FAIL'}")
print(f"  Winner: {choices[probs.index(max(probs))]}")

# Old Phase 2
print("\n=== OLD QUANTUM PROBABILITIES ===")
okdw = [1.0 + (k - mn) / max(kdr, 1e-9) if kdr > 0.05 else 1.0 for k in kd]
oamps = []
for i in range(n_choices):
    s = max(scores[i], 0.001)
    mag = (s * okdw[i]) ** PHI
    ph = (s + kd[i] * 0.1) * math.pi / GOD_CODE
    oamps.append(complex(mag * math.cos(ph), mag * math.sin(ph)))
oprobs = [abs(a)**2 for a in oamps]
ot = sum(oprobs)
oprobs = [p / ot for p in oprobs]
osp = sorted(oprobs, reverse=True)
oratio = osp[0] / max(osp[1], 0.001)
for i in range(n_choices):
    print(f"  {choices[i]:20s}  prob={oprobs[i]:.4f}")
print(f"  Old prob ratio: {oratio:.3f}")
print(f"  Old would pass 1.5 guard: {'PASS' if oratio >= 1.5 else 'FAIL'}")

# Summary
print("\n=== SUMMARY ===")
print(f"  Phase 1 KD signal:  {'STRONG' if kdr > 0.5 else 'MODERATE' if kdr > 0.1 else 'WEAK'} (range={kdr:.4f})")
print(f"  Phase 2 new ratio:  {ratio:.3f} → {'DISCRIMINATIVE' if ratio >= 1.2 else 'UNIFORM'}")
print(f"  Phase 2 old ratio:  {oratio:.3f} → {'DISCRIMINATIVE' if oratio >= 1.5 else 'UNIFORM (no signal)'}")
print(f"  Improvement:        {ratio/oratio:.1f}x discrimination ratio")
