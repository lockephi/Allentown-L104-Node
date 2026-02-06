import json

junk_markers = ['defines:', '__init__', 'Header:', 'primal_calculus', 'resolve_non_dual', '.py defines', '.py implements', 'specialized logic', 'cognitive architecture', 'GOD_CODE coherence at', 'harmonic framework']

clean = []
code = []
for fn in ['kernel_trillion_data.jsonl', 'kernel_training_data.jsonl', 'kernel_full_merged.jsonl']:
    try:
        with open(fn) as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                comp = d.get('completion', '')
                has_junk = any(j in comp for j in junk_markers)
                if not has_junk and len(comp) > 30:
                    clean.append(d)
                else:
                    code.append(d)
    except Exception as e:
        print(f'Error with {fn}: {e}')

print(f'CLEAN entries (usable for conversation): {len(clean)}')
print(f'CODE/JUNK entries (should be filtered): {len(code)}')
print()
print('=== SAMPLE CLEAN ENTRIES ===')
for s in clean[:15]:
    cat = s.get('category', '?')
    print(f'  [{cat}]')
    print(f'    Q: {s["prompt"][:90]}')
    print(f'    A: {s["completion"][:200]}')
    print()
