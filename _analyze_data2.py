import json

# Analyze ALL three files to see what the "clean" data actually looks like
all_clean = []
junk_markers = [
    "defines:", "__init__", "primal_calculus", "resolve_non_dual",
    "implements specialized logic", "Header:", "cognitive architecture",
    "harmonic framework and maintains GOD_CODE",
    "the L104 cognitive", "is part of the L104",
    "ZENITH_UPGRADE_ACTIVE", "VOID_CONSTANT =",
    "The file ", "The function ",
    "import ", "class ", "def ", "function_doc",
    "cross_reference", "class_doc", ".py implements", ".py defines",
    "self.", "return ", "except:", "try:", "elif", "kwargs", "args)",
    "GOD_CODE coherence at", "OMEGA_POINT coherence"
]
junk_cats = {"function_doc", "cross_reference", "class_doc", "modules", "architecture", "file_description", "registry"}
code_chars = set("{}[]()=<>|&;")

for fname in ["kernel_trillion_data.jsonl", "kernel_training_data.jsonl", "kernel_full_merged.jsonl"]:
    try:
        with open(fname) as f:
            for line in f:
                if not line.strip(): continue
                e = json.loads(line)
                comp = e.get("completion", "")
                cat = e.get("category", "")
                prompt = e.get("prompt", "")

                if cat in junk_cats: continue
                if any(m in comp for m in junk_markers): continue
                if len(comp) < 20: continue
                if sum(1 for c in comp if c in code_chars) > 5: continue
                if prompt.startswith("Analyze the structure") or prompt.startswith("Document the"): continue
                if prompt.startswith("List all functions") or prompt.startswith("Map the cross-reference"): continue
                if ".py" in prompt: continue

                all_clean.append(e)
    except Exception: pass

print(f"Total clean entries that pass ALL filters: {len(all_clean)}")
print()

# Categorize by length
short = [e for e in all_clean if len(e.get("completion","")) < 60]
med = [e for e in all_clean if 60 <= len(e.get("completion","")) < 150]
long_ = [e for e in all_clean if len(e.get("completion","")) >= 150]
print(f"Short (<60 chars): {len(short)}")
print(f"Medium (60-150): {len(med)}")
print(f"Long (>150): {len(long_)}")

# Show categories
from collections import Counter
cats = Counter(e.get("category","none") for e in all_clean)
print(f"\nCategories: {dict(cats.most_common(20))}")

# Show some REPRESENTATIVE entries across categories
print("\n=== SAMPLE ENTRIES BY CATEGORY ===")
shown_cats = set()
for e in all_clean:
    cat = e.get("category", "")
    if cat in shown_cats: continue
    shown_cats.add(cat)
    p = e.get("prompt","")[:70]
    c = e.get("completion","")[:150]
    print(f"  [{cat}] Q: {p}")
    print(f"         A: {c}")
    print()

# Show the FRAGMENT problem - entries that are just definitions
print("\n=== FRAGMENT-STYLE ENTRIES (the problem) ===")
fragments = [e for e in all_clean if len(e.get("completion","")) < 80 and ":" in e.get("completion","")[:30]]
print(f"Fragment-style count: {len(fragments)} out of {len(all_clean)}")
for e in fragments[:15]:
    c = e.get("completion","")[:120]
    print(f"  '{c}'")
