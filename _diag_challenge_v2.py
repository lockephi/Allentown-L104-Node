#!/usr/bin/env python3
"""Diagnose Challenge failures q0-499 — full set with categorization."""
import requests, re, sys, collections

API = "https://datasets-server.huggingface.co/rows"
PARAMS = {"dataset": "allenai/ai2_arc", "config": "ARC-Challenge", "split": "test"}

sys.path.insert(0, ".")
from l104_asi.commonsense_reasoning import CommonsenseReasoningEngine
cr = CommonsenseReasoningEngine()

results = []
for offset in range(0, 500, 100):
    print(f"Fetching Challenge q{offset}-{offset+99}...")
    p = {**PARAMS, "offset": offset, "length": 100}
    resp = requests.get(API, params=p, timeout=30)
    rows = resp.json().get("rows", [])
    for i, row in enumerate(rows):
        r = row["row"]
        q = r["question"]
        choices = r["choices"]["text"]
        labels = r["choices"]["label"]
        answer_key = r["answerKey"]
        expected = labels.index(answer_key) if answer_key in labels else -1
        result = cr.answer_mcq(q, choices)
        got = result.get("answer_index", -1)
        correct = (got == expected)
        results.append({"idx": offset+i, "q": q, "choices": choices, "expected": expected, "got": got, "correct": correct})

total = len(results)
correct_count = sum(1 for r in results if r["correct"])
print(f"\nTotal: {correct_count}/{total} = {100*correct_count/total:.1f}%")

got_dist = [0,0,0,0]
exp_dist = [0,0,0,0]
for r in results:
    if 0 <= r["got"] < 4: got_dist[r["got"]] += 1
    if 0 <= r["expected"] < 4: exp_dist[r["expected"]] += 1
print(f"Got: {got_dist}  Expected: {exp_dist}")

failures = [r for r in results if not r["correct"]]
print(f"Failures: {len(failures)}")

# Categorize
categories = {
    "ENERGY_HEAT": r"energy|heat|thermal|temperature|conduct|insulat|transfer|watt|calori|joule|kinetic|potential",
    "FORCE_MOTION": r"force|motion|speed|velocity|accelerat|gravity|friction|momentum|weight|mass|inertia|newton|push|pull|magnet",
    "WEATHER_CLIMATE": r"weather|climate|rain|snow|cloud|wind|humid|storm|season|warm|cold|forecast|atmosphere|air\s+pressure|ocean\s+current",
    "EARTH_GEOLOGY": r"earth|rock|mineral|igneous|sediment|metamorphic|erosion|fossil|soil|crust|mantle|volcano|earthquake|plate|tectonic|lava|magma|continent",
    "SPACE_ASTRONOMY": r"sun|moon|planet|star|orbit|solar\s+system|rotation|revolution|eclipse|galaxy|comet|asteroid|meteor|satellite|telescope",
    "LIFE_BODY": r"cell|organ|tissue|body|blood|heart|lung|brain|nerve|bone|muscle|digest|respirat|circulat|immune|reproduc|membrane|nucleus",
    "ECOLOGY_ENVIRON": r"ecosystem|habitat|food\s+chain|food\s+web|population|predator|prey|producer|consumer|decompos|photosynthes|biome|environment|extinct|adapt|evolut|species|conserv",
    "MATTER_CHEM": r"matter|solid|liquid|gas|atom|molecule|element|compound|mixture|solution|dissolv|chemical|physical|property|density|volume|boil|melt|freez|evaporat|condens|react",
    "GENETICS": r"gene|DNA|chromosom|inherit|trait|dominant|recessive|allele|mutation|offspring|heredit|cross|breed|select",
    "SCIENTIFIC_METHOD": r"hypothesis|experiment|variable|control|data|evidence|conclusion|observ|predict|measure|test|investig|procedure|scien\w+\s+method|peer\s+review|fact\s+vs|opinion",
    "WATER_CYCLE": r"water\s+cycle|evapor|precipit|groundwater|runoff|aquifer|watershed|fresh\s*water",
    "LIGHT_SOUND_WAVE": r"light|shadow|reflect|refract|absorb|transparent|lens|mirror|prism|color|spectrum|sound|vibrat|wave|frequency|pitch|echo",
    "PLANT": r"plant|root|stem|leaf|flower|seed|chlorophyll|chloroplast|xylem|phloem|pollina|germinat|nutrient|fertili",
}

cat_failures = collections.defaultdict(list)
for f in failures:
    matched = False
    for cat, pattern in categories.items():
        if re.search(pattern, f["q"], re.IGNORECASE):
            cat_failures[cat].append(f)
            matched = True
            break
    if not matched:
        cat_failures["OTHER"].append(f)

print("\n=== FAILURE CATEGORIES ===")
for cat, items in sorted(cat_failures.items(), key=lambda x: -len(x[1])):
    print(f"\n{cat} ({len(items)} failures):")
    for item in items[:4]:
        q_short = item["q"][:90]
        exp_text = item["choices"][item["expected"]][:50] if item["expected"] >= 0 else "?"
        got_text = item["choices"][item["got"]][:50] if 0 <= item["got"] < len(item["choices"]) else "?"
        print(f"  q{item['idx']}: {q_short}")
        print(f"    EXP[{item['expected']}]: {exp_text}")
        print(f"    GOT[{item['got']}]: {got_text}")

print(f"\n=== SUMMARY ===")
for cat, items in sorted(cat_failures.items(), key=lambda x: -len(x[1])):
    print(f"  {cat}: {len(items)}")
