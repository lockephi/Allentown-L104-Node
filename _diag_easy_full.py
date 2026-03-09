#!/usr/bin/env python3
"""Diagnose ALL Easy failures (q0-499) — categorize patterns."""
import requests, re, sys, collections

API = "https://datasets-server.huggingface.co/rows"
PARAMS = {"dataset": "allenai/ai2_arc", "config": "ARC-Easy", "split": "test"}

sys.path.insert(0, ".")
from l104_asi.commonsense_reasoning import CommonsenseReasoningEngine
cr = CommonsenseReasoningEngine()

# Fetch in batches of 100
results = []
for offset in range(0, 500, 100):
    print(f"Fetching q{offset}-{offset+99}...")
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
        results.append({
            "idx": offset + i,
            "q": q,
            "choices": choices,
            "expected": expected,
            "got": got,
            "correct": correct
        })

total = len(results)
correct_count = sum(1 for r in results if r["correct"])
print(f"\nTotal: {correct_count}/{total} = {100*correct_count/total:.1f}%")

# Choice distribution
got_dist = [0,0,0,0]
for r in results:
    if 0 <= r["got"] < 4:
        got_dist[r["got"]] += 1
print(f"Got distribution: {got_dist}")

exp_dist = [0,0,0,0]
for r in results:
    if 0 <= r["expected"] < 4:
        exp_dist[r["expected"]] += 1
print(f"Expected distribution: {exp_dist}")

# Analyze failures
failures = [r for r in results if not r["correct"]]
print(f"\nFailures: {len(failures)}")

# Word frequency in failed questions
word_freq = collections.Counter()
for f in failures:
    words = re.findall(r'\b[a-z]{3,}\b', f["q"].lower())
    word_freq.update(set(words))  # unique words per question

# Top words in failures
print("\n=== TOP WORDS IN FAILED QUESTIONS ===")
for word, count in word_freq.most_common(60):
    if word not in {"the","and","which","that","what","from","this","for","are","was","has","have",
                    "with","will","does","not","its","when","how","can","one","most","best","than",
                    "all","they","been","more","into","may","but","use","would","some","also","these",
                    "their","about","used","each","two","many","other","called","below","above",
                    "following","describe","describes","statement","between","example","likely",
                    "result","after","during","could","should","process","found","show","shown",
                    "part","form","group","made","make","type","because","way","both","being","same",
                    "different","change","help","give","take","where","why","over","through","such",
                    "true","only","does","did","his","her","she","like","new","than","any"}:
        print(f"  {word}: {count}")

# Categorize by topic keywords
categories = {
    "ENERGY": r"energy|kinetic|potential|thermal|heat|electricity|electric|power|fuel|solar\s+panel|battery|circuit|watt|voltage|current",
    "FORCE_MOTION": r"force|motion|speed|velocity|accelerat|gravity|friction|momentum|weight|mass|inertia|newton|push|pull|balanced|unbalanced",
    "WEATHER_CLIMATE": r"weather|climate|temperature|rain|snow|cloud|wind|humid|storm|tornado|hurricane|season|warm|cold|forecast|atmosphere|air\s+pressure",
    "EARTH_SPACE": r"earth|sun|moon|planet|star|orbit|solar\s+system|rotation|revolution|season|axis|tilt|eclipse|galaxy|universe|comet|asteroid|meteor",
    "LIFE_SCIENCE": r"cell|organ|tissue|system|body|blood|heart|lung|brain|nerve|bone|muscle|digest|respirat|circulat|immune|reproduc",
    "ECOLOGY": r"ecosystem|habitat|food\s+chain|food\s+web|population|community|predator|prey|producer|consumer|decompos|photosynthes|biome|environment|species|extinct|adapt|evolut",
    "MATTER": r"matter|solid|liquid|gas|atom|molecule|element|compound|mixture|solution|dissolv|chemical|physical|property|density|volume|boil|melt|freez|evaporat|condens",
    "GENETICS": r"gene|DNA|chromosom|inherit|trait|dominant|recessive|allele|mutation|offspring|heredit|purebred|hybrid",
    "ROCK_MINERAL": r"rock|mineral|igneous|sediment|metamorphic|erosion|weather|fossil|soil|layer|crust|mantle|core|volcano|earthquake|plate\s+tectonic|lava|magma",
    "SCIENTIFIC_METHOD": r"hypothesis|experiment|variable|control|data|evidence|conclusion|observ|predict|measure|test|investig|procedure|fair\s+test|scien\w+\s+method",
    "LIGHT_SOUND": r"light|shadow|reflect|refract|absorb|transparent|translucent|opaque|lens|mirror|prism|color|spectrum|sound|vibrat|wave|frequency|pitch|echo",
    "WATER_CYCLE": r"water\s+cycle|evapor|condens|precipit|groundwater|runoff|aquifer|watershed|fresh\s*water|ocean|river|lake|stream|glacier",
    "PLANT_BIOLOGY": r"plant|root|stem|leaf|leaves|flower|seed|photosynthes|chlorophyll|chloroplast|xylem|phloem|pollina|germinat|nutrients|fertiliz|soil",
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
    for item in items[:5]:
        q_short = item["q"][:80]
        exp_text = item["choices"][item["expected"]][:40] if item["expected"] >= 0 else "?"
        got_text = item["choices"][item["got"]][:40] if 0 <= item["got"] < len(item["choices"]) else "?"
        print(f"  q{item['idx']}: {q_short}")
        print(f"    EXP[{item['expected']}]: {exp_text}")
        print(f"    GOT[{item['got']}]: {got_text}")

print(f"\n=== CATEGORY SUMMARY ===")
for cat, items in sorted(cat_failures.items(), key=lambda x: -len(x[1])):
    print(f"  {cat}: {len(items)}")
