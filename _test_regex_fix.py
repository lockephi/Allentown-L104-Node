#!/usr/bin/env python3
"""Test that the regex patterns no longer cause catastrophic backtracking."""
import time

t0 = time.time()
from l104_asi.language_comprehension import RelationTripleExtractor
t1 = time.time()
print(f"Import: {t1-t0:.2f}s")

ext = RelationTripleExtractor()

# Test with diverse facts including ones that previously caused backtracking
test_facts = [
    "The mitochondria is the powerhouse of the cell",
    "DNA stands for deoxyribonucleic acid",
    "Newton discovered gravity by observing a falling apple",
    "Photosynthesis uses sunlight to convert carbon dioxide and water into glucose",
    "The speed of light is approximately 299792458 meters per second",
    "Iron is a chemical element with symbol Fe and atomic number 26",
    "Shakespeare wrote Hamlet and Macbeth and Romeo and Juliet among many other works",
    "The SI unit of force is the newton",
    "Water consists of hydrogen and oxygen atoms bonded together",
    "Smoking causes lung cancer and other respiratory diseases",
    "The Eiffel Tower is located in Paris",
    "E = mc^2",
    "HTTP stands for HyperText Transfer Protocol",
    "Einstein published the theory of general relativity in 1915",
    # Long facts that previously caused backtracking
    "The process of photosynthesis in plants involves the absorption of sunlight by chlorophyll molecules in the chloroplasts which then convert carbon dioxide and water into glucose and oxygen through a series of complex biochemical reactions",
    "In molecular biology the central dogma describes the flow of genetic information within a biological system from DNA to RNA to protein through the processes of transcription and translation",
    "The human immune system is a complex network of cells and proteins that defends the body against infection and disease including both innate immunity and adaptive immunity",
] * 600  # 10,200 facts

t2 = time.time()
ext.index_all_facts(test_facts)
t3 = time.time()

print(f"Indexed {len(test_facts)} facts in {t3-t2:.3f}s")
print(f"Triples extracted: {len(ext._triples)}")
if ext._triples:
    print(f"Sample triples:")
    seen = set()
    for t in ext._triples:
        key = str(t)
        if key not in seen:
            seen.add(key)
            print(f"  {t}")
            if len(seen) >= 8:
                break

print(f"\nTotal time: {t3-t0:.2f}s")
if t3-t2 < 5.0:
    print("PASS: No catastrophic backtracking detected")
else:
    print("WARN: Indexing took over 5s, may still have issues")
