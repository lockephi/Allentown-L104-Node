"""Quick test: verify randomized SVD fix for LSA."""
import time, sys
sys.path.insert(0, ".")

print("=== LSA SVD Timing Test ===")

print("Loading KB data...")
t0 = time.time()
from l104_asi.knowledge_data import KNOWLEDGE_NODES as MMLU_KNOWLEDGE_NODES
facts = []
for n in MMLU_KNOWLEDGE_NODES:
    facts.extend(n.get("facts", []))
print(f"  {len(facts)} facts loaded in {time.time()-t0:.1f}s")

print("Fitting LSA with randomized SVD...")
from l104_asi.language_comprehension import LatentSemanticAnalyzer
lsa = LatentSemanticAnalyzer(n_components=50)
t1 = time.time()
lsa.fit(facts)
t2 = time.time()
print(f"  LSA fit completed in {t2-t1:.3f}s")
print(f"  Fitted: {lsa._fitted}")

if lsa._fitted:
    print(f"  U shape: {lsa._U.shape}")
    print(f"  Sigma[:5]: {lsa._Sigma[:5]}")
    print(f"  Doc vectors shape: {lsa._doc_vectors.shape}")

    # Quick similarity test
    scores = lsa.query_similarity("quantum physics entanglement", top_k=3)
    print(f"  Query test: {len(scores)} hits")
    for doc, score in scores[:3]:
        print(f"    [{score:.3f}] {doc[:80]}...")
else:
    print("  FAILED — LSA did not fit!")

print("DONE")
