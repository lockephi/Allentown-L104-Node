import sys; sys.path.insert(0, '.')
from l104_asi.knowledge_data import KNOWLEDGE_NODES
facts = []
for n in KNOWLEDGE_NODES:
    facts.extend(n.get('facts', []))
from l104_asi.language_comprehension import LatentSemanticAnalyzer
lsa = LatentSemanticAnalyzer(n_components=50)
lsa.fit(facts)
print(list(lsa._vocab.keys())[:20])
