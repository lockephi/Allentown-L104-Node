import time
from l104_local_intellect import LocalIntellect
li = LocalIntellect()
print("v11.3 ULTRA-BANDWIDTH LocalIntellect")
print("EPR Links:", li.entanglement_state.get("epr_links", 0))
print()

queries = ["What is the GOD CODE?", "Explain quantum entanglement", "hello", "What is phi?", "How does L104 work?", "What is consciousness?"]
for q in queries:
    t1=time.time()
    r=li.think(q)
    cold = int((time.time()-t1)*1000)
    t2=time.time()
    r=li.think(q)
    cached = round((time.time()-t2)*1000, 3)
    print(f"{q[:28]:28} Cold: {cold:5}ms  Cached: {cached}ms")
