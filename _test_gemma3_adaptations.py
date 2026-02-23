"""Quick test for Gemma 3 architectural adaptations — standalone (no full module load)."""
import math, time, sys

PHI = 1.618033988749895
GOD_CODE = 527.5184818492612

class G3:
    GEMMA3_SLIDING_WINDOW = 5
    GEMMA3_FINAL_SOFTCAP = 30.0
    GEMMA3_RMS_EPS = 1e-06

    def _gemma3_softcap_confidence(self, confidence, cap_value=None):
        if cap_value is None: cap_value = self.GEMMA3_FINAL_SOFTCAP
        if cap_value <= 0: return confidence
        return math.tanh(confidence / cap_value) * cap_value

    def _gemma3_rms_normalize(self, scores, eps=None):
        if eps is None: eps = self.GEMMA3_RMS_EPS
        if not scores: return scores
        numeric = [s for s in scores if isinstance(s, (int, float))]
        if not numeric: return scores
        mean_sq = sum(x * x for x in numeric) / len(numeric)
        rms = math.sqrt(mean_sq + eps)
        if rms < eps: return scores
        return [s / rms if isinstance(s, (int, float)) else s for s in scores]

    def _gemma3_sliding_window_context(self, message, memory):
        if not memory: return {"local_window": [], "global_summary": "", "window_coherence": 0.0}
        ws = self.GEMMA3_SLIDING_WINDOW
        local = memory[-ws:]
        glob = memory[:-ws] if len(memory) > ws else []
        concepts = {}
        stop = {"this","that","with","from","have","been","were","what","when","where","they","about","would","could","should","there"}
        for e in glob:
            for w in [w.lower().strip(".,!?") for w in e.get("content","").split() if len(w)>3]:
                if w.isalpha() and w not in stop: concepts[w] = concepts.get(w,0)+1
        top = [c for c,_ in sorted(concepts.items(), key=lambda x:x[1], reverse=True)[:max(10,len(concepts)//10)]]
        local_text = " ".join(e.get("content","") for e in local).lower()
        qw = set(w.lower().strip(".,!?") for w in message.split() if len(w)>2)
        overlap = sum(1 for w in qw if w in local_text)
        coh = math.tanh(min(1.0, overlap/max(len(qw),1)) * PHI)
        return {"local_window":local, "global_summary":" ".join(top), "global_concept_count":len(top),
                "local_count":len(local), "global_count":len(glob), "window_coherence":coh}

    def _gemma3_positional_decay(self, results, mode="sliding"):
        if not results: return results
        now = time.time()
        for i, r in enumerate(results):
            if not isinstance(r, dict): continue
            ts = r.get("timestamp", now - (len(results)-i)*3600)
            age_h = max(0, (now-ts)/3600)
            decay = math.exp(-age_h/(PHI*24)) if mode=="sliding" else math.exp(-age_h/(GOD_CODE*24))
            sc = r.get("score", 0.5)
            if isinstance(sc,(int,float)):
                r["score"] = sc * (0.3 + 0.7*decay)
        return results

m = G3()
ok = 0

print("=== Test 1: Logit Soft-Capping ===")
assert abs(m._gemma3_softcap_confidence(0.5, 30.0) - 0.5) < 0.02
assert m._gemma3_softcap_confidence(100.0, 30.0) < 30.1
assert m._gemma3_softcap_confidence(-50.0, 30.0) > -30.1
print(f"  softcap(0.5)={m._gemma3_softcap_confidence(0.5,30):.4f}  softcap(100)={m._gemma3_softcap_confidence(100,30):.4f}")
ok += 1; print("  PASS")

print("=== Test 2: RMSNorm ===")
s = [1.0, 2.0, 3.0, 4.0, 5.0]
n = m._gemma3_rms_normalize(s)
rms = math.sqrt(sum(x*x for x in s)/len(s))
for i,(a,b) in enumerate(zip(n,[x/rms for x in s])): assert abs(a-b)<0.001
print(f"  {s} -> {[round(x,4) for x in n]}")
ok += 1; print("  PASS")

print("=== Test 3: Sliding Window Context ===")
m2 = G3(); m2.GEMMA3_SLIDING_WINDOW = 3
mem = [{"role":"user","content":"quantum mechanics","timestamp":100},
       {"role":"assistant","content":"QM is fundamental","timestamp":101},
       {"role":"user","content":"entanglement","timestamp":200},
       {"role":"assistant","content":"particles linked","timestamp":201},
       {"role":"user","content":"decoherence","timestamp":300}]
r = m2._gemma3_sliding_window_context("decoherence quantum", mem)
assert r["local_count"]==3 and r["global_count"]==2 and r["window_coherence"]>0
print(f"  local={r['local_count']} global={r['global_count']} coherence={r['window_coherence']:.4f}")
ok += 1; print("  PASS")

print("=== Test 4: Positional Decay (Dual RoPE) ===")
now = time.time()
sl = m._gemma3_positional_decay([{"score":1.0,"timestamp":now},{"score":1.0,"timestamp":now-86400},{"score":1.0,"timestamp":now-864000}],mode="sliding")
gl = m._gemma3_positional_decay([{"score":1.0,"timestamp":now},{"score":1.0,"timestamp":now-86400},{"score":1.0,"timestamp":now-864000}],mode="global")
print(f"  sliding: {[round(r['score'],4) for r in sl]}")
print(f"  global:  {[round(r['score'],4) for r in gl]}")
assert sl[2]["score"] < gl[2]["score"], "Sliding must decay faster!"
ok += 1; print("  PASS")

print(f"\nResults: {ok}/4 tests passed")
print("ALL GEMMA 3 ADAPTATIONS VERIFIED" if ok==4 else "FAILED")
sys.exit(0 if ok==4 else 1)
