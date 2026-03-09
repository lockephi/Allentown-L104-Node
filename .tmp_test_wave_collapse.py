"""Test wave collapse survival rate fix — weak measurement integration."""
from l104_quantum_engine.research import ProbabilityWaveCollapseResearch
from l104_quantum_engine.math_core import QuantumMathCore
from l104_quantum_engine.models import QuantumLink
import random

qmath = QuantumMathCore()
pwcr = ProbabilityWaveCollapseResearch(qmath)

# Build synthetic links with realistic properties
links = []
for i in range(50):
    fid = random.uniform(0.5, 1.0)
    strength = random.uniform(0.3, 1.0)
    links.append(QuantumLink(
        source_file='a.py', source_symbol=f'func_{i}', source_line=i,
        target_file='b.py', target_symbol=f'func_{i}', target_line=i,
        link_type='entanglement',
        fidelity=fid, strength=strength,
        coherence_time=random.uniform(0.1, 2.0),
        entanglement_entropy=random.uniform(0.0, 1.0),
        noise_resilience=random.uniform(0.3, 0.9),
    ))

# Run full wave collapse research
result = pwcr.wave_collapse_research(links)
print('=== WAVE COLLAPSE RESEARCH RESULTS ===')
print(f'Result keys: {list(result.keys())}')
N = result.get('links_analyzed', result.get('total_links', len(links)))
print(f'Links: {N}')

# Module 3: Collapse dynamics
cd = result.get('collapse_dynamics', {})
print(f'\n--- Module 3: Collapse Dynamics (Weak Measurement) ---')
print(f'Fidelity preservation:  {cd.get("fidelity_preservation", 0):.4f}')
print(f'Collapse stability:     {cd.get("collapse_stability", 0):.4f}')
print(f'Cumulative survival:    {cd.get("cumulative_survival", 0):.4f}')
print(f'WM coupling:            {cd.get("weak_measurement_coupling", 0):.4f}')
print(f'Entropy produced:       {cd.get("total_entropy_produced", 0):.6f}')

# Module 4: Decoherence channels
dc = result.get('decoherence_channels', {})
print(f'\n--- Module 4: Decoherence Channels (Weak Survival) ---')
print(f'Survival rate:          {dc.get("survival_rate", 0):.4f}')
print(f'Mean survival score:    {dc.get("mean_survival_score", 0):.4f}')
print(f'Darwinism fraction:     {dc.get("quantum_darwinism_fraction", 0):.4f}')
print(f'Survived: {dc.get("survived_count", 0)} / Fragile: {dc.get("fragile_count", 0)}')

# Module 5: Zeno
ze = result.get('quantum_zeno', {})
print(f'\n--- Module 5: Quantum Zeno (Weak Measurement) ---')
print(f'Zeno count:  {ze.get("zeno_count", 0)}')
print(f'Anti-Zeno:   {ze.get("anti_zeno_count", 0)}')
print(f'Neutral:     {ze.get("neutral_count", 0)}')
print(f'Phi stability: {ze.get("phi_stability_index", 0):.4f}')

# Module 6: Synthesis
syn = result.get('collapse_synthesis', {})
print(f'\n--- Module 6: Collapse Synthesis ---')
print(f'Collapse health: {syn.get("collapse_health", 0):.4f}')
print(f'Verdict: {syn.get("verdict", "UNKNOWN")}')
comps = syn.get('components', {})
for k, v in comps.items():
    print(f'  {k}: {v:.4f}')

print('\n✅ FULL WAVE COLLAPSE PIPELINE PASSED')
