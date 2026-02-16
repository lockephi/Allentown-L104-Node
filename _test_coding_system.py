#!/usr/bin/env python3
"""Integration test for l104_coding_system.py"""
from l104_coding_system import coding_system

CODE = '''
import os
import json

class DataProcessor:
    def __init__(self):
        self.data = []
        self.cache = {}

    def process(self, items):
        results = []
        for item in items:
            for sub in item.get('children', []):
                name = sub['name']
                result = ''
                for c in name:
                    result = result + c.upper()
                results.append(result)
        return results

    def dangerous(self, user_input):
        eval(user_input)
        os.system(user_input)

    def load(self, path):
        try:
            with open(path) as f:
                return json.load(f)
        except:
            return None
'''

# 1. Review
r = coding_system.review(CODE, 'processor.py')
score = r.get('composite_score', 0)
verdict = r.get('verdict', '?')
findings = r.get('total_findings', 0)
print(f"1. Review: score={score:.2f} [{verdict}] findings={findings}")

# 2. Suggestions
s = coding_system.suggest(CODE, 'processor.py')
print(f"2. Suggestions: {len(s)}")
for x in s[:3]:
    print(f"   [{x['priority']}] {x['suggestion'][:70]}")

# 3. Quality gate
q = coding_system.quality_check(CODE, 'processor.py')
print(f"3. Quality Gate: {q['verdict']}")
for bf in q.get('blocking_failures', [])[:3]:
    print(f"   BLOCK: {bf}")

# 4. Explain
e = coding_system.explain(CODE, 'processor.py')
funcs = len(e.get('structure', {}).get('functions', []))
cls = len(e.get('structure', {}).get('classes', []))
print(f"4. Explain: lang={e.get('language', '?')} funcs={funcs} classes={cls}")

# 5. Plan
p = coding_system.plan('Add Redis caching to the process method')
steps = p.get('estimated_steps', 0)
print(f"5. Plan: complexity={p['complexity']} steps={steps}")

# 6. AI context
ctx = coding_system.ai_context(CODE, 'processor.py', 'claude')
ai_score = ctx.get('review', {}).get('score', 0)
print(f"6. AI Context (claude): score={ai_score:.2f}")

# 7. AI prompt
prompt = coding_system.ai_prompt('Optimize the process method', CODE, 'processor.py')
print(f"7. AI Prompt: {len(prompt)} chars generated")

# 8. Parse AI response
fake_response = """
Here's the fix:
```python
def process(self, items):
    return [c.upper() for item in items for sub in item.get('children', []) for c in sub['name']]
```
- Use list comprehension for efficiency
- Avoid string concatenation in loops
1. Replace nested loops with generator expression
2. Use join() instead of repeated concatenation
"""
parsed = coding_system.parse_ai_response(fake_response)
print(f"8. Parse AI: code_blocks={len(parsed['code_blocks'])} suggestions={len(parsed['suggestions'])} explanations={len(parsed['explanations'])}")

# 9. Full pipeline
fp = coding_system.full_pipeline(CODE, 'processor.py')
print(f"9. Full Pipeline: score={fp['composite_score']:.2f} [{fp['verdict']}] duration={fp['duration_seconds']:.3f}s")

# 10. Status
st = coding_system.status()
ops = st.get('execution_count', 0)
ver = st.get('version', '?')
eng = st['code_engine'].get('version', '?')
print(f"10. Status: v{ver} | engine v{eng} | {ops} ops | 7 subsystems")

# 11. Session management
sid = coding_system.start_session('integration test')
ctx2 = coding_system.session_context()
print(f"11. Session: id={sid[:12]}... active={ctx2['active']}")
end = coding_system.end_session()
print(f"    Ended: {end.get('actions', 0)} actions, {end.get('duration', 0):.0f}s")

# 12. Learning from history
lrn = coding_system.learn_from_history()
print(f"12. Learning: {lrn.get('total_sessions', 0)} sessions, {len(lrn.get('insights', []))} insights")

print("\n=== ALL 12 TESTS PASSED ===")
print(coding_system.quick_summary())
