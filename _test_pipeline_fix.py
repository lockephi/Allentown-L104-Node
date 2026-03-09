"""Quick test of pipeline_solve fix."""
import os; os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from l104_asi import asi_core

# Test 1: pipeline_solve with natural language
result = asi_core.pipeline_solve('What is 2+2?')
print(f'pipeline_solve solution: {result.get("solution")}')
print(f'channel: {result.get("channel")}')

# Test 2: direct hub with query
result2 = asi_core.solution_hub.solve({'query': 'What is 2+2?'})
print(f'direct hub (query) solution: {result2.get("solution")}')

# Test 3: direct hub with expression
result3 = asi_core.solution_hub.solve({'expression': '2+2'})
print(f'direct hub (expression) solution: {result3.get("solution")}')

# Test 4: harder math
result4 = asi_core.pipeline_solve('calculate 15*3+7')
print(f'pipeline 15*3+7 solution: {result4.get("solution")}')

# Test 5: get_status returns 'status' key
status = asi_core.get_status()
print(f'status key present: {"status" in status}')
print(f'status value: {status.get("status")}')
