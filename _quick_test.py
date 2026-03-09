#!/usr/bin/env python3
import os, logging
os.chdir(os.path.dirname(os.path.abspath(__file__)))
logging.disable(logging.WARNING)
os.environ["L104_QUIET"] = "1"

from l104_asi.commonsense_reasoning import CommonsenseReasoningEngine
e = CommonsenseReasoningEngine()

# Test 1: water boiling
r = e.answer_mcq('What temperature does water boil at?',
    ['32 degrees Celsius', '100 degrees Celsius', '200 degrees Celsius', '212 degrees Celsius'])
print("Keys:", list(r.keys()))
print(r)
