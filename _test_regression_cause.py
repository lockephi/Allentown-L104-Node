#!/usr/bin/env python3
"""Check which v25d rules match the regression questions."""
import re

q1 = 'a certain type of hybrid car utilizes a braking system in which energy is recovered and stored in batteries during braking this reclaims energy that was previously lost this is an example of which energy conversion'
q2 = 'a student heats the same amount of two different liquids over bunsen burners each liquid is at room temperature and reaches its boiling point liquid a reaches the boiling point first compared with liquid a liquid b will'

rules_to_check = [
    ('nitrogen', r'(?:nitrogen.+fertiliz|fertiliz.+nitrogen|nitrogen\s+content.+(?:lake|bay|water)|nitrogen.+compound.+aquatic)'),
    ('neutraliz', r'(?:neutraliz|acid.+base.+react|HCl.+NaHCO|double\s+replacement)'),
    ('food_chain', r'(?:energy\s+transfer.+(?:animal|ecosystem|shoreline)|food\s+chain|energy\s+flow.+(?:between|animal|ecosystem))'),
    ('food_chain_v25', r'(?:energy.+transfer|transfer.+energy|food\s+chain|food\s+web).+(?:order|show|best|correct|flow)'),
    ('recycle', r'(?:(?:conserv|best|proper).+(?:resource|material)|(?:recycle|reuse).+(?:marker|carton|milk))'),
    ('speed', r'(?:average\s+speed.+(?:trip|whole|entire|day)|(?:rode|drove|travel).+stop.+(?:rode|drove|travel).+(?:average|speed))'),
    ('work', r'(?:work.+(?:force|distance|product)|(?:force|distance).+work|example.+work\b)'),
    ('energy_conv_old', r'energy.+(?:stored|recover|reclai|convert|lost)'),
    ('braking', r'brak.+energy'),
    ('boiling', r'boil.+point|heat.+liquid'),
]

print("Hybrid car question:")
for name, pat in rules_to_check:
    m = re.search(pat, q1, re.IGNORECASE)
    print(f'  {name}: {"MATCH" if m else "no match"}')

print("\nHeating liquids question:")
for name, pat in rules_to_check:
    m = re.search(pat, q2, re.IGNORECASE)
    print(f'  {name}: {"MATCH" if m else "no match"}')

# Check what _SCIENCE_RULES patterns might accidentally match
# The most risky new rules are:
# 1. work rule: 'example.+work\b' - does this match "example of which energy conversion"?
wk_qp = r'(?:work.+(?:force|distance|product)|(?:force|distance).+work|example.+work\b)'
print(f"\nWork rule matches hybrid car Q: {bool(re.search(wk_qp, q1, re.IGNORECASE))}")
print(f"Work rule matches heating Q: {bool(re.search(wk_qp, q2, re.IGNORECASE))}")

# The food chain rule changed wrong_pat to include 'plants?→bird'
# Does 'energy transfer' match the broader food chain q_pat?
fc_qp = r'(?:energy\s+transfer.+(?:animal|ecosystem|shoreline)|food\s+chain|energy\s+flow.+(?:between|animal|ecosystem))'
print(f"\nFood chain rule matches hybrid car Q: {bool(re.search(fc_qp, q1, re.IGNORECASE))}")
