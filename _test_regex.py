#!/usr/bin/env python3
"""Quick regex tests for the 6 failing rules."""
import re

q = 'in one day, a family in a car rode for 2 hours, stopped for 3 hours, and then rode for another 5 hours. during the day, the family traveled a total distance of 400 kilometers. what was their average speed for the whole trip?'
pat = r'(?:average\s+speed.+(?:trip|whole|entire|day)|(?:rode|drove|travel).+stop.+(?:rode|drove|travel).+(?:average|speed))'
m = re.search(pat, q, re.IGNORECASE)
print(f'Speed q_pat match: {bool(m)}')
if m: print(f'  Matched: {m.group()[:80]}')

# Test correct/wrong patterns for speed
for c in ['10 km/h', '20 km/h', '40 km/h', '50 km/h']:
    cm = re.search(r'\b40\b', c, re.IGNORECASE)
    wm = re.search(r'\b10\b|\b20\b|\b50\b|\b80\b|\b100\b', c, re.IGNORECASE)
    print(f'  "{c}": correct={bool(cm)}, wrong={bool(wm)}')

# Test H2O correct_pat (v25 rule)
print('\nH2O v25 rule:')
cp3 = r'H_\{2\}O(?!_)|\bH2O\b(?!2)|\bwater\b'
wp3 = r'H_\{2\}O_\{2\}|H2O2|hydrogen\s+peroxide|HO_\{2\}|\b2HO\b'
for c in ['2ho', 'ho_{2}', 'h_{2}o_{2}', 'h_{2}o']:
    cm = re.search(cp3, c, re.IGNORECASE)
    wm = re.search(wp3, c, re.IGNORECASE)
    print(f'  "{c}": correct={bool(cm)}, wrong={bool(wm)}')

# Test MIDDLE rule (line 6764)
print('\nH2O middle rule (line 6764):')
cp_mid = r'H_?\{?2\}?O(?!\d)|water|H2O(?!2)'
wp_mid = r'H_?\{?2\}?O_?\{?2\}?|H2O2|peroxide'
for c in ['2ho', 'ho_{2}', 'h_{2}o_{2}', 'h_{2}o']:
    cm = re.search(cp_mid, c, re.IGNORECASE)
    wm = re.search(wp_mid, c, re.IGNORECASE)
    print(f'  "{c}": correct={bool(cm)}, wrong={bool(wm)}, correct_group={cm.group() if cm else ""}')

# Test nitrogen/fish
print('\nNitrogen fish rule:')
nq = 'excess nitrogen fertilizers sometimes drain into waterways that flow into the chesapeake bay. this nitrogen may cause algae blooms, which reduce dissolved oxygen in the water. how does nitrogen negatively affect the chesapeake bay?'
npat = r'(?:nitrogen.+(?:drain|runoff|flow)|fertiliz.+(?:drain|flow|waterway))'
m = re.search(npat, nq, re.IGNORECASE)
print(f'  q_pat match: {bool(m)}')
ncp = r'(?:fish|population).+decrease|decrease|decline|die|fewer|less|reduce'
for c in ['fish births increase', 'fish populations decrease', 'sediment on the bottom of the bay decreases', 'the rate of water runoff into the bay increases']:
    cm = re.search(ncp, c, re.IGNORECASE)
    print(f'  correct_pat match "{c[:40]}": {bool(cm)} -> {cm.group() if cm else ""}')

# Test food chain
print('\nFood chain v25 rule:')
fc = r'(?:plant|grass|producer|sun).{0,15}(?:fish|shrimp|insect|worm).{0,15}(?:bird|hawk|owl|eagle|fox)'
for c in ['Fish -> Plants -> Birds', 'Plants -> Birds -> Fish', 'Plants -> Fish -> Birds', 'Fish -> Birds -> Plants']:
    cm = re.search(fc, c, re.IGNORECASE)
    print(f'  correct_pat match "{c}": {bool(cm)}')

# Test recycle/reuse
print('\nRecycle/reuse rule:')
rcp = r'reuse.+marker|marker.+reuse|recycle.+(?:carton|milk|container)|(?:carton|milk|container).+recycle'
rwp = r'recycle.+marker|marker.+recycle|reuse.+(?:carton|milk)'
for c in ['recycle the markers, reuse the milk cartons', 'reuse the markers, discard the milk cartons', 'discard the markers, reuse the milk cartons', 'reuse the markers, recycle the milk cartons']:
    cm = re.search(rcp, c, re.IGNORECASE)
    wm = re.search(rwp, c, re.IGNORECASE)
    print(f'  "{c[:50]}": correct={bool(cm)}, wrong={bool(wm)}')
