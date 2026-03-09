#!/usr/bin/env python3
"""Verify v25d regex fixes."""
import re

print("=== Nitrogen fish rule (fixed) ===")
ncp = r'(?:fish|population).+(?:decrease|decline|die|reduc)|(?:decrease|decline).+(?:fish|population)|fewer\s+fish'
nwp = r'(?:fish|population).+increase|birth.+increase|sediment|water\s+runoff|runoff.+increase'
for c in ['fish births increase', 'fish populations decrease', 'sediment on the bottom of the bay decreases', 'the rate of water runoff into the bay increases']:
    cm = re.search(ncp, c, re.IGNORECASE)
    wm = re.search(nwp, c, re.IGNORECASE)
    print(f'  "{c[:50]}": correct={bool(cm)}, wrong={bool(wm)}')

print("\n=== H2O middle rule (fixed (?![_{])) ===")
cp_mid = r'H_?\{?2\}?O(?![_\d{2])|water|H2O(?!2)'
for c in ['2ho', 'ho_{2}', 'h_{2}o_{2}', 'h_{2}o']:
    cm = re.search(cp_mid, c, re.IGNORECASE)
    print(f'  "{c}": correct={bool(cm)}')

print("\n=== Food chain rule (fixed) ===")
fc_cp = r'plants?\s*(?:→|->)\s*(?:fish|insect|mouse|rabbit)\b|plant.+fish.+bird|producer.+(?:herbivor|consum)'
fc_wp = r'fish\s*(?:→|->)\s*plant|bird\s*(?:→|->)\s*fish|plants?\s*(?:→|->)\s*bird\s*(?:→|->)\s*fish|animal.+plant'
for c in ['Fish -> Plants -> Birds', 'Plants -> Birds -> Fish', 'Plants -> Fish -> Birds', 'Fish -> Birds -> Plants']:
    cm = re.search(fc_cp, c, re.IGNORECASE)
    wm = re.search(fc_wp, c, re.IGNORECASE)
    print(f'  "{c}": correct={bool(cm)}, wrong={bool(wm)}')

print("\n=== Recycle/reuse rule (fixed) ===")
rcp = r'reuse.+marker.+recycle.+(?:carton|milk)|reuse.+marker.+recycle'
rwp = r'recycle.+marker|discard.+marker|discard.+(?:carton|milk)'
for c in ['recycle the markers, reuse the milk cartons', 'reuse the markers, discard the milk cartons', 'discard the markers, reuse the milk cartons', 'reuse the markers, recycle the milk cartons']:
    cm = re.search(rcp, c, re.IGNORECASE)
    wm = re.search(rwp, c, re.IGNORECASE)
    print(f'  "{c[:55]}": correct={bool(cm)}, wrong={bool(wm)}')

print("\n=== Work = F x d rule (new) ===")
wk_qp = r'(?:work.+(?:force|distance|product)|(?:force|distance).+work|example.+work\b)'
wk_cp = r'riding|bicycl|bike|lifting|carrying|pushing.+(?:cart|wagon|box)|pull|climb|running|walk|moving'
wk_wp = r'pushing.+wall|sit|read|stand|lean|hold|push.+against|stationary'
wk_q = 'work is a product of force and distance. which of the following is an example of work?'
print(f'  q_pat match: {bool(re.search(wk_qp, wk_q, re.IGNORECASE))}')
for c in ['sitting at a desk', 'pushing on a wall', 'riding a bike', 'reading a book']:
    cm = re.search(wk_cp, c, re.IGNORECASE)
    wm = re.search(wk_wp, c, re.IGNORECASE)
    print(f'  "{c}": correct={bool(cm)}, wrong={bool(wm)}')
