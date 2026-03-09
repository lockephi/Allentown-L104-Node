#!/usr/bin/env python3
"""Diagnose ARC Challenge regression v9→v10"""
import json, sys
sys.path.insert(0, '.')

with open('benchmark_full_online_results.json') as f:
    d = json.load(f)

dr = d.get('detailed_results', {})
print('Detailed results keys:', list(dr.keys()))

if 'ARC' in dr:
    arc = dr['ARC']
    if isinstance(arc, list):
        ch_fail = [r for r in arc if not r.get('correct') and r.get('category') == 'arc_challenge']
        ea_fail = [r for r in arc if not r.get('correct') and r.get('category') == 'arc_easy']
        ch_pass = [r for r in arc if r.get('correct') and r.get('category') == 'arc_challenge']
        ea_pass = [r for r in arc if r.get('correct') and r.get('category') == 'arc_easy']
        print(f'ARC Challenge: {len(ch_pass)}/{len(ch_pass)+len(ch_fail)} correct')
        print(f'ARC Easy: {len(ea_pass)}/{len(ea_pass)+len(ea_fail)} correct')
        print()

        # Show 10 challenge failures
        print('=== ARC CHALLENGE FAILURES (first 15) ===')
        for i, f in enumerate(ch_fail[:15]):
            q = f.get('question', '')[:130]
            choices = f.get('choices', [])
            exp = f.get('expected')
            got = f.get('predicted')
            print(f'\n{i+1}. Q: {q}')
            for ci, c in enumerate(choices):
                marker = '✓' if ci == exp else ('✗' if ci == got else ' ')
                print(f'   [{marker}] {ci}: {c[:80]}')
            print(f'   Expected={exp}  Got={got}')

        # Show 10 easy failures
        print('\n\n=== ARC EASY FAILURES (first 15) ===')
        for i, f in enumerate(ea_fail[:15]):
            q = f.get('question', '')[:130]
            choices = f.get('choices', [])
            exp = f.get('expected')
            got = f.get('predicted')
            print(f'\n{i+1}. Q: {q}')
            for ci, c in enumerate(choices):
                marker = '✓' if ci == exp else ('✗' if ci == got else ' ')
                print(f'   [{marker}] {ci}: {c[:80]}')
            print(f'   Expected={exp}  Got={got}')

    elif isinstance(arc, dict):
        print('ARC is dict with keys:', list(arc.keys())[:10])
        by_cat = arc.get('by_category', {})
        print('By category:', {k: v for k, v in by_cat.items()})
        details = arc.get('details', [])
        if isinstance(details, list):
            ch_fail = [r for r in details if not r.get('correct') and r.get('category') == 'arc_challenge']
            ea_fail = [r for r in details if not r.get('correct') and r.get('category') == 'arc_easy']
            ch_pass = [r for r in details if r.get('correct') and r.get('category') == 'arc_challenge']
            ea_pass = [r for r in details if r.get('correct') and r.get('category') == 'arc_easy']
            print(f'ARC Challenge: {len(ch_pass)}/{len(ch_pass)+len(ch_fail)} correct')
            print(f'ARC Easy: {len(ea_pass)}/{len(ea_pass)+len(ea_fail)} correct')

            # Show challenge failures
            print('\n=== ARC CHALLENGE FAILURES (first 15) ===')
            for i, f in enumerate(ch_fail[:15]):
                q = f.get('question', '')[:130]
                choices = f.get('choices', [])
                exp = f.get('expected')
                got = f.get('predicted')
                print(f'\n{i+1}. Q: {q}')
                for ci, c in enumerate(choices):
                    marker = '✓' if ci == exp else ('✗' if ci == got else ' ')
                    print(f'   [{marker}] {ci}: {c[:80]}')
                print(f'   Expected={exp}  Got={got}')

            # Show easy failures
            print('\n\n=== ARC EASY FAILURES (first 15) ===')
            for i, f in enumerate(ea_fail[:15]):
                q = f.get('question', '')[:130]
                choices = f.get('choices', [])
                exp = f.get('expected')
                got = f.get('predicted')
                print(f'\n{i+1}. Q: {q}')
                for ci, c in enumerate(choices):
                    marker = '✓' if ci == exp else ('✗' if ci == got else ' ')
                    print(f'   [{marker}] {ci}: {c[:80]}')
                print(f'   Expected={exp}  Got={got}')
        else:
            print('details is not a list:', type(details))
