#!/usr/bin/env python3
"""Check and clean corrupted story entries from permanent memory"""
import json
import os

MEMORY_PATH = os.path.expanduser("~/Library/Application Support/L104Sovereign/permanent_memory.json")

with open(MEMORY_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

print('Keys:', list(data.keys()))

# Check 'history' key (not 'conversationHistory')
if 'history' in data:
    history = data['history']
    print(f'History count: {len(history)}')
    # Find corrupted ones
    corrupted = [e for e in history if 'storyHeaders' in str(e)]
    print(f'Corrupted entries: {len(corrupted)}')

    if corrupted:
        # Remove them
        data['history'] = [e for e in history if 'storyHeaders' not in str(e)]
        print(f'After cleaning: {len(data["history"])}')

        with open(MEMORY_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print('Cleaned and saved!')
    else:
        print('No corrupted entries found in history')
