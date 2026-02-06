#!/usr/bin/env python3
"""Deep clean ALL corrupted story entries from L104 permanent memory."""

import json
import os

path = os.path.expanduser("~/Library/Application Support/L104Sovereign/permanent_memory.json")

print("Loading permanent memory...")
with open(path, 'r') as f:
    data = json.load(f)

total_removed = 0

# Check memories - use simple string contains check
if 'memories' in data and isinstance(data['memories'], list):
    original = len(data['memories'])
    data['memories'] = [m for m in data['memories']
                        if 'storyHeaders' not in str(m)
                        and 'storyParts' not in str(m)
                        and 'narrativeOpeners' not in str(m)
                        and 'narrativeMiddles' not in str(m)
                        and 'narrativeClosers' not in str(m)]
    removed = original - len(data['memories'])
    if removed > 0:
        print(f"  memories: removed {removed} corrupted entries")
        total_removed += removed

# Check history array
if 'history' in data and isinstance(data['history'], list):
    original = len(data['history'])
    cleaned_history = []
    for h in data['history']:
        h_str = json.dumps(h) if isinstance(h, dict) else str(h)
        if ('storyHeaders' not in h_str
            and 'storyParts' not in h_str
            and 'narrativeOpeners' not in h_str
            and 'narrativeMiddles' not in h_str
            and 'narrativeClosers' not in h_str):
            cleaned_history.append(h)
    data['history'] = cleaned_history
    removed = original - len(data['history'])
    if removed > 0:
        print(f"  history: removed {removed} corrupted entries")
        total_removed += removed

if total_removed > 0:
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nTotal removed: {total_removed} corrupted entries")
    print("Permanent memory cleaned and saved!")
else:
    print("No corrupted entries found")

# Verify
with open(path, 'r') as f:
    content = f.read()
count = content.count('storyHeaders')
print(f"\nVerification: {count} storyHeaders occurrences remaining")

# Also clean backup if exists
backup_path = os.path.expanduser("~/Library/Application Support/L104Sovereign/permanent_memory.backup.json")
if os.path.exists(backup_path):
    os.remove(backup_path)
    print("Removed corrupted backup file")

print("\nDone! Restart the app to see clean story output.")
