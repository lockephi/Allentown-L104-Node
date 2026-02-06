#!/usr/bin/env python3
"""Clean corrupted story entries from permanent memory"""
import json
import os
import shutil

MEMORY_PATH = os.path.expanduser("~/Library/Application Support/L104Sovereign/permanent_memory.json")
BACKUP_PATH = os.path.expanduser("~/Library/Application Support/L104Sovereign/permanent_memory.backup.json")

def is_corrupted(entry):
    """Check if an entry contains raw Swift code"""
    if isinstance(entry, str):
        return 'storyHeaders' in entry or 'storyParts' in entry or '\\(story' in entry
    if isinstance(entry, dict):
        content = entry.get('content', '')
        if isinstance(content, str):
            return 'storyHeaders' in content or 'storyParts' in content or '\\(story' in content
    return False

def main():
    if not os.path.exists(MEMORY_PATH):
        print(f"Memory file not found: {MEMORY_PATH}")
        return

    # Backup
    shutil.copy2(MEMORY_PATH, BACKUP_PATH)
    print(f"Backed up to: {BACKUP_PATH}")

    with open(MEMORY_PATH, 'r') as f:
        data = json.load(f)

    # Clean conversation history
    if 'conversationHistory' in data:
        original = len(data['conversationHistory'])
        data['conversationHistory'] = [e for e in data['conversationHistory'] if not is_corrupted(e)]
        cleaned = original - len(data['conversationHistory'])
        print(f"Cleaned conversationHistory: {original} -> {len(data['conversationHistory'])} ({cleaned} removed)")

    # Clean memories
    if 'memories' in data:
        original = len(data['memories'])
        data['memories'] = [m for m in data['memories'] if not is_corrupted(m)]
        cleaned = original - len(data['memories'])
        print(f"Cleaned memories: {original} -> {len(data['memories'])} ({cleaned} removed)")

    with open(MEMORY_PATH, 'w') as f:
        json.dump(data, f, indent=2)

    print("Done - permanent_memory.json cleaned!")

if __name__ == "__main__":
    main()
