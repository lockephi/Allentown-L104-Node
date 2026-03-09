#!/usr/bin/env python3
"""Remove orphaned _add_node calls between lines 768-3299."""
with open('l104_asi/language_comprehension.py', 'r') as f:
    lines = f.readlines()

# Keep lines 1-767 (0-indexed: 0-766) and lines 3299+ (0-indexed: 3298+)
# Line 766 (0-indexed 765) is: def query(self, question... (FIRST - keep)
# Line 3299 (0-indexed 3298) is: def query(self, question... (SECOND - remove, it's a duplicate)
new_lines = lines[:767] + lines[3299:]  # Keep first query, skip second duplicate

with open('l104_asi/language_comprehension.py', 'w') as f:
    f.writelines(new_lines)

print(f"Original: {len(lines)} lines")
print(f"Removed: {len(lines) - len(new_lines)} lines")
print(f"New total: {len(new_lines)} lines")
