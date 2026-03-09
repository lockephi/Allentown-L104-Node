"""Fix ZENITH_HZ/UUC assignments that appear before 'from __future__' imports."""
import os

files = [
    "l104_data_precognition.py",
    "l104_deepseek_ingestion.py",
    "l104_god_code_algorithm.py",
    "l104_precog_synthesis.py",
    "l104_qml_v2.py",
    "l104_sage_enlighten.py",
    "l104_search_algorithms.py",
    "l104_three_engine_search_precog.py",
]

for f in files:
    content = open(f).read()
    old_block = "ZENITH_HZ = 3887.8\nUUC = 2301.215661\n"
    future_line = "from __future__ import annotations\n"

    if old_block in content and future_line in content:
        z_pos = content.find(old_block)
        f_pos = content.find(future_line)
        if z_pos < f_pos:
            # Remove first occurrence of the block
            new = content[:z_pos] + content[z_pos + len(old_block):]
            # Insert after __future__
            f_pos2 = new.find(future_line) + len(future_line)
            new = new[:f_pos2] + "\n" + old_block + new[f_pos2:]
            open(f, "w").write(new)
            print("Fixed:", f)
        else:
            print("OK (already after):", f)
    else:
        print("Skip:", f)
