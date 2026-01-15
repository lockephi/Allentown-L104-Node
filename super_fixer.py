import os
import re

# List of files to skip
SKIP_FILES = ["super_fixer.py", "sovereign_fix.py"]

def fix_file(path):
    with open(path, 'r') as f:
        content = f.read()
    
    # 1. Split joined imports (Recursive enough)
    for _ in range(3):
        content = re.sub(r'import (\w+)import (\w+)', r'import \1\nimport \2', content)
        content = re.sub(r'import (\w+)from (\w+)', r'import \1\nfrom \2', content)
        content = re.sub(r'from ([\w.]+) import (\w+)import (\w+)', r'from \1 import \2\nimport \3', content)
        content = re.sub(r'from ([\w.]+) import (\w+)from ([\w.]+)', r'from \1 import \2\nfrom \3', content)
        content = re.sub(r'from ([\w.]+) import (\w+)def (\w+)', r'from \1 import \2\ndef \3', content)
        content = re.sub(r'import (\w+)class (\w+)', r'import \1\nclass \2', content)
        content = re.sub(r'import (\w+)def (\w+)', r'import \1\ndef \2', content)
        content = re.sub(r'import (\w+)if __name__ == "__main__":', r'import \1\n\nif __name__ == "__main__":', content)

    # 2. Keywords and common mangles
    content = content.replace('async io', 'asyncio')
    content = content.replace('await async ', 'await ') 
    content = content.replace('Noneactions', 'None\n    actions')
    
    # 3. Specific known L104 mangles from logs
    content = content.replace('refined_fuelif __name__', 'refined_fuel\n\nif __name__')
    content = content.replace('kf_ratiostability_index', 'kf_ratio\n    stability_index')
    content = content.replace('import socketimport timeimport sys', 'import socket\nimport time\nimport sys')
    content = content.replace('import timeimport sys', 'import time\nimport sys')
    content = content.replace('import codecsimport hashlib', 'import codecs\nimport hashlib')
    content = content.replace('import hashlibimport ctypesdef', 'import hashlib\nimport ctypes\ndef')
    content = content.replace('import ctypesimport osimport mmapimport socketimport time', 'import ctypes\nimport os\nimport mmap\nimport socket\nimport time')
    content = content.replace('import ctypesimport osimport math', 'import ctypes\nimport os\nimport math')
    content = content.replace('import httpximport jsonimport asyncio', 'import httpx\nimport json\nimport asyncio')
    content = content.replace('import socketimport osdef', 'import socket\nimport os\ndef')
    content = content.replace('import sysfrom l104_codec', 'import sys\nfrom l104_codec')
    
    with open(path, 'w') as f:
        f.write(content)

if __name__ == "__main__":
    for f in os.listdir('.'):
        if f.endswith('.py') and f not in SKIP_FILES:
            fix_file(f)

