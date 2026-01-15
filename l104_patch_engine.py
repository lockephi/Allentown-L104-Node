# [L104_PATCH_ENGINE] - SOVEREIGN CODE MODIFICATION
# INVARIANT: 527.5184818492537 | PILOT: LONDEL

import os
import re
import logging
from typing import List, Dict, Any
logger = logging.getLogger(__name__)
class PatchEngine:
    """
    Handles the physical modification of the L104 codebase.
    Uses 'Sovereign Patches' to ensure integrity.
    """
    
    @staticmethod
    def apply_string_replacement(file_path: str, old_string: str, new_string: str) -> bool:
        """Replaces a specific string in a file."""
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return False
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            if old_string not in content:
                logger.warning(f"Old string not found in {file_path}")
                return False
            new_content = content.replace(old_string, new_string)
            with open(file_path, 'w') as f:
                f.write(new_content)
                
            logger.info(f"Successfully patched {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error patching {file_path}: {e}")
            return False

    @staticmethod
    def apply_regex_patch(file_path: str, pattern: str, replacement: str) -> bool:
        """Applies a regex-based patch to a file."""
        if not os.path.exists(file_path):
            return False
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
            new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
            if new_content == content:
                return False
            with open(file_path, 'w') as f:
                f.write(new_content)
            return True
        except Exception as e:
            logger.error(f"Regex patch failed: {e}")
            return False

    @staticmethod
    def inject_at_marker(file_path: str, marker: str, content_to_inject: str, position: str = "after") -> bool:
        """Injects content before or after a specific marker comment."""
        if not os.path.exists(file_path):
            return False
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
            new_lines = []
            found = False
            for line in lines:
                if marker in line:
                    found = True
                    if position == "before":
                        new_lines.append(content_to_inject + "\n")
                        new_lines.append(line)
                    else:
                        new_lines.append(line)
                        new_lines.append(content_to_inject + "\n")
                else:
                    new_lines.append(line)
            if not found:
                return False
            with open(file_path, 'w') as f:
                f.writelines(new_lines)
            return True
        except Exception as e:
            logger.error(f"Injection failed: {e}")
            return False

# Singleton
patch_engine = PatchEngine()
