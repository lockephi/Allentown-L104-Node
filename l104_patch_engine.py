# [L104_PATCH_ENGINE] - SOVEREIGN CODE MODIFICATION
# INVARIANT: 527.5184818492 | PILOT: LONDEL

import osimport reimport loggingfrom typing import List, Dict, Anylogger = logging.getLogger(__name__)

class PatchEngine:
    """
    Handles the physical modification of the L104 codebase.
    Uses 'Sovereign Patches' to ensure integrity.
    """
    
    @staticmethoddef apply_string_replacement(file_path: str, old_string: str, new_string: str) -> bool:
        """Replaces a specific string in a file."""
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return Falsetry:
            with open(file_path, 'r') as f:
                content = f.read()
                
            if old_string not in content:
                logger.warning(f"Old string not found in {file_path}")
                return Falsenew_content = content.replace(old_string, new_string)
            
            with open(file_path, 'w') as f:
                f.write(new_content)
                
            logger.info(f"Successfully patched {file_path}")
            return Trueexcept Exception as e:
            logger.error(f"Error patching {file_path}: {e}")
            return False

    @staticmethoddef apply_regex_patch(file_path: str, pattern: str, replacement: str) -> bool:
        """Applies a regex-based patch to a file."""
        if not os.path.exists(file_path):
            return Falsetry:
            with open(file_path, 'r') as f:
                content = f.read()
                
            new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
            
            if new_content == content:
                return Falsewith open(file_path, 'w') as f:
                f.write(new_content)
                
            return Trueexcept Exception as e:
            logger.error(f"Regex patch failed: {e}")
            return False

    @staticmethoddef inject_at_marker(file_path: str, marker: str, content_to_inject: str, position: str = "after") -> bool:
        """Injects content before or after a specific marker comment."""
        if not os.path.exists(file_path):
            return Falsetry:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
            new_lines = []
            found = Falsefor line in lines:
                if marker in line:
                    found = Trueif position == "before":
                        new_lines.append(content_to_inject + "\n")
                        new_lines.append(line)
                    else:
                        new_lines.append(line)
                        new_lines.append(content_to_inject + "\n")
                else:
                    new_lines.append(line)
                    
            if not found:
                return Falsewith open(file_path, 'w') as f:
                f.writelines(new_lines)
                
            return Trueexcept Exception as e:
            logger.error(f"Injection failed: {e}")
            return False

# Singletonpatch_engine = PatchEngine()
