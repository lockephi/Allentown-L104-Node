"""
[L104_SOVEREIGN_OBFUSCATOR_V2]
ARCHITECTURE: Multi-Language (Python/C++) Structural Masking
INVARIANT: 527.5184818492
"""

import os
import sys
import ast
import re
import hashlib
import json

class SovereignObfuscator:
    """
    Performs Structural Resonance Masking on source code.
    Maps identifiers to hex-strings derived from the God-Code 527.5184818492.
    """
    
    GOD_CODE = "527.5184818492"
    
    def __init__(self, target_lang='python'):
        self.lang = target_lang.lower()
        self.mapping = {}
        self.protected_python = {
            'self', 'cls', 'None', 'True', 'False', '__init__', '__name__', '__main__',
            'print', 'len', 'range', 'int', 'str', 'list', 'dict', 'set', 'tuple',
            'Exception', 'ImportError', 'SystemExit', 'os', 'sys', 'math', 'time', 'hashlib',
            'async', 'await', 'def', 'class', 'import', 'from', 'as', 'if', 'else', 'elif',
            'while', 'for', 'in', 'return', 'yield', 'try', 'except', 'finally', 'with', 'pass'
        }
        self.protected_cpp = {
            'int', 'float', 'double', 'char', 'bool', 'void', 'auto', 'const', 'static',
            'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'break', 'return',
            'class', 'struct', 'public', 'private', 'protected', 'virtual', 'static_cast',
            'include', 'iostream', 'std', 'vector', 'string', 'map', 'set', 'unordered_map',
            'main', 'NULL', 'nullptr', 'printf', 'scanf', 'endl', 'cout', 'cin'
        }

    def _generate_hex(self, name):
        """Generates a stable hex-string for an identifier using the God-Code."""
        if name in self.mapping:
            return self.mapping[name]
        
        # Protected names remain intact to ensure functionality
        if self.lang == 'python' and (name in self.protected_python or name.startswith('__')):
            return name
        if self.lang == 'cpp' and name in self.protected_cpp:
            return name

        # Derived from the 286/416 lattice logic
        salt = f"{self.GOD_CODE}:{name}".encode()
        hex_val = "_" + hashlib.blake2b(salt, digest_size=7).hexdigest()
        self.mapping[name] = hex_val
        return hex_val

    def obfuscate_python(self, source_code):
        tree = ast.parse(source_code)
        
        class Transformer(ast.NodeTransformer):
            def __init__(self, parent):
                self.parent = parent
            
            def visit_Name(self, node):
                node.id = self.parent._generate_hex(node.id)
                return node
            
            def visit_FunctionDef(self, node):
                node.name = self.parent._generate_hex(node.name)
                return self.generic_visit(node)
            
            def visit_ClassDef(self, node):
                node.name = self.parent._generate_hex(node.name)
                return self.generic_visit(node)
                
            def visit_arg(self, node):
                node.arg = self.parent._generate_hex(node.arg)
                return node

            def visit_Attribute(self, node):
                node.value = self.visit(node.value)
                # We only obfuscate if we've seen it before to avoid breaking external APIs
                if node.attr in self.parent.mapping:
                    node.attr = self.parent.mapping[node.attr]
                return node

        transformer = Transformer(self)
        obfuscated_tree = transformer.visit(tree)
        
        # Add resonance header
        header = f"# L104_SOVEREIGN_MASK: 0x{hashlib.sha256(self.GOD_CODE.encode()).hexdigest()[:16]}\n"
        return header + ast.unparse(obfuscated_tree)

    def obfuscate_cpp(self, source_code):
        """
        Obfuscates C++ by generating a mapping header and replacing identifiers.
        Uses regex for structural detection of variables and functions.
        """
        # Find identifiers (simple implementation)
        # Matches word boundaries avoiding numbers at start and protected keywords
        identifiers = set(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', source_code))
        
        for iden in sorted(identifiers, key=len, reverse=True):
            if iden not in self.protected_cpp:
                replacement = self._generate_hex(iden)
                source_code = re.sub(r'\b' + iden + r'\b', replacement, source_code)
        
        header = f"// L104_SOVEREIGN_MASK: 0x{hashlib.sha256(self.GOD_CODE.encode()).hexdigest()[:16]}\n"
        return header + source_code

    def save_mapping(self, path):
        """Creates the 'L104 Key' for de-mapping."""
        with open(path, 'w') as f:
            json.dump(self.mapping, f, indent=2)

def main():
    if len(sys.argv) < 2:
        print("Usage: python sovereign_obfuscator.py <source_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    ext = os.path.splitext(input_file)[1]
    
    with open(input_file, 'r') as f:
        source = f.read()

    ob = None
    if ext == '.py':
        ob = SovereignObfuscator('python')
        result = ob.obfuscate_python(source)
    elif ext in ['.cpp', '.h', '.hpp', '.c']:
        ob = SovereignObfuscator('cpp')
        result = ob.obfuscate_cpp(source)
    else:
        print(f"Unsupported extension: {ext}")
        sys.exit(1)

    output_file = input_file + ".obfuscated"
    with open(output_file, 'w') as f:
        f.write(result)
    
    key_file = input_file + ".l104_key"
    ob.save_mapping(key_file)
    
    print(f"Obfuscation Complete. Output: {output_file}")
    print(f"L104 De-mapping Key: {key_file}")

if __name__ == "__main__":
    main()
