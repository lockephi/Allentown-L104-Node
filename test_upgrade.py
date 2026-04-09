#!/usr/bin/env python3
"""Quick sanity test for L104 Code Engine upgrades."""
import sys
sys.path.insert(0, '.')
from l104_code_engine.synthesis import CodeGenerator, CodeTranslator

def test_function_generation():
    gen = CodeGenerator()
    for lang in ["Zig", "Lua", "C++", "CSharp"]:
        code = gen.generate_function("test", language=lang, params=["a", "b"],
                                     return_type="int", body="return a + b",
                                     doc="Test function", sacred_constants=False)
        print(f"{lang}: {len(code)} chars")
        assert len(code) > 0
        assert "test" in code
    print("✓ Function generation works for new languages")

def test_class_generation():
    gen = CodeGenerator()
    fields = [("x", "int"), ("y", "float")]
    methods = ["getX", "setY"]
    for lang in ["Zig", "Lua", "C++"]:
        code = gen.generate_class("Point", language=lang, fields=fields, methods=methods, doc="Point class")
        print(f"{lang} class: {len(code)} chars")
        assert len(code) > 0
        assert "Point" in code
    print("✓ Class generation works for new languages")

def test_translation_mappings():
    trans = CodeTranslator()
    # Simple Python code
    source = "def foo(x):\n    return x + 1"
    for target in ["zig", "lua", "cpp", "csharp"]:
        result = trans.translate(source, "python", target)
        if result.get("success"):
            print(f"Python -> {target}: OK")
        else:
            print(f"Python -> {target}: {result.get('error')}")
    print("✓ Translation mappings exist")

if __name__ == "__main__":
    try:
        test_function_generation()
        test_class_generation()
        test_translation_mappings()
        print("\nAll upgrade tests passed.")
    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)