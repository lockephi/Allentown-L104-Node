{'type': 'class', 'name': 'PythonGateAnalyzer', 'methods': ['__init__', 'analyze_file', '_is_gate_related', '_extract_function_gate', '_extract_class_gate', '_regex_fallback'], 'bases': 0, 'description': 'Python gate analyzer', 'docstring': '"""Python gate analyzer.\n\nMethods: __init__, analyze_file, _is_gate_related, _extract_function_gate, _extract_class_gate, _regex_fallback\n"""'}

{'type': 'function', 'name': '__init__', 'params': [], 'param_types': {}, 'return_type': 'None', 'has_return': False, 'decorators': [], 'docstring': '"""Init.\n\nArgs:\n        None\n\nReturns:\n    None\n\nRaises:\n    None\n"""', 'description': 'Init'}

{'type': 'function', 'name': 'analyze_file', 'params': ['filepath'], 'param_types': {'filepath': 'Path'}, 'return_type': 'Any', 'has_return': True, 'decorators': [], 'docstring': '"""Analyze file.\n\nArgs:\n        filepath (Path): Description of filepath\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'Analyze file'}

{'type': 'function', 'name': '_is_gate_related', 'params': ['name'], 'param_types': {'name': 'str'}, 'return_type': 'bool', 'has_return': True, 'decorators': [], 'docstring': '"""Is gate related.\n\nArgs:\n        name (str): Description of name\n\nReturns:\n    bool\n\nRaises:\n    None\n"""', 'description': 'Is gate related'}

{'type': 'function', 'name': '_extract_function_gate', 'params': ['node', 'rel_path', 'source', 'class_name'], 'param_types': {'node': 'FunctionDef', 'rel_path': 'str', 'source': 'str', 'class_name': 'str'}, 'return_type': 'LogicGate', 'has_return': True, 'decorators': [], 'docstring': '"""Extract function gate.\n\nArgs:\n        node (FunctionDef): Description of node\n    rel_path (str): Description of rel_path\n    source (str): Description of source\n    class_name (str): Description of class_name\n\nReturns:\n    LogicGate\n\nRaises:\n    None\n"""', 'description': 'Extract function gate'}

{'type': 'function', 'name': '_extract_class_gate', 'params': ['node', 'rel_path', 'source'], 'param_types': {'node': 'ClassDef', 'rel_path': 'str', 'source': 'str'}, 'return_type': 'LogicGate', 'has_return': True, 'decorators': [], 'docstring': '"""Extract class gate.\n\nArgs:\n        node (ClassDef): Description of node\n    rel_path (str): Description of rel_path\n    source (str): Description of source\n\nReturns:\n    LogicGate\n\nRaises:\n    None\n"""', 'description': 'Extract class gate'}

{'type': 'function', 'name': '_regex_fallback', 'params': ['filepath'], 'param_types': {'filepath': 'Path'}, 'return_type': 'Any', 'has_return': True, 'decorators': [], 'docstring': '"""Regex fallback.\n\nArgs:\n        filepath (Path): Description of filepath\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'Regex fallback'}

