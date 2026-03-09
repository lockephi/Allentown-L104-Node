{'type': 'class', 'name': '_HuggingFaceFetcher', 'methods': ['_fetch', 'fetch_mmlu', 'fetch_arc', 'fetch_humaneval'], 'bases': 0, 'description': 'Hugging face fetcher', 'docstring': '"""Hugging face fetcher.\n\nMethods: _fetch, fetch_mmlu, fetch_arc, fetch_humaneval\n"""'}

{'type': 'class', 'name': '_MMLURunner', 'methods': ['__init__', '_get_engine', 'evaluate'], 'bases': 0, 'description': 'Mmlur unner', 'docstring': '"""Mmlur unner.\n\nMethods: __init__, _get_engine, evaluate\n"""'}

{'type': 'class', 'name': '_HumanEvalRunner', 'methods': ['__init__', '_get_engine', 'evaluate', '_manual_test'], 'bases': 0, 'description': 'Human eval runner', 'docstring': '"""Human eval runner.\n\nMethods: __init__, _get_engine, evaluate, _manual_test\n"""'}

{'type': 'class', 'name': '_MATHRunner', 'methods': ['__init__', '_get_solver', 'evaluate', '_check_math_answer'], 'bases': 0, 'description': 'Mathr unner', 'docstring': '"""Mathr unner.\n\nMethods: __init__, _get_solver, evaluate, _check_math_answer\n"""'}

{'type': 'class', 'name': '_ARCRunner', 'methods': ['__init__', '_get_engine', 'evaluate'], 'bases': 0, 'description': 'Arcr unner', 'docstring': '"""Arcr unner.\n\nMethods: __init__, _get_engine, evaluate\n"""'}

{'type': 'class', 'name': 'BenchmarkHarness', 'methods': ['__init__', 'run_all', '_run_humaneval_online', 'run_benchmark', 'get_score', 'get_status', 'print_report', '_verdict'], 'bases': 0, 'description': 'Benchmark harness', 'docstring': '"""Benchmark harness.\n\nMethods: __init__, run_all, _run_humaneval_online, run_benchmark, get_score, get_status, print_report, _verdict\n"""'}

{'type': 'function', 'name': '_fetch', 'params': ['cls', 'dataset', 'config', 'split', 'offset', 'length'], 'param_types': {'dataset': 'str', 'config': 'str', 'split': 'str', 'offset': 'int', 'length': 'int'}, 'return_type': 'Any', 'has_return': True, 'decorators': ['classmethod'], 'docstring': '"""Fetch.\n\nArgs:\n        cls (Any): Description of cls\n    dataset (str): Description of dataset\n    config (str): Description of config\n    split (str): Description of split\n    offset (int): Description of offset\n    length (int): Description of length\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'Fetch'}

{'type': 'function', 'name': 'fetch_mmlu', 'params': ['cls', 'max_questions'], 'param_types': {'max_questions': 'int'}, 'return_type': 'Any', 'has_return': True, 'decorators': ['classmethod'], 'docstring': '"""Fetch mmlu.\n\nArgs:\n        cls (Any): Description of cls\n    max_questions (int): Description of max_questions\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'Fetch mmlu'}

{'type': 'function', 'name': 'fetch_arc', 'params': ['cls', 'max_questions', 'include_easy'], 'param_types': {'max_questions': 'int', 'include_easy': 'bool'}, 'return_type': 'Any', 'has_return': True, 'decorators': ['classmethod'], 'docstring': '"""Fetch arc.\n\nArgs:\n        cls (Any): Description of cls\n    max_questions (int): Description of max_questions\n    include_easy (bool): Description of include_easy\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'Fetch arc'}

{'type': 'function', 'name': 'fetch_humaneval', 'params': ['cls'], 'param_types': {}, 'return_type': 'Any', 'has_return': True, 'decorators': ['classmethod'], 'docstring': '"""Fetch humaneval.\n\nArgs:\n        cls (Any): Description of cls\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'Fetch humaneval'}

{'type': 'function', 'name': '__init__', 'params': [], 'param_types': {}, 'return_type': 'None', 'has_return': False, 'decorators': [], 'docstring': '"""Init.\n\nArgs:\n        None\n\nReturns:\n    None\n\nRaises:\n    None\n"""', 'description': 'Init'}

{'type': 'function', 'name': '_get_engine', 'params': [], 'param_types': {}, 'return_type': 'None', 'has_return': True, 'decorators': [], 'docstring': '"""Get engine.\n\nArgs:\n        None\n\nReturns:\n    None\n\nRaises:\n    None\n"""', 'description': 'Get engine'}

{'type': 'function', 'name': 'evaluate', 'params': ['samples'], 'param_types': {'samples': "Subscript(value=Name(id='Optional', ctx=Load()), slice=Subscript(value=Name(id='List', ctx=Load()), slice=Name(id='Dict', ctx=Load()), ctx=Load()), ctx=Load())"}, 'return_type': 'Any', 'has_return': True, 'decorators': [], 'docstring': '"""Evaluate.\n\nArgs:\n        samples (Subscript(value=Name(id=\'Optional\', ctx=Load()), slice=Subscript(value=Name(id=\'List\', ctx=Load()), slice=Name(id=\'Dict\', ctx=Load()), ctx=Load()), ctx=Load())): Description of samples\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'Evaluate'}

{'type': 'function', 'name': '__init__', 'params': [], 'param_types': {}, 'return_type': 'None', 'has_return': False, 'decorators': [], 'docstring': '"""Init.\n\nArgs:\n        None\n\nReturns:\n    None\n\nRaises:\n    None\n"""', 'description': 'Init'}

{'type': 'function', 'name': '_get_engine', 'params': [], 'param_types': {}, 'return_type': 'None', 'has_return': True, 'decorators': [], 'docstring': '"""Get engine.\n\nArgs:\n        None\n\nReturns:\n    None\n\nRaises:\n    None\n"""', 'description': 'Get engine'}

{'type': 'function', 'name': 'evaluate', 'params': ['samples'], 'param_types': {'samples': "Subscript(value=Name(id='Optional', ctx=Load()), slice=Subscript(value=Name(id='List', ctx=Load()), slice=Name(id='Dict', ctx=Load()), ctx=Load()), ctx=Load())"}, 'return_type': 'Any', 'has_return': True, 'decorators': [], 'docstring': '"""Evaluate.\n\nArgs:\n        samples (Subscript(value=Name(id=\'Optional\', ctx=Load()), slice=Subscript(value=Name(id=\'List\', ctx=Load()), slice=Name(id=\'Dict\', ctx=Load()), ctx=Load()), ctx=Load())): Description of samples\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'Evaluate'}

{'type': 'function', 'name': '_manual_test', 'params': ['code', 'func_name', 'tests'], 'param_types': {'code': 'str', 'func_name': 'str', 'tests': "Subscript(value=Name(id='List', ctx=Load()), slice=Name(id='Dict', ctx=Load()), ctx=Load())"}, 'return_type': 'bool', 'has_return': True, 'decorators': [], 'docstring': '"""Manual test.\n\nArgs:\n        code (str): Description of code\n    func_name (str): Description of func_name\n    tests (Subscript(value=Name(id=\'List\', ctx=Load()), slice=Name(id=\'Dict\', ctx=Load()), ctx=Load())): Description of tests\n\nReturns:\n    bool\n\nRaises:\n    None\n"""', 'description': 'Manual test'}

{'type': 'function', 'name': '__init__', 'params': [], 'param_types': {}, 'return_type': 'None', 'has_return': False, 'decorators': [], 'docstring': '"""Init.\n\nArgs:\n        None\n\nReturns:\n    None\n\nRaises:\n    None\n"""', 'description': 'Init'}

{'type': 'function', 'name': '_get_solver', 'params': [], 'param_types': {}, 'return_type': 'None', 'has_return': True, 'decorators': [], 'docstring': '"""Get solver.\n\nArgs:\n        None\n\nReturns:\n    None\n\nRaises:\n    None\n"""', 'description': 'Get solver'}

{'type': 'function', 'name': 'evaluate', 'params': ['samples'], 'param_types': {'samples': "Subscript(value=Name(id='Optional', ctx=Load()), slice=Subscript(value=Name(id='List', ctx=Load()), slice=Name(id='Dict', ctx=Load()), ctx=Load()), ctx=Load())"}, 'return_type': 'Any', 'has_return': True, 'decorators': [], 'docstring': '"""Evaluate.\n\nArgs:\n        samples (Subscript(value=Name(id=\'Optional\', ctx=Load()), slice=Subscript(value=Name(id=\'List\', ctx=Load()), slice=Name(id=\'Dict\', ctx=Load()), ctx=Load()), ctx=Load())): Description of samples\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'Evaluate'}

{'type': 'function', 'name': '_check_math_answer', 'params': ['predicted', 'expected'], 'param_types': {'predicted': 'str', 'expected': 'str'}, 'return_type': 'bool', 'has_return': True, 'decorators': ['staticmethod'], 'docstring': '"""Check math answer.\n\nArgs:\n        predicted (str): Description of predicted\n    expected (str): Description of expected\n\nReturns:\n    bool\n\nRaises:\n    None\n"""', 'description': 'Check math answer'}

{'type': 'function', 'name': '__init__', 'params': [], 'param_types': {}, 'return_type': 'None', 'has_return': False, 'decorators': [], 'docstring': '"""Init.\n\nArgs:\n        None\n\nReturns:\n    None\n\nRaises:\n    None\n"""', 'description': 'Init'}

{'type': 'function', 'name': '_get_engine', 'params': [], 'param_types': {}, 'return_type': 'None', 'has_return': True, 'decorators': [], 'docstring': '"""Get engine.\n\nArgs:\n        None\n\nReturns:\n    None\n\nRaises:\n    None\n"""', 'description': 'Get engine'}

{'type': 'function', 'name': 'evaluate', 'params': ['samples'], 'param_types': {'samples': "Subscript(value=Name(id='Optional', ctx=Load()), slice=Subscript(value=Name(id='List', ctx=Load()), slice=Name(id='Dict', ctx=Load()), ctx=Load()), ctx=Load())"}, 'return_type': 'Any', 'has_return': True, 'decorators': [], 'docstring': '"""Evaluate.\n\nArgs:\n        samples (Subscript(value=Name(id=\'Optional\', ctx=Load()), slice=Subscript(value=Name(id=\'List\', ctx=Load()), slice=Name(id=\'Dict\', ctx=Load()), ctx=Load()), ctx=Load())): Description of samples\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'Evaluate'}

{'type': 'function', 'name': '__init__', 'params': [], 'param_types': {}, 'return_type': 'None', 'has_return': False, 'decorators': [], 'docstring': '"""Init.\n\nArgs:\n        None\n\nReturns:\n    None\n\nRaises:\n    None\n"""', 'description': 'Init'}

{'type': 'function', 'name': 'run_all', 'params': ['online'], 'param_types': {'online': 'bool'}, 'return_type': 'Any', 'has_return': True, 'decorators': [], 'docstring': '"""Run all.\n\nArgs:\n        online (bool): Description of online\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'Run all'}

{'type': 'function', 'name': '_run_humaneval_online', 'params': ['problems'], 'param_types': {'problems': "Subscript(value=Name(id='List', ctx=Load()), slice=Name(id='Dict', ctx=Load()), ctx=Load())"}, 'return_type': 'Any', 'has_return': True, 'decorators': [], 'docstring': '"""Run humaneval online.\n\nArgs:\n        problems (Subscript(value=Name(id=\'List\', ctx=Load()), slice=Name(id=\'Dict\', ctx=Load()), ctx=Load())): Description of problems\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'Run humaneval online'}

{'type': 'function', 'name': 'run_benchmark', 'params': ['name'], 'param_types': {'name': 'str'}, 'return_type': 'Any', 'has_return': True, 'decorators': [], 'docstring': '"""Run benchmark.\n\nArgs:\n        name (str): Description of name\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'Run benchmark'}

{'type': 'function', 'name': 'get_score', 'params': [], 'param_types': {}, 'return_type': 'float', 'has_return': True, 'decorators': [], 'docstring': '"""Get score.\n\nArgs:\n        None\n\nReturns:\n    float\n\nRaises:\n    None\n"""', 'description': 'Get score'}

{'type': 'function', 'name': 'get_status', 'params': [], 'param_types': {}, 'return_type': 'Any', 'has_return': True, 'decorators': [], 'docstring': '"""Get status.\n\nArgs:\n        None\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'Get status'}

{'type': 'function', 'name': 'print_report', 'params': [], 'param_types': {}, 'return_type': 'None', 'has_return': False, 'decorators': [], 'docstring': '"""Print report.\n\nArgs:\n        None\n\nReturns:\n    None\n\nRaises:\n    None\n"""', 'description': 'Print report'}

{'type': 'function', 'name': '_verdict', 'params': ['score'], 'param_types': {'score': 'float'}, 'return_type': 'str', 'has_return': True, 'decorators': ['staticmethod'], 'docstring': '"""Verdict.\n\nArgs:\n        score (float): Description of score\n\nReturns:\n    str\n\nRaises:\n    None\n"""', 'description': 'Verdict'}

