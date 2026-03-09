{'type': 'class', 'name': 'L104HardwareAdaptiveRuntime', 'methods': ['__init__', '_detect_hardware', 'get_memory_pressure', 'get_thermal_state', 'optimize_for_workload', 'record_perf_sample', 'get_perf_trend', 'get_runtime_status'], 'bases': 0, 'description': 'L hardware adaptive runtime', 'docstring': '"""L hardware adaptive runtime.\n\nMethods: __init__, _detect_hardware, get_memory_pressure, get_thermal_state, optimize_for_workload, record_perf_sample, get_perf_trend, get_runtime_status\n"""'}

{'type': 'class', 'name': 'L104PlatformCompatibilityLayer', 'methods': ['__init__', '_detect_modules', '_compute_feature_flags', 'safe_import', 'get_optimal_dtype', 'get_max_concurrency', 'get_compatibility_report', '_build_dependency_graph', 'resolve_dependency_chain', '_get_degradation_strategies', 'get_degradation_level', '_detect_platform_details', 'get_optimal_config_for_workload', 'classify_performance_tier', 'get_tier_recommendations'], 'bases': 0, 'description': 'L platform compatibility layer', 'docstring': '"""L platform compatibility layer.\n\nMethods: __init__, _detect_modules, _compute_feature_flags, safe_import, get_optimal_dtype, get_max_concurrency, get_compatibility_report, _build_dependency_graph, resolve_dependency_chain, _get_degradation_strategies, get_degradation_level, _detect_platform_details, get_optimal_config_for_workload, classify_performance_tier, get_tier_recommendations\n"""'}

{'type': 'function', 'name': '__init__', 'params': [], 'param_types': {}, 'return_type': 'None', 'has_return': False, 'decorators': [], 'docstring': '"""Init.\n\nArgs:\n        None\n\nReturns:\n    None\n\nRaises:\n    None\n"""', 'description': 'Init'}

{'type': 'function', 'name': '_detect_hardware', 'params': [], 'param_types': {}, 'return_type': 'Dict', 'has_return': True, 'decorators': [], 'docstring': '"""Detect hardware.\n\nArgs:\n        None\n\nReturns:\n    Dict\n\nRaises:\n    None\n"""', 'description': 'Detect hardware'}

{'type': 'function', 'name': 'get_memory_pressure', 'params': [], 'param_types': {}, 'return_type': 'Dict', 'has_return': True, 'decorators': [], 'docstring': '"""Get memory pressure.\n\nArgs:\n        None\n\nReturns:\n    Dict\n\nRaises:\n    None\n"""', 'description': 'Get memory pressure'}

{'type': 'function', 'name': 'get_thermal_state', 'params': [], 'param_types': {}, 'return_type': 'Dict', 'has_return': True, 'decorators': [], 'docstring': '"""Get thermal state.\n\nArgs:\n        None\n\nReturns:\n    Dict\n\nRaises:\n    None\n"""', 'description': 'Get thermal state'}

{'type': 'function', 'name': 'optimize_for_workload', 'params': ['workload_type'], 'param_types': {'workload_type': 'str'}, 'return_type': 'Dict', 'has_return': True, 'decorators': [], 'docstring': '"""Optimize for workload.\n\nArgs:\n        workload_type (str): Description of workload_type\n\nReturns:\n    Dict\n\nRaises:\n    None\n"""', 'description': 'Optimize for workload'}

{'type': 'function', 'name': 'record_perf_sample', 'params': ['operation', 'duration_ms', 'memory_delta_mb'], 'param_types': {'operation': 'str', 'duration_ms': 'float', 'memory_delta_mb': 'float'}, 'return_type': 'None', 'has_return': False, 'decorators': [], 'docstring': '"""Record perf sample.\n\nArgs:\n        operation (str): Description of operation\n    duration_ms (float): Description of duration_ms\n    memory_delta_mb (float): Description of memory_delta_mb\n\nReturns:\n    None\n\nRaises:\n    None\n"""', 'description': 'Record perf sample'}

{'type': 'function', 'name': 'get_perf_trend', 'params': ['operation', 'window'], 'param_types': {'operation': 'str', 'window': 'int'}, 'return_type': 'Dict', 'has_return': True, 'decorators': [], 'docstring': '"""Get perf trend.\n\nArgs:\n        operation (str): Description of operation\n    window (int): Description of window\n\nReturns:\n    Dict\n\nRaises:\n    None\n"""', 'description': 'Get perf trend'}

{'type': 'function', 'name': 'get_runtime_status', 'params': [], 'param_types': {}, 'return_type': 'Dict', 'has_return': True, 'decorators': [], 'docstring': '"""Get runtime status.\n\nArgs:\n        None\n\nReturns:\n    Dict\n\nRaises:\n    None\n"""', 'description': 'Get runtime status'}

{'type': 'function', 'name': '__init__', 'params': [], 'param_types': {}, 'return_type': 'None', 'has_return': False, 'decorators': [], 'docstring': '"""Init.\n\nArgs:\n        None\n\nReturns:\n    None\n\nRaises:\n    None\n"""', 'description': 'Init'}

{'type': 'function', 'name': '_detect_modules', 'params': [], 'param_types': {}, 'return_type': 'Any', 'has_return': True, 'decorators': [], 'docstring': '"""Detect modules.\n\nArgs:\n        None\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'Detect modules'}

{'type': 'function', 'name': '_compute_feature_flags', 'params': [], 'param_types': {}, 'return_type': 'Any', 'has_return': True, 'decorators': [], 'docstring': '"""Compute feature flags.\n\nArgs:\n        None\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'Compute feature flags'}

{'type': 'function', 'name': 'safe_import', 'params': ['module_name', 'fallback'], 'param_types': {'module_name': 'str'}, 'return_type': 'None', 'has_return': True, 'decorators': [], 'docstring': '"""Safe import.\n\nArgs:\n        module_name (str): Description of module_name\n    fallback (Any): Description of fallback\n\nReturns:\n    None\n\nRaises:\n    None\n"""', 'description': 'Safe import'}

{'type': 'function', 'name': 'get_optimal_dtype', 'params': [], 'param_types': {}, 'return_type': 'str', 'has_return': True, 'decorators': [], 'docstring': '"""Get optimal dtype.\n\nArgs:\n        None\n\nReturns:\n    str\n\nRaises:\n    None\n"""', 'description': 'Get optimal dtype'}

{'type': 'function', 'name': 'get_max_concurrency', 'params': [], 'param_types': {}, 'return_type': 'int', 'has_return': True, 'decorators': [], 'docstring': '"""Get max concurrency.\n\nArgs:\n        None\n\nReturns:\n    int\n\nRaises:\n    None\n"""', 'description': 'Get max concurrency'}

{'type': 'function', 'name': 'get_compatibility_report', 'params': [], 'param_types': {}, 'return_type': 'Dict', 'has_return': True, 'decorators': [], 'docstring': '"""Get compatibility report.\n\nArgs:\n        None\n\nReturns:\n    Dict\n\nRaises:\n    None\n"""', 'description': 'Get compatibility report'}

{'type': 'function', 'name': '_build_dependency_graph', 'params': [], 'param_types': {}, 'return_type': 'Any', 'has_return': True, 'decorators': [], 'docstring': '"""Build dependency graph.\n\nArgs:\n        None\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'Build dependency graph'}

{'type': 'function', 'name': 'resolve_dependency_chain', 'params': ['target_feature'], 'param_types': {'target_feature': 'str'}, 'return_type': 'Dict', 'has_return': True, 'decorators': [], 'docstring': '"""Resolve dependency chain.\n\nArgs:\n        target_feature (str): Description of target_feature\n\nReturns:\n    Dict\n\nRaises:\n    None\n"""', 'description': 'Resolve dependency chain'}

{'type': 'function', 'name': '_get_degradation_strategies', 'params': [], 'param_types': {}, 'return_type': 'Any', 'has_return': True, 'decorators': [], 'docstring': '"""Get degradation strategies.\n\nArgs:\n        None\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'Get degradation strategies'}

{'type': 'function', 'name': 'get_degradation_level', 'params': ['feature'], 'param_types': {'feature': 'str'}, 'return_type': 'str', 'has_return': True, 'decorators': [], 'docstring': '"""Get degradation level.\n\nArgs:\n        feature (str): Description of feature\n\nReturns:\n    str\n\nRaises:\n    None\n"""', 'description': 'Get degradation level'}

{'type': 'function', 'name': '_detect_platform_details', 'params': [], 'param_types': {}, 'return_type': 'Dict', 'has_return': True, 'decorators': [], 'docstring': '"""Detect platform details.\n\nArgs:\n        None\n\nReturns:\n    Dict\n\nRaises:\n    None\n"""', 'description': 'Detect platform details'}

{'type': 'function', 'name': 'get_optimal_config_for_workload', 'params': ['workload_type'], 'param_types': {'workload_type': 'str'}, 'return_type': 'Dict', 'has_return': True, 'decorators': [], 'docstring': '"""Get optimal config for workload.\n\nArgs:\n        workload_type (str): Description of workload_type\n\nReturns:\n    Dict\n\nRaises:\n    None\n"""', 'description': 'Get optimal config for workload'}

{'type': 'function', 'name': 'classify_performance_tier', 'params': [], 'param_types': {}, 'return_type': 'str', 'has_return': True, 'decorators': [], 'docstring': '"""Classify performance tier.\n\nArgs:\n        None\n\nReturns:\n    str\n\nRaises:\n    None\n"""', 'description': 'Classify performance tier'}

{'type': 'function', 'name': 'get_tier_recommendations', 'params': [], 'param_types': {}, 'return_type': 'Dict', 'has_return': True, 'decorators': [], 'docstring': '"""Get tier recommendations.\n\nArgs:\n        None\n\nReturns:\n    Dict\n\nRaises:\n    None\n"""', 'description': 'Get tier recommendations'}

