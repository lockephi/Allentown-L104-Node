{'type': 'class', 'name': 'PropOp', 'methods': [], 'bases': 1, 'description': 'Prop op', 'docstring': '"""Prop op.\n\nMethods: \n"""'}

{'type': 'class', 'name': 'PropFormula', 'methods': ['is_atom', 'variables', 'evaluate', '__repr__'], 'bases': 0, 'description': 'Prop formula', 'docstring': '"""Prop formula.\n\nMethods: is_atom, variables, evaluate, __repr__\n"""'}

{'type': 'function', 'name': 'Atom', 'params': ['name'], 'param_types': {'name': 'str'}, 'return_type': 'PropFormula', 'has_return': True, 'decorators': [], 'docstring': '"""Atom.\n\nArgs:\n        name (str): Description of name\n\nReturns:\n    PropFormula\n\nRaises:\n    None\n"""', 'description': 'Atom'}

{'type': 'function', 'name': 'Not', 'params': ['f'], 'param_types': {'f': 'PropFormula'}, 'return_type': 'PropFormula', 'has_return': True, 'decorators': [], 'docstring': '"""Not.\n\nArgs:\n        f (PropFormula): Description of f\n\nReturns:\n    PropFormula\n\nRaises:\n    None\n"""', 'description': 'Not'}

{'type': 'function', 'name': 'And', 'params': ['l', 'r'], 'param_types': {'l': 'PropFormula', 'r': 'PropFormula'}, 'return_type': 'PropFormula', 'has_return': True, 'decorators': [], 'docstring': '"""And.\n\nArgs:\n        l (PropFormula): Description of l\n    r (PropFormula): Description of r\n\nReturns:\n    PropFormula\n\nRaises:\n    None\n"""', 'description': 'And'}

{'type': 'function', 'name': 'Or', 'params': ['l', 'r'], 'param_types': {'l': 'PropFormula', 'r': 'PropFormula'}, 'return_type': 'PropFormula', 'has_return': True, 'decorators': [], 'docstring': '"""Or.\n\nArgs:\n        l (PropFormula): Description of l\n    r (PropFormula): Description of r\n\nReturns:\n    PropFormula\n\nRaises:\n    None\n"""', 'description': 'Or'}

{'type': 'function', 'name': 'Implies', 'params': ['l', 'r'], 'param_types': {'l': 'PropFormula', 'r': 'PropFormula'}, 'return_type': 'PropFormula', 'has_return': True, 'decorators': [], 'docstring': '"""Implies.\n\nArgs:\n        l (PropFormula): Description of l\n    r (PropFormula): Description of r\n\nReturns:\n    PropFormula\n\nRaises:\n    None\n"""', 'description': 'Implies'}

{'type': 'function', 'name': 'Iff', 'params': ['l', 'r'], 'param_types': {'l': 'PropFormula', 'r': 'PropFormula'}, 'return_type': 'PropFormula', 'has_return': True, 'decorators': [], 'docstring': '"""Iff.\n\nArgs:\n        l (PropFormula): Description of l\n    r (PropFormula): Description of r\n\nReturns:\n    PropFormula\n\nRaises:\n    None\n"""', 'description': 'Iff'}

{'type': 'function', 'name': 'Xor', 'params': ['l', 'r'], 'param_types': {'l': 'PropFormula', 'r': 'PropFormula'}, 'return_type': 'PropFormula', 'has_return': True, 'decorators': [], 'docstring': '"""Xor.\n\nArgs:\n        l (PropFormula): Description of l\n    r (PropFormula): Description of r\n\nReturns:\n    PropFormula\n\nRaises:\n    None\n"""', 'description': 'Xor'}

{'type': 'class', 'name': 'TruthTableGenerator', 'methods': ['generate', 'is_tautology', 'is_contradiction', 'is_satisfiable', 'are_equivalent', 'entails'], 'bases': 0, 'description': 'Truth table generator', 'docstring': '"""Truth table generator.\n\nMethods: generate, is_tautology, is_contradiction, is_satisfiable, are_equivalent, entails\n"""'}

{'type': 'class', 'name': 'NormalFormConverter', 'methods': ['to_nnf', 'to_cnf', '_distribute_or_over_and', 'to_dnf', '_distribute_and_over_or'], 'bases': 0, 'description': 'Normal form converter', 'docstring': '"""Normal form converter.\n\nMethods: to_nnf, to_cnf, _distribute_or_over_and, to_dnf, _distribute_and_over_or\n"""'}

{'type': 'class', 'name': 'QuantifierType', 'methods': [], 'bases': 1, 'description': 'Quantifier type', 'docstring': '"""Quantifier type.\n\nMethods: \n"""'}

{'type': 'class', 'name': 'PredicateFormula', 'methods': ['is_atomic', 'free_variables', '__repr__'], 'bases': 0, 'description': 'Predicate formula', 'docstring': '"""Predicate formula.\n\nMethods: is_atomic, free_variables, __repr__\n"""'}

{'type': 'class', 'name': 'PredicateModel', 'methods': ['__init__', 'add_predicate', 'add_constant', 'evaluate'], 'bases': 0, 'description': 'Predicate model', 'docstring': '"""Predicate model.\n\nMethods: __init__, add_predicate, add_constant, evaluate\n"""'}

{'type': 'class', 'name': 'SyllogismFigure', 'methods': [], 'bases': 1, 'description': 'Syllogism figure', 'docstring': '"""Syllogism figure.\n\nMethods: \n"""'}

{'type': 'class', 'name': 'SyllogismMood', 'methods': [], 'bases': 1, 'description': 'Syllogism mood', 'docstring': '"""Syllogism mood.\n\nMethods: \n"""'}

{'type': 'class', 'name': 'CategoricalProposition', 'methods': ['__repr__', 'evaluate'], 'bases': 0, 'description': 'Categorical proposition', 'docstring': '"""Categorical proposition.\n\nMethods: __repr__, evaluate\n"""'}

{'type': 'class', 'name': 'Syllogism', 'methods': ['get_mood', 'detect_figure', 'is_valid'], 'bases': 0, 'description': 'Syllogism', 'docstring': '"""Syllogism.\n\nMethods: get_mood, detect_figure, is_valid\n"""'}

{'type': 'class', 'name': 'SyllogisticEngine', 'methods': ['analyze', '_check_rules', 'construct_from_text'], 'bases': 0, 'description': 'Syllogistic engine', 'docstring': '"""Syllogistic engine.\n\nMethods: analyze, _check_rules, construct_from_text\n"""'}

{'type': 'class', 'name': 'LogicalLaw', 'methods': [], 'bases': 0, 'description': 'Logical law', 'docstring': '"""Logical law.\n\nMethods: \n"""'}

{'type': 'class', 'name': 'EquivalenceProver', 'methods': ['__init__', 'prove_equivalence', 'identify_applicable_laws', 'simplify', 'list_laws'], 'bases': 0, 'description': 'Equivalence prover', 'docstring': '"""Equivalence prover.\n\nMethods: __init__, prove_equivalence, identify_applicable_laws, simplify, list_laws\n"""'}

{'type': 'class', 'name': 'Fallacy', 'methods': [], 'bases': 0, 'description': 'Fallacy', 'docstring': '"""Fallacy.\n\nMethods: \n"""'}

{'type': 'class', 'name': 'FallacyDetector', 'methods': ['__init__', '_compile_patterns', 'detect', '_structural_analysis', 'get_fallacy_by_name', 'list_all'], 'bases': 0, 'description': 'Fallacy detector', 'docstring': '"""Fallacy detector.\n\nMethods: __init__, _compile_patterns, detect, _structural_analysis, get_fallacy_by_name, list_all\n"""'}

{'type': 'class', 'name': 'ModalOperator', 'methods': [], 'bases': 1, 'description': 'Modal operator', 'docstring': '"""Modal operator.\n\nMethods: \n"""'}

{'type': 'class', 'name': 'ModalFormula', 'methods': ['__repr__'], 'bases': 0, 'description': 'Modal formula', 'docstring': '"""Modal formula.\n\nMethods: __repr__\n"""'}

{'type': 'class', 'name': 'KripkeFrame', 'methods': ['__init__', 'add_world', 'add_accessibility', 'make_reflexive', 'make_symmetric', 'make_transitive', 'make_s5', 'evaluate'], 'bases': 0, 'description': 'Kripke frame', 'docstring': '"""Kripke frame.\n\nMethods: __init__, add_world, add_accessibility, make_reflexive, make_symmetric, make_transitive, make_s5, evaluate\n"""'}

{'type': 'class', 'name': 'NLToLogicTranslator', 'methods': ['__init__', 'translate', '_proposition_name', '_predicate_name'], 'bases': 0, 'description': 'Nlt o logic translator', 'docstring': '"""Nlt o logic translator.\n\nMethods: __init__, translate, _proposition_name, _predicate_name\n"""'}

{'type': 'class', 'name': 'Argument', 'methods': [], 'bases': 0, 'description': 'Argument', 'docstring': '"""Argument.\n\nMethods: \n"""'}

{'type': 'class', 'name': 'ArgumentAnalyzer', 'methods': ['__init__', 'analyze', 'evaluate_deductive_validity'], 'bases': 0, 'description': 'Argument analyzer', 'docstring': '"""Argument analyzer.\n\nMethods: __init__, analyze, evaluate_deductive_validity\n"""'}

{'type': 'class', 'name': 'Clause', 'methods': ['from_formula', 'is_empty', '__repr__', '__hash__', '__eq__'], 'bases': 0, 'description': 'Clause', 'docstring': '"""Clause.\n\nMethods: from_formula, is_empty, __repr__, __hash__, __eq__\n"""'}

{'type': 'function', 'name': '_collect_clause_lits', 'params': ['f', 'out'], 'param_types': {'f': 'PropFormula', 'out': 'set'}, 'return_type': 'None', 'has_return': False, 'decorators': [], 'docstring': '"""Collect clause lits.\n\nArgs:\n        f (PropFormula): Description of f\n    out (set): Description of out\n\nReturns:\n    None\n\nRaises:\n    None\n"""', 'description': 'Collect clause lits'}

{'type': 'function', 'name': '_collect_cnf_clauses', 'params': ['f'], 'param_types': {'f': 'PropFormula'}, 'return_type': 'Any', 'has_return': True, 'decorators': [], 'docstring': '"""Collect cnf clauses.\n\nArgs:\n        f (PropFormula): Description of f\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'Collect cnf clauses'}

{'type': 'class', 'name': 'ResolutionProver', 'methods': ['__init__', 'resolve_pair', 'prove'], 'bases': 0, 'description': 'Resolution prover', 'docstring': '"""Resolution prover.\n\nMethods: __init__, resolve_pair, prove\n"""'}

{'type': 'class', 'name': 'DeductionRule', 'methods': [], 'bases': 1, 'description': 'Deduction rule', 'docstring': '"""Deduction rule.\n\nMethods: \n"""'}

{'type': 'class', 'name': 'ProofStep', 'methods': ['__repr__'], 'bases': 0, 'description': 'Proof step', 'docstring': '"""Proof step.\n\nMethods: __repr__\n"""'}

{'type': 'class', 'name': 'NaturalDeductionEngine', 'methods': ['__init__', 'modus_ponens_proof', 'hypothetical_syllogism_proof', 'conjunction_elim_proof', 'double_negation_proof', 'auto_prove'], 'bases': 0, 'description': 'Natural deduction engine', 'docstring': '"""Natural deduction engine.\n\nMethods: __init__, modus_ponens_proof, hypothetical_syllogism_proof, conjunction_elim_proof, double_negation_proof, auto_prove\n"""'}

{'type': 'class', 'name': 'InferenceStep', 'methods': [], 'bases': 0, 'description': 'Inference step', 'docstring': '"""Inference step.\n\nMethods: \n"""'}

{'type': 'class', 'name': 'InferenceChainBuilder', 'methods': ['__init__', 'add_rule', 'forward_chain', 'build_chain'], 'bases': 0, 'description': 'Inference chain builder', 'docstring': '"""Inference chain builder.\n\nMethods: __init__, add_rule, forward_chain, build_chain\n"""'}

{'type': 'class', 'name': 'FormalLogicEngine', 'methods': ['__init__', '_init_engines', 'three_engine_logic_score', 'analyze_argument', 'detect_fallacies', 'translate_to_logic', 'check_validity', 'generate_truth_table', 'prove_equivalence', 'analyze_syllogism', 'simplify_formula', 'to_cnf', 'to_dnf', 'list_fallacies', 'list_logical_laws', 'resolve_proof', 'natural_deduction_proof', 'build_inference_chain', 'comprehensive_proof', 'logic_depth_score', 'status'], 'bases': 0, 'description': 'Formal logic engine', 'docstring': '"""Formal logic engine.\n\nMethods: __init__, _init_engines, three_engine_logic_score, analyze_argument, detect_fallacies, translate_to_logic, check_validity, generate_truth_table, prove_equivalence, analyze_syllogism, simplify_formula, to_cnf, to_dnf, list_fallacies, list_logical_laws, resolve_proof, natural_deduction_proof, build_inference_chain, comprehensive_proof, logic_depth_score, status\n"""'}

{'type': 'function', 'name': 'is_atom', 'params': [], 'param_types': {}, 'return_type': 'bool', 'has_return': True, 'decorators': [], 'docstring': '"""Is atom.\n\nArgs:\n        None\n\nReturns:\n    bool\n\nRaises:\n    None\n"""', 'description': 'Is atom'}

{'type': 'function', 'name': 'variables', 'params': [], 'param_types': {}, 'return_type': 'Any', 'has_return': True, 'decorators': [], 'docstring': '"""Variables.\n\nArgs:\n        None\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'Variables'}

{'type': 'function', 'name': 'evaluate', 'params': ['assignment'], 'param_types': {'assignment': "Subscript(value=Name(id='Dict', ctx=Load()), slice=Tuple(elts=[Name(id='str', ctx=Load()), Name(id='bool', ctx=Load())], ctx=Load()), ctx=Load())"}, 'return_type': 'bool', 'has_return': True, 'decorators': [], 'docstring': '"""Evaluate.\n\nArgs:\n        assignment (Subscript(value=Name(id=\'Dict\', ctx=Load()), slice=Tuple(elts=[Name(id=\'str\', ctx=Load()), Name(id=\'bool\', ctx=Load())], ctx=Load()), ctx=Load())): Description of assignment\n\nReturns:\n    bool\n\nRaises:\n    None\n"""', 'description': 'Evaluate'}

{'type': 'function', 'name': '__repr__', 'params': [], 'param_types': {}, 'return_type': 'str', 'has_return': True, 'decorators': [], 'docstring': '"""Repr.\n\nArgs:\n        None\n\nReturns:\n    str\n\nRaises:\n    None\n"""', 'description': 'Repr'}

{'type': 'function', 'name': 'generate', 'params': ['formula'], 'param_types': {'formula': 'PropFormula'}, 'return_type': 'Any', 'has_return': True, 'decorators': [], 'docstring': '"""Generate.\n\nArgs:\n        formula (PropFormula): Description of formula\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'Generate'}

{'type': 'function', 'name': 'is_tautology', 'params': ['formula'], 'param_types': {'formula': 'PropFormula'}, 'return_type': 'bool', 'has_return': True, 'decorators': [], 'docstring': '"""Is tautology.\n\nArgs:\n        formula (PropFormula): Description of formula\n\nReturns:\n    bool\n\nRaises:\n    None\n"""', 'description': 'Is tautology'}

{'type': 'function', 'name': 'is_contradiction', 'params': ['formula'], 'param_types': {'formula': 'PropFormula'}, 'return_type': 'bool', 'has_return': True, 'decorators': [], 'docstring': '"""Is contradiction.\n\nArgs:\n        formula (PropFormula): Description of formula\n\nReturns:\n    bool\n\nRaises:\n    None\n"""', 'description': 'Is contradiction'}

{'type': 'function', 'name': 'is_satisfiable', 'params': ['formula'], 'param_types': {'formula': 'PropFormula'}, 'return_type': 'bool', 'has_return': True, 'decorators': [], 'docstring': '"""Is satisfiable.\n\nArgs:\n        formula (PropFormula): Description of formula\n\nReturns:\n    bool\n\nRaises:\n    None\n"""', 'description': 'Is satisfiable'}

{'type': 'function', 'name': 'are_equivalent', 'params': ['f1', 'f2'], 'param_types': {'f1': 'PropFormula', 'f2': 'PropFormula'}, 'return_type': 'bool', 'has_return': True, 'decorators': [], 'docstring': '"""Are equivalent.\n\nArgs:\n        f1 (PropFormula): Description of f1\n    f2 (PropFormula): Description of f2\n\nReturns:\n    bool\n\nRaises:\n    None\n"""', 'description': 'Are equivalent'}

{'type': 'function', 'name': 'entails', 'params': ['premises', 'conclusion'], 'param_types': {'premises': "Subscript(value=Name(id='List', ctx=Load()), slice=Name(id='PropFormula', ctx=Load()), ctx=Load())", 'conclusion': 'PropFormula'}, 'return_type': 'bool', 'has_return': True, 'decorators': [], 'docstring': '"""Entails.\n\nArgs:\n        premises (Subscript(value=Name(id=\'List\', ctx=Load()), slice=Name(id=\'PropFormula\', ctx=Load()), ctx=Load())): Description of premises\n    conclusion (PropFormula): Description of conclusion\n\nReturns:\n    bool\n\nRaises:\n    None\n"""', 'description': 'Entails'}

{'type': 'function', 'name': 'to_nnf', 'params': ['formula'], 'param_types': {'formula': 'PropFormula'}, 'return_type': 'PropFormula', 'has_return': True, 'decorators': [], 'docstring': '"""To nnf.\n\nArgs:\n        formula (PropFormula): Description of formula\n\nReturns:\n    PropFormula\n\nRaises:\n    None\n"""', 'description': 'To nnf'}

{'type': 'function', 'name': 'to_cnf', 'params': ['formula'], 'param_types': {'formula': 'PropFormula'}, 'return_type': 'PropFormula', 'has_return': True, 'decorators': [], 'docstring': '"""To cnf.\n\nArgs:\n        formula (PropFormula): Description of formula\n\nReturns:\n    PropFormula\n\nRaises:\n    None\n"""', 'description': 'To cnf'}

{'type': 'function', 'name': '_distribute_or_over_and', 'params': ['f'], 'param_types': {'f': 'PropFormula'}, 'return_type': 'PropFormula', 'has_return': True, 'decorators': [], 'docstring': '"""Distribute or over and.\n\nArgs:\n        f (PropFormula): Description of f\n\nReturns:\n    PropFormula\n\nRaises:\n    None\n"""', 'description': 'Distribute or over and'}

{'type': 'function', 'name': 'to_dnf', 'params': ['formula'], 'param_types': {'formula': 'PropFormula'}, 'return_type': 'PropFormula', 'has_return': True, 'decorators': [], 'docstring': '"""To dnf.\n\nArgs:\n        formula (PropFormula): Description of formula\n\nReturns:\n    PropFormula\n\nRaises:\n    None\n"""', 'description': 'To dnf'}

{'type': 'function', 'name': '_distribute_and_over_or', 'params': ['f'], 'param_types': {'f': 'PropFormula'}, 'return_type': 'PropFormula', 'has_return': True, 'decorators': [], 'docstring': '"""Distribute and over or.\n\nArgs:\n        f (PropFormula): Description of f\n\nReturns:\n    PropFormula\n\nRaises:\n    None\n"""', 'description': 'Distribute and over or'}

{'type': 'function', 'name': 'is_atomic', 'params': [], 'param_types': {}, 'return_type': 'bool', 'has_return': True, 'decorators': [], 'docstring': '"""Is atomic.\n\nArgs:\n        None\n\nReturns:\n    bool\n\nRaises:\n    None\n"""', 'description': 'Is atomic'}

{'type': 'function', 'name': 'free_variables', 'params': [], 'param_types': {}, 'return_type': 'Any', 'has_return': True, 'decorators': [], 'docstring': '"""Free variables.\n\nArgs:\n        None\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'Free variables'}

{'type': 'function', 'name': '__repr__', 'params': [], 'param_types': {}, 'return_type': 'str', 'has_return': True, 'decorators': [], 'docstring': '"""Repr.\n\nArgs:\n        None\n\nReturns:\n    str\n\nRaises:\n    None\n"""', 'description': 'Repr'}

{'type': 'function', 'name': '__init__', 'params': ['domain'], 'param_types': {'domain': "Subscript(value=Name(id='Set', ctx=Load()), slice=Name(id='str', ctx=Load()), ctx=Load())"}, 'return_type': 'None', 'has_return': False, 'decorators': [], 'docstring': '"""Init.\n\nArgs:\n        domain (Subscript(value=Name(id=\'Set\', ctx=Load()), slice=Name(id=\'str\', ctx=Load()), ctx=Load())): Description of domain\n\nReturns:\n    None\n\nRaises:\n    None\n"""', 'description': 'Init'}

{'type': 'function', 'name': 'add_predicate', 'params': ['name', 'extension'], 'param_types': {'name': 'str', 'extension': "Subscript(value=Name(id='Set', ctx=Load()), slice=Subscript(value=Name(id='Tuple', ctx=Load()), slice=Tuple(elts=[Name(id='str', ctx=Load()), Constant(value=Ellipsis)], ctx=Load()), ctx=Load()), ctx=Load())"}, 'return_type': 'None', 'has_return': False, 'decorators': [], 'docstring': '"""Add predicate.\n\nArgs:\n        name (str): Description of name\n    extension (Subscript(value=Name(id=\'Set\', ctx=Load()), slice=Subscript(value=Name(id=\'Tuple\', ctx=Load()), slice=Tuple(elts=[Name(id=\'str\', ctx=Load()), Constant(value=Ellipsis)], ctx=Load()), ctx=Load()), ctx=Load())): Description of extension\n\nReturns:\n    None\n\nRaises:\n    None\n"""', 'description': 'Add predicate'}

{'type': 'function', 'name': 'add_constant', 'params': ['name', 'value'], 'param_types': {'name': 'str', 'value': 'str'}, 'return_type': 'None', 'has_return': False, 'decorators': [], 'docstring': '"""Add constant.\n\nArgs:\n        name (str): Description of name\n    value (str): Description of value\n\nReturns:\n    None\n\nRaises:\n    None\n"""', 'description': 'Add constant'}

{'type': 'function', 'name': 'evaluate', 'params': ['formula', 'assignment'], 'param_types': {'formula': 'PredicateFormula', 'assignment': "Subscript(value=Name(id='Dict', ctx=Load()), slice=Tuple(elts=[Name(id='str', ctx=Load()), Name(id='str', ctx=Load())], ctx=Load()), ctx=Load())"}, 'return_type': 'bool', 'has_return': True, 'decorators': [], 'docstring': '"""Evaluate.\n\nArgs:\n        formula (PredicateFormula): Description of formula\n    assignment (Subscript(value=Name(id=\'Dict\', ctx=Load()), slice=Tuple(elts=[Name(id=\'str\', ctx=Load()), Name(id=\'str\', ctx=Load())], ctx=Load()), ctx=Load())): Description of assignment\n\nReturns:\n    bool\n\nRaises:\n    None\n"""', 'description': 'Evaluate'}

{'type': 'function', 'name': '__repr__', 'params': [], 'param_types': {}, 'return_type': 'str', 'has_return': True, 'decorators': [], 'docstring': '"""Repr.\n\nArgs:\n        None\n\nReturns:\n    str\n\nRaises:\n    None\n"""', 'description': 'Repr'}

{'type': 'function', 'name': 'evaluate', 'params': ['domain'], 'param_types': {'domain': "Subscript(value=Name(id='Dict', ctx=Load()), slice=Tuple(elts=[Name(id='str', ctx=Load()), Subscript(value=Name(id='Set', ctx=Load()), slice=Name(id='str', ctx=Load()), ctx=Load())], ctx=Load()), ctx=Load())"}, 'return_type': 'bool', 'has_return': True, 'decorators': [], 'docstring': '"""Evaluate.\n\nArgs:\n        domain (Subscript(value=Name(id=\'Dict\', ctx=Load()), slice=Tuple(elts=[Name(id=\'str\', ctx=Load()), Subscript(value=Name(id=\'Set\', ctx=Load()), slice=Name(id=\'str\', ctx=Load()), ctx=Load())], ctx=Load()), ctx=Load())): Description of domain\n\nReturns:\n    bool\n\nRaises:\n    None\n"""', 'description': 'Evaluate'}

{'type': 'function', 'name': 'get_mood', 'params': [], 'param_types': {}, 'return_type': 'str', 'has_return': True, 'decorators': [], 'docstring': '"""Get mood.\n\nArgs:\n        None\n\nReturns:\n    str\n\nRaises:\n    None\n"""', 'description': 'Get mood'}

{'type': 'function', 'name': 'detect_figure', 'params': [], 'param_types': {}, 'return_type': 'int', 'has_return': True, 'decorators': [], 'docstring': '"""Detect figure.\n\nArgs:\n        None\n\nReturns:\n    int\n\nRaises:\n    None\n"""', 'description': 'Detect figure'}

{'type': 'function', 'name': 'is_valid', 'params': [], 'param_types': {}, 'return_type': 'bool', 'has_return': True, 'decorators': [], 'docstring': '"""Is valid.\n\nArgs:\n        None\n\nReturns:\n    bool\n\nRaises:\n    None\n"""', 'description': 'Is valid'}

{'type': 'function', 'name': 'analyze', 'params': ['syllogism'], 'param_types': {'syllogism': 'Syllogism'}, 'return_type': 'Any', 'has_return': True, 'decorators': [], 'docstring': '"""Analyze.\n\nArgs:\n        syllogism (Syllogism): Description of syllogism\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'Analyze'}

{'type': 'function', 'name': '_check_rules', 'params': ['s'], 'param_types': {'s': 'Syllogism'}, 'return_type': 'Any', 'has_return': True, 'decorators': [], 'docstring': '"""Check rules.\n\nArgs:\n        s (Syllogism): Description of s\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'Check rules'}

{'type': 'function', 'name': 'construct_from_text', 'params': ['major', 'minor', 'conclusion'], 'param_types': {'major': 'str', 'minor': 'str', 'conclusion': 'str'}, 'return_type': 'Any', 'has_return': True, 'decorators': [], 'docstring': '"""Construct from text.\n\nArgs:\n        major (str): Description of major\n    minor (str): Description of minor\n    conclusion (str): Description of conclusion\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'Construct from text'}

{'type': 'function', 'name': '__init__', 'params': [], 'param_types': {}, 'return_type': 'None', 'has_return': False, 'decorators': [], 'docstring': '"""Init.\n\nArgs:\n        None\n\nReturns:\n    None\n\nRaises:\n    None\n"""', 'description': 'Init'}

{'type': 'function', 'name': 'prove_equivalence', 'params': ['f1', 'f2'], 'param_types': {'f1': 'PropFormula', 'f2': 'PropFormula'}, 'return_type': 'Any', 'has_return': True, 'decorators': [], 'docstring': '"""Prove equivalence.\n\nArgs:\n        f1 (PropFormula): Description of f1\n    f2 (PropFormula): Description of f2\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'Prove equivalence'}

{'type': 'function', 'name': 'identify_applicable_laws', 'params': ['f1', 'f2'], 'param_types': {'f1': 'PropFormula', 'f2': 'PropFormula'}, 'return_type': 'Any', 'has_return': True, 'decorators': [], 'docstring': '"""Identify applicable laws.\n\nArgs:\n        f1 (PropFormula): Description of f1\n    f2 (PropFormula): Description of f2\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'Identify applicable laws'}

{'type': 'function', 'name': 'simplify', 'params': ['formula'], 'param_types': {'formula': 'PropFormula'}, 'return_type': 'PropFormula', 'has_return': True, 'decorators': [], 'docstring': '"""Simplify.\n\nArgs:\n        formula (PropFormula): Description of formula\n\nReturns:\n    PropFormula\n\nRaises:\n    None\n"""', 'description': 'Simplify'}

{'type': 'function', 'name': 'list_laws', 'params': [], 'param_types': {}, 'return_type': 'Any', 'has_return': True, 'decorators': [], 'docstring': '"""List laws.\n\nArgs:\n        None\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'List laws'}

{'type': 'function', 'name': '__init__', 'params': [], 'param_types': {}, 'return_type': 'None', 'has_return': False, 'decorators': [], 'docstring': '"""Init.\n\nArgs:\n        None\n\nReturns:\n    None\n\nRaises:\n    None\n"""', 'description': 'Init'}

{'type': 'function', 'name': '_compile_patterns', 'params': [], 'param_types': {}, 'return_type': 'None', 'has_return': False, 'decorators': [], 'docstring': '"""Compile patterns.\n\nArgs:\n        None\n\nReturns:\n    None\n\nRaises:\n    None\n"""', 'description': 'Compile patterns'}

{'type': 'function', 'name': 'detect', 'params': ['text'], 'param_types': {'text': 'str'}, 'return_type': 'Any', 'has_return': True, 'decorators': [], 'docstring': '"""Detect.\n\nArgs:\n        text (str): Description of text\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'Detect'}

{'type': 'function', 'name': '_structural_analysis', 'params': ['text', 'fallacy'], 'param_types': {'text': 'str', 'fallacy': 'Fallacy'}, 'return_type': 'float', 'has_return': True, 'decorators': [], 'docstring': '"""Structural analysis.\n\nArgs:\n        text (str): Description of text\n    fallacy (Fallacy): Description of fallacy\n\nReturns:\n    float\n\nRaises:\n    None\n"""', 'description': 'Structural analysis'}

{'type': 'function', 'name': 'get_fallacy_by_name', 'params': ['name'], 'param_types': {'name': 'str'}, 'return_type': 'Any', 'has_return': True, 'decorators': [], 'docstring': '"""Get fallacy by name.\n\nArgs:\n        name (str): Description of name\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'Get fallacy by name'}

{'type': 'function', 'name': 'list_all', 'params': [], 'param_types': {}, 'return_type': 'Any', 'has_return': True, 'decorators': [], 'docstring': '"""List all.\n\nArgs:\n        None\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'List all'}

{'type': 'function', 'name': '__repr__', 'params': [], 'param_types': {}, 'return_type': 'str', 'has_return': True, 'decorators': [], 'docstring': '"""Repr.\n\nArgs:\n        None\n\nReturns:\n    str\n\nRaises:\n    None\n"""', 'description': 'Repr'}

{'type': 'function', 'name': '__init__', 'params': [], 'param_types': {}, 'return_type': 'None', 'has_return': False, 'decorators': [], 'docstring': '"""Init.\n\nArgs:\n        None\n\nReturns:\n    None\n\nRaises:\n    None\n"""', 'description': 'Init'}

{'type': 'function', 'name': 'add_world', 'params': ['name', 'props'], 'param_types': {'name': 'str', 'props': "Subscript(value=Name(id='Dict', ctx=Load()), slice=Tuple(elts=[Name(id='str', ctx=Load()), Name(id='bool', ctx=Load())], ctx=Load()), ctx=Load())"}, 'return_type': 'None', 'has_return': False, 'decorators': [], 'docstring': '"""Add world.\n\nArgs:\n        name (str): Description of name\n    props (Subscript(value=Name(id=\'Dict\', ctx=Load()), slice=Tuple(elts=[Name(id=\'str\', ctx=Load()), Name(id=\'bool\', ctx=Load())], ctx=Load()), ctx=Load())): Description of props\n\nReturns:\n    None\n\nRaises:\n    None\n"""', 'description': 'Add world'}

{'type': 'function', 'name': 'add_accessibility', 'params': ['from_world', 'to_world'], 'param_types': {'from_world': 'str', 'to_world': 'str'}, 'return_type': 'None', 'has_return': False, 'decorators': [], 'docstring': '"""Add accessibility.\n\nArgs:\n        from_world (str): Description of from_world\n    to_world (str): Description of to_world\n\nReturns:\n    None\n\nRaises:\n    None\n"""', 'description': 'Add accessibility'}

{'type': 'function', 'name': 'make_reflexive', 'params': [], 'param_types': {}, 'return_type': 'None', 'has_return': False, 'decorators': [], 'docstring': '"""Make reflexive.\n\nArgs:\n        None\n\nReturns:\n    None\n\nRaises:\n    None\n"""', 'description': 'Make reflexive'}

{'type': 'function', 'name': 'make_symmetric', 'params': [], 'param_types': {}, 'return_type': 'None', 'has_return': False, 'decorators': [], 'docstring': '"""Make symmetric.\n\nArgs:\n        None\n\nReturns:\n    None\n\nRaises:\n    None\n"""', 'description': 'Make symmetric'}

{'type': 'function', 'name': 'make_transitive', 'params': [], 'param_types': {}, 'return_type': 'None', 'has_return': False, 'decorators': [], 'docstring': '"""Make transitive.\n\nArgs:\n        None\n\nReturns:\n    None\n\nRaises:\n    None\n"""', 'description': 'Make transitive'}

{'type': 'function', 'name': 'make_s5', 'params': [], 'param_types': {}, 'return_type': 'None', 'has_return': False, 'decorators': [], 'docstring': '"""Make s.\n\nArgs:\n        None\n\nReturns:\n    None\n\nRaises:\n    None\n"""', 'description': 'Make s'}

{'type': 'function', 'name': 'evaluate', 'params': ['formula', 'world'], 'param_types': {'formula': 'ModalFormula', 'world': 'str'}, 'return_type': 'bool', 'has_return': True, 'decorators': [], 'docstring': '"""Evaluate.\n\nArgs:\n        formula (ModalFormula): Description of formula\n    world (str): Description of world\n\nReturns:\n    bool\n\nRaises:\n    None\n"""', 'description': 'Evaluate'}

{'type': 'function', 'name': '__init__', 'params': [], 'param_types': {}, 'return_type': 'None', 'has_return': False, 'decorators': [], 'docstring': '"""Init.\n\nArgs:\n        None\n\nReturns:\n    None\n\nRaises:\n    None\n"""', 'description': 'Init'}

{'type': 'function', 'name': 'translate', 'params': ['text'], 'param_types': {'text': 'str'}, 'return_type': 'Any', 'has_return': True, 'decorators': [], 'docstring': '"""Translate.\n\nArgs:\n        text (str): Description of text\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'Translate'}

{'type': 'function', 'name': '_proposition_name', 'params': ['text'], 'param_types': {'text': 'str'}, 'return_type': 'str', 'has_return': True, 'decorators': [], 'docstring': '"""Proposition name.\n\nArgs:\n        text (str): Description of text\n\nReturns:\n    str\n\nRaises:\n    None\n"""', 'description': 'Proposition name'}

{'type': 'function', 'name': '_predicate_name', 'params': ['text'], 'param_types': {'text': 'str'}, 'return_type': 'str', 'has_return': True, 'decorators': [], 'docstring': '"""Predicate name.\n\nArgs:\n        text (str): Description of text\n\nReturns:\n    str\n\nRaises:\n    None\n"""', 'description': 'Predicate name'}

{'type': 'function', 'name': '__init__', 'params': [], 'param_types': {}, 'return_type': 'None', 'has_return': False, 'decorators': [], 'docstring': '"""Init.\n\nArgs:\n        None\n\nReturns:\n    None\n\nRaises:\n    None\n"""', 'description': 'Init'}

{'type': 'function', 'name': 'analyze', 'params': ['argument'], 'param_types': {'argument': 'Argument'}, 'return_type': 'Any', 'has_return': True, 'decorators': [], 'docstring': '"""Analyze.\n\nArgs:\n        argument (Argument): Description of argument\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'Analyze'}

{'type': 'function', 'name': 'evaluate_deductive_validity', 'params': ['premises', 'conclusion'], 'param_types': {'premises': "Subscript(value=Name(id='List', ctx=Load()), slice=Name(id='PropFormula', ctx=Load()), ctx=Load())", 'conclusion': 'PropFormula'}, 'return_type': 'Any', 'has_return': True, 'decorators': [], 'docstring': '"""Evaluate deductive validity.\n\nArgs:\n        premises (Subscript(value=Name(id=\'List\', ctx=Load()), slice=Name(id=\'PropFormula\', ctx=Load()), ctx=Load())): Description of premises\n    conclusion (PropFormula): Description of conclusion\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'Evaluate deductive validity'}

{'type': 'function', 'name': 'from_formula', 'params': ['formula'], 'param_types': {'formula': 'PropFormula'}, 'return_type': 'Clause', 'has_return': True, 'decorators': ['staticmethod'], 'docstring': '"""From formula.\n\nArgs:\n        formula (PropFormula): Description of formula\n\nReturns:\n    Clause\n\nRaises:\n    None\n"""', 'description': 'From formula'}

{'type': 'function', 'name': 'is_empty', 'params': [], 'param_types': {}, 'return_type': 'bool', 'has_return': True, 'decorators': ['property'], 'docstring': '"""Is empty.\n\nArgs:\n        None\n\nReturns:\n    bool\n\nRaises:\n    None\n"""', 'description': 'Is empty'}

{'type': 'function', 'name': '__repr__', 'params': [], 'param_types': {}, 'return_type': 'str', 'has_return': True, 'decorators': [], 'docstring': '"""Repr.\n\nArgs:\n        None\n\nReturns:\n    str\n\nRaises:\n    None\n"""', 'description': 'Repr'}

{'type': 'function', 'name': '__hash__', 'params': [], 'param_types': {}, 'return_type': 'None', 'has_return': True, 'decorators': [], 'docstring': '"""Hash.\n\nArgs:\n        None\n\nReturns:\n    None\n\nRaises:\n    None\n"""', 'description': 'Hash'}

{'type': 'function', 'name': '__eq__', 'params': ['other'], 'param_types': {}, 'return_type': 'None', 'has_return': True, 'decorators': [], 'docstring': '"""Eq.\n\nArgs:\n        other (Any): Description of other\n\nReturns:\n    None\n\nRaises:\n    None\n"""', 'description': 'Eq'}

{'type': 'function', 'name': '__init__', 'params': [], 'param_types': {}, 'return_type': 'None', 'has_return': False, 'decorators': [], 'docstring': '"""Init.\n\nArgs:\n        None\n\nReturns:\n    None\n\nRaises:\n    None\n"""', 'description': 'Init'}

{'type': 'function', 'name': 'resolve_pair', 'params': ['c1', 'c2'], 'param_types': {'c1': 'Clause', 'c2': 'Clause'}, 'return_type': 'Any', 'has_return': True, 'decorators': [], 'docstring': '"""Resolve pair.\n\nArgs:\n        c1 (Clause): Description of c1\n    c2 (Clause): Description of c2\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'Resolve pair'}

{'type': 'function', 'name': 'prove', 'params': ['premises', 'conclusion'], 'param_types': {'premises': "Subscript(value=Name(id='List', ctx=Load()), slice=Name(id='PropFormula', ctx=Load()), ctx=Load())", 'conclusion': 'PropFormula'}, 'return_type': 'Any', 'has_return': True, 'decorators': [], 'docstring': '"""Prove.\n\nArgs:\n        premises (Subscript(value=Name(id=\'List\', ctx=Load()), slice=Name(id=\'PropFormula\', ctx=Load()), ctx=Load())): Description of premises\n    conclusion (PropFormula): Description of conclusion\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'Prove'}

{'type': 'function', 'name': '__repr__', 'params': [], 'param_types': {}, 'return_type': 'str', 'has_return': True, 'decorators': [], 'docstring': '"""Repr.\n\nArgs:\n        None\n\nReturns:\n    str\n\nRaises:\n    None\n"""', 'description': 'Repr'}

{'type': 'function', 'name': '__init__', 'params': [], 'param_types': {}, 'return_type': 'None', 'has_return': False, 'decorators': [], 'docstring': '"""Init.\n\nArgs:\n        None\n\nReturns:\n    None\n\nRaises:\n    None\n"""', 'description': 'Init'}

{'type': 'function', 'name': 'modus_ponens_proof', 'params': ['p', 'p_implies_q'], 'param_types': {'p': 'PropFormula', 'p_implies_q': 'PropFormula'}, 'return_type': 'Any', 'has_return': True, 'decorators': [], 'docstring': '"""Modus ponens proof.\n\nArgs:\n        p (PropFormula): Description of p\n    p_implies_q (PropFormula): Description of p_implies_q\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'Modus ponens proof'}

{'type': 'function', 'name': 'hypothetical_syllogism_proof', 'params': ['p_implies_q', 'q_implies_r'], 'param_types': {'p_implies_q': 'PropFormula', 'q_implies_r': 'PropFormula'}, 'return_type': 'Any', 'has_return': True, 'decorators': [], 'docstring': '"""Hypothetical syllogism proof.\n\nArgs:\n        p_implies_q (PropFormula): Description of p_implies_q\n    q_implies_r (PropFormula): Description of q_implies_r\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'Hypothetical syllogism proof'}

{'type': 'function', 'name': 'conjunction_elim_proof', 'params': ['conj', 'side'], 'param_types': {'conj': 'PropFormula', 'side': 'str'}, 'return_type': 'Any', 'has_return': True, 'decorators': [], 'docstring': '"""Conjunction elim proof.\n\nArgs:\n        conj (PropFormula): Description of conj\n    side (str): Description of side\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'Conjunction elim proof'}

{'type': 'function', 'name': 'double_negation_proof', 'params': ['nn_p'], 'param_types': {'nn_p': 'PropFormula'}, 'return_type': 'Any', 'has_return': True, 'decorators': [], 'docstring': '"""Double negation proof.\n\nArgs:\n        nn_p (PropFormula): Description of nn_p\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'Double negation proof'}

{'type': 'function', 'name': 'auto_prove', 'params': ['premises', 'conclusion'], 'param_types': {'premises': "Subscript(value=Name(id='List', ctx=Load()), slice=Name(id='PropFormula', ctx=Load()), ctx=Load())", 'conclusion': 'PropFormula'}, 'return_type': 'Any', 'has_return': True, 'decorators': [], 'docstring': '"""Auto prove.\n\nArgs:\n        premises (Subscript(value=Name(id=\'List\', ctx=Load()), slice=Name(id=\'PropFormula\', ctx=Load()), ctx=Load())): Description of premises\n    conclusion (PropFormula): Description of conclusion\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'Auto prove'}

{'type': 'function', 'name': '__init__', 'params': [], 'param_types': {}, 'return_type': 'None', 'has_return': False, 'decorators': [], 'docstring': '"""Init.\n\nArgs:\n        None\n\nReturns:\n    None\n\nRaises:\n    None\n"""', 'description': 'Init'}

{'type': 'function', 'name': 'add_rule', 'params': ['name', 'condition', 'derive', 'explanation_template'], 'param_types': {'name': 'str', 'condition': "Subscript(value=Name(id='Callable', ctx=Load()), slice=Tuple(elts=[List(elts=[Name(id='Dict', ctx=Load())], ctx=Load()), Name(id='bool', ctx=Load())], ctx=Load()), ctx=Load())", 'derive': "Subscript(value=Name(id='Callable', ctx=Load()), slice=Tuple(elts=[List(elts=[Name(id='Dict', ctx=Load())], ctx=Load()), Subscript(value=Name(id='Optional', ctx=Load()), slice=Name(id='str', ctx=Load()), ctx=Load())], ctx=Load()), ctx=Load())", 'explanation_template': 'str'}, 'return_type': 'None', 'has_return': False, 'decorators': [], 'docstring': '"""Add rule.\n\nArgs:\n        name (str): Description of name\n    condition (Subscript(value=Name(id=\'Callable\', ctx=Load()), slice=Tuple(elts=[List(elts=[Name(id=\'Dict\', ctx=Load())], ctx=Load()), Name(id=\'bool\', ctx=Load())], ctx=Load()), ctx=Load())): Description of condition\n    derive (Subscript(value=Name(id=\'Callable\', ctx=Load()), slice=Tuple(elts=[List(elts=[Name(id=\'Dict\', ctx=Load())], ctx=Load()), Subscript(value=Name(id=\'Optional\', ctx=Load()), slice=Name(id=\'str\', ctx=Load()), ctx=Load())], ctx=Load()), ctx=Load())): Description of derive\n    explanation_template (str): Description of explanation_template\n\nReturns:\n    None\n\nRaises:\n    None\n"""', 'description': 'Add rule'}

{'type': 'function', 'name': 'forward_chain', 'params': ['initial_facts', 'max_steps'], 'param_types': {'initial_facts': "Subscript(value=Name(id='Dict', ctx=Load()), slice=Tuple(elts=[Name(id='str', ctx=Load()), Name(id='str', ctx=Load())], ctx=Load()), ctx=Load())", 'max_steps': 'int'}, 'return_type': 'Any', 'has_return': True, 'decorators': [], 'docstring': '"""Forward chain.\n\nArgs:\n        initial_facts (Subscript(value=Name(id=\'Dict\', ctx=Load()), slice=Tuple(elts=[Name(id=\'str\', ctx=Load()), Name(id=\'str\', ctx=Load())], ctx=Load()), ctx=Load())): Description of initial_facts\n    max_steps (int): Description of max_steps\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'Forward chain'}

{'type': 'function', 'name': 'build_chain', 'params': ['premises', 'target', 'max_steps'], 'param_types': {'premises': "Subscript(value=Name(id='List', ctx=Load()), slice=Name(id='str', ctx=Load()), ctx=Load())", 'target': 'str', 'max_steps': 'int'}, 'return_type': 'Any', 'has_return': True, 'decorators': [], 'docstring': '"""Build chain.\n\nArgs:\n        premises (Subscript(value=Name(id=\'List\', ctx=Load()), slice=Name(id=\'str\', ctx=Load()), ctx=Load())): Description of premises\n    target (str): Description of target\n    max_steps (int): Description of max_steps\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'Build chain'}

{'type': 'function', 'name': '__init__', 'params': [], 'param_types': {}, 'return_type': 'None', 'has_return': False, 'decorators': [], 'docstring': '"""Init.\n\nArgs:\n        None\n\nReturns:\n    None\n\nRaises:\n    None\n"""', 'description': 'Init'}

{'type': 'function', 'name': '_init_engines', 'params': [], 'param_types': {}, 'return_type': 'None', 'has_return': False, 'decorators': [], 'docstring': '"""Init engines.\n\nArgs:\n        None\n\nReturns:\n    None\n\nRaises:\n    None\n"""', 'description': 'Init engines'}

{'type': 'function', 'name': 'three_engine_logic_score', 'params': [], 'param_types': {}, 'return_type': 'Any', 'has_return': True, 'decorators': [], 'docstring': '"""Three engine logic score.\n\nArgs:\n        None\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'Three engine logic score'}

{'type': 'function', 'name': 'analyze_argument', 'params': ['premises', 'conclusion', 'argument_type'], 'param_types': {'premises': "Subscript(value=Name(id='List', ctx=Load()), slice=Name(id='str', ctx=Load()), ctx=Load())", 'conclusion': 'str', 'argument_type': 'str'}, 'return_type': 'Any', 'has_return': True, 'decorators': [], 'docstring': '"""Analyze argument.\n\nArgs:\n        premises (Subscript(value=Name(id=\'List\', ctx=Load()), slice=Name(id=\'str\', ctx=Load()), ctx=Load())): Description of premises\n    conclusion (str): Description of conclusion\n    argument_type (str): Description of argument_type\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'Analyze argument'}

{'type': 'function', 'name': 'detect_fallacies', 'params': ['text'], 'param_types': {'text': 'str'}, 'return_type': 'Any', 'has_return': True, 'decorators': [], 'docstring': '"""Detect fallacies.\n\nArgs:\n        text (str): Description of text\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'Detect fallacies'}

{'type': 'function', 'name': 'translate_to_logic', 'params': ['text'], 'param_types': {'text': 'str'}, 'return_type': 'Any', 'has_return': True, 'decorators': [], 'docstring': '"""Translate to logic.\n\nArgs:\n        text (str): Description of text\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'Translate to logic'}

{'type': 'function', 'name': 'check_validity', 'params': ['premises', 'conclusion'], 'param_types': {'premises': "Subscript(value=Name(id='List', ctx=Load()), slice=Name(id='PropFormula', ctx=Load()), ctx=Load())", 'conclusion': 'PropFormula'}, 'return_type': 'bool', 'has_return': True, 'decorators': [], 'docstring': '"""Check validity.\n\nArgs:\n        premises (Subscript(value=Name(id=\'List\', ctx=Load()), slice=Name(id=\'PropFormula\', ctx=Load()), ctx=Load())): Description of premises\n    conclusion (PropFormula): Description of conclusion\n\nReturns:\n    bool\n\nRaises:\n    None\n"""', 'description': 'Check validity'}

{'type': 'function', 'name': 'generate_truth_table', 'params': ['formula'], 'param_types': {'formula': 'PropFormula'}, 'return_type': 'Any', 'has_return': True, 'decorators': [], 'docstring': '"""Generate truth table.\n\nArgs:\n        formula (PropFormula): Description of formula\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'Generate truth table'}

{'type': 'function', 'name': 'prove_equivalence', 'params': ['f1', 'f2'], 'param_types': {'f1': 'PropFormula', 'f2': 'PropFormula'}, 'return_type': 'Any', 'has_return': True, 'decorators': [], 'docstring': '"""Prove equivalence.\n\nArgs:\n        f1 (PropFormula): Description of f1\n    f2 (PropFormula): Description of f2\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'Prove equivalence'}

{'type': 'function', 'name': 'analyze_syllogism', 'params': ['major', 'minor', 'conclusion'], 'param_types': {'major': 'str', 'minor': 'str', 'conclusion': 'str'}, 'return_type': 'Any', 'has_return': True, 'decorators': [], 'docstring': '"""Analyze syllogism.\n\nArgs:\n        major (str): Description of major\n    minor (str): Description of minor\n    conclusion (str): Description of conclusion\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'Analyze syllogism'}

{'type': 'function', 'name': 'simplify_formula', 'params': ['formula'], 'param_types': {'formula': 'PropFormula'}, 'return_type': 'PropFormula', 'has_return': True, 'decorators': [], 'docstring': '"""Simplify formula.\n\nArgs:\n        formula (PropFormula): Description of formula\n\nReturns:\n    PropFormula\n\nRaises:\n    None\n"""', 'description': 'Simplify formula'}

{'type': 'function', 'name': 'to_cnf', 'params': ['formula'], 'param_types': {'formula': 'PropFormula'}, 'return_type': 'PropFormula', 'has_return': True, 'decorators': [], 'docstring': '"""To cnf.\n\nArgs:\n        formula (PropFormula): Description of formula\n\nReturns:\n    PropFormula\n\nRaises:\n    None\n"""', 'description': 'To cnf'}

{'type': 'function', 'name': 'to_dnf', 'params': ['formula'], 'param_types': {'formula': 'PropFormula'}, 'return_type': 'PropFormula', 'has_return': True, 'decorators': [], 'docstring': '"""To dnf.\n\nArgs:\n        formula (PropFormula): Description of formula\n\nReturns:\n    PropFormula\n\nRaises:\n    None\n"""', 'description': 'To dnf'}

{'type': 'function', 'name': 'list_fallacies', 'params': [], 'param_types': {}, 'return_type': 'Any', 'has_return': True, 'decorators': [], 'docstring': '"""List fallacies.\n\nArgs:\n        None\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'List fallacies'}

{'type': 'function', 'name': 'list_logical_laws', 'params': [], 'param_types': {}, 'return_type': 'Any', 'has_return': True, 'decorators': [], 'docstring': '"""List logical laws.\n\nArgs:\n        None\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'List logical laws'}

{'type': 'function', 'name': 'resolve_proof', 'params': ['premises', 'conclusion'], 'param_types': {'premises': "Subscript(value=Name(id='List', ctx=Load()), slice=Name(id='PropFormula', ctx=Load()), ctx=Load())", 'conclusion': 'PropFormula'}, 'return_type': 'Any', 'has_return': True, 'decorators': [], 'docstring': '"""Resolve proof.\n\nArgs:\n        premises (Subscript(value=Name(id=\'List\', ctx=Load()), slice=Name(id=\'PropFormula\', ctx=Load()), ctx=Load())): Description of premises\n    conclusion (PropFormula): Description of conclusion\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'Resolve proof'}

{'type': 'function', 'name': 'natural_deduction_proof', 'params': ['premises', 'conclusion'], 'param_types': {'premises': "Subscript(value=Name(id='List', ctx=Load()), slice=Name(id='PropFormula', ctx=Load()), ctx=Load())", 'conclusion': 'PropFormula'}, 'return_type': 'Any', 'has_return': True, 'decorators': [], 'docstring': '"""Natural deduction proof.\n\nArgs:\n        premises (Subscript(value=Name(id=\'List\', ctx=Load()), slice=Name(id=\'PropFormula\', ctx=Load()), ctx=Load())): Description of premises\n    conclusion (PropFormula): Description of conclusion\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'Natural deduction proof'}

{'type': 'function', 'name': 'build_inference_chain', 'params': ['premises', 'target'], 'param_types': {'premises': "Subscript(value=Name(id='List', ctx=Load()), slice=Name(id='str', ctx=Load()), ctx=Load())", 'target': 'str'}, 'return_type': 'Any', 'has_return': True, 'decorators': [], 'docstring': '"""Build inference chain.\n\nArgs:\n        premises (Subscript(value=Name(id=\'List\', ctx=Load()), slice=Name(id=\'str\', ctx=Load()), ctx=Load())): Description of premises\n    target (str): Description of target\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'Build inference chain'}

{'type': 'function', 'name': 'comprehensive_proof', 'params': ['premises', 'conclusion'], 'param_types': {'premises': "Subscript(value=Name(id='List', ctx=Load()), slice=Name(id='PropFormula', ctx=Load()), ctx=Load())", 'conclusion': 'PropFormula'}, 'return_type': 'Any', 'has_return': True, 'decorators': [], 'docstring': '"""Comprehensive proof.\n\nArgs:\n        premises (Subscript(value=Name(id=\'List\', ctx=Load()), slice=Name(id=\'PropFormula\', ctx=Load()), ctx=Load())): Description of premises\n    conclusion (PropFormula): Description of conclusion\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'Comprehensive proof'}

{'type': 'function', 'name': 'logic_depth_score', 'params': [], 'param_types': {}, 'return_type': 'float', 'has_return': True, 'decorators': [], 'docstring': '"""Logic depth score.\n\nArgs:\n        None\n\nReturns:\n    float\n\nRaises:\n    None\n"""', 'description': 'Logic depth score'}

{'type': 'function', 'name': 'status', 'params': [], 'param_types': {}, 'return_type': 'Any', 'has_return': True, 'decorators': [], 'docstring': '"""Status.\n\nArgs:\n        None\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'Status'}

{'type': 'function', 'name': 'parse_prop', 'params': ['text'], 'param_types': {'text': 'str'}, 'return_type': 'Any', 'has_return': True, 'decorators': [], 'docstring': '"""Parse prop.\n\nArgs:\n        text (str): Description of text\n\nReturns:\n    Any\n\nRaises:\n    None\n"""', 'description': 'Parse prop'}

