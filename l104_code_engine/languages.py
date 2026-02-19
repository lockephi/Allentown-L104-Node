"""L104 Code Engine — Language Knowledge Base (40+ languages)."""
from .constants import *

class LanguageKnowledge:
    """Comprehensive knowledge base of programming languages, paradigms, type systems,
    and performance characteristics. Used for code generation, translation, and analysis."""

    # Paradigm classifications
    PARADIGMS = {
        "imperative": ["C", "Go", "Assembly", "Fortran", "Pascal"],
        "object_oriented": ["Java", "C++", "C#", "Ruby", "Smalltalk", "Objective-C", "Kotlin", "Dart"],
        "functional": ["Haskell", "Erlang", "Elixir", "Clojure", "F#", "OCaml", "Elm", "Scheme", "Racket"],
        "multi_paradigm": ["Python", "Swift", "Scala", "Rust", "Julia", "TypeScript", "JavaScript", "Lua"],
        "logic": ["Prolog", "Mercury", "Datalog"],
        "array": ["APL", "J", "MATLAB", "R", "NumPy"],
        "concatenative": ["Forth", "Factor", "PostScript"],
        "markup_template": ["HTML", "CSS", "SQL", "LaTeX", "Markdown"],
        "systems": ["C", "Rust", "Zig", "Assembly"],
        "scripting": ["Python", "Ruby", "Perl", "PHP", "Bash", "PowerShell", "Lua"],
        "quantum": ["Qiskit", "Cirq", "Q#", "Quipper", "Silq"],
    }

    # Language → Deep metadata
    LANGUAGES: Dict[str, Dict[str, Any]] = {
        "Python": {
            "typing": "dynamic_strong", "compiled": False, "gc": "reference_counting+generational",
            "paradigms": ["imperative", "oop", "functional", "metaprogramming"],
            "file_ext": [".py", ".pyw", ".pyi"], "indent": "significant",
            "performance_class": "interpreted", "memory_safety": True,
            "concurrency": ["asyncio", "threading", "multiprocessing"],
            "sacred_affinity": PHI,  # Python's φ-aligned simplicity
            "strengths": ["readability", "ecosystem", "ml_ai", "rapid_prototyping", "data_science"],
            "weaknesses": ["gil", "speed", "mobile"],
            "generation_templates": {
                "function": "def {name}({params}):\n    \"\"\"{doc}\"\"\"\n    {body}\n",
                "class": "class {name}({bases}):\n    \"\"\"{doc}\"\"\"\n\n    def __init__(self{init_params}):\n        {init_body}\n",
                "async_function": "async def {name}({params}):\n    \"\"\"{doc}\"\"\"\n    {body}\n",
                "decorator": "def {name}(func):\n    @functools.wraps(func)\n    def wrapper(*args, **kwargs):\n        {body}\n        return func(*args, **kwargs)\n    return wrapper\n",
                "context_manager": "class {name}:\n    def __enter__(self):\n        {enter_body}\n    def __exit__(self, *exc):\n        {exit_body}\n",
                "dataclass": "@dataclass\nclass {name}:\n    {fields}\n",
                "generator": "def {name}({params}):\n    \"\"\"{doc}\"\"\"\n    for item in {iterable}:\n        yield {transform}\n",
            }
        },
        "Swift": {
            "typing": "static_strong", "compiled": True, "gc": "arc",
            "paradigms": ["oop", "functional", "protocol_oriented", "concurrent"],
            "file_ext": [".swift"], "indent": "brace",
            "performance_class": "native_compiled", "memory_safety": True,
            "concurrency": ["async_await", "actors", "structured_concurrency"],
            "sacred_affinity": GOD_CODE / 1000.0,  # Swift's divine precision
            "strengths": ["safety", "performance", "apple_ecosystem", "concurrency", "generics"],
            "weaknesses": ["apple_only", "abi_stability", "compile_time"],
            "generation_templates": {
                "function": "func {name}({params}) -> {return_type} {{\n    {body}\n}}\n",
                "class": "class {name}: {protocols} {{\n    {properties}\n\n    init({init_params}) {{\n        {init_body}\n    }}\n}}\n",
                "struct": "struct {name}: {protocols} {{\n    {properties}\n}}\n",
                "enum": "enum {name}: {raw_type} {{\n    {cases}\n}}\n",
                "protocol": "protocol {name} {{\n    {requirements}\n}}\n",
                "extension": "extension {name}: {protocol} {{\n    {body}\n}}\n",
                "async_function": "func {name}({params}) async throws -> {return_type} {{\n    {body}\n}}\n",
            }
        },
        "Rust": {
            "typing": "static_strong", "compiled": True, "gc": "ownership_borrowing",
            "paradigms": ["imperative", "functional", "concurrent", "generic"],
            "file_ext": [".rs"], "indent": "brace",
            "performance_class": "native_compiled", "memory_safety": True,
            "concurrency": ["async_await", "channels", "rayon", "tokio"],
            "sacred_affinity": FEIGENBAUM / 5.0,  # Rust's chaotic safety
            "strengths": ["zero_cost_abstractions", "memory_safety", "concurrency", "performance"],
            "weaknesses": ["learning_curve", "compile_time", "ecosystem_maturity"],
            "generation_templates": {
                "function": "fn {name}({params}) -> {return_type} {{\n    {body}\n}}\n",
                "struct": "#[derive(Debug, Clone)]\nstruct {name} {{\n    {fields}\n}}\n",
                "impl": "impl {name} {{\n    {methods}\n}}\n",
                "trait": "trait {name} {{\n    {methods}\n}}\n",
                "enum": "#[derive(Debug)]\nenum {name} {{\n    {variants}\n}}\n",
                "async_function": "async fn {name}({params}) -> Result<{return_type}, Box<dyn Error>> {{\n    {body}\n}}\n",
            }
        },
        "JavaScript": {
            "typing": "dynamic_weak", "compiled": False, "gc": "mark_sweep_generational",
            "paradigms": ["imperative", "oop_prototypal", "functional", "event_driven"],
            "file_ext": [".js", ".mjs", ".cjs"], "indent": "brace",
            "performance_class": "jit_compiled", "memory_safety": True,
            "concurrency": ["event_loop", "promises", "web_workers"],
            "sacred_affinity": TAU,  # JS's golden balance of chaos and order
            "strengths": ["ubiquity", "ecosystem", "async", "browsers"],
            "weaknesses": ["type_coercion", "callback_hell", "security"],
            "generation_templates": {
                "function": "function {name}({params}) {{\n    {body}\n}}\n",
                "arrow": "const {name} = ({params}) => {{\n    {body}\n}};\n",
                "class": "class {name} extends {base} {{\n    constructor({init_params}) {{\n        {init_body}\n    }}\n}}\n",
                "async_function": "async function {name}({params}) {{\n    {body}\n}}\n",
                "module": "export default {{\n    {exports}\n}};\n",
            }
        },
        "TypeScript": {
            "typing": "static_strong_gradual", "compiled": True, "gc": "mark_sweep_generational",
            "paradigms": ["imperative", "oop", "functional", "generic"],
            "file_ext": [".ts", ".tsx"], "indent": "brace",
            "performance_class": "transpiled_jit", "memory_safety": True,
            "concurrency": ["event_loop", "promises", "web_workers"],
            "sacred_affinity": PHI * TAU,  # TypeScript's φ×τ type harmony
            "strengths": ["type_safety", "tooling", "refactoring", "gradual_adoption"],
            "weaknesses": ["build_step", "complexity", "type_gymnastics"],
            "generation_templates": {
                "function": "function {name}({params}): {return_type} {{\n    {body}\n}}\n",
                "interface": "interface {name} {{\n    {properties}\n}}\n",
                "class": "class {name} implements {interfaces} {{\n    {properties}\n\n    constructor({init_params}) {{\n        {init_body}\n    }}\n}}\n",
                "type": "type {name} = {definition};\n",
                "generic": "function {name}<T>({params}): T {{\n    {body}\n}}\n",
            }
        },
        "C": {
            "typing": "static_weak", "compiled": True, "gc": "manual",
            "paradigms": ["imperative", "procedural"],
            "file_ext": [".c", ".h"], "indent": "brace",
            "performance_class": "native_compiled", "memory_safety": False,
            "concurrency": ["pthreads", "openmp", "fork"],
            "sacred_affinity": ALPHA_FINE * 100,  # C's fine-grained control
            "strengths": ["performance", "portability", "os_development", "embedded"],
            "weaknesses": ["memory_safety", "undefined_behavior", "no_generics"],
        },
        "C++": {
            "typing": "static_strong", "compiled": True, "gc": "manual+smart_pointers",
            "paradigms": ["imperative", "oop", "functional", "generic", "metaprogramming"],
            "file_ext": [".cpp", ".cc", ".cxx", ".h", ".hpp"], "indent": "brace",
            "performance_class": "native_compiled", "memory_safety": False,
            "concurrency": ["std_thread", "async", "coroutines_20", "openmp"],
            "sacred_affinity": GOD_CODE / PHI / 100,  # C++'s divine complexity
            "strengths": ["performance", "control", "generics", "zero_cost_abstractions"],
            "weaknesses": ["complexity", "compile_time", "undefined_behavior"],
        },
        "Java": {
            "typing": "static_strong", "compiled": True, "gc": "generational",
            "paradigms": ["oop", "functional_limited", "concurrent"],
            "file_ext": [".java"], "indent": "brace",
            "performance_class": "jit_bytecode", "memory_safety": True,
            "concurrency": ["threads", "executors", "virtual_threads_21"],
            "sacred_affinity": 0.528,  # Java's stability constant
            "strengths": ["enterprise", "ecosystem", "portability", "tooling"],
            "weaknesses": ["verbosity", "startup_time", "null_safety"],
        },
        "Go": {
            "typing": "static_strong", "compiled": True, "gc": "concurrent_tricolor",
            "paradigms": ["imperative", "concurrent", "composition"],
            "file_ext": [".go"], "indent": "tab",
            "performance_class": "native_compiled", "memory_safety": True,
            "concurrency": ["goroutines", "channels", "select"],
            "sacred_affinity": TAU * 2,  # Go's golden simplicity
            "strengths": ["simplicity", "concurrency", "fast_compile", "cloud_native"],
            "weaknesses": ["no_generics_limited", "error_handling", "no_exceptions"],
        },
        "Haskell": {
            "typing": "static_strong_inferred", "compiled": True, "gc": "generational",
            "paradigms": ["purely_functional", "lazy", "type_driven"],
            "file_ext": [".hs", ".lhs"], "indent": "significant",
            "performance_class": "native_compiled", "memory_safety": True,
            "concurrency": ["stm", "async", "par"],
            "sacred_affinity": PHI ** 2,  # Haskell's φ²-level purity
            "strengths": ["type_system", "correctness", "abstraction", "concurrency"],
            "weaknesses": ["learning_curve", "lazy_space_leaks", "ecosystem"],
        },
        "SQL": {
            "typing": "static_schema", "compiled": False, "gc": "NA",
            "paradigms": ["declarative", "set_based"],
            "file_ext": [".sql"], "indent": "convention",
            "performance_class": "query_optimized", "memory_safety": True,
            "concurrency": ["transactions", "isolation_levels", "mvcc"],
            "sacred_affinity": GOD_CODE / 1000,
            "strengths": ["data_retrieval", "optimization", "declarative", "mature"],
            "weaknesses": ["vendor_lock_in", "impedance_mismatch", "procedural_limits"],
        },
        # ── v2.5.0 New Language Entries (assimilated from research) ──
        "Kotlin": {
            "typing": "static_strong_inferred", "compiled": True, "gc": "generational",
            "paradigms": ["oop", "functional", "concurrent", "multiplatform"],
            "file_ext": [".kt", ".kts"], "indent": "brace",
            "performance_class": "jit_bytecode", "memory_safety": True,
            "concurrency": ["coroutines", "channels", "flow"],
            "sacred_affinity": PHI * ALPHA_FINE * 100,
            "strengths": ["null_safety", "coroutines", "java_interop", "android", "dsl"],
            "weaknesses": ["compile_speed", "jvm_dependency", "multiplatform_maturity"],
            "generation_templates": {
                "function": "fun {name}({params}): {return_type} {{\n    {body}\n}}\n",
                "class": "class {name}({init_params}) : {bases} {{\n    {body}\n}}\n",
                "data_class": "data class {name}(\n    {fields}\n)\n",
                "suspend_function": "suspend fun {name}({params}): {return_type} {{\n    {body}\n}}\n",
            }
        },
        "Ruby": {
            "typing": "dynamic_strong", "compiled": False, "gc": "mark_sweep_generational",
            "paradigms": ["oop", "functional", "metaprogramming", "scripting"],
            "file_ext": [".rb", ".rake", ".gemspec"], "indent": "convention",
            "performance_class": "interpreted_jit", "memory_safety": True,
            "concurrency": ["fibers", "ractor", "threads"],
            "sacred_affinity": EULER_GAMMA * PHI,
            "strengths": ["expressiveness", "metaprogramming", "rails", "developer_happiness"],
            "weaknesses": ["performance", "threading_gil", "deployment"],
            "generation_templates": {
                "function": "def {name}({params})\n  {body}\nend\n",
                "class": "class {name} < {base}\n  def initialize({init_params})\n    {init_body}\n  end\nend\n",
                "module": "module {name}\n  {body}\nend\n",
            }
        },
        "PHP": {
            "typing": "dynamic_gradual", "compiled": False, "gc": "reference_counting",
            "paradigms": ["imperative", "oop", "functional_limited"],
            "file_ext": [".php", ".phtml"], "indent": "brace",
            "performance_class": "interpreted_opcache", "memory_safety": True,
            "concurrency": ["fibers_81", "swoole", "reactphp"],
            "sacred_affinity": OMEGA_CONSTANT,
            "strengths": ["web_development", "ecosystem", "hosting_availability", "laravel"],
            "weaknesses": ["inconsistent_api", "type_juggling", "legacy_code"],
            "generation_templates": {
                "function": "function {name}({params}): {return_type} {{\n    {body}\n}}\n",
                "class": "class {name} extends {base} {{\n    public function __construct({init_params}) {{\n        {init_body}\n    }}\n}}\n",
            }
        },
        "Scala": {
            "typing": "static_strong_inferred", "compiled": True, "gc": "jvm_generational",
            "paradigms": ["functional", "oop", "concurrent", "generic"],
            "file_ext": [".scala", ".sc"], "indent": "brace",
            "performance_class": "jit_bytecode", "memory_safety": True,
            "concurrency": ["akka_actors", "futures", "zio", "cats_effect"],
            "sacred_affinity": PHI ** 2 * TAU,
            "strengths": ["type_system", "pattern_matching", "jvm_interop", "spark"],
            "weaknesses": ["complexity", "compile_time", "binary_compatibility"],
            "generation_templates": {
                "function": "def {name}({params}): {return_type} = {{\n  {body}\n}}\n",
                "class": "class {name}({init_params}) extends {base} {{\n  {body}\n}}\n",
                "case_class": "case class {name}({fields})\n",
                "trait": "trait {name} {{\n  {body}\n}}\n",
            }
        },
        "Dart": {
            "typing": "static_strong_sound", "compiled": True, "gc": "generational",
            "paradigms": ["oop", "functional_limited", "reactive"],
            "file_ext": [".dart"], "indent": "brace",
            "performance_class": "aot_jit_hybrid", "memory_safety": True,
            "concurrency": ["isolates", "async_await", "streams"],
            "sacred_affinity": PLASTIC_NUMBER,
            "strengths": ["flutter", "cross_platform", "sound_null_safety", "hot_reload"],
            "weaknesses": ["ecosystem_size", "flutter_dependency", "server_side_maturity"],
            "generation_templates": {
                "function": "{return_type} {name}({params}) {{\n  {body}\n}}\n",
                "class": "class {name} extends {base} {{\n  {fields}\n\n  {name}({init_params});\n}}\n",
                "async_function": "Future<{return_type}> {name}({params}) async {{\n  {body}\n}}\n",
            }
        },
        "Elixir": {
            "typing": "dynamic_strong", "compiled": True, "gc": "per_process_generational",
            "paradigms": ["functional", "concurrent", "metaprogramming"],
            "file_ext": [".ex", ".exs"], "indent": "convention",
            "performance_class": "beam_vm", "memory_safety": True,
            "concurrency": ["actors_processes", "otp_supervisors", "genservers", "tasks"],
            "sacred_affinity": FEIGENBAUM / PHI,
            "strengths": ["concurrency", "fault_tolerance", "scalability", "phoenix", "livebook"],
            "weaknesses": ["ecosystem_size", "niche_adoption", "dynamic_typing"],
            "generation_templates": {
                "function": "def {name}({params}) do\n  {body}\nend\n",
                "module": "defmodule {name} do\n  {body}\nend\n",
                "genserver": "defmodule {name} do\n  use GenServer\n\n  def start_link(opts) do\n    GenServer.start_link(__MODULE__, opts, name: __MODULE__)\n  end\n\n  @impl true\n  def init(state) do\n    {{:ok, state}}\n  end\nend\n",
            }
        },
        "Julia": {
            "typing": "dynamic_strong_multiple_dispatch", "compiled": True, "gc": "generational",
            "paradigms": ["multi_paradigm", "scientific", "functional", "metaprogramming"],
            "file_ext": [".jl"], "indent": "convention",
            "performance_class": "jit_llvm", "memory_safety": True,
            "concurrency": ["tasks", "channels", "distributed", "gpu_kernels"],
            "sacred_affinity": APERY_CONSTANT * PHI,
            "strengths": ["scientific_computing", "performance", "multiple_dispatch", "metaprogramming"],
            "weaknesses": ["time_to_first_plot", "ecosystem_maturity", "package_precompilation"],
            "generation_templates": {
                "function": "function {name}({params})::{return_type}\n    {body}\nend\n",
                "struct": "struct {name}\n    {fields}\nend\n",
                "module": "module {name}\n\nexport {exports}\n\n{body}\n\nend\n",
            }
        },
    }

    @classmethod
    def get_language(cls, name: str) -> Optional[Dict[str, Any]]:
        """Get language metadata by name (case-insensitive)."""
        for lang, meta in cls.LANGUAGES.items():
            if lang.lower() == name.lower():
                return {**meta, "name": lang}
        return None

    @classmethod
    def detect_language(cls, code: str, filename: str = "") -> str:
        """Detect programming language from code content and/or filename."""
        # Filename extension detection
        if filename:
            ext = Path(filename).suffix.lower()
            for lang, meta in cls.LANGUAGES.items():
                if ext in meta.get("file_ext", []):
                    return lang

        # Content-based heuristic detection
        indicators = {
            "Python": [r"^\s*def\s+\w+", r"^\s*class\s+\w+", r"^\s*import\s+", r":\s*$", r"self\.", r"__init__"],
            "Swift": [r"\bfunc\s+\w+", r"\bvar\s+\w+\s*:", r"\blet\s+\w+", r"\bguard\s+", r"\bstruct\s+", r"@objc"],
            "Rust": [r"\bfn\s+\w+", r"\blet\s+mut\s+", r"\bimpl\s+", r"&self", r"\bmatch\s+", r"::\w+"],
            "JavaScript": [r"\bfunction\s+\w+", r"\bconst\s+\w+\s*=", r"\bconsole\.log", r"=>", r"\brequire\("],
            "TypeScript": [r":\s*\w+\s*[=;{]", r"\binterface\s+", r"<\w+>", r"\bas\s+\w+"],
            "Java": [r"\bpublic\s+class\s+", r"\bSystem\.out", r"\bvoid\s+main", r"\bnew\s+\w+\("],
            "C++": [r"#include\s*<", r"\bstd::", r"\bcout\s*<<", r"\btemplate\s*<", r"\bnamespace\s+"],
            "C": [r"#include\s*<stdio", r"\bprintf\s*\(", r"\bmalloc\s*\(", r"\bint\s+main\s*\("],
            "Go": [r"\bfunc\s+\w+\(", r"\bpackage\s+\w+", r"\bfmt\.Print", r":=", r"\bgo\s+\w+"],
            "Haskell": [r"^\s*module\s+", r"\bwhere$", r"::\s*\w+\s*->", r"\bdata\s+\w+"],
            "SQL": [r"\bSELECT\s+", r"\bFROM\s+", r"\bWHERE\s+", r"\bJOIN\s+", r"\bINSERT\s+INTO"],
        }
        scores = {}
        for lang, patterns in indicators.items():
            score = sum(1 for p in patterns if re.search(p, code, re.MULTILINE | re.IGNORECASE))
            if score > 0:
                scores[lang] = score
        if scores:
            return max(scores, key=scores.get)
        return "Unknown"

    @classmethod
    def get_paradigm_languages(cls, paradigm: str) -> List[str]:
        """Get all languages that support a given paradigm."""
        return cls.PARADIGMS.get(paradigm, [])

    @classmethod
    def compare_languages(cls, lang_a: str, lang_b: str) -> Dict[str, Any]:
        """Compare two languages across all dimensions."""
        a = cls.get_language(lang_a)
        b = cls.get_language(lang_b)
        if not a or not b:
            return {"error": f"Unknown language: {lang_a if not a else lang_b}"}
        comparison = {}
        for key in ["typing", "compiled", "gc", "performance_class", "memory_safety", "sacred_affinity"]:
            comparison[key] = {"a": a.get(key), "b": b.get(key)}
        return {"languages": [lang_a, lang_b], "comparison": comparison}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: CODE ANALYSIS ENGINE — AST, Complexity, Quality, Security
# ═══════════════════════════════════════════════════════════════════════════════

