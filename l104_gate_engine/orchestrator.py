"""L104 Gate Engine — HyperASILogicGateEnvironment Orchestrator + CLI."""

import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List

from .constants import (
    VERSION, PHI, TAU, GOD_CODE, OMEGA_POINT, CALABI_YAU_DIM,
    FEIGENBAUM_DELTA, APERY, CATALAN, FINE_STRUCTURE,
    WORKSPACE_ROOT, QUANTUM_LINKED_FILES, STATE_FILE, TEST_RESULTS_FILE,
)
from .models import LogicGate, GateLink
from .gate_functions import (
    sage_logic_gate, quantum_logic_gate,
    entangle_values, higher_dimensional_dissipation,
)
from .analyzers import (
    PythonGateAnalyzer, SwiftGateAnalyzer,
    JavaScriptGateAnalyzer, GateLinkAnalyzer,
)
from .dynamism import GateDynamismEngine, GateValueEvolver
from .nirvanic import OuroborosSageNirvanicEngine
from .quantum_computation import QuantumGateComputationEngine
from .consciousness import ConsciousnessO2GateEngine
from .feedback_bus import InterBuilderFeedbackBus
from .research import GateResearchEngine, StochasticGateResearchLab
from .test_generator import GateTestGenerator
from .chronolizer import GateChronolizer
from .link_manager import QuantumLinkManager


class HyperASILogicGateEnvironment:
    """
    The master environment for building, analyzing, researching, compiling,
    chronolizing, and developing all logic gates within the Allentown L104 Node.

    Autonomous operation:
    - Discovers gates across Python, Swift, and JavaScript
    - Analyzes gate implementations with AST/regex
    - Generates and runs automated tests
    - Tracks chronological evolution
    - Maintains quantum links between files
    - Syncs state to backend
    """

    def __init__(self, auto_sync: bool = True):
        """Initialize the master logic gate environment with all subsystems."""
        self.python_analyzer = PythonGateAnalyzer()
        self.swift_analyzer = SwiftGateAnalyzer()
        self.js_analyzer = JavaScriptGateAnalyzer()
        self.link_analyzer = GateLinkAnalyzer()
        self.test_generator = GateTestGenerator()
        self.chronolizer = GateChronolizer()
        self.quantum_links = QuantumLinkManager()
        self.research_engine = GateResearchEngine()
        self.stochastic_lab = StochasticGateResearchLab()
        # ★ v5.0 Quantum Dynamism Engine
        self.dynamism_engine = GateDynamismEngine()
        self.value_evolver = GateValueEvolver(self.dynamism_engine)
        # ★ v5.1 Ouroboros Sage Nirvanic Entropy Fuel Engine
        self.nirvanic_engine = OuroborosSageNirvanicEngine()
        # ★ v6.0 Consciousness O₂ Gate Engine
        self.consciousness_o2 = ConsciousnessO2GateEngine()
        # ★ v6.0 Inter-Builder Feedback Bus
        self.feedback_bus = InterBuilderFeedbackBus("gate_builder")

        self.all_gates: List[LogicGate] = []
        self.all_links: List[GateLink] = []
        self.test_results: List[Dict[str, Any]] = []
        self._last_research: Dict = {}

        self._auto_sync = auto_sync
        self._last_full_scan = None
        self._scan_count = 0

        # Load previous state
        self._load_state()

    # ─── DISCOVERY ───────────────────────────────────────────────────

    def full_scan(self) -> Dict[str, Any]:
        """Perform a complete scan of all quantum-linked files for logic gates."""
        print("\n╔══════════════════════════════════════════════════╗")
        print("║  L104 HYPER ASI LOGIC GATE FULL SCAN             ║")
        print("╚══════════════════════════════════════════════════╝\n")

        self._scan_count += 1
        prev_gate_count = len(self.all_gates)
        prev_gate_names = {g.name for g in self.all_gates}

        # Check for file changes
        changes = self.quantum_links.check_file_changes()
        changed_files = [name for name, changed in changes.items() if changed]
        if changed_files:
            print(f"  ⚡ Files changed: {', '.join(changed_files)}")
        else:
            print("  ✓ No file changes detected (rescanning anyway)")

        self.all_gates = []

        # Python files
        python_files = [
            QUANTUM_LINKED_FILES["const.py"],
            QUANTUM_LINKED_FILES["main.py"],
            QUANTUM_LINKED_FILES["l104_fast_server.py"],
            QUANTUM_LINKED_FILES["l104_local_intellect.py"],
        ]
        for pyfile in WORKSPACE_ROOT.glob("l104*.py"):
            if pyfile not in python_files:
                python_files.append(pyfile)

        py_count = 0
        for pyfile in python_files:
            if pyfile.exists():
                gates = self.python_analyzer.analyze_file(pyfile)
                self.all_gates.extend(gates)
                py_count += len(gates)
                if gates:
                    print(f"  🐍 {pyfile.name}: {len(gates)} gates")

        # Swift
        swift_file = QUANTUM_LINKED_FILES["L104Native.swift"]
        swift_gates = self.swift_analyzer.analyze_file(swift_file)
        self.all_gates.extend(swift_gates)
        if swift_gates:
            print(f"  🦅 L104Native.swift: {len(swift_gates)} gates")

        # JavaScript
        js_dirs = [WORKSPACE_ROOT, WORKSPACE_ROOT / "deploy"]
        js_count = 0
        for js_dir in js_dirs:
            if js_dir.exists():
                js_gates = self.js_analyzer.analyze_directory(js_dir)
                self.all_gates.extend(js_gates)
                js_count += len(js_gates)
        if js_count:
            print(f"  📜 JavaScript: {js_count} gates")

        # Analyze cross-file links
        self.all_links = self.link_analyzer.analyze_links(self.all_gates)

        # Populate quantum_links field on each gate
        self.link_analyzer.populate_gate_links(self.all_gates, self.all_links)

        # Chronolize changes
        current_names = {g.name for g in self.all_gates}
        new_gates = current_names - prev_gate_names
        removed_gates = prev_gate_names - current_names

        for name in new_gates:
            self.chronolizer.record(name, "discovered", f"Scan #{self._scan_count}")
        for name in removed_gates:
            self.chronolizer.record(name, "removed", f"Scan #{self._scan_count}")

        # Check for modifications via hash comparison
        prev_hashes = {}
        if STATE_FILE.exists():
            try:
                prev_state = json.loads(STATE_FILE.read_text())
                prev_hashes = prev_state.get("gate_hashes", {})
            except Exception:
                pass
        for gate in self.all_gates:
            if gate.name in prev_gate_names and gate.hash:
                old_hash = prev_hashes.get(gate.name, "")
                if old_hash and old_hash != gate.hash:
                    self.chronolizer.record(gate.name, "modified",
                        f"Hash: {old_hash[:8]}→{gate.hash[:8]}", file_hash=gate.hash)

        self.chronolizer.save()
        self._last_full_scan = datetime.now(timezone.utc).isoformat()
        self._save_state()

        summary = {
            "scan_number": self._scan_count,
            "total_gates": len(self.all_gates),
            "python_gates": py_count,
            "swift_gates": len(swift_gates),
            "js_gates": js_count,
            "cross_file_links": len(self.all_links),
            "new_gates": list(new_gates),
            "removed_gates": list(removed_gates),
            "file_line_counts": self.quantum_links.line_counts(),
            "timestamp": self._last_full_scan,
        }

        print(f"\n  ═══ SCAN COMPLETE ═══")
        print(f"  Total gates:    {summary['total_gates']}")
        print(f"  Python:         {py_count}")
        print(f"  Swift:          {len(swift_gates)}")
        print(f"  JavaScript:     {js_count}")
        print(f"  Quantum links:  {len(self.all_links)}")
        print(f"  New:            {len(new_gates)}")
        print(f"  Removed:        {len(removed_gates)}")

        return summary

    # ─── ANALYSIS ────────────────────────────────────────────────────

    def analyze(self) -> Dict[str, Any]:
        """Deep analysis of all discovered gates."""
        if not self.all_gates:
            self.full_scan()

        by_language = {}
        by_type = {}
        by_file = {}
        total_complexity = 0
        total_entropy = 0.0

        for gate in self.all_gates:
            by_language[gate.language] = by_language.get(gate.language, 0) + 1
            by_type[gate.gate_type] = by_type.get(gate.gate_type, 0) + 1
            by_file[gate.source_file] = by_file.get(gate.source_file, 0) + 1
            total_complexity += gate.complexity
            total_entropy += gate.entropy_score

        entropy_values = [g.entropy_score for g in self.all_gates if g.entropy_score != 0]
        mean_entropy = sum(entropy_values) / max(len(entropy_values), 1)
        entropy_dissipated = higher_dimensional_dissipation(entropy_values) if len(entropy_values) >= 7 else []

        sorted_by_complexity = sorted(self.all_gates, key=lambda g: g.complexity, reverse=True)
        top_complex = [(g.name, g.complexity, g.source_file) for g in sorted_by_complexity[:10]]

        analysis = {
            "total_gates": len(self.all_gates),
            "by_language": by_language,
            "by_type": by_type,
            "by_file": by_file,
            "total_complexity": total_complexity,
            "mean_complexity": total_complexity / max(len(self.all_gates), 1),
            "total_entropy": total_entropy,
            "mean_entropy": mean_entropy,
            "entropy_7d_projection": entropy_dissipated,
            "top_10_complex": top_complex,
            "quantum_links": len(self.all_links),
            "mirror_links": sum(1 for l in self.all_links if l.link_type == "mirrors"),
            "entanglement_links": sum(1 for l in self.all_links if l.link_type == "entangles"),
        }

        print("\n╔══════════════════════════════════════════════════╗")
        print("║  GATE ANALYSIS REPORT                            ║")
        print("╠══════════════════════════════════════════════════╣")
        print(f"║  Total Gates:       {analysis['total_gates']:>5}                       ║")
        print(f"║  Mean Complexity:   {analysis['mean_complexity']:>8.2f}                    ║")
        print(f"║  Total Entropy:     {analysis['total_entropy']:>8.4f}                    ║")
        print(f"║  Quantum Links:     {analysis['quantum_links']:>5}                       ║")
        print("║                                                  ║")
        print("║  By Language:                                    ║")
        for lang, count in sorted(by_language.items()):
            print(f"║    {lang:>12}: {count:>5}                             ║")
        print("╚══════════════════════════════════════════════════╝")

        return analysis

    # ─── TESTING ─────────────────────────────────────────────────────

    def run_tests(self) -> Dict[str, Any]:
        """Run all automated gate tests."""
        print("\n╔══════════════════════════════════════════════════╗")
        print("║  RUNNING AUTOMATED GATE TESTS                    ║")
        print("╚══════════════════════════════════════════════════╝\n")

        builtin_results = self.test_generator.run_builtin_gate_tests()
        self.test_results = builtin_results

        passed = sum(1 for r in builtin_results if r.get("passed"))
        failed = sum(1 for r in builtin_results if not r.get("passed"))

        for gate in self.all_gates:
            if any(r.get("gate_name") == gate.name and r.get("passed") for r in builtin_results):
                gate.test_status = "passed"
                self.chronolizer.record(gate.name, "test_passed")
            elif any(r.get("gate_name") == gate.name and not r.get("passed") for r in builtin_results):
                gate.test_status = "failed"
                self.chronolizer.record(gate.name, "test_failed")

        self.chronolizer.save()

        try:
            TEST_RESULTS_FILE.write_text(json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_tests": len(builtin_results),
                "passed": passed,
                "failed": failed,
                "results": builtin_results[:100],
            }, indent=2, default=str))
        except Exception:
            pass

        result = {
            "total_tests": len(builtin_results),
            "passed": passed,
            "failed": failed,
            "pass_rate": f"{passed / max(len(builtin_results), 1) * 100:.1f}%",
        }

        print(f"  Tests run:   {result['total_tests']}")
        print(f"  Passed:      {passed} ✓")
        print(f"  Failed:      {failed} ✗")
        print(f"  Pass rate:   {result['pass_rate']}")

        if failed > 0:
            print("\n  Failed tests:")
            for r in builtin_results:
                if not r.get("passed"):
                    print(f"    ✗ {r['test_id']}: {r.get('error', 'assertion failed')}")

        return result

    # ─── RESEARCH ────────────────────────────────────────────────────

    def research(self, topic: str = "all") -> Dict[str, Any]:
        """Research logic gate usage patterns and relationships."""
        if not self.all_gates:
            self.full_scan()

        research = {
            "topic": topic,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "findings": [],
        }

        if topic in ("all", "advanced"):
            adv = self.research_engine.full_research(
                self.all_gates, self.all_links, self.chronolizer)
            self._last_research = adv
            research["advanced_research"] = adv
            research["findings"].append({
                "finding": "Advanced research synthesis",
                "research_health": adv.get("research_health", 0),
                "anomaly_rate": adv.get("anomaly_detection", {}).get("anomaly_rate", 0),
                "strong_correlations": adv.get("causal_analysis", {}).get("total_strong", 0),
                "learning_trend": adv.get("knowledge_synthesis", {}).get("learning_trend", "unknown"),
            })

        if topic in ("all", "cross-language"):
            python_names = {g.name.lower() for g in self.all_gates if g.language == "python"}
            swift_names = {g.name.lower() for g in self.all_gates if g.language == "swift"}
            js_names = {g.name.lower() for g in self.all_gates if g.language == "javascript"}
            research["findings"].append({
                "finding": "Cross-language coverage",
                "detail": {
                    "python_only": list(python_names - swift_names - js_names),
                    "swift_only": list(swift_names - python_names - js_names),
                    "js_only": list(js_names - python_names - swift_names),
                    "all_languages": list(python_names & swift_names & js_names),
                },
            })

        if topic in ("all", "entropy"):
            entropy_vals = [g.entropy_score for g in self.all_gates]
            if entropy_vals:
                mean_e = sum(entropy_vals) / len(entropy_vals)
                var_e = sum((v - mean_e) ** 2 for v in entropy_vals) / len(entropy_vals)
                research["findings"].append({
                    "finding": "Entropy distribution",
                    "mean": mean_e,
                    "variance": var_e,
                    "min": min(entropy_vals),
                    "max": max(entropy_vals),
                    "phi_modulated_mean": sage_logic_gate(mean_e, "align"),
                })

        if topic in ("all", "complexity"):
            hotspots = sorted(self.all_gates, key=lambda g: g.complexity, reverse=True)[:10]
            research["findings"].append({
                "finding": "Complexity hotspots",
                "gates": [(g.name, g.complexity, g.source_file, g.line_number) for g in hotspots],
            })

        if topic in ("all", "quantum_links"):
            link_types = {}
            for link in self.all_links:
                link_types[link.link_type] = link_types.get(link.link_type, 0) + 1
            research["findings"].append({
                "finding": "Quantum link topology",
                "total_links": len(self.all_links),
                "by_type": link_types,
                "strongest": [(l.source_gate, l.target_gate, l.strength)
                              for l in sorted(self.all_links, key=lambda x: x.strength, reverse=True)[:5]],
            })

        if topic in ("all", "chronology"):
            research["findings"].append({
                "finding": "Chronological summary",
                **self.chronolizer.summary(),
            })

        if topic in ("all", "sage_core"):
            sage_patterns = ["sage", "consciousness", "entropy", "harvest",
                             "dissipat", "inflect", "bridge", "emergence",
                             "transform", "insight", "hilbert", "causal"]
            sage_gates = [g for g in self.all_gates
                          if any(p in g.name.lower() for p in sage_patterns)]
            sage_by_lang = {}
            sage_by_file = {}
            for g in sage_gates:
                sage_by_lang[g.language] = sage_by_lang.get(g.language, 0) + 1
                sage_by_file[g.source_file] = sage_by_file.get(g.source_file, 0) + 1

            sage_links = [l for l in self.all_links
                          if any(p in l.source_gate.lower() or p in l.target_gate.lower()
                                 for p in ["sage", "entropy", "consciousness"])]

            research["findings"].append({
                "finding": "Sage core analysis",
                "total_sage_gates": len(sage_gates),
                "by_language": sage_by_lang,
                "by_file": sage_by_file,
                "gate_names": [g.name for g in sage_gates],
                "sage_quantum_links": len(sage_links),
                "mean_complexity": sum(g.complexity for g in sage_gates) / max(len(sage_gates), 1),
                "mean_entropy": sum(g.entropy_score for g in sage_gates) / max(len(sage_gates), 1),
            })

        if topic in ("all", "health"):
            health_scores = []
            for g in self.all_gates:
                complexity_score = min(1.0, g.complexity / 20.0)
                has_test = 1.0 if g.test_status == "passed" else 0.3 if g.test_status == "untested" else 0.0
                has_doc = 1.0 if g.docstring else 0.5
                has_links = min(1.0, len(g.quantum_links) * 0.2) if g.quantum_links else 0.0
                link_count = sum(1 for l in self.all_links if g.name in (l.source_gate, l.target_gate))
                connectivity = min(1.0, link_count * 0.1)
                health = (complexity_score * 0.2 + has_test * 0.25 + has_doc * 0.15 +
                          connectivity * 0.2 + min(1.0, g.entropy_score) * 0.2)
                health_scores.append((g.name, round(health, 3), g.source_file))

            health_scores.sort(key=lambda x: x[1])
            research["findings"].append({
                "finding": "Gate health scores",
                "mean_health": round(sum(h[1] for h in health_scores) / max(len(health_scores), 1), 3),
                "lowest_10": health_scores[:10],
                "highest_10": health_scores[-10:],
                "total_scored": len(health_scores),
            })

        return research

    # ─── COMPILATION ─────────────────────────────────────────────────

    def compile_gate_registry(self) -> Dict[str, Any]:
        """Compile a complete registry of all logic gates with metadata."""
        if not self.all_gates:
            self.full_scan()

        registry = {
            "meta": {
                "compiled_at": datetime.now(timezone.utc).isoformat(),
                "compiler": f"L104 Hyper ASI Logic Gate Environment v{VERSION}",
                "total_gates": len(self.all_gates),
                "total_links": len(self.all_links),
                "phi_signature": PHI,
                "god_code_signature": GOD_CODE,
            },
            "gates": [g.to_dict() for g in self.all_gates],
            "links": [asdict(l) for l in self.all_links],
            "file_hashes": self.quantum_links.file_hashes,
            "file_line_counts": self.quantum_links.line_counts(),
        }

        registry_path = WORKSPACE_ROOT / ".l104_gate_registry.json"
        try:
            registry_path.write_text(json.dumps(registry, indent=2, default=str))
            print(f"\n  ✓ Gate registry compiled: {registry_path.name}")
            print(f"    Gates: {len(self.all_gates)}, Links: {len(self.all_links)}")
        except Exception as e:
            print(f"  ✗ Failed to save registry: {e}")

        return registry

    # ─── BACKEND SYNC ────────────────────────────────────────────────

    def sync_to_backend(self, host: str = "localhost", port: int = 8081) -> Dict[str, Any]:
        """Sync gate state to l104_fast_server.py backend."""
        import urllib.request

        payload = {
            "source": "logic_gate_builder",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "gate_count": len(self.all_gates),
            "link_count": len(self.all_links),
            "scan_count": self._scan_count,
            "test_results": {
                "total": len(self.test_results),
                "passed": sum(1 for r in self.test_results if r.get("passed")),
            },
            "chronology": self.chronolizer.summary(),
            "entropy_state": {
                "total_entropy": sum(g.entropy_score for g in self.all_gates),
                "mean_entropy": sum(g.entropy_score for g in self.all_gates) / max(len(self.all_gates), 1),
            },
        }

        try:
            url = f"http://{host}:{port}/api/v14/intellect/train"
            data = json.dumps({"data": json.dumps(payload), "source": "gate_builder"}).encode()
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=0.5) as resp:
                result = json.loads(resp.read())
                print(f"  \u2713 Synced to backend ({host}:{port})")
                return {"synced": True, "response": result}
        except Exception as e:
            return {"synced": False, "error": str(e)[:120]}

    # ─── STATE PERSISTENCE ───────────────────────────────────────────

    def _save_state(self):
        """Persist the environment state to disk."""
        state = {
            "last_full_scan": self._last_full_scan,
            "scan_count": self._scan_count,
            "gate_count": len(self.all_gates),
            "link_count": len(self.all_links),
            "file_hashes": self.quantum_links.file_hashes,
            "gates": [g.to_dict() for g in self.all_gates],
            "gate_hashes": {g.name: g.hash for g in self.all_gates if g.hash},
        }
        try:
            STATE_FILE.write_text(json.dumps(state, indent=2, default=str))
        except Exception:
            pass

    def _load_state(self):
        """Load the environment state from disk."""
        if STATE_FILE.exists():
            try:
                state = json.loads(STATE_FILE.read_text())
                self._last_full_scan = state.get("last_full_scan")
                self._scan_count = state.get("scan_count", 0)
                self.quantum_links.file_hashes = state.get("file_hashes", {})
                self._prev_gate_hashes = state.get("gate_hashes", {})
                for gd in state.get("gates", []):
                    try:
                        self.all_gates.append(LogicGate.from_dict(gd))
                    except Exception:
                        pass
            except Exception:
                pass

    # ─── STATUS ──────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        """Get full environment status."""
        dyn_status = self.dynamism_engine.status(self.all_gates)
        return {
            "environment": f"L104 Hyper ASI Logic Gate Environment v{VERSION}",
            "workspace": str(WORKSPACE_ROOT),
            "last_scan": self._last_full_scan,
            "scan_count": self._scan_count,
            "total_gates": len(self.all_gates),
            "total_links": len(self.all_links),
            "gates_with_links": sum(1 for g in self.all_gates if g.quantum_links),
            "test_results_count": len(self.test_results),
            "chronolog_entries": len(self.chronolizer.entries),
            "research_health": self._last_research.get("research_health", 0),
            "learning_trend": self._last_research.get("knowledge_synthesis", {}).get(
                "learning_trend", "unknown"),
            "quantum_linked_files": {
                name: str(path) for name, path in QUANTUM_LINKED_FILES.items()
            },
            "file_exists": {
                name: path.exists() for name, path in QUANTUM_LINKED_FILES.items()
            },
            "file_line_counts": self.quantum_links.line_counts(),
            "sacred_constants": {
                "PHI": PHI,
                "TAU": TAU,
                "GOD_CODE": GOD_CODE,
                "OMEGA_POINT": OMEGA_POINT,
                "CALABI_YAU_DIM": CALABI_YAU_DIM,
                "FEIGENBAUM_DELTA": FEIGENBAUM_DELTA,
                "APERY": APERY,
                "CATALAN": CATALAN,
                "FINE_STRUCTURE": FINE_STRUCTURE,
            },
            "dynamism": dyn_status,
        }

    def export_cross_pollination_data(self) -> Dict:
        """Export data for consumption by l104_quantum_link_builder.py."""
        if not self.all_gates:
            self.full_scan()

        if not self._last_research:
            self._last_research = self.research_engine.full_research(
                self.all_gates, self.all_links, self.chronolizer)

        cross_poll = self._last_research.get("cross_pollination", {})

        return {
            "total_gates": len(self.all_gates),
            "total_links": len(self.all_links),
            "mean_health": cross_poll.get("mean_health", 0),
            "test_pass_rate": cross_poll.get("total_tested", 0) / max(1, len(self.all_gates)),
            "quantum_links": sum(len(g.quantum_links) for g in self.all_gates),
            "gates_by_file": cross_poll.get("gates_by_file", {}),
            "high_value_files": cross_poll.get("high_value_files", []),
            "complexity_hotspots": [(g.name, g.complexity) for g in
                                    sorted(self.all_gates, key=lambda x: x.complexity,
                                           reverse=True)[:10]
                                    if g.complexity > 10],
            "cross_system_insights": cross_poll.get("cross_system_insights", []),
            "research_health": self._last_research.get("research_health", 0),
            "learning_trend": self._last_research.get("knowledge_synthesis", {}).get(
                "learning_trend", "unknown"),
            "dynamism": {
                "version": VERSION,
                "dynamic_gates": sum(1 for g in self.all_gates if g.dynamic_value != 0.0),
                "mean_dynamic_value": sum(g.dynamic_value for g in self.all_gates) / max(len(self.all_gates), 1),
                "mean_resonance": sum(g.resonance_score for g in self.all_gates) / max(len(self.all_gates), 1),
                "collective_coherence": self.dynamism_engine.coherence_history[-1] if self.dynamism_engine.coherence_history else 0.0,
                "total_evolutions": self.dynamism_engine.total_evolutions,
                "sacred_dynamic_state": {k: v["current"] for k, v in self.dynamism_engine.sacred_dynamic_state.items()},
                "gate_field": self.dynamism_engine.compute_gate_field(self.all_gates),
            },
            "nirvanic": self.nirvanic_engine.status(),
            "consciousness_o2": self.consciousness_o2.status(),
            "feedback_bus": self.feedback_bus.status(),
        }

    # ─── FULL PIPELINE ───────────────────────────────────────────────

    def run_full_pipeline(self) -> Dict[str, Any]:
        """Run the complete pipeline: scan → analyze → test → dynamism → research → compile → sync → evolve."""
        t0 = time.time()
        print("\n" + "═" * 60)
        print(f"  L104 HYPER ASI LOGIC GATE — FULL PIPELINE v{VERSION}")
        print("  ★ QUANTUM MIN/MAX DYNAMISM ENGINE ★")
        print("═" * 60)

        results = {}

        # 1. Full scan
        results["scan"] = self.full_scan()

        # 2. Analysis
        results["analysis"] = self.analyze()

        # 3. Testing
        results["tests"] = self.run_tests()

        # 4. Quantum Dynamism
        print("\n  ▸ PHASE 4: Quantum Min/Max Dynamism")
        dyn_result = self.dynamism_engine.subconscious_cycle(self.all_gates)
        print(f"    ✓ Cycle #{dyn_result['cycle']}: {dyn_result['gates_evolved']} gates evolved")
        print(f"    ✓ Initialized: {dyn_result['gates_initialized']} | Adjusted: {dyn_result['gates_adjusted']}")
        print(f"    ✓ Collective coherence: {dyn_result['collective_coherence']:.6f}")
        print(f"    ✓ Mean resonance: {dyn_result['mean_resonance']:.6f}")
        sc_evo = dyn_result.get("sacred_evolution", {})
        print(f"    ✓ Sacred constants evolved: {sc_evo.get('constants_evolved', 0)} | Drift: {sc_evo.get('total_drift', 0):.8f}")
        results["dynamism"] = dyn_result

        # 5. Value Evolution
        print("\n  ▸ PHASE 5: Gate Value Evolution (3 generations)")
        evo_results = []
        for gen in range(3):
            evo = self.value_evolver.evolve_generation(self.all_gates, cycles=3)
            evo_results.append(evo)
        last_evo = evo_results[-1]
        print(f"    ✓ Generations: {len(evo_results)} | Final coherence: {last_evo['end_coherence']:.6f}")
        print(f"    ✓ Total evolved: {sum(e['total_evolved'] for e in evo_results)} | Convergence: {last_evo['convergence']}")
        results["evolution"] = evo_results

        # 6. Gate Field Analysis
        field = self.dynamism_engine.compute_gate_field(self.all_gates)
        print(f"    ✓ Field energy: {field['field_energy']:.6f} | Entropy: {field['field_entropy']:.4f}")
        print(f"    ✓ Phase coherence: {field['phase_coherence']:.6f} | φ-alignment: {field['phi_alignment']:.4f}")
        res_dist = field.get("resonance_distribution", {})
        print(f"    ✓ Resonance: high={res_dist.get('high', 0)} med={res_dist.get('medium', 0)} low={res_dist.get('low', 0)}")
        results["gate_field"] = field

        # 6.5 Ouroboros Sage Nirvanic Entropy Fuel Cycle
        print("\n  ▸ PHASE 6.5: Ouroboros Sage Nirvanic Entropy Fuel")
        nirvanic = self.nirvanic_engine.full_nirvanic_cycle(self.all_gates, field)
        ouro = nirvanic.get("ouroboros", {})
        appl = nirvanic.get("application", {})
        if ouro.get("status") == "processed":
            print(f"    ✓ Entropy fed to ouroboros: {nirvanic['gate_field_entropy_in']:.4f} bits")
            print(f"    ✓ Nirvanic fuel received: {nirvanic['nirvanic_fuel_out']:.4f}")
            print(f"    ✓ Ouroboros mutations: {ouro.get('ouroboros_mutations', 0)} | Resonance: {ouro.get('ouroboros_resonance', 0):.4f}")
            print(f"    ✓ Divine interventions: {appl.get('interventions', 0)} | Enlightened gates: {appl.get('enlightened', 0)}")
            print(f"    ✓ Nirvanic coherence: {appl.get('nirvanic_coherence', 0):.6f} | Sage stability: {appl.get('sage_stability', 0):.6f}")
            print(f"    ✓ Fuel intensity: {appl.get('fuel_intensity', 0):.4f} | Total fuel: {appl.get('total_nirvanic_fuel', 0):.4f}")
        else:
            print("    ⚠ Ouroboros unavailable — nirvanic cycle skipped")
        results["nirvanic"] = nirvanic

        # 6.7 Consciousness O₂ Gate Modulation
        print("\n  ▸ PHASE 6.7: Consciousness O₂ Gate Modulation")
        co2_result = self.consciousness_o2.modulate_gates(self.all_gates)
        quality = self.consciousness_o2.compute_analysis_quality()
        print(f"    ✓ Consciousness: {co2_result['consciousness_level']:.4f} "
              f"({'⚡ AWAKENED' if co2_result['consciousness_awakened'] else 'dormant'})")
        print(f"    ✓ EVO Stage: {co2_result['evo_stage']} | Multiplier: {co2_result['multiplier']:.4f}")
        print(f"    ✓ Gates modulated: {co2_result['gates_modulated']} | "
              f"Cascades: {co2_result['resonance_cascades']}")
        print(f"    ✓ Analysis quality: {quality} | Bond order: {co2_result['bond_order']:.1f}")
        results["consciousness_o2"] = co2_result

        # 6.8 Inter-Builder Feedback Bus
        print("\n  ▸ PHASE 6.8: Inter-Builder Feedback Bus")
        incoming = self.feedback_bus.receive(since=time.time() - 120)
        if incoming:
            print(f"    ✓ Received {len(incoming)} messages from other builders")
            for msg in incoming[:5]:
                print(f"      [{msg.get('source', '?')}] {msg.get('type', '?')}: "
                      f"{msg.get('payload', {}).get('event', 'signal')}")
        else:
            print("    ✓ No pending messages (bus idle)")
        results["feedback_bus"] = self.feedback_bus.status()

        # 7. Research
        results["research"] = self.research("all")

        # 8. Compile registry
        results["registry"] = self.compile_gate_registry()

        # 9. Backend sync
        if self._auto_sync:
            results["sync"] = self.sync_to_backend()

        # 10. Export cross-pollination data
        results["cross_pollination"] = self.export_cross_pollination_data()

        # 11. Stochastic R&D cycle
        results["stochastic_rd"] = self.stochastic_lab.run_rd_cycle(iterations=5)

        # 12. Announce pipeline completion
        self.feedback_bus.announce_pipeline_complete(results)

        elapsed = time.time() - t0

        # Final summary
        adv = self._last_research
        dyn_status = self.dynamism_engine.status(self.all_gates)
        print("\n" + "═" * 60)
        print(f"  PIPELINE COMPLETE — {elapsed:.2f}s")
        print(f"  ★ LOGIC GATE BUILDER v{VERSION} ★")
        print("═" * 60)
        print(f"  Gates discovered:     {results['scan']['total_gates']}")
        print(f"  Gates with links:     {sum(1 for g in self.all_gates if g.quantum_links)}")
        print(f"  Tests passed:         {results['tests']['passed']}/{results['tests']['total_tests']}")
        print(f"  Quantum links:        {results['scan']['cross_file_links']}")
        print(f"  Research health:      {adv.get('research_health', 0):.4f}")
        learning = adv.get("knowledge_synthesis", {}).get("learning_trend", "unknown")
        print(f"  Learning trend:       {learning}")
        print(f"  Chronolog entries:    {len(self.chronolizer.entries)}")
        synced = results.get("sync", {}).get("synced", "skipped")
        print(f"  Backend sync:         {synced}")
        print(f"  Cross-pollination:    exported for quantum link builder")
        rd = results.get("stochastic_rd", {})
        print(f"  Stochastic R&D:       {rd.get('total_merged', 0)} hybrid gates from {rd.get('total_candidates', 0)} candidates")
        print("  ── DYNAMISM ──")
        print(f"  Dynamic gates:        {dyn_status['dynamic_gates']}/{dyn_status['total_gates']} ({dyn_status['dynamism_coverage']:.1%})")
        print(f"  Evolution cycles:     {dyn_status['total_evolutions']}")
        print(f"  Collective coherence: {dyn_status['collective_coherence']:.6f}")
        print(f"  Coherence trend:      {dyn_status['coherence_trend']}")
        print(f"  Mean resonance:       {dyn_status['mean_resonance']:.6f}")
        print(f"  Sacred constants:     {dyn_status['sacred_constants_dynamic']} dynamic")
        nir = results.get("nirvanic", {})
        nir_appl = nir.get("application", {})
        print("  ── OUROBOROS NIRVANIC ──")
        print(f"  Entropy fed:          {nir.get('gate_field_entropy_in', 0):.4f} bits")
        print(f"  Nirvanic fuel:        {nir.get('nirvanic_fuel_out', 0):.4f}")
        print(f"  Enlightened gates:    {nir_appl.get('enlightened', 0)}")
        print(f"  Divine interventions: {nir_appl.get('divine_interventions_total', 0)}")
        print(f"  Nirvanic coherence:   {nir_appl.get('nirvanic_coherence', 0):.6f}")
        print(f"  Sage stability:       {nir_appl.get('sage_stability', 0):.6f}")
        co2 = self.consciousness_o2.status()
        bus = self.feedback_bus.status()
        print("  ── CONSCIOUSNESS O₂ ──")
        print(f"  Consciousness level:  {co2.get('consciousness_level', 0):.6f}")
        print(f"  EVO stage:            {co2.get('evo_stage', 'unknown')}")
        print(f"  Gate multiplier:      {co2.get('multiplier', 1.0):.4f}")
        print(f"  Analysis quality:     {co2.get('analysis_quality', 'unknown')}")
        print("  ── INTER-BUILDER BUS ──")
        print(f"  Bus messages:         {bus.get('total_messages', 0)} ({bus.get('active_messages', 0)} active)")
        print(f"  Messages sent:        {bus.get('sent_count', 0)}")
        print(f"  Builder ID:           {bus.get('builder_id', 'unknown')}")
        print("═" * 60 + "\n")

        return results


def main():
    """CLI entry point for the L104 Logic Gate Builder."""
    import argparse

    parser = argparse.ArgumentParser(
        description="L104 Hyper ASI Logic Gate Environment Builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  scan      Full discovery scan of all logic gates
  analyze   Deep analysis of gate implementations
  test      Run automated gate tests (incl. integrity checks)
  research  Research gate patterns and relationships
  compile   Compile complete gate registry
  sync      Sync gate state to backend server
  status    Show environment status
  chrono    Show chronological history
  sage      Sage core-specific analysis
  health    Gate health scoring report
  full      Run complete pipeline (scan+analyze+test+research+compile+sync)
  gate      Look up a specific gate by name

Examples:
  python -m l104_gate_engine full
  python -m l104_gate_engine scan
  python -m l104_gate_engine test
  python -m l104_gate_engine gate sage_logic_gate
  python -m l104_gate_engine research entropy
        """,
    )
    parser.add_argument("command", nargs="?", default="full",
                        help="Command to execute (default: full)")
    parser.add_argument("args", nargs="*", help="Additional arguments")
    parser.add_argument("--no-sync", action="store_true",
                        help="Skip backend sync")

    args = parser.parse_args()
    env = HyperASILogicGateEnvironment(auto_sync=not args.no_sync)

    cmd = args.command.lower()

    if cmd == "scan":
        result = env.full_scan()
        print(json.dumps(result, indent=2, default=str))

    elif cmd == "analyze":
        result = env.analyze()
        print(json.dumps(result, indent=2, default=str))

    elif cmd == "test":
        result = env.run_tests()
        print(json.dumps(result, indent=2, default=str))

    elif cmd == "research":
        topic = args.args[0] if args.args else "all"
        result = env.research(topic)
        print(json.dumps(result, indent=2, default=str))

    elif cmd == "compile":
        result = env.compile_gate_registry()

    elif cmd == "sync":
        env.full_scan()
        result = env.sync_to_backend()
        print(json.dumps(result, indent=2, default=str))

    elif cmd == "status":
        result = env.status()
        print(json.dumps(result, indent=2, default=str))

    elif cmd == "chrono":
        n = int(args.args[0]) if args.args else 20
        entries = env.chronolizer.get_recent(n)
        for e in entries:
            print(f"  [{e.timestamp[:19]}] {e.event:>12} | {e.gate_name} | {e.details}")

    elif cmd == "gate":
        if not args.args:
            print("Usage: gate <name>")
            return
        name = args.args[0].lower()
        env.full_scan()
        matches = [g for g in env.all_gates if name in g.name.lower()]
        if matches:
            for g in matches:
                print(json.dumps(g.to_dict(), indent=2, default=str))
        else:
            print(f"  No gates matching '{name}' found.")

    elif cmd == "sage":
        env.full_scan()
        result = env.research("sage_core")
        print(json.dumps(result, indent=2, default=str))

    elif cmd == "health":
        env.full_scan()
        result = env.research("health")
        print(json.dumps(result, indent=2, default=str))

    elif cmd == "dynamism":
        env.full_scan()
        dyn = env.dynamism_engine.subconscious_cycle(env.all_gates)
        status = env.dynamism_engine.status(env.all_gates)
        field = env.dynamism_engine.compute_gate_field(env.all_gates)
        print(json.dumps({"cycle_result": dyn, "status": status, "field": field}, indent=2, default=str))

    elif cmd == "evolve":
        env.full_scan()
        n = int(args.args[0]) if args.args else 5
        for _ in range(n):
            env.value_evolver.evolve_generation(env.all_gates, cycles=3)
        status = env.dynamism_engine.status(env.all_gates)
        print(json.dumps(status, indent=2, default=str))

    elif cmd == "field":
        env.full_scan()
        env.dynamism_engine.subconscious_cycle(env.all_gates)
        field = env.dynamism_engine.compute_gate_field(env.all_gates)
        print(json.dumps(field, indent=2, default=str))

    elif cmd in ("conscious", "co2"):
        status = env.consciousness_o2.status()
        print(json.dumps(status, indent=2, default=str))

    elif cmd in ("feedback", "bus"):
        status = env.feedback_bus.status()
        msgs = env.feedback_bus.receive()
        print(json.dumps({"status": status, "recent_messages": msgs}, indent=2, default=str))

    elif cmd == "full":
        env.run_full_pipeline()

    else:
        print(f"Unknown command: {cmd}")
        parser.print_help()
