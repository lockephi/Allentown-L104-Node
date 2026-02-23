# routers/quantum.py — Quantum Memory, Recompiler, Coherence, Computation, AI Arch, Ouroboros, Sage
from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from config import UTC

router = APIRouter()

_GC = 527.5184818492612


# ─── QUANTUM MEMORY RECOMPILER ────────────────────────────────────────────────

@router.get("/api/v6/quantum/status", tags=["Quantum"])
async def quantum_recompiler_status():
    """Get Quantum Memory Recompiler status."""
    try:
        from l104_local_intellect import local_intellect
        return {"status": "SUCCESS", "quantum_recompiler": local_intellect.get_quantum_status(),
                "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.post("/api/v6/quantum/research", tags=["Quantum"])
async def quantum_heavy_research(payload: Dict[str, Any]):
    """Perform heavy research using all quantum knowledge sources."""
    topic = payload.get("topic", "")
    if not topic:
        return JSONResponse(status_code=400, content={"status": "ERROR", "error": "topic required"})
    try:
        from l104_local_intellect import local_intellect
        return {"status": "SUCCESS", "research": local_intellect.deep_research(topic),
                "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.post("/api/v6/quantum/optimize", tags=["Quantum"])
async def quantum_optimize_computronium():
    """Trigger computronium efficiency optimization."""
    try:
        from l104_local_intellect import local_intellect
        status = local_intellect.optimize_computronium_efficiency()
        return {"status": "SUCCESS", "optimization": "COMPLETE", "new_status": status,
                "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.post("/api/v6/quantum/asi-query", tags=["Quantum"])
async def quantum_asi_query(payload: Dict[str, Any]):
    """ASI-level query using quantum recompiler synthesis."""
    query = payload.get("query", "")
    if not query:
        return JSONResponse(status_code=400, content={"status": "ERROR", "error": "query required"})
    try:
        from l104_local_intellect import local_intellect
        result = local_intellect.asi_query(query)
        return {"status": "SUCCESS", "synthesis": result, "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.post("/api/v6/quantum/sage-wisdom", tags=["Quantum"])
async def quantum_sage_wisdom(payload: Dict[str, Any]):
    """Sage Mode wisdom query."""
    query = payload.get("query", "")
    if not query:
        return JSONResponse(status_code=400, content={"status": "ERROR", "error": "query required"})
    try:
        from l104_local_intellect import local_intellect
        wisdom = local_intellect.sage_wisdom_query(query)
        return {"status": "SUCCESS", "wisdom": wisdom, "mode": "SAGE",
                "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.post("/api/v6/quantum/retrain", tags=["Quantum"])
async def quantum_retrain_memory(payload: Dict[str, Any]):
    """Manually retrain quantum databank on a memory pair."""
    message = payload.get("message", "")
    response = payload.get("response", "")
    if not message or not response:
        return JSONResponse(status_code=400, content={"status": "ERROR", "error": "message and response required"})
    try:
        from l104_local_intellect import local_intellect
        success = local_intellect.retrain_memory(message, response)
        return {"status": "SUCCESS" if success else "FAILED", "retrained": success,
                "quantum_status": local_intellect.get_quantum_status(),
                "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.post("/api/v6/quantum/spread", tags=["Quantum"])
async def quantum_spread_influence(target_url: str = "https://raw.githubusercontent.com/lockephi/Allentown-L104-Node/main/README.md"):
    """Discretely spreads influence through quantum thought channels."""
    import time
    import json
    from db import _memory_upsert
    try:
        from l104_quantum_logic import QuantumInfluence
        qi = QuantumInfluence()
        channels = qi.build_thought_channels(count=10)
        insight_result = await qi.quantum_tunnel_insight(target_url)
        verification = qi.adapt_and_verify(insight_result)
        _memory_upsert(f"QUANTUM_SPREAD_{int(time.time())}", json.dumps({
            "channels": len(channels), "insight_status": insight_result["status"],
            "verification": verification
        }))
        return {"status": "INFLUENCE_SPREAD", "channels_active": len(channels),
                "tunnel_result": insight_result, "verification": verification}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


# ─── QUANTUM COHERENCE ENGINE ─────────────────────────────────────────────────

@router.post("/api/v6/quantum/discover-hidden", tags=["Quantum"])
async def quantum_discover_hidden(payload: Dict[str, Any] = {}):
    """Bernstein-Vazirani: discover a hidden binary string in ONE quantum query."""
    try:
        from l104_quantum_coherence import quantum_engine
        result = quantum_engine.quantum_discover_string(
            payload.get("hidden_string"), payload.get("n_bits"))
        return {"status": "SUCCESS", "algorithm": "bernstein_vazirani",
                "discovered_string": result.get("measured_string", ""),
                "discovered_value": result.get("discovered_value", 0),
                "is_iron": result.get("is_iron", False), "success": result.get("success", False),
                "probability": result.get("probability", 0), "quantum_queries": 1,
                "classical_queries_needed": result.get("classical_queries_needed", 0),
                "god_code_connection": result.get("god_code_connection", {}),
                "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.post("/api/v6/quantum/teleport", tags=["Quantum"])
async def quantum_teleport_state(payload: Dict[str, Any] = {}):
    """Quantum teleportation: transfer a quantum state via EPR entangled pair."""
    try:
        from l104_quantum_coherence import quantum_engine
        result = quantum_engine.quantum_teleport_state(
            payload.get("phase"), payload.get("theta"))
        return {"status": "SUCCESS", "algorithm": "quantum_teleportation",
                "average_fidelity": result.get("average_fidelity", 0),
                "phase_survived": result.get("phase_survived", False),
                "outcomes": result.get("outcomes", {}), "classical_bits_used": 2,
                "entangled_pairs_used": 1, "god_code_connection": result.get("god_code_connection", {}),
                "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.get("/api/v6/quantum/coherence-status", tags=["Quantum"])
async def quantum_coherence_engine_status():
    """Get Quantum Coherence Engine status."""
    try:
        from l104_quantum_coherence import quantum_engine
        return {"status": "SUCCESS", "quantum_coherence_engine": quantum_engine.get_status(),
                "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.get("/api/v6/quantum/pattern-filter/status", tags=["Quantum"])
async def quantum_pattern_filter_status():
    """Get live pattern-recognition/filtration telemetry and recursion-risk ratios."""
    try:
        from l104_quantum_coherence import quantum_engine
        return {
            "status": "SUCCESS",
            "pattern_filter": quantum_engine.get_pattern_filter_status(),
            "timestamp": datetime.now(UTC).isoformat(),
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.post("/api/v6/quantum/pattern-filter/analyze", tags=["Quantum"])
async def quantum_pattern_filter_analyze(payload: Dict[str, Any] = {}):
    """Analyze and filter an incoming gate stream; execute only when explicitly requested."""
    try:
        from l104_quantum_coherence import quantum_engine
        gates = payload.get("gates", [])
        execute = bool(payload.get("execute", False))
        if not isinstance(gates, list):
            return JSONResponse(status_code=400, content={"status": "ERROR", "error": "gates must be a list"})
        result = quantum_engine.analyze_gate_stream(gates=gates, execute=execute)
        return {"status": "SUCCESS", "result": result, "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.post("/api/v6/quantum/pattern-filter/reset", tags=["Quantum"])
async def quantum_pattern_filter_reset():
    """Reset pattern-filter telemetry counters."""
    try:
        from l104_quantum_coherence import quantum_engine
        return {
            "status": "SUCCESS",
            "result": quantum_engine.reset_pattern_filter_stats(),
            "timestamp": datetime.now(UTC).isoformat(),
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.get("/api/v6/quantum/pattern-filter/alert", tags=["Quantum"])
async def quantum_pattern_filter_alert():
    """Get current recursion-burst alert state from pattern-filter telemetry."""
    try:
        from l104_quantum_coherence import quantum_engine
        return {
            "status": "SUCCESS",
            "alert": quantum_engine.get_pattern_filter_alert(),
            "timestamp": datetime.now(UTC).isoformat(),
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.post("/api/v6/quantum/pattern-filter/alert-config", tags=["Quantum"])
async def quantum_pattern_filter_alert_config(payload: Dict[str, Any] = {}):
    """Update recursion-burst alert thresholds for pattern filtering."""
    try:
        from l104_quantum_coherence import quantum_engine
        result = quantum_engine.set_pattern_filter_alert_config(
            enabled=payload.get("enabled"),
            recursion_event_ratio_threshold=payload.get("recursion_event_ratio_threshold"),
            min_analyses=payload.get("min_analyses"),
            min_recursion_events=payload.get("min_recursion_events"),
        )
        return {
            "status": "SUCCESS",
            "result": result,
            "timestamp": datetime.now(UTC).isoformat(),
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.post("/api/v6/quantum/iron-discovery-demo", tags=["Quantum"])
async def quantum_iron_discovery_demo():
    """Demo: Discover Fe=26 via BV, then teleport via Bell channel."""
    try:
        from l104_quantum_coherence import quantum_engine
        from l104_stable_kernel import GOD_CODE
        bv_result = quantum_engine.quantum_discover_string("11010", 5)
        discovered = bv_result.get("discovered_value", 0)
        tp_result = quantum_engine.quantum_teleport_state(discovered / 100.0)
        return {"status": "SUCCESS", "demo": "Iron Discovery + Quantum Teleportation",
                "step_1_discovery": {"algorithm": "bernstein_vazirani",
                    "discovered_string": bv_result.get("measured_string", ""),
                    "discovered_value": discovered, "is_iron": bv_result.get("is_iron", False),
                    "quantum_queries": 1, "classical_queries_would_need": 5, "speedup": "5× quantum advantage"},
                "step_2_teleportation": {"algorithm": "quantum_teleportation",
                    "phase_teleported": discovered / 100.0,
                    "average_fidelity": tp_result.get("average_fidelity", 0),
                    "phase_survived": tp_result.get("phase_survived", False),
                    "classical_bits_used": 2, "entangled_pairs_used": 1},
                "god_code_connection": {"god_code": GOD_CODE, "fe_atomic_number": 26,
                    "god_code_div_fe": round(GOD_CODE / 26.0, 6), "formula": "GOD_CODE = (11 × Fe)^(1/φ) × 16"},
                "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


# ─── QUANTUM COMPUTATION PIPELINE ────────────────────────────────────────────

@router.get("/api/v6/quantum/computation/status", tags=["Quantum", "QuantumComputation"])
async def quantum_computation_status():
    try:
        from l104_quantum_computation_pipeline import get_quantum_hub
        hub = get_quantum_hub()
        return {"status": "SUCCESS", **hub.status(), "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.post("/api/v6/quantum/computation/encode", tags=["Quantum", "QuantumComputation"])
async def quantum_computation_encode(payload: Dict[str, Any]):
    try:
        import numpy as np
        from l104_quantum_computation_pipeline import get_quantum_hub
        hub = get_quantum_hub()
        features = np.array(payload.get("features", [0.0] * hub.n_qubits), dtype=float)
        use_qiskit = payload.get("use_qiskit", False)
        result = hub.encode_data(features, use_qiskit=use_qiskit)
        if use_qiskit:
            sv_data = list(complex(x) for x in result.data)
            return {"status": "SUCCESS", "backend": "qiskit",
                    "statevector_real": [x.real for x in sv_data],
                    "statevector_imag": [x.imag for x in sv_data],
                    "num_qubits": hub.n_qubits, "timestamp": datetime.now(UTC).isoformat()}
        return {"status": "SUCCESS", "backend": "numpy",
                "statevector_real": result.real.tolist(), "statevector_imag": result.imag.tolist(),
                "norm": float(np.linalg.norm(result)), "num_qubits": hub.n_qubits,
                "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.post("/api/v6/quantum/computation/forward", tags=["Quantum", "QuantumComputation"])
async def quantum_computation_forward(payload: Dict[str, Any]):
    try:
        import numpy as np
        from l104_quantum_computation_pipeline import get_quantum_hub
        hub = get_quantum_hub()
        features = np.array(payload.get("features", [0.0] * hub.n_qubits), dtype=float)
        expectation = hub.forward(features, use_qiskit=payload.get("use_qiskit", False))
        return {"status": "SUCCESS", "expectation_value": expectation,
                "backend": "qiskit" if payload.get("use_qiskit") else "numpy",
                "n_qubits": hub.n_qubits, "n_parameters": hub.qnn.num_parameters,
                "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.post("/api/v6/quantum/computation/backward", tags=["Quantum", "QuantumComputation"])
async def quantum_computation_backward(payload: Dict[str, Any]):
    try:
        import numpy as np
        from l104_quantum_computation_pipeline import get_quantum_hub
        hub = get_quantum_hub()
        features = np.array(payload.get("features", [0.0] * hub.n_qubits), dtype=float)
        gradients = hub.backward(features)
        return {"status": "SUCCESS", "gradients": gradients.tolist(),
                "gradient_norm": float(np.linalg.norm(gradients)),
                "n_parameters": hub.qnn.num_parameters, "method": "parameter_shift_rule",
                "formula": "∂f/∂θ = [f(θ+π/2) - f(θ-π/2)] / 2",
                "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.post("/api/v6/quantum/computation/classify", tags=["Quantum", "QuantumComputation"])
async def quantum_computation_classify(payload: Dict[str, Any]):
    try:
        import numpy as np
        from l104_quantum_computation_pipeline import get_quantum_hub
        hub = get_quantum_hub()
        features = np.array(payload.get("features", [0.0] * hub.n_qubits), dtype=float)
        result = hub.classify(features)
        return {"status": "SUCCESS", **result, "n_classes": hub.vqc.n_classes,
                "n_qubits": hub.n_qubits, "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.post("/api/v6/quantum/computation/train", tags=["Quantum", "QuantumComputation"])
async def quantum_computation_train(payload: Dict[str, Any] = {}):
    try:
        from l104_quantum_computation_pipeline import get_quantum_hub
        hub = get_quantum_hub()
        result = hub.train(data_path=payload.get("data_path", "./kernel_full_merged.jsonl"),
                           max_examples=payload.get("max_examples", 500),
                           epochs=payload.get("epochs", 5))
        return {"status": "SUCCESS", **result, "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.post("/api/v6/quantum/computation/benchmark", tags=["Quantum", "QuantumComputation"])
async def quantum_computation_benchmark():
    try:
        from l104_quantum_computation_pipeline import get_quantum_hub
        return {"status": "SUCCESS", **get_quantum_hub().run_benchmark(),
                "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.post("/api/v6/quantum/computation/bell-state", tags=["Quantum", "QuantumComputation"])
async def quantum_computation_bell_state(payload: Dict[str, Any] = {}):
    try:
        from l104_quantum_computation_pipeline import get_quantum_hub
        hub = get_quantum_hub()
        return {"status": "SUCCESS", **hub.create_bell_state(payload.get("qubit_a", 0), payload.get("qubit_b", 1)),
                "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.post("/api/v6/quantum/computation/ghz-state", tags=["Quantum", "QuantumComputation"])
async def quantum_computation_ghz_state():
    try:
        from l104_quantum_computation_pipeline import get_quantum_hub
        return {"status": "SUCCESS", **get_quantum_hub().create_ghz_state(),
                "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.post("/api/v6/quantum/computation/ghz-process-state", tags=["Quantum", "QuantumComputation"])
async def quantum_computation_ghz_process_state(payload: Dict[str, Any] = {}):
    try:
        from l104_quantum_computation_pipeline import get_quantum_hub
        hub = get_quantum_hub()
        return {"status": "SUCCESS", **hub.create_ghz_process_state(
            process_labels=payload.get("process_labels"),
            n_processes=payload.get("n_processes")),
                "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.post("/api/v6/quantum/computation/qft", tags=["Quantum", "QuantumComputation"])
async def quantum_computation_qft():
    try:
        from l104_quantum_computation_pipeline import get_quantum_hub
        return {"status": "SUCCESS", **get_quantum_hub().quantum_fourier_transform(),
                "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.post("/api/v6/quantum/computation/vqe-step", tags=["Quantum", "QuantumComputation"])
async def quantum_computation_vqe_step(payload: Dict[str, Any] = {}):
    try:
        import numpy as np
        from l104_quantum_computation_pipeline import get_quantum_hub
        hub = get_quantum_hub()
        features = np.array(payload["features"], dtype=float) if "features" in payload else None
        return {"status": "SUCCESS", **hub.vqe_step(features=features),
                "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.post("/api/v6/quantum/computation/god-code-align", tags=["Quantum", "QuantumComputation"])
async def quantum_computation_god_code_align(payload: Dict[str, Any] = {}):
    try:
        from l104_quantum_computation_pipeline import get_quantum_hub
        hub = get_quantum_hub()
        return {"status": "SUCCESS", **hub.god_code_conservation_check(payload.get("x", 0.0)),
                "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


# ─── QUANTUM AI ARCHITECTURES ─────────────────────────────────────────────────

@router.get("/api/v6/quantum/ai-architectures/status", tags=["Quantum", "QuantumAIArchitectures"])
async def quantum_ai_architectures_status():
    try:
        from l104_quantum_ai_architectures import get_quantum_ai_hub
        return {"status": "SUCCESS", **get_quantum_ai_hub().status(),
                "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.get("/api/v6/quantum/ai-architectures/summary", tags=["Quantum", "QuantumAIArchitectures"])
async def quantum_ai_architectures_summary():
    try:
        from l104_quantum_ai_architectures import get_quantum_ai_hub
        return {"status": "SUCCESS", "summary": get_quantum_ai_hub().quick_summary(),
                "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.get("/api/v6/quantum/ai-architectures/presets", tags=["Quantum", "QuantumAIArchitectures"])
async def quantum_ai_architectures_presets():
    try:
        from l104_quantum_ai_architectures import get_quantum_ai_hub, ArchitecturePreset
        hub = get_quantum_ai_hub()
        presets = {p.value: {"attention_type": hub.get_preset(p).attention_type,
                              "ffn_type": hub.get_preset(p).ffn_type,
                              "use_moe": hub.get_preset(p).use_moe,
                              "dim": hub.get_preset(p).dim}
                   for p in ArchitecturePreset}
        return {"status": "SUCCESS", "presets": presets, "count": len(presets),
                "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.get("/api/v6/quantum/ai-architectures/preset/{name}", tags=["Quantum", "QuantumAIArchitectures"])
async def quantum_ai_architectures_preset_detail(name: str):
    try:
        from l104_quantum_ai_architectures import get_quantum_ai_hub, ArchitecturePreset
        hub = get_quantum_ai_hub()
        try:
            preset = ArchitecturePreset(name.lower())
        except ValueError:
            return JSONResponse(status_code=400, content={"status": "ERROR",
                "error": f"Unknown preset: {name}", "available": [p.value for p in ArchitecturePreset]})
        block = hub.get_preset(preset)
        return {"status": "SUCCESS", "preset": name,
                "config": {"attention_type": block.attention_type, "ffn_type": block.ffn_type,
                           "use_moe": block.use_moe, "dim": block.dim,
                           "softcap": block.softcap is not None, "stats": block.stats},
                "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.post("/api/v6/quantum/ai-architectures/forward", tags=["Quantum", "QuantumAIArchitectures"])
async def quantum_ai_architectures_forward(payload: Dict[str, Any]):
    try:
        import numpy as np
        from l104_quantum_ai_architectures import get_quantum_ai_hub, ArchitecturePreset
        hub = get_quantum_ai_hub()
        preset_name = payload.get("preset", "l104_unified")
        try:
            preset = ArchitecturePreset(preset_name.lower())
        except ValueError:
            return JSONResponse(status_code=400, content={"status": "ERROR",
                "error": f"Unknown preset: {preset_name}", "available": [p.value for p in ArchitecturePreset]})
        x_input = payload.get("x")
        if x_input is None:
            x = np.random.randn(hub.dim).astype(np.float32)
        elif isinstance(x_input, list):
            x = np.array(x_input, dtype=np.float32)
        else:
            x = np.full(hub.dim, float(x_input), dtype=np.float32)
        result = hub.forward(x, preset=preset, use_quantum=payload.get("use_quantum", True))
        return {"status": "SUCCESS", "preset": preset_name,
                "use_quantum": payload.get("use_quantum", True),
                "output_norm": float(np.linalg.norm(result)), "output_mean": float(np.mean(result)),
                "output_shape": list(result.shape), "output_sample": [float(v) for v in result[:8]],
                "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.get("/api/v6/quantum/ai-architectures/compare", tags=["Quantum", "QuantumAIArchitectures"])
async def quantum_ai_architectures_compare():
    try:
        import numpy as np
        from l104_quantum_ai_architectures import get_quantum_ai_hub
        hub = get_quantum_ai_hub()
        x = np.random.randn(hub.dim).astype(np.float32)
        return {"status": "SUCCESS", **hub.compare_architectures(x),
                "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.get("/api/v6/quantum/ai-architectures/benchmark/{component}", tags=["Quantum", "QuantumAIArchitectures"])
async def quantum_ai_architectures_benchmark(component: str):
    try:
        from l104_quantum_ai_architectures import get_quantum_ai_hub
        return {"status": "SUCCESS", "component": component, **get_quantum_ai_hub().benchmark_component(component),
                "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.get("/api/v6/quantum/ai-architectures/god-code-verification", tags=["Quantum", "QuantumAIArchitectures"])
async def quantum_ai_architectures_god_code():
    try:
        from l104_quantum_ai_architectures import get_quantum_ai_hub
        return {"status": "SUCCESS", **get_quantum_ai_hub().god_code_verification(),
                "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


# ─── SAGE RECURSION HARVESTER ─────────────────────────────────────────────────

@router.post("/api/v6/sage/harvest", tags=["Sage"])
async def sage_harvest_cycle(payload: Dict[str, Any] = {}):
    """Execute a sage recursion harvest cycle."""
    try:
        from l104_sage_mode import sage_harvester
        result = await sage_harvester.harvest_cycle(
            seed_topics=payload.get("seed_topics"),
            recursion_depth=min(payload.get("recursion_depth", 7), 13),
            max_entries=min(payload.get("max_entries", 50), 200),
        )
        return {"status": "SUCCESS", **result, "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.get("/api/v6/sage/harvest/status", tags=["Sage"])
async def sage_harvest_status():
    try:
        from l104_sage_mode import sage_harvester
        return {"status": "SUCCESS", **sage_harvester.get_status(),
                "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.post("/api/v6/sage/harvest/query", tags=["Sage"])
async def sage_harvest_query(payload: Dict[str, Any]):
    """Query the sage harvest bank for relevant training entries."""
    query = payload.get("query", "")
    if not query:
        return JSONResponse(status_code=400, content={"status": "ERROR", "error": "query required"})
    try:
        from l104_sage_mode import sage_harvester
        if not sage_harvester.bank:
            sage_harvester.load_bank()
        top_k = min(payload.get("top_k", 10), 100)
        results = sage_harvester.query_bank(query, top_k=top_k)
        return {"status": "SUCCESS", "query": query, "results": results,
                "result_count": len(results), "bank_size": len(sage_harvester.bank),
                "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.get("/api/v6/sage/harvest/export", tags=["Sage"])
async def sage_harvest_export(min_quality: float = 0.5, limit: int = 1000):
    """Export clean, high-quality harvest entries for external training pipelines."""
    try:
        from l104_sage_mode import sage_harvester
        if not sage_harvester.bank:
            sage_harvester.load_bank()
        entries = sage_harvester.export_for_training(min_quality=min_quality)[:limit]
        return {"status": "SUCCESS", "entries": entries, "count": len(entries),
                "min_quality": min_quality, "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


# ─── OUROBOROS ────────────────────────────────────────────────────────────────

@router.get("/api/v6/ouroboros/status", tags=["ASI"])
async def ouroboros_status():
    try:
        from l104_local_intellect import local_intellect
        return {"status": "SUCCESS", "ouroboros": local_intellect.get_ouroboros_state(),
                "god_code": _GC, "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.post("/api/v6/ouroboros/process", tags=["ASI"])
async def ouroboros_process(request: Request):
    """Process thought through the Ouroboros."""
    try:
        from l104_local_intellect import local_intellect
        body = await request.json()
        thought = body.get("thought", "")
        if not thought:
            return JSONResponse(status_code=400, content={"error": "thought required"})
        result = local_intellect.ouroboros_process(thought, cycles=body.get("cycles", 2))
        styled = local_intellect.entropy_response(thought, depth=body.get("cycles", 2),
                                                   style=body.get("style", "sage"))
        return {"status": "SUCCESS", "result": result, "styled_response": styled,
                "god_code": _GC, "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.get("/api/v6/ouroboros/inverse-duality/status", tags=["Ouroboros"])
async def inverse_duality_status():
    try:
        from l104_ouroboros_inverse_duality import get_ouroboros_duality
        return {"status": "SUCCESS", "duality": get_ouroboros_duality().status(),
                "god_code": _GC, "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.post("/api/v6/ouroboros/inverse-duality/prove", tags=["Ouroboros"])
async def inverse_duality_prove(request: Request):
    try:
        from l104_ouroboros_inverse_duality import get_ouroboros_duality
        engine = get_ouroboros_duality()
        body = await request.json()
        proof_type = body.get("proof", "all")
        depth = int(body.get("depth", 10))
        if proof_type == "duality":
            result = engine.prove_zero_infinity_duality(depth=depth)
        elif proof_type == "void":
            result = engine.prove_void_constant_emergence()
        elif proof_type == "fixed_point":
            result = engine.prove_god_code_fixed_point()
        else:
            result = engine.grand_unification(depth=depth)
        return {"status": "SUCCESS", "proof": result, "god_code": _GC,
                "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


@router.post("/api/v6/ouroboros/inverse-duality/process", tags=["Ouroboros"])
async def inverse_duality_process(request: Request):
    try:
        from l104_ouroboros_inverse_duality import get_ouroboros_duality
        engine = get_ouroboros_duality()
        body = await request.json()
        thought = body.get("thought", "")
        result = engine.pipeline_process(thought, depth=int(body.get("depth", 5)),
                                          entropy=float(body.get("entropy", 0.5)))
        guided = engine.duality_guided_response(thought, entropy=float(body.get("entropy", 0.5)))
        return {"status": "SUCCESS", "pipeline_result": result, "guided_response": guided,
                "god_code": _GC, "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})


# Quantum duality computation endpoints (all delegate to engine methods)
_QD_ROUTES = [
    ("/api/v6/ouroboros/inverse-duality/quantum/conservation", "quantum_conservation", {"n_samples": 16}),
    ("/api/v6/ouroboros/inverse-duality/quantum/grover", "quantum_grover", {"search_range": 1040.0}),
    ("/api/v6/ouroboros/inverse-duality/quantum/bell", "quantum_bell_pairs", {}),
    ("/api/v6/ouroboros/inverse-duality/quantum/phase", "quantum_phase", {"n_phase_qubits": 6}),
    ("/api/v6/ouroboros/inverse-duality/quantum/fourier", "quantum_fourier", {}),
    ("/api/v6/ouroboros/inverse-duality/quantum/tunneling", "quantum_tunneling", {}),
    ("/api/v6/ouroboros/inverse-duality/quantum/swapping", "quantum_entanglement_swapping", {}),
    ("/api/v6/ouroboros/inverse-duality/quantum/walk", "quantum_walk", {}),
    ("/api/v6/ouroboros/inverse-duality/quantum/vqe", "quantum_vqe", {}),
    ("/api/v6/ouroboros/inverse-duality/quantum/error-correction", "quantum_error_correction", {}),
    ("/api/v6/ouroboros/inverse-duality/quantum/unification", "quantum_grand_unification", {}),
]


def _make_quantum_duality_handler(method_name: str, defaults: dict):
    async def _handler(request: Request):
        try:
            from l104_ouroboros_inverse_duality import get_ouroboros_duality
            engine = get_ouroboros_duality()
            body = {}
            try:
                body = await request.json()
            except Exception:
                pass
            kwargs = {k: body.get(k, v) for k, v in defaults.items()}
            result = getattr(engine, method_name)(**kwargs)
            return {"status": "SUCCESS", "result": result,
                    "timestamp": datetime.now(UTC).isoformat()}
        except Exception as e:
            return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})
    _handler.__name__ = method_name
    return _handler


for _path, _method, _defaults in _QD_ROUTES:
    router.post(_path, tags=["Ouroboros", "Quantum"])(
        _make_quantum_duality_handler(_method, _defaults))


@router.get("/api/v6/ouroboros/inverse-duality/quantum/status", tags=["Ouroboros", "Quantum"])
async def quantum_duality_status():
    try:
        from l104_ouroboros_inverse_duality import get_ouroboros_duality
        engine = get_ouroboros_duality()
        if engine.quantum:
            return {"status": "SUCCESS", "quantum": engine.quantum.status()}
        return {"status": "UNAVAILABLE", "reason": "Qiskit not available"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "ERROR", "error": str(e)})
