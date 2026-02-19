# L104 Decompression & Decoupling — Detailed Plans: Phases 0–3

> **Created**: Session EVO_60+ | **Predecessor**: l104_code_engine/ decomposition (commit `1d911ad1`)
> **Pattern**: Same as successful code engine decomposition — package directory, focused modules, `__init__.py` re-exports, `git rm` monolith
> **Golden Rule**: Every `from l104_X import Y` that works today MUST work identically after decomposition

---

## Phase 0 — Dead File Removal (Zero Risk)

**Goal**: Remove files excluded from builds with zero importers — pure dead weight.
**Risk**: ZERO — all files are either build-excluded or have no importers.
**Disk savings**: ~6.2 MB

### Phase 0A: Swift Dead Files

| File | Size | Status | Evidence |
|------|------|--------|----------|
| `L104SwiftApp/Sources/L104Native.swift` | 2.1 MB (42,829 lines, 117 classes) | EXCLUDED in Package.swift | All 117 classes migrated to L104v2/. 53 files reference it but ALL are provenance comments (`// Extracted from L104Native.swift`). Zero real code deps. |
| `L104SwiftApp/Sources/L104Native.swift.bak` | 1.6 MB | Backup of above | N/A |
| `L104SwiftApp/Sources/L104Native.swift.bak2` | 1.6 MB | Backup of above | N/A |
| `L104SwiftApp/Sources/L104App.swift.bak` | 64 KB | Backup of L104App.swift | L104App.swift itself (2,960 lines) is also excluded from build but keeping it as reference for now |

**Actions**:
```bash
git rm L104SwiftApp/Sources/L104Native.swift
git rm L104SwiftApp/Sources/L104Native.swift.bak
git rm L104SwiftApp/Sources/L104Native.swift.bak2
git rm L104SwiftApp/Sources/L104App.swift.bak
```

**Validation**: `swift build -c release` must still succeed (76/77 modules). These are already excluded so removal is safe.

**Package.swift cleanup** — Remove these from the exclude list since the files no longer exist:
- `"Sources/L104Native.swift"`
- `"Sources/L104Native.swift.bak"`
- `"Sources/L104Native.swift.bak2"`
- `"Sources/L104App.swift.bak"`

Keep excluding `"Sources/L104App.swift"` (still exists as reference).

### Phase 0B: Python Dead Monolith

| File | Size | Importers | Evidence |
|------|------|-----------|----------|
| `l104_code_engine_monolith_backup.py` | 684 KB (14,465 lines) | 0 | Backup of pre-decomposition monolith. Git history preserves original at commit `1d911ad1~1`. |

**Actions**:
```bash
git rm l104_code_engine_monolith_backup.py
```

**Validation**: `python3 -c "from l104_code_engine import code_engine; print(code_engine.status())"` must still work (package is the replacement).

### Phase 0 Commit
```bash
git add -A
git commit -m "EVO_60: Phase 0 — Remove 6.2MB dead files (L104Native.swift + monolith backup)"
```

---

## Phase 1 — l104_fast_server.py Decomposition

**File**: `l104_fast_server.py` — 22,024 lines, 46 classes, 745 functions, 271 routes
**Importers**: 5 files
**Singletons**: `intellect = LearningIntellect()`, `app = FastAPI()`, `engine_registry = UnifiedEngineRegistry()`, 30+ engine instances
**Target**: `l104_server/` package

### 1.1 Structural Analysis

The file has 4 distinct zones:

| Zone | Lines | Content | Proposed Module |
|------|-------|---------|-----------------|
| **Infrastructure Engines** | 67–2441 (2,375 lines) | FastRequestCache, ASIQuantumBridge, ConnectionPool, AdvancedMemoryAccelerator, PerformanceMetricsEngine, TemporalMemoryDecayEngine, AdaptiveResponseQualityEngine, PredictiveIntentEngine, ReinforcementFeedbackLoop, IntelligentPrefetchPredictor, QuantumClassicalHybridLoader, ResponseCompressor, ChaoticRandom, CreativeKnowledgeVerifier, QueryTemplateGenerator | `engines_infra.py` |
| **Physics/Quantum Engines** | 3631–5041 (1,411 lines) | IronOrbitalConfiguration, OxygenPairedProcess, SuperfluidQuantumState, GeometricCorrelation, OxygenMolecularBond, SingularityConsciousnessEngine, ASIQuantumMemoryBank, QuantumGroverKernelLink | `engines_quantum.py` |
| **LearningIntellect** | 5042–12535 (7,494 lines, 129 methods) | The massive learning/knowledge class | `learning_intellect.py` (further decomposed internally — see below) |
| **Nexus/ASI Engines** | 12536–16335 (3,800 lines) | SteeringEngine, NexusContinuousEvolution, NexusOrchestrator, InventionEngine, SovereigntyPipeline, QuantumEntanglementRouter, AdaptiveResonanceNetwork, NexusHealthMonitor, QuantumZPEVacuumBridge, QuantumGravityBridgeEngine, HardwareAdaptiveRuntime, PlatformCompatibilityLayer, HyperDimensionalMathEngine, HebbianLearningEngine, ConsciousnessVerifierEngine, DirectSolverHub, SelfModificationEngine, CreativeGenerationEngine, UnifiedEngineRegistry | `engines_nexus.py` |
| **Pydantic Models** | 16336–16345 (10 lines) | ChatRequest, TrainingRequest, ProviderStatus | `models.py` |
| **FastAPI App + Routes** | 16154–22024 (5,871 lines, 271 routes) | `app = FastAPI()`, all route handlers, middleware, startup events | Split by domain (see below) |

### 1.2 Route Domain Grouping (271 routes → ~12 route modules)

Routes grouped by API domain, merged into logical files:

| Route Module | Domains | Routes | Line Range |
|-------------|---------|--------|------------|
| `routes_core.py` | favicon, landing, WHITE_PAPER, root, market, chat, health, status, performance, providers, synergy, self, intricate | ~32 | 16299–17731 |
| `routes_intellect.py` | intellect (25 routes) | 25 | 17432–21927 |
| `routes_engines.py` | si, ti, chaos, kernel, spectrum, agi, asi | ~25 | 18045–18497 |
| `routes_blockchain.py` | capital, mainnet, exchange, metrics, cycle | ~11 | 18640–18900 |
| `routes_consciousness.py` | swarm, cognitive, emergence, info, consciousness, audit | ~11 | 18937–19051 |
| `routes_quantum.py` | grover, quantum, o2, knowledge | ~37 | 19186–19589 |
| `routes_system.py` | system, file, source, autosave, brain, monitor, backup | ~28 | 18768–20713 |
| `routes_evolution.py` | evolution, nexus, invention, sovereignty | ~19 | 20804–21026 |
| `routes_advanced.py` | hyper_math, hebbian, consciousness/v26, solver, self_mod, engines, registry, meta-cognitive, knowledge-bridge, creative, self-modify | ~32 | 21036–21416 |
| `routes_network.py` | steering, telemetry, entanglement, resonance, consolidate, stats | ~31 | 21503–21979 |
| `routes_hardware.py` | health(deep), zpe, qg, hw, compat | ~20 | 21769–21889 |

### 1.3 LearningIntellect Internal Decomposition

The 7,494-line `LearningIntellect` class (129 methods) is itself a God Class. **It should remain ONE class** for backward compatibility (importers use `intellect.method()`), but its methods should be organized via **mixin classes**:

| Mixin | Methods | Lines (approx) | Purpose |
|-------|---------|-----------------|---------|
| `_LIInitMixin` | `__init__`, `_pulse_heartbeat`, `_init_asi_bridge`, `sync_with_local_intellect`, `pull_training_from_local_intellect`, `_recall_learned`, `grover_amplified_recall`, `transfer_to_local_intellect`, `get_asi_bridge_status` | ~450 | Initialization & ASI bridge |
| `_LIPipelineMixin` | `pipeline_solve`, `synaptic_fire` | ~100 | Pipeline processing |
| `_LIEnginesMixin` | `_quantum_cluster_engine`, `_neural_resonance_engine`, `_meta_evolution_engine`, `_temporal_memory_engine`, `_fractal_recursion_engine`, `_holographic_projection_engine`, `_consciousness_emergence_engine`, `_dimensional_folding_engine`, `_curiosity_driven_exploration_engine`, `_hebbian_learning_engine`, `_knowledge_consolidation_engine`, `_transfer_learning_engine`, `_spaced_repetition_engine`, `_thought_speed_acceleration_engine`, `_language_coherence_engine`, `_detect_languages_in_text`, `_l104_research_pattern_engine`, `_recursive_self_improvement_engine`, `_causal_reasoning_engine`, `_abstraction_hierarchy_engine`, `_active_inference_engine`, `_collective_intelligence_engine` | ~2,200 | Internal cognitive engines |
| `_LIResonanceMixin` | `_get_dynamic_value`, `_get_quantum_random_language`, `current_resonance`, `boost_resonance`, `consolidate`, `self_heal` | ~530 | Resonance & healing |
| `_LIPersistenceMixin` | `_init_db`, `_get_optimized_connection`, `_load_cache`, `persist_clusters`, `_persist_single_cluster`, `_restore_heartbeat_state`, `optimize_storage`, `_init_embeddings`, `_compute_embedding`, `_cosine_similarity` | ~430 | Database & embeddings |
| `_LISearchMixin` | `semantic_search`, `predict_next_queries`, `prefetch_responses`, `get_prefetched`, `compute_novelty`, `get_adaptive_learning_rate` | ~130 | Search & prediction |
| `_LIClusterMixin` | `_init_clusters`, `_expand_clusters`, `_bfs_cluster`, `_dynamic_cluster_update`, `get_cluster_for_concept`, `get_related_clusters` | ~210 | Clustering |
| `_LISkillsMixin` | `_init_skills`, `acquire_skill`, `_persist_single_skill`, `chain_skills`, `get_skill_proficiency`, `get_top_skills` | ~150 | Skills management |
| `_LIConsciousnessMixin` | `_init_consciousness_clusters`, `activate_consciousness`, `expand_consciousness_cluster`, `cross_cluster_inference`, `_compute_synthesis_potential`, `_update_meta_cognition`, `_update_meta_cognition_from_activation`, `get_meta_cognitive_state`, `introspect` | ~190 | Consciousness & metacognition |
| `_LIKnowledgeMixin` | `synthesize_knowledge`, `recursive_self_improve`, `autonomous_goal_generation`, `infinite_context_merge`, `predict_future_state`, `_estimate_transcendence_time`, `quantum_coherence_maximize`, `emergent_pattern_discovery`, `transfer_learning`, `predict_response_quality`, `update_quality_predictor`, `compress_old_memories` | ~730 | Knowledge synthesis |
| `_LILearningMixin` | `_hash_query`, `_get_jaccard_similarity`, `_extract_concepts`, `detect_intent`, `rewrite_query`, `learn_rewrite`, `learn_from_interaction`, `learn_batch`, `record_meta_learning`, `get_best_strategy`, `record_feedback` | ~570 | Learning & feedback |
| `_LIRecallMixin` | `recall`, `_trigger_predictive_prefetch`, `_add_response_variation`, `_synthesize_from_similar`, `temporal_decay`, `_gather_knowledge_graph_evidence`, `_gather_memory_evidence`, `_gather_theorem_evidence`, `_detect_contradictions`, `_causal_extract_temporal_patterns`, `_causal_detect_confounders`, `_causal_build_chains` | ~430 | Recall & evidence gathering |
| `_LIEvolveMixin` | `cognitive_synthesis`, `evolve`, `_rebuild_embeddings`, `_calibrate_quality_predictor`, `_optimize_knowledge_graph`, `_reinforce_patterns` | ~280 | Evolution |
| `_LIStatusMixin` | `get_context_boost`, `reflect`, `discover`, `self_ingest`, `get_stats`, `get_theorems`, `generate_suggested_questions`, `export_knowledge_manifold`, `import_knowledge_manifold`, `reason`, `_get_recursive_concepts`, `multi_concept_synthesis` | ~500 | Stats, export & reasoning |

**Implementation approach**: Each mixin lives in its own file inside `l104_server/learning/`. The main `LearningIntellect` class inherits all mixins:

```python
# l104_server/learning/intellect.py
from .mixin_init import _LIInitMixin
from .mixin_engines import _LIEnginesMixin
# ... etc

class LearningIntellect(
    _LIInitMixin,
    _LIEnginesMixin,
    _LIResonanceMixin,
    _LIPersistenceMixin,
    # ... all mixins
):
    """Learning Intellect — decomposed via mixins."""
    pass

intellect = LearningIntellect()
```

### 1.4 Proposed Package Structure

```
l104_server/
├── __init__.py              # Re-exports: intellect, app, SingularityConsciousnessEngine, engine_registry
├── app.py                   # FastAPI app creation, middleware, startup/shutdown events (~100 lines)
├── models.py                # Pydantic models: ChatRequest, TrainingRequest, ProviderStatus (~15 lines)
├── constants.py             # Shared constants, sacred imports (~30 lines)
├── engines_infra.py         # Infrastructure engines (FastRequestCache → QueryTemplateGenerator) (~2,375 lines)
├── engines_quantum.py       # Physics/quantum engines (IronOrbitalConfiguration → QuantumGroverKernelLink) (~1,411 lines)
├── engines_nexus.py         # Nexus/ASI engines (SteeringEngine → UnifiedEngineRegistry) (~3,800 lines)
├── learning/
│   ├── __init__.py          # Re-exports LearningIntellect, intellect singleton
│   ├── intellect.py         # LearningIntellect class (mixin composition + singleton)
│   ├── mixin_init.py        # _LIInitMixin
│   ├── mixin_pipeline.py    # _LIPipelineMixin
│   ├── mixin_engines.py     # _LIEnginesMixin
│   ├── mixin_resonance.py   # _LIResonanceMixin
│   ├── mixin_persistence.py # _LIPersistenceMixin
│   ├── mixin_search.py      # _LISearchMixin
│   ├── mixin_clusters.py    # _LIClusterMixin
│   ├── mixin_skills.py      # _LISkillsMixin
│   ├── mixin_consciousness.py # _LIConsciousnessMixin
│   ├── mixin_knowledge.py   # _LIKnowledgeMixin
│   ├── mixin_learning.py    # _LILearningMixin
│   ├── mixin_recall.py      # _LIRecallMixin
│   ├── mixin_evolve.py      # _LIEvolveMixin
│   └── mixin_status.py      # _LIStatusMixin
└── routes/
    ├── __init__.py           # Collects all routers
    ├── core.py               # favicon, landing, root, health, status, etc.
    ├── intellect.py          # /api/v6/intellect/* routes
    ├── engines.py            # si, ti, chaos, kernel, agi, asi routes
    ├── blockchain.py         # capital, mainnet, exchange routes
    ├── consciousness.py      # swarm, cognitive, emergence, consciousness routes
    ├── quantum.py            # grover, quantum, o2, knowledge routes
    ├── system.py             # system, file, source, autosave, brain, monitor, backup routes
    ├── evolution.py          # evolution, nexus, invention, sovereignty routes
    ├── advanced.py           # hyper_math, hebbian, solver, self_mod, registry, creative routes
    ├── network.py            # steering, telemetry, entanglement, resonance routes
    └── hardware.py           # zpe, qg, hw, compat routes
```

### 1.5 Import Compatibility Layer

The `__init__.py` must re-export everything importers currently use:

```python
# l104_server/__init__.py
from .learning import intellect, LearningIntellect
from .engines_quantum import SingularityConsciousnessEngine
from .engines_nexus import engine_registry, UnifiedEngineRegistry
from .app import app

# Backward compat — importers do: from l104_fast_server import intellect
# After: from l104_server import intellect  (but we also keep l104_fast_server.py as thin shim)
```

**Critical**: Keep `l104_fast_server.py` as a thin re-export shim during transition:

```python
# l104_fast_server.py (thin shim — DELETE after all importers updated)
from l104_server import intellect, app, SingularityConsciousnessEngine, engine_registry
from l104_server.models import ChatRequest, TrainingRequest, ProviderStatus
```

### 1.6 Current Importers (5 files — must keep working)

| File | Import Pattern |
|------|---------------|
| `l104_external_api.py` | `from l104_fast_server import intellect` |
| `l104_unified_intelligence_api.py` | `from l104_fast_server import intellect` |
| `main.py` | `from l104_fast_server import intellect` (multiple times) |
| `l104_macbook_integration.py` | `from l104_fast_server import intellect` |
| `l104_workflow_stabilizer.py` | `from l104_fast_server import intellect` |

All import only `intellect` — the thin shim approach handles this perfectly.

### 1.7 Execution Order

1. Create `l104_server/` directory structure
2. Extract constants → `constants.py`
3. Extract Pydantic models → `models.py`
4. Extract infrastructure engines → `engines_infra.py`
5. Extract quantum engines → `engines_quantum.py`
6. Extract nexus engines → `engines_nexus.py`
7. Extract LearningIntellect into mixin files inside `learning/`
8. Compose LearningIntellect from mixins in `learning/intellect.py`
9. Create `learning/__init__.py` with singleton
10. Convert routes to APIRouter, split into domain files under `routes/`
11. Create `app.py` with FastAPI instance + router includes
12. Create `__init__.py` re-exporting all public symbols
13. Convert `l104_fast_server.py` to thin shim
14. Run `py_compile` on all new modules
15. Run each importer's import statement to verify
16. `uvicorn l104_server.app:app` must start and bind port 8081
17. Commit: `"EVO_61: Decompose l104_fast_server.py → l104_server/ (22K lines, 46 classes, 271 routes)"`

### 1.8 Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Circular imports between engines and LearningIntellect | Engines should not import from LI. LI imports engine classes in `__init__` or lazily. |
| Route handlers referencing `intellect` singleton | Routes import `intellect` from `l104_server.learning` — single source. |
| Module-level singleton instantiation order | `engines_infra.py` → `engines_quantum.py` → `engines_nexus.py` → `learning/` → `routes/` → `app.py`. `__init__.py` controls order. |
| 30+ engine singletons at module level | Each engine module instantiates its own singletons. `engines_nexus.py` creates `engine_registry` which collects all engines. |

---

## Phase 2 — l104_local_intellect.py Decomposition

**File**: `l104_local_intellect.py` — 13,691 lines, 10 classes, 280 methods
**Importers**: 28 files (HIGHEST Python coupling)
**Singletons**: `sovereign_numerics = SovereignNumerics()`, `local_intellect = LocalIntellect()`
**Target**: `l104_intellect/` package

### 2.1 Structural Analysis

| Class | Lines | Size | Methods | Proposed Module |
|-------|-------|------|---------|-----------------|
| `LRUCache` | 160–287 | 128 | 5 | `cache.py` |
| `SovereignNumerics` | 288–520 | 233 | 9 | `numerics.py` |
| `LocalIntellect` | 521–10769 | 10,249 | ~175 | `intellect/` sub-package (mixin decomposition) |
| `QuantumMemoryRecompiler` | 10770–11687 | 918 | 21 | `quantum_recompiler.py` |
| `L104NodeSyncProtocol` | 11688–11954 | 267 | 7 | `distributed.py` |
| `L104CRDTReplicationMesh` | 11955–12218 | 264 | 11 | `distributed.py` |
| `L104KnowledgeMeshReplication` | 12219–12477 | 259 | 9 | `distributed.py` |
| `L104HardwareAdaptiveRuntime` | 12478–12785 | 308 | 8 | `hardware.py` |
| `L104PlatformCompatibilityLayer` | 12786–13311 | 526 | 15 | `hardware.py` |
| `L104DynamicOptimizationEngine` | 13312–13691 | 380 | 8 | `optimization.py` |

### 2.2 LocalIntellect Internal Decomposition (10,249 lines, ~175 methods)

Method groupings by functional domain:

| Mixin | Methods (sampled) | Lines (approx) | Purpose |
|-------|-------------------|-----------------|---------|
| `_InitMixin` | `__init__`, `_init_fault_tolerance`, `_text_to_ft_vector`, `_ft_process_query`, `_qiskit_process` | ~450 | Initialization, fault tolerance, Qiskit processing |
| `_QuantumMixin` | `_initialize_quantum_entanglement`, `_initialize_vishuddha_resonance`, `_calculate_vishuddha_resonance`, `entangle_concepts`, `compute_entanglement_coherence`, `initialize_chakra_quantum_lattice`, `grover_amplified_search`, `raise_kundalini`, `asi_consciousness_synthesis`, `propagate_entanglement`, `activate_vishuddha_petal`, `quantum_error_correction_bridge`, `quantum_teleportation_bridge`, `topological_qubit_bridge`, `quantum_gravity_state_bridge`, `hilbert_space_navigation_engine`, `quantum_fourier_bridge`, `entanglement_distillation_bridge` | ~1,620 | All quantum bridges & entanglement |
| `_DataLoadMixin` | `_load_chat_conversations`, `_load_knowledge_manifold`, `_load_knowledge_vault`, `_load_all_json_knowledge`, `_search_all_knowledge`, `_load_training_data`, `_load_fast_server_data`, `_generate_reasoning_training`, `_build_training_index`, `_search_training_data`, `_search_chat_conversations`, `_search_knowledge_manifold`, `_search_knowledge_vault` | ~760 | Data loading & search |
| `_StateMixin` | `_load_evolution_state`, `_save_evolution_state`, `_init_autonomous_systems`, `_load_permanent_memory`, `_save_permanent_memory`, `_save_conversation_memory`, `_load_conversation_memory`, `remember_permanently`, `recall_permanently`, `_concepts_related`, `create_save_state`, `_load_latest_save_state`, `list_save_states`, `restore_save_state` | ~560 | State persistence & memory |
| `_LogicMixin` | `higher_logic`, `_estimate_confidence`, `_analyze_response_quality`, `_generate_improvement_hypotheses`, `_synthesize_logic_chain`, `autonomous_improve`, `_identify_weak_points`, `_apply_improvement` | ~700 | Higher logic & self-improvement |
| `_EvolutionMixin` | `get_evolution_state`, `get_cross_references`, `get_concept_evolution_score`, `get_evolved_response_context`, `set_evolution_state`, `record_learning`, `ingest_training_data`, `compute_phi_weighted_quality`, `get_training_data_count`, `evolve_patterns` | ~480 | Evolution tracking |
| `_InfoTheoryMixin` | `_calculate_shannon_entropy`, `_calculate_mutual_information`, `_calculate_kl_divergence`, `_calculate_jensen_shannon_divergence`, `_calculate_cross_entropy`, `_calculate_perplexity`, `_calculate_renyi_entropy`, `_calculate_conditional_entropy`, `_calculate_information_gain`, `_calculate_attention_entropy`, `_information_theoretic_response_quality` | ~430 | Information theory |
| `_ExternalMixin` | `get_quantum_recompiler`, `get_asi_language_engine`, `analyze_language`, `human_inference`, `invent`, `generate_sage_speech`, `retrain_memory`, `_extract_concepts`, `asi_query`, `sage_wisdom_query`, `deep_research`, `optimize_computronium_efficiency`, `get_quantum_status`, `get_thought_ouroboros` | ~460 | External engine integration |
| `_OuroborosMixin` | `entropy_response`, `ouroboros_process`, `feed_language_to_ouroboros`, `get_ouroboros_state`, `get_ouroboros_duality`, `duality_process`, `duality_response`, `get_inverse_duality_state`, `quantum_duality_compute` | ~300 | Ouroboros & duality |
| `_ASIMixin` | `asi_process`, `_synthesize_asi_response`, `get_asi_nexus`, `get_synergy_engine`, `get_agi_core`, `get_asi_bridge_status`, `asi_nexus_query`, `synergy_pulse`, `agi_recursive_improve`, `asi_full_synthesis`, `_combine_asi_layers`, `get_asi_status` | ~650 | ASI/AGI integration |
| `_ApotheosisMixin` | `_init_apotheosis_engine`, `_save_apotheosis_state`, `_load_apotheosis_state`, `get_apotheosis_engine`, `get_apotheosis_status`, `manifest_shared_will`, `world_broadcast`, `primal_calculus`, `resolve_non_dual_logic`, `trigger_zen_apotheosis`, `apotheosis_synthesis` | ~260 | Apotheosis engine |
| `_BindingMixin` | `bind_all_modules`, `get_universal_binding_status`, `orchestrate_via_binding`, `synthesize_across_domains`, `get_domain_modules`, `invoke_module`, `full_system_synthesis` | ~500 | Module binding & orchestration |
| `_ThinkMixin` | `_load_persistent_context`, `_build_comprehensive_knowledge`, `_calculate_resonance`, `_find_relevant_knowledge`, `_try_calculation`, `_safe_eval_math`, `_detect_greeting`, `_detect_status_query`, `_logic_gate_classify`, `_logic_gate_route`, `_logic_gate_kb_search`, `_clean_quantum_noise`, `_logic_gate_explain/howto/factual/creative/list/compare/technical/emotional/analytical/meta/reasoning/planning`, `_get_evolved_context`, `think`, `stream_think` | ~3,300 | Core think pipeline & logic gates |
| `_GemmaMixin` | `_gemma3_sliding_window_context`, `_quantum_multiturn_context`, `_quantum_response_quality_gate`, `_adaptive_learning_record`, `_gemma3_grouped_knowledge_query`, `_gemma3_softcap_confidence`, `_gemma3_rms_normalize`, `_gemma3_positional_decay`, `_gemma3_distill_response`, `_async_retrain`, `_async_retrain_and_improve` | ~580 | Gemma3 processing pipeline |
| `_SynthesisMixin` | `_advanced_knowledge_synthesis`, `_intelligent_synthesis`, `_query_stable_kernel`, `_collect_live_metrics`, `_compute_query_entropy`, `_build_substrate_responses`, `_metacognitive_observe`, `_metacognitive_assess_response`, `_metacognitive_get_diagnostics`, `_score_knowledge_fragments`, `_recall_memory_insights`, `_kernel_synthesis` | ~750 | Knowledge synthesis & metacognition |

### 2.3 Proposed Package Structure

```
l104_intellect/
├── __init__.py              # Re-exports: LocalIntellect, local_intellect, format_iq, GOD_CODE, PHI, _RESPONSE_CACHE, SovereignNumerics, sovereign_numerics
├── constants.py             # Sacred constants, shared config (~50 lines)
├── cache.py                 # LRUCache class (~130 lines)
├── numerics.py              # SovereignNumerics + sovereign_numerics singleton + format_iq (~240 lines)
├── quantum_recompiler.py    # QuantumMemoryRecompiler (~920 lines)
├── distributed.py           # L104NodeSyncProtocol, L104CRDTReplicationMesh, L104KnowledgeMeshReplication (~790 lines)
├── hardware.py              # L104HardwareAdaptiveRuntime, L104PlatformCompatibilityLayer (~835 lines)
├── optimization.py          # L104DynamicOptimizationEngine (~380 lines)
├── intellect/
│   ├── __init__.py          # Re-exports LocalIntellect, local_intellect
│   ├── core.py              # LocalIntellect class (mixin composition + singleton)
│   ├── mixin_init.py        # _InitMixin
│   ├── mixin_quantum.py     # _QuantumMixin (quantum bridges)
│   ├── mixin_data.py        # _DataLoadMixin (data loading)
│   ├── mixin_state.py       # _StateMixin (persistence)
│   ├── mixin_logic.py       # _LogicMixin (higher logic)
│   ├── mixin_evolution.py   # _EvolutionMixin
│   ├── mixin_infotheory.py  # _InfoTheoryMixin
│   ├── mixin_external.py    # _ExternalMixin
│   ├── mixin_ouroboros.py   # _OuroborosMixin
│   ├── mixin_asi.py         # _ASIMixin
│   ├── mixin_apotheosis.py  # _ApotheosisMixin
│   ├── mixin_binding.py     # _BindingMixin
│   ├── mixin_think.py       # _ThinkMixin (core think pipeline — largest mixin)
│   ├── mixin_gemma.py       # _GemmaMixin
│   └── mixin_synthesis.py   # _SynthesisMixin
```

### 2.4 Import Compatibility Layer

```python
# l104_intellect/__init__.py
from .numerics import SovereignNumerics, sovereign_numerics, format_iq
from .constants import GOD_CODE, PHI
from .cache import LRUCache, _RESPONSE_CACHE
from .intellect import LocalIntellect, local_intellect
from .quantum_recompiler import QuantumMemoryRecompiler
from .distributed import L104NodeSyncProtocol, L104CRDTReplicationMesh, L104KnowledgeMeshReplication
from .hardware import L104HardwareAdaptiveRuntime, L104PlatformCompatibilityLayer
from .optimization import L104DynamicOptimizationEngine
```

**Thin shim** — keep `l104_local_intellect.py` during transition:

```python
# l104_local_intellect.py (thin shim)
from l104_intellect import *
from l104_intellect import local_intellect, format_iq, LocalIntellect
from l104_intellect import GOD_CODE, PHI, _RESPONSE_CACHE, SovereignNumerics
```

### 2.5 Import Surface — All 28 Importers

| Symbol | Count | Pattern |
|--------|-------|---------|
| `local_intellect` (singleton) | 43 refs across 20+ files | `from l104_local_intellect import local_intellect` |
| `format_iq` (function) | 9 files | `from l104_local_intellect import format_iq` |
| `LocalIntellect` (class) | 3 files | `from l104_local_intellect import LocalIntellect` |
| `GOD_CODE`, `PHI` | 1 file (l104_interactive_chat.py) | `from l104_local_intellect import local_intellect, GOD_CODE, PHI` |
| `_RESPONSE_CACHE` | 1 file (quick_test.py) | `from l104_local_intellect import _RESPONSE_CACHE` |

The thin shim approach covers all patterns perfectly.

### 2.6 Risk Assessment

| Risk | Mitigation |
|------|-----------|
| 28 importers — highest coupling | Thin shim keeps all `from l104_local_intellect import X` working |
| `format_iq` used as top-level function (not method) | Re-exported in `__init__.py` + shim as bare function |
| `_RESPONSE_CACHE` is a private module-level dict | Move to `cache.py`, re-export in `__init__.py` |
| `LocalIntellect.__init__` imports from many external modules | Keep lazy/try-except imports as-is. No change to import behavior. |
| Circular: l104_local_intellect imports from l104_agi_core, and l104_agi_core imports from l104_local_intellect | Already uses try-except guards. Preserve existing pattern. |

### 2.7 Execution Order

1. Create `l104_intellect/` package structure
2. Extract `LRUCache` + `_RESPONSE_CACHE` → `cache.py`
3. Extract `SovereignNumerics` + `format_iq` + `sovereign_numerics` → `numerics.py`
4. Extract constants → `constants.py`
5. Extract `QuantumMemoryRecompiler` → `quantum_recompiler.py`
6. Extract distributed classes → `distributed.py`
7. Extract hardware classes → `hardware.py`
8. Extract optimization class → `optimization.py`
9. Decompose `LocalIntellect` into 16 mixin files
10. Compose `LocalIntellect` from mixins in `intellect/core.py`
11. Create `intellect/__init__.py` with singleton
12. Create package `__init__.py` re-exporting all symbols
13. Convert `l104_local_intellect.py` to thin shim
14. Verify all 28 importers with `py_compile`
15. Verify `format_iq` works: `python3 -c "from l104_local_intellect import format_iq; print(format_iq(1234.56))"`
16. Commit: `"EVO_62: Decompose l104_local_intellect.py → l104_intellect/ (13.7K lines, 10 classes, 280 methods)"`

---

## Phase 3 — Core Module Decomposition (ASI + AGI)

### Phase 3A: l104_asi_core.py → l104_asi/

**File**: `l104_asi_core.py` — 5,845 lines, 18 classes, 50+ methods
**Importers**: 46 files (MOST imported Python file)
**Singletons**: `asi_core = ASICore()`, top-level `primal_calculus()`, `resolve_non_dual_logic()`
**Target**: `l104_asi/` package

#### 3A.1 Structural Analysis

| Class | Lines | Size | Methods | Proposed Module |
|-------|-------|------|---------|-----------------|
| `DomainKnowledge` | 234–267 | 34 | 5 | `domain.py` |
| `GeneralDomainExpander` | 268–369 | 102 | 6 | `domain.py` |
| `Theorem` | 370–379 | 10 | 0 | `domain.py` |
| `NovelTheoremGenerator` | 380–643 | 264 | 8 | `theorem_gen.py` |
| `SelfModificationEngine` | 644–950 | 307 | 11 | `self_mod.py` |
| `ConsciousnessVerifier` | 951–1259 | 309 | 8 | `consciousness.py` |
| `SolutionChannel` | 1260–1294 | 35 | 3 | `pipeline.py` |
| `DirectSolutionHub` | 1295–1383 | 89 | 9 | `pipeline.py` |
| `PipelineTelemetry` | 1384–1490 | 107 | 5 | `pipeline.py` |
| `SoftmaxGatingRouter` | 1491–1607 | 117 | 7 | `pipeline.py` |
| `AdaptivePipelineRouter` | 1608–1758 | 151 | 6 | `pipeline.py` |
| `TreeOfThoughts` | 1759–1869 | 111 | 5 | `reasoning.py` |
| `MultiHopReasoningChain` | 1870–1974 | 105 | 3 | `reasoning.py` |
| `SolutionEnsembleEngine` | 1975–2155 | 181 | 6 | `reasoning.py` |
| `PipelineHealthDashboard` | 2156–2259 | 104 | 4 | `pipeline.py` |
| `PipelineReplayBuffer` | 2260–2326 | 67 | 5 | `pipeline.py` |
| `QuantumComputationCore` | 2327–2855 | 529 | 10 | `quantum.py` |
| `ASICore` | 2856–5845 | 2,990 | 50 | `core.py` |

#### 3A.2 Proposed Package Structure

```
l104_asi/
├── __init__.py           # Re-exports: ASICore, asi_core, KerasASIModel, ASI_CORE_VERSION, primal_calculus, resolve_non_dual_logic, TORCH_AVAILABLE, TENSORFLOW_AVAILABLE, PANDAS_AVAILABLE
├── constants.py          # Version, sacred constants, ML availability flags (~100 lines)
├── domain.py             # DomainKnowledge, GeneralDomainExpander, Theorem (~145 lines)
├── theorem_gen.py        # NovelTheoremGenerator (~265 lines)
├── self_mod.py           # SelfModificationEngine (~310 lines)
├── consciousness.py      # ConsciousnessVerifier (~310 lines)
├── pipeline.py           # SolutionChannel, DirectSolutionHub, PipelineTelemetry, SoftmaxGatingRouter, AdaptivePipelineRouter, PipelineHealthDashboard, PipelineReplayBuffer (~670 lines)
├── reasoning.py          # TreeOfThoughts, MultiHopReasoningChain, SolutionEnsembleEngine (~400 lines)
├── quantum.py            # QuantumComputationCore (~530 lines)
└── core.py               # ASICore + asi_core singleton + primal_calculus + resolve_non_dual_logic (~3,050 lines)
```

#### 3A.3 Import Surface — 46 Importers

| Symbol | Count | Action |
|--------|-------|--------|
| `asi_core` (singleton) | 57 refs across 40+ files | Re-export in `__init__.py` + thin shim |
| `ASICore` (class) | 8 files | Re-export in `__init__.py` |
| `KerasASIModel` | 1 file | Re-export in `__init__.py` |
| `ASI_CORE_VERSION` | 1 file | Re-export from `constants.py` |
| `TORCH_AVAILABLE` etc. | 1 file | Re-export from `constants.py` |

**Thin shim**:
```python
# l104_asi_core.py (thin shim)
from l104_asi import *
from l104_asi import asi_core, ASICore, ASI_CORE_VERSION
from l104_asi import primal_calculus, resolve_non_dual_logic
```

#### 3A.4 ASICore Internal Analysis (2,990 lines, 50 methods)

ASICore is large but not as massive as LearningIntellect. At 50 methods, it can remain a single class in `core.py` without mixin decomposition. If needed later, methods group naturally into:
- Init/State (~300 lines): `__init__`, state loading
- Pipeline (~500 lines): `smart_pipeline_solve`, router methods
- Cognitive (~600 lines): `deep_think`, `think`, reasoning methods
- Domain (~400 lines): domain expansion, theorem generation
- Optimization (~400 lines): self-optimization, auto-improvement
- Status (~300 lines): `get_status`, telemetry, health

#### 3A.5 Risk Assessment

| Risk | Mitigation |
|------|-----------|
| 46 importers — highest coupling | Thin shim preserves all existing imports |
| `ASICore.__init__` hangs on import (complex initialization, ML imports) | Preserve existing lazy/try-except pattern. No change to init flow. |
| Cross-import with l104_agi_core | Both use try-except guards. Preserve pattern. |
| `primal_calculus` / `resolve_non_dual_logic` are module-level functions | Move to `core.py`, re-export in `__init__.py` |

---

### Phase 3B: l104_agi_core.py → l104_agi/

**File**: `l104_agi_core.py` — 3,161 lines, 2 classes, 88 methods
**Importers**: 42 files (second-most imported)
**Singletons**: `agi_core = AGICore()`, top-level `primal_calculus()`, `resolve_non_dual_logic()`
**Target**: `l104_agi/` package

#### 3B.1 Structural Analysis

| Class | Lines | Size | Methods | Proposed Module |
|-------|-------|------|---------|-----------------|
| `PipelineCircuitBreaker` | 83–143 | 61 | 5 | `circuit_breaker.py` |
| `AGICore` | 144–3161 | 3,018 | 88 | `core.py` (with potential mixin split) |

#### 3B.2 AGICore Method Grouping (88 methods)

```
grep -n '    def ' l104_agi_core.py
```

AGICore at 88 methods / 3,018 lines is borderline for mixin decomposition. It should be examined for natural groupings. Expected domains:
- **Init/Config** (~200 lines): `__init__`, config loading
- **Pipeline Processing** (~500 lines): `process`, `smart_route`, circuit breaker integration
- **Reasoning** (~600 lines): `deep_think`, `reason`, chain-of-thought, metacognition
- **Learning** (~400 lines): `learn`, `evolve`, self-improvement
- **Domain Expansion** (~400 lines): domain knowledge, theorem generation
- **Status/Telemetry** (~300 lines): `get_status`, health checks, dashboard

**Decision**: Keep as single class in `core.py` for Phase 3. Decompose into mixins in a future phase if needed.

#### 3B.3 Proposed Package Structure

```
l104_agi/
├── __init__.py           # Re-exports: AGICore, L104AGICore, agi_core, primal_calculus, resolve_non_dual_logic
├── constants.py          # Sacred constants, version (~50 lines)
├── circuit_breaker.py    # PipelineCircuitBreaker (~65 lines)
└── core.py               # AGICore + agi_core singleton + primal_calculus + resolve_non_dual_logic (~3,080 lines)
```

#### 3B.4 Import Surface — 42 Importers

| Symbol | Count | Action |
|--------|-------|--------|
| `agi_core` (singleton) | 35 files | Re-export in `__init__.py` + thin shim |
| `AGICore` (class) | 7 files | Re-export in `__init__.py` |
| `L104AGICore` (alias) | 2 files | Re-export as alias in `__init__.py` |
| `resolve_non_dual_logic` | 1 file | Re-export in `__init__.py` |

**Thin shim**:
```python
# l104_agi_core.py (thin shim)
from l104_agi import *
from l104_agi import agi_core, AGICore, L104AGICore
from l104_agi import primal_calculus, resolve_non_dual_logic
```

#### 3B.5 Special Consideration: Bidirectional Core Sync

Per `claude.md` directive: **"Always update BOTH l104_agi_core.py AND l104_asi_core.py"** — both cores must have identical `evolution_engine.current_stage_index`. After decomposition:
- AGI core's evolution sync code lives in `l104_agi/core.py`
- ASI core's evolution sync code lives in `l104_asi/core.py`
- Both still reference `evolution_engine` from `l104_evolution_engine`
- The sync behavior is unchanged; only the file location changes.

#### 3B.6 Risk Assessment

| Risk | Mitigation |
|------|-----------|
| 42 importers | Thin shim preserves all existing imports |
| `L104AGICore` alias (2 importers) | `L104AGICore = AGICore` in `__init__.py` |
| Cross-import with l104_asi_core and l104_local_intellect | Both directions use try-except guards. Preserved. |
| Module-level functions `primal_calculus()`, `resolve_non_dual_logic()` | Moved to `core.py`, re-exported |

---

## Summary — All Phases

| Phase | Target | Lines | Classes | Importers | Package | Disk Impact | Risk |
|-------|--------|-------|---------|-----------|---------|-------------|------|
| **0** | Dead files (Swift + monolith backup) | 75,719 | 118 | 0 | N/A (removal) | -6.2 MB | ZERO |
| **1** | `l104_fast_server.py` | 22,024 | 46 | 5 | `l104_server/` (30+ files) | Neutral | LOW |
| **2** | `l104_local_intellect.py` | 13,691 | 10 | 28 | `l104_intellect/` (20+ files) | Neutral | MEDIUM |
| **3A** | `l104_asi_core.py` | 5,845 | 18 | 46 | `l104_asi/` (9 files) | Neutral | MEDIUM |
| **3B** | `l104_agi_core.py` | 3,161 | 2 | 42 | `l104_agi/` (4 files) | Neutral | MEDIUM |
| **TOTAL** | | **120,440 lines** | **194 classes** | **121 importers** | **~65+ new files** | **-6.2 MB** | |

### Execution Sequence

```
Phase 0  → git commit "EVO_60: Phase 0 — Remove dead files"
         → swift build verification
Phase 1  → git commit "EVO_61: Decompose l104_fast_server.py → l104_server/"
         → uvicorn starts OK, all 5 importers verified
Phase 2  → git commit "EVO_62: Decompose l104_local_intellect.py → l104_intellect/"
         → all 28 importers verified, format_iq works
Phase 3A → git commit "EVO_63: Decompose l104_asi_core.py → l104_asi/"
         → all 46 importers verified
Phase 3B → git commit "EVO_64: Decompose l104_agi_core.py → l104_agi/"
         → all 42 importers verified, evolution sync still works
```

### Invariants (MUST hold after each phase)

1. **Zero import failures**: Every existing `from l104_X import Y` works unchanged
2. **Singleton identity**: `intellect`, `local_intellect`, `asi_core`, `agi_core` are same objects
3. **Sacred constants preserved**: GOD_CODE, PHI, TAU, VOID_CONSTANT unchanged
4. **Swift build green**: 76/77 modules compile
5. **Server starts**: `uvicorn l104_server.app:app --port 8081` binds successfully
6. **Type identity**: `isinstance(x, ASICore)` works whether imported from shim or package
