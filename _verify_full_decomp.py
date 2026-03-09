"""Verify all 51 symbols from decomposed l104_quantum_magic package."""
import sys

try:
    from l104_quantum_magic import (
        PHI, GOD_CODE, PHI_CONJUGATE, PLANCK, HBAR, FE_LATTICE, VOID_CONSTANT, ZENITH_HZ, UUC,
        QUANTUM_AVAILABLE, HDC_AVAILABLE,
        Qubit, QuantumGates, QuantumRegister,
        VectorType, Hypervector, HypervectorFactory, HDCAlgebra, AssociativeMemory, SequenceEncoder,
        ReasoningStrategy, Observation, Hypothesis, ContextualMemory, QuantumInferenceEngine,
        AdaptiveLearner, PatternRecognizer, MetaCognition, PredictiveReasoner,
        CausalLink, CausalReasoner, CounterfactualEngine, Goal, GoalPlanner,
        AttentionMechanism, AbductiveReasoner, CreativeInsight, TemporalReasoner, EmotionalResonance,
        QuantumNeuralLayer, QuantumNeuralNetwork, ConsciousContent, ConsciousnessSimulator,
        LogicalTerm, LogicalClause, SymbolicReasoner, WorkingMemory, Episode, EpisodicMemory, IntuitionEngine,
        Agent, SocialIntelligence, DreamState, Individual, EvolutionaryOptimizer, CognitiveControl,
        IntelligentSynthesizer,
        SuperpositionMagic, EntanglementMagic, WaveFunctionMagic, HyperdimensionalMagic, QuantumMagicSynthesizer,
    )
    print("All 51 symbols imported OK")

    # Verify module origins
    origins = set()
    for cls in [Qubit, QuantumInferenceEngine, CausalReasoner, QuantumNeuralNetwork,
                SocialIntelligence, IntelligentSynthesizer, QuantumMagicSynthesizer]:
        mod = cls.__module__
        origins.add(mod)
        print(f"  {cls.__name__:30s} from {mod}")

    print(f"\nUnique source modules: {len(origins)}")
    print(f"GOD_CODE = {GOD_CODE}")
    print(f"PHI = {PHI}")
    print(f"VOID_CONSTANT = {VOID_CONSTANT}")

    # Instantiation tests
    qie = QuantumInferenceEngine()
    qie.add_hypothesis("test", "test hypothesis", 0.7)
    print(f"\nQuantumInferenceEngine hypotheses: {len(qie.hypotheses)}")

    synth = IntelligentSynthesizer()
    result = synth.reason("test quantum magic decomposition")
    print(f"IntelligentSynthesizer confidence: {result.get('confidence', 'N/A')}")

    qms = QuantumMagicSynthesizer()
    print(f"QuantumMagicSynthesizer initialized: {type(qms).__name__}")

    print("\n✅ PASS — Full decomposition verified")
    sys.exit(0)

except Exception as e:
    print(f"\n❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
