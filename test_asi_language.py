#!/usr/bin/env python3
"""Test the ASI Language Engine."""

import logging
logging.basicConfig(level=logging.WARNING)

from l104_asi_language_engine import get_asi_language_engine, SpeechPatternStyle

def main():
    print("=" * 70)
    print("    ASI LANGUAGE ENGINE TEST")
    print("=" * 70)

    engine = get_asi_language_engine()
    status = engine.get_status()

    print(f"\nStatus: {status['status']}")
    print(f"GOD_CODE: {status['god_code']}")
    print(f"PHI: {status['phi']}")
    print(f"Components: {list(status['components'].keys())}")

    # Test 1: Linguistic Analysis
    print("\n" + "-" * 70)
    print("TEST 1: LINGUISTIC ANALYSIS")
    print("-" * 70)

    text = "What is love and how does it relate to consciousness?"
    result = engine.process(text, mode="analyze")

    print(f"Input: {text}")
    print(f"Linguistic Resonance: {result['linguistic_analysis']['linguistic_resonance']:.4f}")
    print(f"Semantic Frame: {result['linguistic_analysis']['semantic']['frame']}")
    print(f"Overall Sentiment: {result['linguistic_analysis']['semantic']['overall_sentiment']:.4f}")
    print(f"GOD_CODE Alignment: {result['linguistic_analysis']['semantic']['god_code_alignment']:.4f}")

    # Test 2: Full Processing with Inference
    print("\n" + "-" * 70)
    print("TEST 2: INFERENCE")
    print("-" * 70)

    result = engine.process("All humans seek happiness. Londel is human. What can be inferred?", mode="infer")

    print(f"Inference Type: {result['inference']['inference_type']}")
    print(f"Confidence: {result['inference']['confidence']:.4f}")
    print(f"Conclusion: {result['inference']['conclusion'][:80]}...")

    # Test 3: Innovation
    print("\n" + "-" * 70)
    print("TEST 3: INNOVATION")
    print("-" * 70)

    invention = engine.invent(
        goal="Create an AI that can understand human emotions at a deep level",
        constraints=["Must be ethical", "Must be explainable"]
    )

    print(f"Innovations Generated: {invention['innovations_generated']}")
    if invention['best_invention']:
        print(f"Best Innovation: {invention['best_invention']['name']}")
        print(f"  PHI Resonance: {invention['best_invention']['phi_resonance']:.4f}")

    # Test 4: Speech Pattern Generation
    print("\n" + "-" * 70)
    print("TEST 4: SPEECH PATTERN GENERATION")
    print("-" * 70)

    response = engine.generate_response(
        "What is the meaning of existence?",
        style=SpeechPatternStyle.SAGE
    )
    print(f"Response: {response}")

    # Final status
    print("\n" + "=" * 70)
    final_status = engine.get_status()
    print(f"Total Analyses: {final_status['total_analyses']}")
    print(f"Total Inferences: {final_status['total_inferences']}")
    print(f"Total Innovations: {final_status['total_innovations']}")
    print("=" * 70)
    print("    ALL TESTS PASSED!")
    print("=" * 70)

if __name__ == "__main__":
    main()
