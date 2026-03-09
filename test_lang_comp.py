#!/usr/bin/env python3
"""Test script for LanguageComprehensionEngine initialization."""

try:
    print("Importing l104_asi.language_comprehension...")
    from l104_asi.language_comprehension import LanguageComprehensionEngine
    print("Import successful!")

    print("Creating LanguageComprehensionEngine...")
    engine = LanguageComprehensionEngine()
    print("Engine created successfully!")

    print("Initializing engine...")
    engine.initialize()
    print("Engine initialized successfully!")

    print("Testing lazy-loaded components...")
    print("Accessing subject_detector...")
    sd = engine.subject_detector
    print(f"SubjectDetector: {type(sd).__name__}")

    print("Accessing frame_analyzer...")
    fa = engine.frame_analyzer
    print(f"SemanticFrameAnalyzer: {type(fa).__name__}")

    print("Accessing pragmatics...")
    prag = engine.pragmatics
    print(f"PragmaticInferenceEngine: {type(prag).__name__}")

    print("Getting final status...")
    status = engine.get_status()
    print(f"Version: {status.get('version')}")
    print(f"Initialized: {status.get('initialized')}")
    print(f"Subject detector available: {status.get('layers', {}).get('4b_subject_detector')}")
    print(f"Frame analyzer available: {status.get('layers', {}).get('21_semantic_frames')}")
    print(f"Pragmatic inference available: {status.get('layers', {}).get('24_pragmatic_inference')}")
    print("All tests PASSED!")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()