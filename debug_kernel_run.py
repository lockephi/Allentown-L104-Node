from pathlib import Path
#!/usr/bin/env python3
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
"""
L104 Kernel Debug & Run Script
Loads kernel, runs calculations, and verifies state.
"""
import sys
import json
from dataclasses import asdict

sys.path.insert(0, str(Path(__file__).parent.absolute()))

from l104_kernel_llm_trainer import KernelLLMTrainer, TrainingExample

print("═" * 60)
print("🚀 L104 KERNEL DEBUG & RUN")
print("═" * 60)

# Initialize
kernel = KernelLLMTrainer()

# Load existing data
print("\n📚 Loading training data...")
try:
    with open("kernel_training_data.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            ex = TrainingExample(
                prompt=obj.get("prompt", ""),
                completion=obj.get("completion", ""),
                category=obj.get("category", "unknown"),
                difficulty=obj.get("difficulty", 0.5),
                importance=obj.get("importance", 0.5),
                metadata=obj.get("metadata", {})
            )
            kernel.training_data.append(ex)
    print(f"✅ Loaded {len(kernel.training_data)} examples.")
except FileNotFoundError:
    print("⚠️ No existing data. Generating fresh...")
    kernel.generate_training_data()

# Train
print("\n🧠 Training neural network...")
kernel.train()
vocab_size = len(kernel.neural_net.vocabulary)
params = vocab_size * len(kernel.training_data)
print(f"✅ Vocabulary: {vocab_size}")
print(f"✅ Parameters: {params:,}")

# Run Calculations / Queries
print("\n" + "═" * 60)
print("🔍 RUNNING KERNEL CALCULATIONS")
print("═" * 60)

test_queries = [
    "What is GOD_CODE?",
    "What is a ZK-SNARK?",
    "Explain Learning With Errors (LWE).",
    "Define PHI constant.",
    "What is the Schnorr Protocol?"
]

for q in test_queries:
    print(f"\n❓ {q}")
    response = kernel.query(q)
    # Truncate for display
    if len(response) > 150:
        response = response[:150] + "..."
    print(f"💡 {response}")

# Summary
print("\n" + "═" * 60)
print("📊 KERNEL STATUS")
print("═" * 60)
print(f"Total Examples:  {len(kernel.training_data)}")
print(f"Vocabulary Size: {vocab_size}")
print(f"Parameters:      {params:,}")
print(f"Trained:         {kernel.trained}")

# Load and display manifest
try:
    with open("KERNEL_MANIFEST.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    stages = manifest.get("evolution_stages", [])
    print(f"\n🧬 Evolution Stages: {len(stages)}")
    for s in stages[-5:]:  # Last 5 stages
        print(f"   • {s}")
except Exception:
    pass

print("\n✨ Kernel Debug Complete.")
