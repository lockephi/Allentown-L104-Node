from pathlib import Path
#!/usr/bin/env python3
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
"""
L104 Kernel Training Script
Loads, trains, and verifies the kernel.
"""
import sys
import json
sys.path.insert(0, str(Path(__file__).parent.absolute()))

from l104_kernel_llm_trainer import KernelLLMTrainer, TrainingExample

print("=" * 60)
print("🚀 L104 KERNEL TRAINING")
print("=" * 60)

# Initialize
kernel = KernelLLMTrainer()

# Load existing data
print("\n📚 Loading training data...")
try:
    with open("kernel_training_data.jsonl", "r") as f:
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
    print(f"✅ Loaded {len(kernel.training_data)} examples")
except FileNotFoundError:
    print("⚠️ No existing data. Generating fresh...")
    kernel.generate_training_data()
    print(f"✅ Generated {len(kernel.training_data)} examples")

# Train
print("\n🧠 Training neural network...")
kernel.train()

# Stats
params = kernel.neural_net.get_parameter_count()
vocab = len(kernel.neural_net.vocabulary)
print(f"✅ Vocabulary: {vocab}")
print(f"✅ Parameters: {params:,}")

# Test queries
print("\n" + "=" * 60)
print("🔍 RUNNING TEST QUERIES")
print("=" * 60)

test_queries = [
    "What is GOD_CODE?",
    "What is PHI?",
    "Explain ZK-SNARK"
]

for q in test_queries:
    response = kernel.query(q)
    short_resp = response[:80] + "..." if len(response) > 80 else response
    print(f"\n❓ {q}")
    print(f"💡 {short_resp}")

print("\n" + "=" * 60)
print("✨ KERNEL TRAINING COMPLETE")
print("=" * 60)
