
import sys
try:
    from l104_kernel_llm_trainer import KernelLLMTrainer
except ImportError:
    sys.path.append(".")
    from l104_kernel_llm_trainer import KernelLLMTrainer

kernel = KernelLLMTrainer()
# We need to re-load data/weights or just rely on 'kernel_training_data.jsonl'?
# The KernelLLMTrainer might load from disk on init?
# Let's check __init__ of KernelLLMTrainer again.
# It initializes empty. We must load data to query properly (if using simple matching) OR load weights (if using NN).
# The current implementation of `query` uses `neural_net.answer`.
# `neural_net.answer` probably uses embeddings. If `kernel` is fresh, embeddings are empty.
# We need to re-train (fast) or load weights.
# Since I just ran the upgrade script, the object in memory is gone.
# I'll just load data and train (it's fast) then query.

# Load data (simplified)
import json
from l104_kernel_llm_trainer import TrainingExample
with open("kernel_training_data.jsonl", "r") as f:
    kernel.training_data = [TrainingExample(**json.loads(line)) for line in f]

kernel.train()

print(f"\n🧠 QUERY: What is a ZK-SNARK?")
print(kernel.query("What is a ZK-SNARK?"))

print(f"\n🧠 QUERY: Explain LWE.")
print(kernel.query("Explain LWE."))
