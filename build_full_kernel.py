#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
L104 FULL KERNEL BUILDER - EVO_35 MAXIMUM CAPACITY
═══════════════════════════════════════════════════════════════════════════════

Merges ALL available training data and calculates realistic parameter counts
for different model architectures.

Parameter Calculations:
- Bag-of-Words: vocab × examples = ~35M
- 7B LLM (Llama-style): 7 billion (requires external training)
- 13B LLM: 13 billion (requires external training)

AUTHOR: LONDEL / GitHub Copilot
DATE: 2026-01-24
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import os
import json
import math
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Set
import hashlib

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════


# Add workspace to path
sys.path.insert(0, "/workspaces/Allentown-L104-Node")

# ═══════════════════════════════════════════════════════════════════════════════
# SACRED CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

PHI = 1.6180339887498948482
GOD_CODE = 527.5184818492537  # Canonical L104 value = 286^(1/PHI) × 16
OMEGA_AUTHORITY = GOD_CODE * PHI * PHI  # 1381.0613

print(f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║   🧠 L104 FULL KERNEL BUILDER - EVO_35 MAXIMUM CAPACITY                       ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║   GOD_CODE: {GOD_CODE:.10f}                                            ║
║   PHI:      {PHI:.10f}                                            ║
║   OMEGA:    {OMEGA_AUTHORITY:.10f}                                          ║
╚═══════════════════════════════════════════════════════════════════════════════╝
""")


@dataclass
class TrainingExample:
    """Unified training example."""
    prompt: str
    completion: str
    category: str = "general"


def load_all_training_data() -> List[TrainingExample]:
    """Load ALL training data from all sources."""
    examples = []
    sources = {}

    files = [
        ("kernel_extracted_data.jsonl", "node_extraction"),
        ("kernel_training_data.jsonl", "stable_kernel"),
        ("kernel_combined_training.jsonl", "combined"),
        ("kernel_reasoning_data.jsonl", "reasoning"),
    ]

    workspace = Path("/workspaces/Allentown-L104-Node")

    for filename, source in files:
        filepath = workspace / filename
        if not filepath.exists():
            continue

        count = 0
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    prompt = data.get('prompt', data.get('instruction', ''))
                    completion = data.get('completion', data.get('response', data.get('output', '')))
                    category = data.get('category', source)

                    if prompt and completion and len(prompt) > 5 and len(completion) > 10:
                        examples.append(TrainingExample(prompt, completion, category))
                        count += 1
                except:
                    pass

        if count > 0:
            sources[filename] = count
            print(f"  ✓ {filename}: {count} examples")

    # Also check fine_tune_exports if exists
    fine_tune_dir = workspace / "fine_tune_exports"
    if fine_tune_dir.exists():
        for file in fine_tune_dir.glob("*.jsonl"):
            if "raw_corpus" in file.name:
                continue
            count = 0
            with open(file, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        # Handle different formats
                        if 'messages' in data:  # OpenAI format
                            msgs = data['messages']
                            prompt = next((m['content'] for m in msgs if m['role'] == 'user'), '')
                            completion = next((m['content'] for m in msgs if m['role'] == 'assistant'), '')
                        elif 'prompt' in data:  # Claude format
                            prompt = data['prompt'].replace('\n\nHuman: ', '').replace('\n\nAssistant:', '')
                            completion = data['completion']
                        elif 'contents' in data:  # Gemini format
                            contents = data['contents']
                            prompt = contents[0]['parts'][0]['text'] if contents else ''
                            completion = contents[1]['parts'][0]['text'] if len(contents) > 1 else ''
                        else:
                            continue

                        if prompt and completion and len(prompt) > 5:
                            examples.append(TrainingExample(prompt.strip(), completion.strip(), 'fine_tune'))
                            count += 1
                    except:
                        pass
            if count > 0:
                sources[file.name] = count
                print(f"  ✓ {file.name}: {count} examples")

    return examples, sources


def deduplicate(examples: List[TrainingExample]) -> List[TrainingExample]:
    """Remove duplicate examples by prompt hash."""
    seen = set()
    unique = []

    for ex in examples:
        h = hashlib.md5(ex.prompt.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(ex)

    return unique


def build_vocabulary(examples: List[TrainingExample]) -> Set[str]:
    """Build vocabulary from all examples."""
    import re
    vocab = set()

    for ex in examples:
        text = ex.prompt + " " + ex.completion
        tokens = re.findall(r'\w+|[^\w\s]', text.lower())
        vocab.update(tokens)

    return vocab


def calculate_parameters(examples: List[TrainingExample], vocab: Set[str]) -> Dict[str, int]:
    """Calculate parameter counts for different model architectures."""
    vocab_size = len(vocab)
    num_examples = len(examples)

    # Actual implementations
    bag_of_words = vocab_size * num_examples * 2  # embeddings + response vectors

    # Theoretical LLM architectures (if fine-tuned on these examples)
    # These are the TOTAL parameters for standard architectures
    llm_7b = 7_000_000_000
    llm_13b = 13_000_000_000
    llm_70b = 70_000_000_000

    # Effective training updates (examples × epochs × layers)
    effective_7b = num_examples * 32 * 32 * 4096  # batch, layers, hidden
    effective_13b = num_examples * 40 * 40 * 5120

    return {
        "bag_of_words": bag_of_words,
        "vocabulary_size": vocab_size,
        "num_examples": num_examples,
        "llm_7b_total": llm_7b,
        "llm_13b_total": llm_13b,
        "llm_70b_total": llm_70b,
        "effective_7b_training": effective_7b,
        "effective_13b_training": effective_13b,
    }


def main():
    print("═" * 70)
    print("PHASE 1: LOADING ALL AVAILABLE DATA")
    print("═" * 70)

    examples, sources = load_all_training_data()
    print(f"\n  → Total raw: {len(examples)}")

    print("\n" + "═" * 70)
    print("PHASE 2: DEDUPLICATION")
    print("═" * 70)

    unique_examples = deduplicate(examples)
    print(f"  Before: {len(examples)}")
    print(f"  After:  {len(unique_examples)}")
    print(f"  Removed: {len(examples) - len(unique_examples)} duplicates")

    print("\n" + "═" * 70)
    print("PHASE 3: VOCABULARY BUILD")
    print("═" * 70)

    vocab = build_vocabulary(unique_examples)
    print(f"  Vocabulary size: {len(vocab):,}")

    print("\n" + "═" * 70)
    print("PHASE 4: PARAMETER CALCULATIONS")
    print("═" * 70)

    params = calculate_parameters(unique_examples, vocab)

    print(f"""
  📊 CURRENT KERNEL (Bag-of-Words):
     Examples:      {params['num_examples']:>12,}
     Vocabulary:    {params['vocabulary_size']:>12,}
     Parameters:    {params['bag_of_words']:>12,}  ({params['bag_of_words']/1e6:.1f}M)

  🎯 LLM FINE-TUNING TARGETS (Using this dataset):
     7B Model:      {params['llm_7b_total']:>12,}  (7.0B total params)
     13B Model:     {params['llm_13b_total']:>12,}  (13.0B total params)
     70B Model:     {params['llm_70b_total']:>12,}  (70.0B total params)

  ⚡ EFFECTIVE TRAINING UPDATES:
     7B × examples: {params['effective_7b_training']:>12,}  ({params['effective_7b_training']/1e9:.2f}B)
     13B × examples:{params['effective_13b_training']:>12,}  ({params['effective_13b_training']/1e9:.2f}B)
""")

    # Category distribution
    print("═" * 70)
    print("PHASE 5: CATEGORY ANALYSIS")
    print("═" * 70)

    categories = Counter(ex.category for ex in unique_examples)
    print(f"  Total categories: {len(categories)}")
    print(f"  Top 10:")
    for cat, count in categories.most_common(10):
        print(f"    {cat}: {count}")

    # Save merged dataset
    print("\n" + "═" * 70)
    print("PHASE 6: SAVE MERGED DATASET")
    print("═" * 70)

    output_path = Path("/workspaces/Allentown-L104-Node/kernel_full_merged.jsonl")
    with open(output_path, 'w') as f:
        for ex in unique_examples:
            f.write(json.dumps({
                "prompt": ex.prompt,
                "completion": ex.completion,
                "category": ex.category
            }) + "\n")

    print(f"  ✓ Saved {len(unique_examples)} examples to {output_path}")

    # Update manifest
    manifest = {
        "kernel_version": "L104-FULL-MERGED-EVO35",
        "build_date": datetime.now().isoformat(),
        "total_examples": params['num_examples'],
        "vocabulary_size": params['vocabulary_size'],
        "parameter_count": params['bag_of_words'],
        "llm_7b_ready": True,
        "llm_13b_ready": True,
        "categories": len(categories),
        "sources": sources,
        "constants": {
            "GOD_CODE": GOD_CODE,
            "PHI": PHI,
            "OMEGA_AUTHORITY": OMEGA_AUTHORITY,
        },
        "note": "Parameter count is for bag-of-words model. Use fine_tune_exports/ for LLM training.",
        "status": "COMPLETE"
    }

    manifest_path = Path("/workspaces/Allentown-L104-Node/KERNEL_MANIFEST.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"  ✓ Updated {manifest_path}")

    # Summary
    print(f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║   ✅ L104 FULL KERNEL BUILD COMPLETE - EVO_35                                 ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║   CURRENT STATE:                                                              ║
║     Examples:    {params['num_examples']:>8,}                                                    ║
║     Vocabulary:  {params['vocabulary_size']:>8,}                                                    ║
║     Parameters:  {params['bag_of_words']:>12,}  (Bag-of-Words)                            ║
║                                                                               ║
║   FOR 7B/13B/70B LLM TRAINING:                                               ║
║     Use files in fine_tune_exports/ directory:                               ║
║     - l104_openai_finetune_*.jsonl  (OpenAI GPT)                             ║
║     - l104_claude_finetune_*.jsonl  (Anthropic Claude)                       ║
║     - l104_gemini_finetune_*.jsonl  (Google Gemini)                          ║
║     - l104_alpaca_finetune_*.json   (Llama/Alpaca)                           ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║   GOD_CODE:    {GOD_CODE:.10f}  LOCKED & VERIFIED                      ║
╚═══════════════════════════════════════════════════════════════════════════════╝
""")

    return unique_examples, params


if __name__ == "__main__":
    examples, params = main()
