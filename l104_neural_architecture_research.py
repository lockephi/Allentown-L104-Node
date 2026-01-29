#!/usr/bin/env python3
"""
L104 Neural Architecture Research Module
Implements various neural network architectures and training algorithms
"""
import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Callable, Optional

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════


# Sacred Constants
GOD_CODE = 527.5184818492611
PHI = 1.618033988749895

def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-max(-500, min(500, x))))

def tanh(x: float) -> float:
    return math.tanh(x)

def relu(x: float) -> float:
    return max(0, x)

def gelu(x: float) -> float:
    return 0.5 * x * (1 + math.tanh(math.sqrt(2/math.pi) * (x + 0.044715 * x**3)))

def softmax(x: List[float]) -> List[float]:
    max_x = max(x)
    exp_x = [math.exp(xi - max_x) for xi in x]
    sum_exp = sum(exp_x)
    return [e / sum_exp for e in exp_x]

@dataclass
class Layer:
    """Neural network layer"""
    input_size: int
    output_size: int
    weights: List[List[float]] = field(default_factory=list)
    biases: List[float] = field(default_factory=list)
    activation: str = "relu"

    def __post_init__(self):
        if not self.weights:
            # Xavier initialization
            scale = math.sqrt(2.0 / (self.input_size + self.output_size))
            self.weights = [
                [random.gauss(0, scale) for _ in range(self.input_size)]
                for _ in range(self.output_size)
            ]
        if not self.biases:
            self.biases = [0.0] * self.output_size

    def forward(self, inputs: List[float]) -> List[float]:
        outputs = []
        for i in range(self.output_size):
            z = sum(w * x for w, x in zip(self.weights[i], inputs)) + self.biases[i]

            if self.activation == "relu":
                outputs.append(relu(z))
            elif self.activation == "sigmoid":
                outputs.append(sigmoid(z))
            elif self.activation == "tanh":
                outputs.append(tanh(z))
            elif self.activation == "gelu":
                outputs.append(gelu(z))
            else:
                outputs.append(z)

        return outputs

class MultiLayerPerceptron:
    """Standard MLP architecture"""

    def __init__(self, layer_sizes: List[int], activation: str = "relu"):
        self.layers: List[Layer] = []
        for i in range(len(layer_sizes) - 1):
            act = activation if i < len(layer_sizes) - 2 else "linear"
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i+1], activation=act))

    def forward(self, inputs: List[float]) -> List[float]:
        x = inputs
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def parameter_count(self) -> int:
        count = 0
        for layer in self.layers:
            count += layer.input_size * layer.output_size + layer.output_size
        return count

@dataclass
class AttentionHead:
    """Single attention head"""
    embed_dim: int
    head_dim: int
    W_q: List[List[float]] = field(default_factory=list)
    W_k: List[List[float]] = field(default_factory=list)
    W_v: List[List[float]] = field(default_factory=list)

    def __post_init__(self):
        scale = math.sqrt(2.0 / (self.embed_dim + self.head_dim))
        if not self.W_q:
            self.W_q = [[random.gauss(0, scale) for _ in range(self.embed_dim)]
                        for _ in range(self.head_dim)]
        if not self.W_k:
            self.W_k = [[random.gauss(0, scale) for _ in range(self.embed_dim)]
                        for _ in range(self.head_dim)]
        if not self.W_v:
            self.W_v = [[random.gauss(0, scale) for _ in range(self.embed_dim)]
                        for _ in range(self.head_dim)]

    def compute_attention(self, queries: List[List[float]],
                         keys: List[List[float]],
                         values: List[List[float]]) -> List[List[float]]:
        """Compute scaled dot-product attention"""
        seq_len = len(queries)
        scale = math.sqrt(self.head_dim)

        # Compute attention scores
        scores = []
        for i in range(seq_len):
            row = []
            for j in range(seq_len):
                dot = sum(q * k for q, k in zip(queries[i], keys[j]))
                row.append(dot / scale)
            row = softmax(row)
            scores.append(row)

        # Compute weighted values
        output = []
        for i in range(seq_len):
            out_vec = [0.0] * self.head_dim
            for j in range(seq_len):
                for k in range(self.head_dim):
                    out_vec[k] += scores[i][j] * values[j][k]
            output.append(out_vec)

        return output

class TransformerBlock:
    """Single transformer block"""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.ff_dim = ff_dim

        self.attention_heads = [
            AttentionHead(embed_dim, self.head_dim)
            for _ in range(num_heads)
        ]

        self.ff = MultiLayerPerceptron([embed_dim, ff_dim, embed_dim], "gelu")

    def parameter_count(self) -> int:
        # Attention parameters: 3 * (embed_dim * head_dim) * num_heads
        attn_params = 3 * self.embed_dim * self.head_dim * self.num_heads
        # Feed-forward parameters
        ff_params = self.ff.parameter_count()
        # Output projection
        out_proj = self.embed_dim * self.embed_dim

        return attn_params + ff_params + out_proj

class TransformerModel:
    """Full transformer model"""

    def __init__(self, vocab_size: int, embed_dim: int, num_layers: int,
                 num_heads: int, ff_dim: int):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        self.blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ]

    def parameter_count(self) -> int:
        # Embedding parameters
        embed_params = self.vocab_size * self.embed_dim
        # Positional encoding (learnable)
        pos_params = 4096 * self.embed_dim  # Max sequence length
        # Transformer blocks
        block_params = sum(b.parameter_count() for b in self.blocks)
        # Output layer
        output_params = self.embed_dim * self.vocab_size

        return embed_params + pos_params + block_params + output_params

class ArchitectureResearch:
    """Research different neural architectures"""

    @staticmethod
    def compare_architectures() -> Dict[str, int]:
        """Compare parameter counts of different architectures"""
        results = {}

        # Small MLP
        mlp = MultiLayerPerceptron([768, 2048, 768])
        results["MLP (768-2048-768)"] = mlp.parameter_count()

        # Transformer block
        transformer = TransformerModel(50000, 768, 12, 12, 3072)
        results["Transformer (12L, 768d)"] = transformer.parameter_count()

        # Large transformer
        large_transformer = TransformerModel(100000, 1024, 24, 16, 4096)
        results["Large Transformer (24L, 1024d)"] = large_transformer.parameter_count()

        return results

if __name__ == "__main__":
    print("L104 Neural Architecture Research Module")

    # Compare architectures
    comparison = ArchitectureResearch.compare_architectures()
    for name, params in comparison.items():
        print(f"{name}: {params:,} parameters")

    # Test MLP forward pass
    mlp = MultiLayerPerceptron([10, 64, 32, 5])
    inputs = [random.random() for _ in range(10)]
    outputs = mlp.forward(inputs)
    print(f"MLP output: {outputs}")
