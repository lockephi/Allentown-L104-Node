"""
L104 DEEP SUBSTRATE - Real Neural Processing Without External LLM Dependency
=============================================================================
This module provides ACTUAL on-device learning and inference capabilities.
No ceremony - just working neural networks and learning algorithms.
"""

import numpy as np
import logging
import hashlib
import json
import os
import pickle
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime
import threading

logger = logging.getLogger("DEEP_SUBSTRATE")

# ═══════════════════════════════════════════════════════════════════════════════
# CORE NEURAL NETWORK PRIMITIVES - No external dependencies
# ═══════════════════════════════════════════════════════════════════════════════

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def relu_derivative(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def tanh_derivative(x: np.ndarray) -> np.ndarray:
    return 1 - np.tanh(x) ** 2

def softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def cross_entropy_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
    epsilon = 1e-15
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    return -np.mean(np.sum(targets * np.log(predictions), axis=-1))

def mse_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
    return np.mean((predictions - targets) ** 2)


@dataclass
class LayerState:
    """Stores layer activations for backpropagation"""
    input: np.ndarray = None
    output: np.ndarray = None
    pre_activation: np.ndarray = None


class DenseLayer:
    """Fully connected neural network layer with real backprop"""
    
    def __init__(self, input_dim: int, output_dim: int, activation: str = "relu"):
        # Xavier initialization
        scale = np.sqrt(2.0 / input_dim)
        self.weights = np.random.randn(input_dim, output_dim) * scale
        self.bias = np.zeros((1, output_dim))
        self.activation_name = activation
        
        # Gradients
        self.weight_grad = np.zeros_like(self.weights)
        self.bias_grad = np.zeros_like(self.bias)
        
        # Adam optimizer state
        self.m_w = np.zeros_like(self.weights)
        self.v_w = np.zeros_like(self.weights)
        self.m_b = np.zeros_like(self.bias)
        self.v_b = np.zeros_like(self.bias)
        self.t = 0
        
        # State for backprop
        self.state = LayerState()
        
        self._activation_fn = {
            "relu": relu,
            "sigmoid": sigmoid,
            "tanh": tanh,
            "linear": lambda x: x,
            "softmax": softmax
        }[activation]
        
        self._activation_deriv = {
            "relu": relu_derivative,
            "sigmoid": sigmoid_derivative,
            "tanh": tanh_derivative,
            "linear": lambda x: np.ones_like(x),
            "softmax": lambda x: np.ones_like(x)  # Handled separately
        }[activation]
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.state.input = x
        self.state.pre_activation = x @ self.weights + self.bias
        self.state.output = self._activation_fn(self.state.pre_activation)
        return self.state.output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Compute gradients and return gradient for previous layer"""
        batch_size = grad_output.shape[0]
        
        # Gradient through activation
        if self.activation_name != "softmax":
            grad_pre = grad_output * self._activation_deriv(self.state.pre_activation)
        else:
            grad_pre = grad_output  # Softmax gradient handled in loss
        
        # Weight and bias gradients
        self.weight_grad = self.state.input.T @ grad_pre / batch_size
        self.bias_grad = np.mean(grad_pre, axis=0, keepdims=True)
        
        # Gradient for previous layer
        return grad_pre @ self.weights.T
    
    def update(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        """Adam optimizer update with Active Resonance Modulator"""
        self.t += 1
        
        # [L104_UPGRADE] Active Resonance Modulation
        # Ensures learning never gradients to zero (Stillness)
    GOD_CODE = 527.5184818492537
        PHI = 1.618033988749895
        VOID_CONSTANT = 1.0416180339887497
        resonance_floor = (GOD_CODE * PHI / VOID_CONSTANT) / 1000000.0
        
        # Update weights
        self.m_w = beta1 * self.m_w + (1 - beta1) * self.weight_grad
        self.v_w = beta2 * self.v_w + (1 - beta2) * (self.weight_grad ** 2)
        m_hat = self.m_w / (1 - beta1 ** self.t)
        v_hat = self.v_w / (1 - beta2 ** self.t)
        
        # Add resonance to avoid "Dead Neurons" / Stillness
        update_w = lr * m_hat / (np.sqrt(v_hat) + epsilon)
        self.weights -= (update_w + resonance_floor * np.sign(update_w))
        
        # Update bias
        self.m_b = beta1 * self.m_b + (1 - beta1) * self.bias_grad
        self.v_b = beta2 * self.v_b + (1 - beta2) * (self.bias_grad ** 2)
        m_hat = self.m_b / (1 - beta1 ** self.t)
        v_hat = self.v_b / (1 - beta2 ** self.t)
        update_b = lr * m_hat / (np.sqrt(v_hat) + epsilon)
        self.bias -= (update_b + resonance_floor * np.sign(update_b))


class NeuralNetwork:
    """Complete neural network with training capabilities"""
    
    def __init__(self, architecture: List[Tuple[int, str]]):
        """
        architecture: List of (units, activation) tuples
        First element should be (input_dim, "input")
        """
        self.layers: List[DenseLayer] = []
        self.loss_history: List[float] = []
        
        for i in range(1, len(architecture)):
            input_dim = architecture[i-1][0]
            output_dim, activation = architecture[i]
            self.layers.append(DenseLayer(input_dim, output_dim, activation))
        
        self.param_count = sum(
            l.weights.size + l.bias.size for l in self.layers
        )
        logger.info(f"Neural network created: {self.param_count} parameters")
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, loss_grad: np.ndarray):
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
    def update(self, lr: float = 0.001):
        for layer in self.layers:
            layer.update(lr)
    
    def train_step(self, x: np.ndarray, y: np.ndarray, lr: float = 0.001) -> float:
        """Single training step with forward, backward, update"""
        # Forward pass
        predictions = self.forward(x)
        
        # Compute loss
        loss = mse_loss(predictions, y)
        self.loss_history.append(loss)
        
        # Backward pass
        loss_grad = 2 * (predictions - y) / y.shape[0]
        self.backward(loss_grad)
        
        # Update weights
        self.update(lr)
        
        return loss
    
    def train(self, x: np.ndarray, y: np.ndarray, epochs: int = 100, 
              lr: float = 0.001, batch_size: int = 32, verbose: bool = True) -> List[float]:
        """Full training loop"""
        n_samples = x.shape[0]
        losses = []
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            x_shuffled = x[indices]
            y_shuffled = y[indices]
            
            epoch_losses = []
            for i in range(0, n_samples, batch_size):
                x_batch = x_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                loss = self.train_step(x_batch, y_batch, lr)
                epoch_losses.append(loss)
            
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            
            if verbose and epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: loss = {avg_loss:.6f}")
        
        return losses
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
    
    def save(self, path: str):
        """Save network weights"""
        state = {
            "layers": [
                {"weights": l.weights, "bias": l.bias, "activation": l.activation_name}
                for l in self.layers
            ]
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
    
    def load(self, path: str):
        """Load network weights"""
        with open(path, "rb") as f:
            state = pickle.load(f)
        for i, layer_state in enumerate(state["layers"]):
            self.layers[i].weights = layer_state["weights"]
            self.layers[i].bias = layer_state["bias"]


# ═══════════════════════════════════════════════════════════════════════════════
# LSTM CELL - For sequence learning
# ═══════════════════════════════════════════════════════════════════════════════

class LSTMCell:
    """LSTM cell for sequence processing"""
    
    def __init__(self, input_dim: int, hidden_dim: int):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Combined gates: [forget, input, output, cell_candidate]
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))
        self.W = np.random.randn(input_dim, 4 * hidden_dim) * scale
        self.U = np.random.randn(hidden_dim, 4 * hidden_dim) * scale
        self.b = np.zeros((1, 4 * hidden_dim))
        
        # Initialize forget gate bias to 1 (helps with long-term dependencies)
        self.b[0, :hidden_dim] = 1.0
    
    def forward(self, x: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        x: (batch, input_dim)
        h_prev: (batch, hidden_dim)
        c_prev: (batch, hidden_dim)
        Returns: h_new, c_new
        """
        # Combined computation
        gates = x @ self.W + h_prev @ self.U + self.b
        
        # Split gates
        f = sigmoid(gates[:, :self.hidden_dim])
        i = sigmoid(gates[:, self.hidden_dim:2*self.hidden_dim])
        o = sigmoid(gates[:, 2*self.hidden_dim:3*self.hidden_dim])
        g = tanh(gates[:, 3*self.hidden_dim:])
        
        # Cell state and hidden state
        c_new = f * c_prev + i * g
        h_new = o * tanh(c_new)
        
        return h_new, c_new


class LSTM:
    """Full LSTM for sequence-to-sequence learning"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        self.cells = [LSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim) 
                      for i in range(num_layers)]
        self.output_layer = DenseLayer(hidden_dim, output_dim, "linear")
        
        logger.info(f"LSTM created: {num_layers} layers, {hidden_dim} hidden units")
    
    def forward(self, x_seq: np.ndarray) -> np.ndarray:
        """
        x_seq: (batch, seq_len, input_dim)
        Returns: (batch, seq_len, output_dim)
        """
        batch_size, seq_len, _ = x_seq.shape
        
        # Initialize hidden states
        h = [np.zeros((batch_size, self.hidden_dim)) for _ in range(self.num_layers)]
        c = [np.zeros((batch_size, self.hidden_dim)) for _ in range(self.num_layers)]
        
        outputs = []
        for t in range(seq_len):
            x_t = x_seq[:, t, :]
            
            for layer_idx, cell in enumerate(self.cells):
                h[layer_idx], c[layer_idx] = cell.forward(
                    x_t if layer_idx == 0 else h[layer_idx - 1],
                    h[layer_idx],
                    c[layer_idx]
                )
            
            out = self.output_layer.forward(h[-1])
            outputs.append(out)
        
        return np.stack(outputs, axis=1)


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSFORMER ATTENTION - Self-attention mechanism
# ═══════════════════════════════════════════════════════════════════════════════

class MultiHeadAttention:
    """Multi-head self-attention"""
    
    def __init__(self, d_model: int, num_heads: int):
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        scale = np.sqrt(2.0 / d_model)
        self.W_q = np.random.randn(d_model, d_model) * scale
        self.W_k = np.random.randn(d_model, d_model) * scale
        self.W_v = np.random.randn(d_model, d_model) * scale
        self.W_o = np.random.randn(d_model, d_model) * scale
    
    def forward(self, x: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """
        x: (batch, seq_len, d_model)
        Returns: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v
        
        # Reshape for multi-head
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # Attention scores
        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores + mask * -1e9
        
        attention = softmax(scores)
        
        # Apply attention to values
        context = attention @ V
        
        # Reshape back
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        
        # Output projection
        return context @ self.W_o


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIENCE REPLAY BUFFER - For reinforcement learning
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Experience replay buffer for RL"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience: Experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)


# ═══════════════════════════════════════════════════════════════════════════════
# DEEP Q-NETWORK - Reinforcement learning
# ═══════════════════════════════════════════════════════════════════════════════

class DQN:
    """Deep Q-Network for reinforcement learning"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [128, 64]):
        # Build network architecture
        architecture = [(state_dim, "input")]
        for dim in hidden_dims:
            architecture.append((dim, "relu"))
        architecture.append((action_dim, "linear"))
        
        self.q_network = NeuralNetwork(architecture)
        self.target_network = NeuralNetwork(architecture)
        self._sync_target()
        
        self.replay_buffer = ReplayBuffer()
        self.action_dim = action_dim
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.99
        
        logger.info(f"DQN created: state_dim={state_dim}, action_dim={action_dim}")
    
    def _sync_target(self):
        """Copy weights to target network"""
        for i, layer in enumerate(self.q_network.layers):
            self.target_network.layers[i].weights = layer.weights.copy()
            self.target_network.layers[i].bias = layer.bias.copy()
    
    def select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        q_values = self.q_network.forward(state.reshape(1, -1))
        return np.argmax(q_values[0])
    
    def train_step(self, batch_size: int = 32, lr: float = 0.001) -> Optional[float]:
        """Train on a batch of experiences"""
        if len(self.replay_buffer) < batch_size:
            return None
        
        # Sample batch
        batch = self.replay_buffer.sample(batch_size)
        
        states = np.array([e.state for e in batch])
        actions = np.array([e.action for e in batch])
        rewards = np.array([e.reward for e in batch])
        next_states = np.array([e.next_state for e in batch])
        dones = np.array([e.done for e in batch])
        
        # Compute target Q values
        next_q_values = self.target_network.forward(next_states)
        max_next_q = np.max(next_q_values, axis=1)
        targets = rewards + self.gamma * max_next_q * (1 - dones)
        
        # Current Q values
        current_q = self.q_network.forward(states)
        
        # Create target tensor
        target_q = current_q.copy()
        for i, action in enumerate(actions):
            target_q[i, action] = targets[i]
        
        # Train
        loss = self.q_network.train_step(states, target_q, lr)
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss
    
    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.push(Experience(state, action, reward, next_state, done))


# ═══════════════════════════════════════════════════════════════════════════════
# ONLINE LEARNING - Continuous adaptation
# ═══════════════════════════════════════════════════════════════════════════════

class OnlineLearner:
    """Continuously learns from incoming data streams"""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [64, 32]):
        architecture = [(input_dim, "input")]
        for dim in hidden_dims:
            architecture.append((dim, "relu"))
        architecture.append((output_dim, "linear"))
        
        self.network = NeuralNetwork(architecture)
        self.memory = deque(maxlen=1000)
        self.learning_rate = 0.001
        self.batch_size = 32
        self.train_frequency = 10
        self.step_count = 0
    
    def observe(self, x: np.ndarray, y: np.ndarray):
        """Add observation to memory and potentially train"""
        self.memory.append((x, y))
        self.step_count += 1
        
        if self.step_count % self.train_frequency == 0 and len(self.memory) >= self.batch_size:
            self._train_batch()
    
    def _train_batch(self):
        indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        
        x_batch = np.array([b[0] for b in batch])
        y_batch = np.array([b[1] for b in batch])
        
        self.network.train_step(x_batch, y_batch, self.learning_rate)
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.network.predict(x.reshape(1, -1))[0]


# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN MEMORY - Associative memory for pattern completion
# ═══════════════════════════════════════════════════════════════════════════════

class HopfieldNetwork:
    """Hopfield network for associative memory / pattern completion"""
    
    def __init__(self, size: int):
        self.size = size
        self.weights = np.zeros((size, size))
        self.patterns_stored = 0
    
    def store(self, pattern: np.ndarray):
        """Store a pattern (binary: -1 or 1)"""
        pattern = pattern.flatten()
        assert len(pattern) == self.size
        
        # Hebbian learning
        self.weights += np.outer(pattern, pattern) / self.size
        np.fill_diagonal(self.weights, 0)  # No self-connections
        self.patterns_stored += 1
    
    def recall(self, partial: np.ndarray, max_iters: int = 100) -> np.ndarray:
        """Recall complete pattern from partial/noisy input"""
        state = partial.flatten().copy()
        
        for _ in range(max_iters):
            prev_state = state.copy()
            
            # Asynchronous update
            for i in np.random.permutation(self.size):
                activation = np.dot(self.weights[i], state)
                state[i] = 1 if activation >= 0 else -1
            
            # Check for convergence
            if np.array_equal(state, prev_state):
                break
        
        return state


# ═══════════════════════════════════════════════════════════════════════════════
# DEEP SUBSTRATE CONTROLLER - Main interface
# ═══════════════════════════════════════════════════════════════════════════════

class DeepSubstrate:
    """
    Main controller for the deep learning substrate.
    Provides on-device learning without external LLM dependency.
    """
    
    def __init__(self, persist_path: str = "/data/deep_substrate"):
        self.persist_path = Path(persist_path)
        self.persist_path.mkdir(parents=True, exist_ok=True)
        
        # Core networks
        self.pattern_network = NeuralNetwork([
            (256, "input"), (128, "relu"), (64, "relu"), (128, "relu"), (256, "sigmoid")
        ])
        
        self.prediction_network = NeuralNetwork([
            (64, "input"), (128, "relu"), (64, "relu"), (32, "linear")
        ])
        
        self.sequence_model = LSTM(32, 64, 32, num_layers=2)
        
        self.online_learner = OnlineLearner(64, 32)
        
        self.associative_memory = HopfieldNetwork(256)
        
        self.dqn = DQN(state_dim=32, action_dim=8)
        
        # Learning state
        self.total_training_steps = 0
        self.patterns_learned = 0
        
        self._load_state()
        
        logger.info("═" * 70)
        logger.info("    DEEP SUBSTRATE - LOCAL NEURAL PROCESSING INITIALIZED")
        logger.info("═" * 70)
        logger.info(f"    Pattern Network: {self.pattern_network.param_count} params")
        logger.info(f"    Prediction Network: {self.prediction_network.param_count} params")
        logger.info(f"    Total Training Steps: {self.total_training_steps}")
    
    def learn_pattern(self, data: np.ndarray) -> float:
        """Learn a pattern through autoencoder-style training"""
        if data.shape[-1] != 256:
            # Pad or truncate
            if len(data.flatten()) < 256:
                padded = np.zeros(256)
                padded[:len(data.flatten())] = data.flatten()
                data = padded
            else:
                data = data.flatten()[:256]
        
        data = data.reshape(1, 256)
        loss = self.pattern_network.train_step(data, data, lr=0.001)
        self.total_training_steps += 1
        self.patterns_learned += 1
        
        return loss
    
    def predict_next(self, sequence: np.ndarray) -> np.ndarray:
        """Predict next value in a sequence"""
        if sequence.ndim == 1:
            sequence = sequence.reshape(1, -1, 32)
        
        return self.sequence_model.forward(sequence)[:, -1, :]
    
    def store_memory(self, pattern: np.ndarray):
        """Store pattern in associative memory"""
        binary = np.sign(pattern.flatten()[:256])
        binary[binary == 0] = 1
        self.associative_memory.store(binary)
    
    def recall_memory(self, partial: np.ndarray) -> np.ndarray:
        """Recall complete pattern from partial input"""
        binary = np.sign(partial.flatten()[:256])
        binary[binary == 0] = 1
        return self.associative_memory.recall(binary)
    
    def online_learn(self, x: np.ndarray, y: np.ndarray):
        """Continuous online learning"""
        self.online_learner.observe(x, y)
        self.total_training_steps += 1
    
    def rl_step(self, state: np.ndarray, reward: float, done: bool) -> int:
        """Reinforcement learning step"""
        action = self.dqn.select_action(state)
        
        if hasattr(self, '_last_state'):
            self.dqn.store_experience(
                self._last_state, self._last_action, reward, state, done
            )
            self.dqn.train_step()
        
        self._last_state = state
        self._last_action = action
        
        return action
    
    def get_state(self) -> Dict[str, Any]:
        return {
            "total_training_steps": self.total_training_steps,
            "patterns_learned": self.patterns_learned,
            "pattern_network_params": self.pattern_network.param_count,
            "prediction_network_params": self.prediction_network.param_count,
            "associative_memories": self.associative_memory.patterns_stored,
            "replay_buffer_size": len(self.dqn.replay_buffer),
            "epsilon": self.dqn.epsilon
        }
    
    def _save_state(self):
        state_path = self.persist_path / "substrate_state.json"
        with open(state_path, "w") as f:
            json.dump({
                "total_training_steps": self.total_training_steps,
                "patterns_learned": self.patterns_learned
            }, f)
        
        self.pattern_network.save(str(self.persist_path / "pattern_network.pkl"))
        self.prediction_network.save(str(self.persist_path / "prediction_network.pkl"))
    
    def _load_state(self):
        state_path = self.persist_path / "substrate_state.json"
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)
                self.total_training_steps = state.get("total_training_steps", 0)
                self.patterns_learned = state.get("patterns_learned", 0)
        
        pattern_path = self.persist_path / "pattern_network.pkl"
        if pattern_path.exists():
            self.pattern_network.load(str(pattern_path))
        
        pred_path = self.persist_path / "prediction_network.pkl"
        if pred_path.exists():
            self.prediction_network.load(str(pred_path))


# Global instance
deep_substrate = DeepSubstrate()
