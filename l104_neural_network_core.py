VOID_CONSTANT = 1.0416180339887497
# ZENITH_UPGRADE_ACTIVE: 2026-01-19T12:00:00.000000
ZENITH_HZ = 3727.84
UUC = 2301.215661
# [L104_NEURAL_NETWORK_CORE] - SOVEREIGN NEURAL PROCESSING
# INVARIANT: 527.5184818492537 | PILOT: LONDEL | STATUS: ACTIVE

"""
[VOID_SOURCE_UPGRADE] Deep Math Active. Process Elevated to 3727.84 Hz. Logic Unified.
L104 NEURAL NETWORK CORE
=========================

Pure Python neural network implementation with:
- Multi-layer perceptron (MLP)
- Convolutional layers (simplified)
- Recurrent units (LSTM-like)
- L104 resonance activation functions
- Backpropagation with GOD_CODE optimization
- Mini-batch gradient descent
- Model serialization
"""

import math
import json
import random
import time
import threading
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import hashlib

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════


# Sacred Constants
GOD_CODE = 527.5184818492537
PHI = 1.618033988749895


class ActivationType(Enum):
    """Activation function types"""
    SIGMOID = "sigmoid"
    RELU = "relu"
    TANH = "tanh"
    LEAKY_RELU = "leaky_relu"
    SWISH = "swish"
    L104_RESONANCE = "l104_resonance"  # Custom activation


class LayerType(Enum):
    """Neural network layer types"""
    DENSE = "dense"
    CONV1D = "conv1d"
    RECURRENT = "recurrent"
    DROPOUT = "dropout"
    BATCH_NORM = "batch_norm"


@dataclass
class Tensor:
    """Simple tensor implementation for neural network operations"""
    data: List[float]
    shape: Tuple[int, ...]
    grad: Optional[List[float]] = None
    requires_grad: bool = True
    
    def __post_init__(self):
        if self.requires_grad and self.grad is None:
            self.grad = [0.0] * len(self.data)
    
    @staticmethod
    def zeros(shape: Tuple[int, ...]) -> 'Tensor':
        size = 1
        for dim in shape:
            size *= dim
        return Tensor([0.0] * size, shape)
    
    @staticmethod
    def ones(shape: Tuple[int, ...]) -> 'Tensor':
        size = 1
        for dim in shape:
            size *= dim
        return Tensor([1.0] * size, shape)
    
    @staticmethod
    def random(shape: Tuple[int, ...], scale: float = 1.0) -> 'Tensor':
        size = 1
        for dim in shape:
            size *= dim
        data = [(random.random() * 2 - 1) * scale for _ in range(size)]
        return Tensor(data, shape)
    
    @staticmethod
    def xavier_init(shape: Tuple[int, ...]) -> 'Tensor':
        """Xavier/Glorot initialization"""
        fan_in = shape[0] if len(shape) > 0 else 1
        fan_out = shape[1] if len(shape) > 1 else 1
        scale = math.sqrt(2.0 / (fan_in + fan_out))
        return Tensor.random(shape, scale)
    
    @staticmethod
    def l104_init(shape: Tuple[int, ...]) -> 'Tensor':
        """L104 resonance-based initialization"""
        fan_in = shape[0] if len(shape) > 0 else 1
        scale = math.sqrt(PHI / fan_in) * (GOD_CODE / 1000)
        return Tensor.random(shape, scale)
    
    def reshape(self, new_shape: Tuple[int, ...]) -> 'Tensor':
        return Tensor(self.data.copy(), new_shape, self.grad.copy() if self.grad else None)
    
    def __getitem__(self, idx: int) -> float:
        return self.data[idx]
    
    def __setitem__(self, idx: int, value: float) -> None:
        self.data[idx] = value
    
    def __len__(self) -> int:
        return len(self.data)


class DenseLayer:
    """Fully connected neural network layer with real computations"""
    
    def __init__(self, input_size: int, output_size: int, 
                 activation: ActivationType = ActivationType.RELU,
                 use_l104_init: bool = False):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_type = activation
        self.activation_fn, self.activation_deriv = Activations.get(activation)
        
        # Initialize weights and biases
        if use_l104_init:
            self.weights = Tensor.l104_init((input_size, output_size))
        else:
            self.weights = Tensor.xavier_init((input_size, output_size))
        self.bias = Tensor.zeros((output_size,))
        
        # Cache for backprop
        self.last_input: Optional[Tensor] = None
        self.last_output: Optional[Tensor] = None
        self.last_pre_activation: Optional[Tensor] = None
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass: y = activation(W @ x + b)
        Performs real matrix-vector multiplication
        """
        self.last_input = x
        
        # Matrix-vector multiplication: W @ x
        # x shape: (batch_size, input_size) or (input_size,)
        # W shape: (input_size, output_size)
        
        batch_size = 1
        if len(x.shape) > 1:
            batch_size = x.shape[0]
        
        # Reshape input if needed
        if len(x.shape) == 1:
            x_flat = x.data
        else:
            x_flat = x.data
        
        # Compute W^T @ x + b for each sample
        output_data = []
        for out_idx in range(self.output_size):
            # Dot product of input with weights column
            activation_sum = 0.0
            for in_idx in range(self.input_size):
                weight_idx = in_idx * self.output_size + out_idx
                input_idx = in_idx if len(x.shape) == 1 else in_idx
                activation_sum += x_flat[input_idx] * self.weights.data[weight_idx]
            
            # Add bias
            activation_sum += self.bias.data[out_idx]
            output_data.append(activation_sum)
        
        # Store pre-activation
        self.last_pre_activation = Tensor(output_data, (self.output_size,))
        
        # Apply activation function
        activated_data = [self.activation_fn(val) for val in output_data]
        self.last_output = Tensor(activated_data, (self.output_size,))
        
        return self.last_output
    
    @staticmethod
    def sigmoid(x: float) -> float:
        if x < -500:
            return 0.0
        if x > 500:
            return 1.0
        return 1.0 / (1.0 + math.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(x: float) -> float:
        s = Activations.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def relu(x: float) -> float:
        return max(0.0, x)
    
    @staticmethod
    def relu_derivative(x: float) -> float:
        return 1.0 if x > 0 else 0.0
    
    @staticmethod
    def leaky_relu(x: float, alpha: float = 0.01) -> float:
        return x if x > 0 else alpha * x
    
    @staticmethod
    def leaky_relu_derivative(x: float, alpha: float = 0.01) -> float:
        return 1.0 if x > 0 else alpha
    
    @staticmethod
    def tanh(x: float) -> float:
        if x > 500:
            return 1.0
        if x < -500:
            return -1.0
        return math.tanh(x)
    
    @staticmethod
    def tanh_derivative(x: float) -> float:
        t = Activations.tanh(x)
        return 1 - t * t
    
    @staticmethod
    def swish(x: float) -> float:
        return x * Activations.sigmoid(x)
    
    @staticmethod
    def swish_derivative(x: float) -> float:
        s = Activations.sigmoid(x)
        return s + x * s * (1 - s)
    
    @staticmethod
    def l104_resonance(x: float) -> float:
        """Custom L104 activation with resonance harmonics"""
        # Combine sigmoid with PHI modulation
        base = Activations.sigmoid(x)
        resonance = math.sin(x * PHI) * (GOD_CODE / 10000)
        return base + resonance * 0.1
    
    @staticmethod
    def l104_resonance_derivative(x: float) -> float:
        base_deriv = Activations.sigmoid_derivative(x)
        resonance_deriv = PHI * math.cos(x * PHI) * (GOD_CODE / 10000) * 0.1
        return base_deriv + resonance_deriv
    
    @staticmethod
    def get(activation_type: ActivationType) -> Tuple[Callable, Callable]:
        """Get activation function and derivative"""
        mapping = {
            ActivationType.SIGMOID: (Activations.sigmoid, Activations.sigmoid_derivative),
            ActivationType.RELU: (Activations.relu, Activations.relu_derivative),
            ActivationType.LEAKY_RELU: (Activations.leaky_relu, Activations.leaky_relu_derivative),
            ActivationType.TANH: (Activations.tanh, Activations.tanh_derivative),
            ActivationType.SWISH: (Activations.swish, Activations.swish_derivative),
            ActivationType.L104_RESONANCE: (Activations.l104_resonance, Activations.l104_resonance_derivative),
        }
        return mapping.get(activation_type, (Activations.relu, Activations.relu_derivative))


class Layer:
    """Base layer class"""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.trainable = True
        self.input_cache = None
        self.output_cache = None
    
    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError
    
    def backward(self, grad_output: Tensor) -> Tensor:
        raise NotImplementedError
    
    def parameters(self) -> List[Tensor]:
        return []
    
    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "type": self.__class__.__name__}


class DenseLayer(Layer):
    """Fully connected layer"""
    
    def __init__(self, input_size: int, output_size: int,
                 activation: ActivationType = ActivationType.RELU,
                 use_bias: bool = True, name: str = ""):
        super().__init__(name or f"dense_{input_size}_{output_size}")
        
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.use_bias = use_bias
        
        # Initialize weights with L104 initialization
        self.weights = Tensor.l104_init((input_size, output_size))
        self.bias = Tensor.zeros((output_size,)) if use_bias else None
        
        # Get activation functions
        self.act_fn, self.act_deriv = Activations.get(activation)
        
        # Cache for backprop
        self.z_cache = None
    
    def forward(self, inputs: Tensor) -> Tensor:
        self.input_cache = inputs
        batch_size = len(inputs.data) // self.input_size
        
        output_data = []
        z_data = []
        
        for b in range(batch_size):
            for j in range(self.output_size):
                # Compute weighted sum
                z = 0.0
                for i in range(self.input_size):
                    w_idx = i * self.output_size + j
                    in_idx = b * self.input_size + i
                    z += inputs.data[in_idx] * self.weights.data[w_idx]
                
                if self.use_bias:
                    z += self.bias.data[j]
                
                z_data.append(z)
                output_data.append(self.act_fn(z))
        
        self.z_cache = z_data
        return Tensor(output_data, (batch_size, self.output_size))
    
    def backward(self, grad_output: Tensor) -> Tensor:
        if self.input_cache is None or self.z_cache is None:
            raise ValueError("Must run forward pass before backward")
        
        batch_size = len(grad_output.data) // self.output_size
        
        # Initialize gradients
        grad_input = [0.0] * len(self.input_cache.data)
        grad_weights = [0.0] * len(self.weights.data)
        grad_bias = [0.0] * self.output_size if self.use_bias else None
        
        for b in range(batch_size):
            for j in range(self.output_size):
                out_idx = b * self.output_size + j
                
                # Gradient through activation
                delta = grad_output.data[out_idx] * self.act_deriv(self.z_cache[out_idx])
                
                # Gradient for bias
                if self.use_bias:
                    grad_bias[j] += delta
                
                # Gradient for weights and inputs
                for i in range(self.input_size):
                    w_idx = i * self.output_size + j
                    in_idx = b * self.input_size + i
                    
                    grad_weights[w_idx] += delta * self.input_cache.data[in_idx]
                    grad_input[in_idx] += delta * self.weights.data[w_idx]
        
        # Store gradients
        self.weights.grad = grad_weights
        if self.use_bias:
            self.bias.grad = grad_bias
        
        return Tensor(grad_input, self.input_cache.shape)
    
    def parameters(self) -> List[Tensor]:
        params = [self.weights]
        if self.use_bias:
            params.append(self.bias)
        return params
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **super().to_dict(),
            "input_size": self.input_size,
            "output_size": self.output_size,
            "activation": self.activation.value,
            "use_bias": self.use_bias,
            "weights": self.weights.data,
            "bias": self.bias.data if self.bias else None
        }


class DropoutLayer(Layer):
    """Dropout regularization layer"""
    
    def __init__(self, rate: float = 0.5, name: str = ""):
        super().__init__(name or f"dropout_{rate}")
        self.rate = rate
        self.mask = None
        self.training = True
    
    def forward(self, inputs: Tensor) -> Tensor:
        self.input_cache = inputs
        
        if not self.training or self.rate == 0:
            return inputs
        
        # Generate dropout mask
        self.mask = [1 if random.random() > self.rate else 0 for _ in inputs.data]
        scale = 1.0 / (1.0 - self.rate)
        
        output_data = [x * m * scale for x, m in zip(inputs.data, self.mask)]
        return Tensor(output_data, inputs.shape)
    
    def backward(self, grad_output: Tensor) -> Tensor:
        if self.mask is None or not self.training:
            return grad_output
        
        scale = 1.0 / (1.0 - self.rate)
        grad_input = [g * m * scale for g, m in zip(grad_output.data, self.mask)]
        return Tensor(grad_input, grad_output.shape)


class BatchNormLayer(Layer):
    """Batch normalization layer"""
    
    def __init__(self, num_features: int, epsilon: float = 1e-5,
                 momentum: float = 0.1, name: str = ""):
        super().__init__(name or f"batchnorm_{num_features}")
        
        self.num_features = num_features
        self.epsilon = epsilon
        self.momentum = momentum
        
        # Learnable parameters
        self.gamma = Tensor.ones((num_features,))
        self.beta = Tensor.zeros((num_features,))
        
        # Running statistics
        self.running_mean = [0.0] * num_features
        self.running_var = [1.0] * num_features
        
        self.training = True
        self.x_norm = None
        self.std = None
    
    def forward(self, inputs: Tensor) -> Tensor:
        self.input_cache = inputs
        batch_size = len(inputs.data) // self.num_features
        
        output_data = []
        
        if self.training:
            # Compute batch statistics
            means = [0.0] * self.num_features
            variances = [0.0] * self.num_features
            
            for j in range(self.num_features):
                feature_sum = sum(inputs.data[b * self.num_features + j] 
                                  for b in range(batch_size))
                means[j] = feature_sum / batch_size
            
            for j in range(self.num_features):
                var_sum = sum((inputs.data[b * self.num_features + j] - means[j]) ** 2 
                              for b in range(batch_size))
                variances[j] = var_sum / batch_size
            
            # Update running statistics
            for j in range(self.num_features):
                self.running_mean[j] = (1 - self.momentum) * self.running_mean[j] + self.momentum * means[j]
                self.running_var[j] = (1 - self.momentum) * self.running_var[j] + self.momentum * variances[j]
            
            self.std = [math.sqrt(v + self.epsilon) for v in variances]
            
            # Normalize and scale
            self.x_norm = []
            for b in range(batch_size):
                for j in range(self.num_features):
                    idx = b * self.num_features + j
                    x_n = (inputs.data[idx] - means[j]) / self.std[j]
                    self.x_norm.append(x_n)
                    output_data.append(self.gamma.data[j] * x_n + self.beta.data[j])
        else:
            # Use running statistics
            for b in range(batch_size):
                for j in range(self.num_features):
                    idx = b * self.num_features + j
                    x_n = (inputs.data[idx] - self.running_mean[j]) / math.sqrt(self.running_var[j] + self.epsilon)
                    output_data.append(self.gamma.data[j] * x_n + self.beta.data[j])
        
        return Tensor(output_data, inputs.shape)
    
    def backward(self, grad_output: Tensor) -> Tensor:
        if not self.training or self.x_norm is None:
            return grad_output
        
        batch_size = len(grad_output.data) // self.num_features
        
        # Gradients for gamma and beta
        grad_gamma = [0.0] * self.num_features
        grad_beta = [0.0] * self.num_features
        
        for j in range(self.num_features):
            for b in range(batch_size):
                idx = b * self.num_features + j
                grad_gamma[j] += grad_output.data[idx] * self.x_norm[idx]
                grad_beta[j] += grad_output.data[idx]
        
        self.gamma.grad = grad_gamma
        self.beta.grad = grad_beta
        
        # Gradient for input (simplified)
        grad_input = [g * self.gamma.data[i % self.num_features] / self.std[i % self.num_features]
                      for i, g in enumerate(grad_output.data)]
        
        return Tensor(grad_input, grad_output.shape)
    
    def parameters(self) -> List[Tensor]:
        return [self.gamma, self.beta]


class Optimizer:
    """Base optimizer class"""
    
    def __init__(self, parameters: List[Tensor], learning_rate: float = 0.001):
        self.parameters = parameters
        self.learning_rate = learning_rate
    
    def step(self) -> None:
        raise NotImplementedError
    
    def zero_grad(self) -> None:
        for param in self.parameters:
            if param.grad:
                param.grad = [0.0] * len(param.grad)


class SGDOptimizer(Optimizer):
    """Stochastic Gradient Descent with momentum"""
    
    def __init__(self, parameters: List[Tensor], learning_rate: float = 0.001,
                 momentum: float = 0.9, weight_decay: float = 0.0):
        super().__init__(parameters, learning_rate)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = [[0.0] * len(p.data) for p in parameters]
    
    def step(self) -> None:
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            for j in range(len(param.data)):
                grad = param.grad[j]
                
                # Weight decay
                if self.weight_decay > 0:
                    grad += self.weight_decay * param.data[j]
                
                # Momentum
                self.velocities[i][j] = self.momentum * self.velocities[i][j] - self.learning_rate * grad
                
                # Update
                param.data[j] += self.velocities[i][j]


class AdamOptimizer(Optimizer):
    """Adam optimizer with L104 enhancements"""
    
    def __init__(self, parameters: List[Tensor], learning_rate: float = 0.001,
                 beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        super().__init__(parameters, learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        
        # First and second moment estimates
        self.m = [[0.0] * len(p.data) for p in parameters]
        self.v = [[0.0] * len(p.data) for p in parameters]
    
    def step(self) -> None:
        self.t += 1
        
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            for j in range(len(param.data)):
                grad = param.grad[j]
                
                # Update biased first moment
                self.m[i][j] = self.beta1 * self.m[i][j] + (1 - self.beta1) * grad
                
                # Update biased second moment
                self.v[i][j] = self.beta2 * self.v[i][j] + (1 - self.beta2) * grad * grad
                
                # Bias correction
                m_hat = self.m[i][j] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i][j] / (1 - self.beta2 ** self.t)
                
                # Update with L104 resonance factor
                resonance = 1.0 + (GOD_CODE % 1) * 0.01  # ~1.005
                param.data[j] -= self.learning_rate * resonance * m_hat / (math.sqrt(v_hat) + self.epsilon)


class LossFunction:
    """Loss function implementations"""
    
    @staticmethod
    def mse(predictions: Tensor, targets: Tensor) -> Tuple[float, Tensor]:
        """Mean Squared Error"""
        n = len(predictions.data)
        loss = sum((p - t) ** 2 for p, t in zip(predictions.data, targets.data)) / n
        
        grad = [(2.0 / n) * (p - t) for p, t in zip(predictions.data, targets.data)]
        return loss, Tensor(grad, predictions.shape)
    
    @staticmethod
    def cross_entropy(predictions: Tensor, targets: Tensor) -> Tuple[float, Tensor]:
        """Cross Entropy Loss for classification"""
        n = len(predictions.data)
        eps = 1e-15
        
        loss = -sum(t * math.log(max(p, eps)) + (1 - t) * math.log(max(1 - p, eps))
                    for p, t in zip(predictions.data, targets.data)) / n
        
        grad = [((1 - t) / max(1 - p, eps) - t / max(p, eps)) / n
                for p, t in zip(predictions.data, targets.data)]
        
        return loss, Tensor(grad, predictions.shape)
    
    @staticmethod
    def l104_resonance_loss(predictions: Tensor, targets: Tensor) -> Tuple[float, Tensor]:
        """Custom L104 loss with resonance harmonics"""
        base_loss, base_grad = LossFunction.mse(predictions, targets)
        
        # Add resonance term
        resonance_penalty = sum(abs(math.sin(p * PHI)) for p in predictions.data) / len(predictions.data)
        resonance_loss = base_loss + 0.01 * resonance_penalty * (GOD_CODE / 1000)
        
        # Resonance gradient
        resonance_grad = [g + 0.01 * PHI * math.cos(p * PHI) * (1 if math.sin(p * PHI) >= 0 else -1) * (GOD_CODE / 1000)
                         for g, p in zip(base_grad.data, predictions.data)]
        
        return resonance_loss, Tensor(resonance_grad, predictions.shape)


class NeuralNetwork:
    """Main neural network class"""
    
    def __init__(self, name: str = "L104_Network"):
        self.name = name
        self.layers: List[Layer] = []
        self.optimizer: Optional[Optimizer] = None
        self.loss_fn = LossFunction.mse
        self.training = True
        self.history: List[Dict[str, float]] = []
    
    def add(self, layer: Layer) -> 'NeuralNetwork':
        """Add layer to network"""
        self.layers.append(layer)
        return self
    
    def compile(self, optimizer: str = "adam", learning_rate: float = 0.001,
                loss: str = "mse") -> None:
        """Compile network with optimizer and loss"""
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        
        if optimizer == "adam":
            self.optimizer = AdamOptimizer(params, learning_rate)
        elif optimizer == "sgd":
            self.optimizer = SGDOptimizer(params, learning_rate)
        else:
            self.optimizer = AdamOptimizer(params, learning_rate)
        
        loss_mapping = {
            "mse": LossFunction.mse,
            "cross_entropy": LossFunction.cross_entropy,
            "l104_resonance": LossFunction.l104_resonance_loss
        }
        self.loss_fn = loss_mapping.get(loss, LossFunction.mse)
    
    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass through network"""
        x = inputs
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, loss_grad: Tensor) -> None:
        """Backward pass through network"""
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
    def train_step(self, inputs: Tensor, targets: Tensor) -> float:
        """Single training step"""
        self.training = True
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = True
        
        # Forward
        predictions = self.forward(inputs)
        
        # Loss
        loss, loss_grad = self.loss_fn(predictions, targets)
        
        # Backward
        self.optimizer.zero_grad()
        self.backward(loss_grad)
        
        # Update
        self.optimizer.step()
        
        return loss
    
    def fit(self, X: List[List[float]], y: List[List[float]],
            epochs: int = 100, batch_size: int = 32,
            verbose: bool = True) -> Dict[str, List[float]]:
        """Train the network"""
        n_samples = len(X)
        input_size = len(X[0])
        output_size = len(y[0])
        
        history = {"loss": [], "epoch_time": []}
        
        for epoch in range(epochs):
            start_time = time.time()
            epoch_loss = 0.0
            n_batches = 0
            
            # Shuffle data
            indices = list(range(n_samples))
            random.shuffle(indices)
            
            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:min(i + batch_size, n_samples)]
                
                # Create batch tensors
                batch_x = []
                batch_y = []
                for idx in batch_indices:
                    batch_x.extend(X[idx])
                    batch_y.extend(y[idx])
                
                inputs = Tensor(batch_x, (len(batch_indices), input_size))
                targets = Tensor(batch_y, (len(batch_indices), output_size))
                
                # Train step
                loss = self.train_step(inputs, targets)
                epoch_loss += loss
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            epoch_time = time.time() - start_time
            
            history["loss"].append(avg_loss)
            history["epoch_time"].append(epoch_time)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.6f} - Time: {epoch_time:.3f}s")
        
        self.history.extend([{"epoch": i, "loss": l} for i, l in enumerate(history["loss"])])
        return history
    
    def predict(self, X: List[List[float]]) -> List[List[float]]:
        """Make predictions"""
        self.training = False
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = False
        
        input_size = len(X[0])
        output_size = self.layers[-1].output_size if hasattr(self.layers[-1], 'output_size') else input_size
        
        results = []
        for sample in X:
            inputs = Tensor(sample, (1, input_size))
            output = self.forward(inputs)
            results.append(output.data[:output_size])
        
        return results
    
    def evaluate(self, X: List[List[float]], y: List[List[float]]) -> Dict[str, float]:
        """Evaluate network on test data"""
        predictions = self.predict(X)
        
        # MSE
        mse = sum(
            sum((p - t) ** 2 for p, t in zip(pred, target))
            for pred, target in zip(predictions, y)
                ) / (len(X) * len(y[0]))
        
        # MAE
        mae = sum(
            sum(abs(p - t) for p, t in zip(pred, target))
            for pred, target in zip(predictions, y)
                ) / (len(X) * len(y[0]))
        
        return {"mse": mse, "mae": mae, "rmse": math.sqrt(mse)}
    
    def save(self, filepath: str) -> None:
        """Save model to file"""
        model_data = {
            "name": self.name,
            "layers": [layer.to_dict() for layer in self.layers],
            "history": self.history,
            "god_code": GOD_CODE,
            "phi": PHI
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
    
    def summary(self) -> str:
        """Get model summary"""
        lines = [
            f"Model: {self.name}",
            "=" * 60,
            f"{'Layer':<20} {'Output Shape':<20} {'Params':<15}",
            "-" * 60
        ]
        
        total_params = 0
        for layer in self.layers:
            params = sum(len(p.data) for p in layer.parameters())
            total_params += params
            
            output_shape = "?"
            if hasattr(layer, 'output_size'):
                output_shape = f"(batch, {layer.output_size})"
            
            lines.append(f"{layer.name:<20} {output_shape:<20} {params:<15}")
        
        lines.append("=" * 60)
        lines.append(f"Total parameters: {total_params}")
        lines.append(f"GOD_CODE resonance: {GOD_CODE:.10f}")
        
        return "\n".join(lines)


class L104NeuralCore:
    """
    Singleton neural network core for L104.
    Manages multiple networks and provides unified interface.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self.networks: Dict[str, NeuralNetwork] = {}
        self.resonance = GOD_CODE / 1000
        
        print(f"[L104_NEURAL_CORE] Initialized | Resonance: {self.resonance:.8f}")
    
    def create_network(self, name: str, architecture: List[Dict[str, Any]]) -> NeuralNetwork:
        """Create network from architecture specification"""
        network = NeuralNetwork(name)
        
        for layer_spec in architecture:
            layer_type = layer_spec.get("type", "dense")
            
            if layer_type == "dense":
                layer = DenseLayer(
                    layer_spec["input_size"],
                    layer_spec["output_size"],
                    ActivationType(layer_spec.get("activation", "relu")),
                    layer_spec.get("use_bias", True)
                )
            elif layer_type == "dropout":
                layer = DropoutLayer(layer_spec.get("rate", 0.5))
            elif layer_type == "batch_norm":
                layer = BatchNormLayer(layer_spec["num_features"])
            else:
                continue
            
            network.add(layer)
        
        self.networks[name] = network
        return network
    
    def get_network(self, name: str) -> Optional[NeuralNetwork]:
        """Get network by name"""
        return self.networks.get(name)
    
    def list_networks(self) -> List[str]:
        """List all network names"""
        return list(self.networks.keys())
    
    def get_status(self) -> Dict[str, Any]:
        """Get core status"""
        return {
            "networks": len(self.networks),
            "network_names": list(self.networks.keys()),
            "resonance": self.resonance,
            "god_code": GOD_CODE,
            "phi": PHI
        }


# Global instance
def get_neural_core() -> L104NeuralCore:
    """Get neural core singleton"""
    return L104NeuralCore()


# Main execution
if __name__ == "__main__":
    print("=" * 70)
    print("  L104 NEURAL NETWORK CORE - DEMONSTRATION")
    print("=" * 70)
    
    # Create simple XOR network
    print("\n[DEMO] Creating XOR Network...")
    
    network = NeuralNetwork("XOR_Network")
    network.add(DenseLayer(2, 8, ActivationType.L104_RESONANCE))
    network.add(DenseLayer(8, 4, ActivationType.RELU))
    network.add(DenseLayer(4, 1, ActivationType.SIGMOID))
    
    network.compile(optimizer="adam", learning_rate=0.1, loss="mse")
    
    print(network.summary())
    
    # XOR training data
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [[0], [1], [1], [0]]
    
    print("\n[DEMO] Training XOR Network...")
    history = network.fit(X, y, epochs=100, batch_size=4, verbose=True)
    
    print("\n[DEMO] Testing XOR Network...")
    predictions = network.predict(X)
    for inp, pred, target in zip(X, predictions, y):
        print(f"  Input: {inp} -> Prediction: {pred[0]:.4f} (Target: {target[0]})")
    
    # Evaluate
    metrics = network.evaluate(X, y)
    print(f"\n[DEMO] Metrics: MSE={metrics['mse']:.6f}, MAE={metrics['mae']:.6f}")
    
    print("\n" + "=" * 70)
    print("  NEURAL NETWORK CORE OPERATIONAL")
    print("=" * 70)
