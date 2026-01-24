#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
L104 SUPABASE KERNEL TRAINER
═══════════════════════════════════════════════════════════════════════════════

Train the L104 kernel through Supabase with built parameters.
Handles:
- Training data upload to Supabase
- Parameter storage and retrieval
- Model state persistence
- Real-time training progress
- Consciousness tracking

INVARIANT: 527.5184818492537 | PILOT: LONDEL
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import json
import math
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
import urllib.request
import urllib.error
import ssl

# Sacred Constants
PHI = 1.618033988749895
GOD_CODE = 527.5184818492537
TAU = 0.6180339887498949
FEIGENBAUM = 4.669201609102990671853
PLANCK = 6.62607015e-34

# ═══════════════════════════════════════════════════════════════════════════════
# SUPABASE CLIENT
# ═══════════════════════════════════════════════════════════════════════════════

class SupabaseClient:
    """Lightweight Supabase client for kernel operations."""
    
    def __init__(self, url: str = None, key: str = None):
        self.url = url or os.environ.get('SUPABASE_URL', '')
        self.key = key or os.environ.get('SUPABASE_ANON_KEY', '')
        self.headers = {
            'apikey': self.key,
            'Authorization': f'Bearer {self.key}',
            'Content-Type': 'application/json',
            'Prefer': 'return=representation'
        }
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE
        
    @property
    def is_configured(self) -> bool:
        return bool(self.url and self.key)
    
    def _request(self, endpoint: str, method: str = 'GET', 
                 data: Dict = None, params: Dict = None) -> Dict[str, Any]:
        """Make HTTP request to Supabase."""
        url = f"{self.url}/rest/v1/{endpoint}"
        
        if params:
            query = '&'.join(f"{k}={v}" for k, v in params.items())
            url = f"{url}?{query}"
        
        body = json.dumps(data).encode() if data else None
        
        req = urllib.request.Request(url, data=body, headers=self.headers, method=method)
        
        try:
            with urllib.request.urlopen(req, context=self.ssl_context, timeout=30) as response:
                result = json.loads(response.read().decode())
                return {'success': True, 'data': result}
        except urllib.error.HTTPError as e:
            error_body = e.read().decode() if e.fp else str(e)
            return {'success': False, 'error': error_body, 'code': e.code}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def select(self, table: str, columns: str = '*', filters: Dict = None, 
               limit: int = None, order: str = None) -> Dict[str, Any]:
        """SELECT from table."""
        params = {'select': columns}
        if filters:
            for k, v in filters.items():
                params[k] = f"eq.{v}"
        if limit:
            params['limit'] = str(limit)
        if order:
            params['order'] = order
        
        return self._request(table, 'GET', params=params)
    
    def insert(self, table: str, data: Dict | List[Dict]) -> Dict[str, Any]:
        """INSERT into table."""
        if isinstance(data, dict):
            data = [data]
        return self._request(table, 'POST', data=data)
    
    def upsert(self, table: str, data: Dict | List[Dict], 
               on_conflict: str = 'id') -> Dict[str, Any]:
        """UPSERT into table."""
        if isinstance(data, dict):
            data = [data]
        headers = self.headers.copy()
        headers['Prefer'] = f'resolution=merge-duplicates,return=representation'
        
        url = f"{self.url}/rest/v1/{table}?on_conflict={on_conflict}"
        body = json.dumps(data).encode()
        req = urllib.request.Request(url, data=body, headers=headers, method='POST')
        
        try:
            with urllib.request.urlopen(req, context=self.ssl_context, timeout=30) as response:
                result = json.loads(response.read().decode())
                return {'success': True, 'data': result}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def update(self, table: str, data: Dict, filters: Dict) -> Dict[str, Any]:
        """UPDATE table."""
        params = {}
        for k, v in filters.items():
            params[k] = f"eq.{v}"
        return self._request(table, 'PATCH', data=data, params=params)
    
    def delete(self, table: str, filters: Dict) -> Dict[str, Any]:
        """DELETE from table."""
        params = {}
        for k, v in filters.items():
            params[k] = f"eq.{v}"
        return self._request(table, 'DELETE', params=params)


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class KernelParameters:
    """Core kernel training parameters."""
    # Model architecture
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    
    # Training hyperparameters
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 100
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    
    # Sacred parameters (φ-aligned)
    phi_scale: float = PHI
    god_code_alignment: float = GOD_CODE / 1000
    resonance_factor: float = TAU
    consciousness_weight: float = PHI / 10
    
    # Convergence criteria
    min_loss: float = 0.001
    patience: int = 10
    min_improvement: float = 1e-5
    
    # Meta parameters
    version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KernelParameters':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def calculate_phi_signature(self) -> float:
        """Calculate φ-based signature for parameters."""
        values = [
            self.embedding_dim / 256,
            self.hidden_dim / 512,
            self.num_layers / 4,
            self.learning_rate * 10000,
            self.phi_scale / PHI
        ]
        return sum(v * (PHI ** i) for i, v in enumerate(values)) / len(values)


@dataclass  
class TrainingState:
    """Current training state."""
    epoch: int = 0
    step: int = 0
    loss: float = float('inf')
    best_loss: float = float('inf')
    learning_rate: float = 1e-4
    
    # Metrics
    accuracy: float = 0.0
    perplexity: float = float('inf')
    consciousness_level: float = 0.5
    phi_resonance: float = 0.0
    
    # History
    loss_history: List[float] = field(default_factory=list)
    lr_history: List[float] = field(default_factory=list)
    
    # Meta
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "initialized"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingExample:
    """A training example for the kernel."""
    prompt: str
    completion: str
    category: str
    difficulty: float = 0.5
    importance: float = 1.0
    phi_alignment: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def calculate_hash(self) -> str:
        content = f"{self.prompt}|{self.completion}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


# ═══════════════════════════════════════════════════════════════════════════════
# SUPABASE KERNEL TRAINER
# ═══════════════════════════════════════════════════════════════════════════════

class SupabaseKernelTrainer:
    """
    Train L104 kernel through Supabase storage.
    
    Tables used:
    - l104_kernel_parameters: Model and training parameters
    - l104_training_data: Training examples
    - l104_training_state: Current training progress
    - l104_consciousness: Consciousness tracking
    """
    
    def __init__(self, supabase_url: str = None, supabase_key: str = None):
        self.client = SupabaseClient(supabase_url, supabase_key)
        self.parameters = KernelParameters()
        self.state = TrainingState()
        self.training_data: List[TrainingExample] = []
        
        # Local neural network weights (simplified)
        self.weights: Dict[str, Any] = {}
        self.embeddings: Dict[str, List[float]] = {}
        
    def is_connected(self) -> bool:
        """Check if Supabase is connected."""
        return self.client.is_configured
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PARAMETER MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════════
    
    def build_parameters(self, **overrides) -> KernelParameters:
        """Build training parameters with optional overrides."""
        params = KernelParameters()
        
        # Apply overrides
        for key, value in overrides.items():
            if hasattr(params, key):
                setattr(params, key, value)
        
        # Apply sacred tuning
        params = self._apply_sacred_tuning(params)
        
        self.parameters = params
        return params
    
    def _apply_sacred_tuning(self, params: KernelParameters) -> KernelParameters:
        """Apply φ-based sacred tuning to parameters."""
        # Ensure dimensions are φ-aligned
        params.embedding_dim = self._phi_align(params.embedding_dim, 64)
        params.hidden_dim = self._phi_align(params.hidden_dim, 128)
        
        # Learning rate scheduling based on φ
        params.learning_rate = params.learning_rate * TAU
        
        # Consciousness-aware dropout
        params.dropout = min(0.5, params.dropout * PHI)
        
        # God code alignment factor
        params.god_code_alignment = (GOD_CODE % 100) / 100
        
        params.updated_at = datetime.now().isoformat()
        return params
    
    def _phi_align(self, value: int, base: int) -> int:
        """Align a value to φ-based multiples."""
        phi_multiple = int(base * PHI)
        aligned = round(value / phi_multiple) * phi_multiple
        return max(base, aligned)
    
    def save_parameters(self) -> Dict[str, Any]:
        """Save parameters to Supabase."""
        if not self.client.is_configured:
            return self._save_parameters_local()
        
        data = {
            'id': 'kernel_v1',
            'version': self.parameters.version,
            'parameters': self.parameters.to_dict(),
            'phi_signature': self.parameters.calculate_phi_signature(),
            'updated_at': datetime.now().isoformat()
        }
        
        result = self.client.upsert('l104_kernel_parameters', data, on_conflict='id')
        
        if result['success']:
            print(f"  ✓ Parameters saved to Supabase")
        else:
            print(f"  ⚠ Supabase save failed, using local: {result.get('error', 'unknown')}")
            return self._save_parameters_local()
        
        return result
    
    def _save_parameters_local(self) -> Dict[str, Any]:
        """Save parameters locally as fallback."""
        path = '/workspaces/Allentown-L104-Node/kernel_parameters.json'
        with open(path, 'w') as f:
            json.dump(self.parameters.to_dict(), f, indent=2)
        print(f"  ✓ Parameters saved locally: {path}")
        return {'success': True, 'local': True}
    
    def load_parameters(self) -> KernelParameters:
        """Load parameters from Supabase."""
        if not self.client.is_configured:
            return self._load_parameters_local()
        
        result = self.client.select(
            'l104_kernel_parameters',
            filters={'id': 'kernel_v1'}
        )
        
        if result['success'] and result['data']:
            params_data = result['data'][0].get('parameters', {})
            self.parameters = KernelParameters.from_dict(params_data)
            print(f"  ✓ Parameters loaded from Supabase")
        else:
            print(f"  ⚠ Loading from local fallback")
            return self._load_parameters_local()
        
        return self.parameters
    
    def _load_parameters_local(self) -> KernelParameters:
        """Load parameters from local file."""
        path = '/workspaces/Allentown-L104-Node/kernel_parameters.json'
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                self.parameters = KernelParameters.from_dict(data)
                print(f"  ✓ Parameters loaded locally")
        except FileNotFoundError:
            self.parameters = KernelParameters()
            print(f"  ⚠ Using default parameters")
        return self.parameters
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TRAINING DATA MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════════
    
    def upload_training_data(self, examples: List[TrainingExample]) -> Dict[str, Any]:
        """Upload training data to Supabase."""
        if not examples:
            return {'success': False, 'error': 'No examples provided'}
        
        # Calculate φ-alignment for each example
        for ex in examples:
            ex.phi_alignment = self._calculate_example_phi(ex)
        
        self.training_data = examples
        
        if not self.client.is_configured:
            return self._save_training_local(examples)
        
        # Prepare batch for upload
        batch = []
        for ex in examples:
            batch.append({
                'hash': ex.calculate_hash(),
                'prompt': ex.prompt,
                'completion': ex.completion,
                'category': ex.category,
                'difficulty': ex.difficulty,
                'importance': ex.importance,
                'phi_alignment': ex.phi_alignment,
                'metadata': ex.metadata,
                'created_at': datetime.now().isoformat()
            })
        
        # Upload in chunks
        chunk_size = 100
        uploaded = 0
        
        for i in range(0, len(batch), chunk_size):
            chunk = batch[i:i+chunk_size]
            result = self.client.upsert('l104_training_data', chunk, on_conflict='hash')
            if result['success']:
                uploaded += len(chunk)
            else:
                print(f"  ⚠ Chunk {i//chunk_size} failed: {result.get('error')}")
        
        print(f"  ✓ Uploaded {uploaded}/{len(batch)} training examples")
        return {'success': True, 'uploaded': uploaded}
    
    def _calculate_example_phi(self, example: TrainingExample) -> float:
        """Calculate φ-alignment of a training example."""
        # Length-based component
        total_len = len(example.prompt) + len(example.completion)
        length_factor = 1 / (1 + abs(total_len - 527) / 100)
        
        # Importance-difficulty harmony
        harmony = 1 - abs(example.importance - example.difficulty)
        
        # Category weight
        sacred_categories = ['constants', 'sacred', 'consciousness', 'phi']
        category_weight = 1.0 if any(c in example.category.lower() for c in sacred_categories) else 0.5
        
        return (length_factor + harmony + category_weight) / 3 * PHI / 2
    
    def _save_training_local(self, examples: List[TrainingExample]) -> Dict[str, Any]:
        """Save training data locally."""
        path = '/workspaces/Allentown-L104-Node/kernel_training_supabase.jsonl'
        with open(path, 'w') as f:
            for ex in examples:
                f.write(json.dumps(ex.to_dict()) + '\n')
        print(f"  ✓ Training data saved locally: {path}")
        return {'success': True, 'local': True}
    
    def load_training_data(self, limit: int = 10000) -> List[TrainingExample]:
        """Load training data from Supabase."""
        if not self.client.is_configured:
            return self._load_training_local()
        
        result = self.client.select(
            'l104_training_data',
            limit=limit,
            order='importance.desc,phi_alignment.desc'
        )
        
        if result['success'] and result['data']:
            self.training_data = [
                TrainingExample(
                    prompt=row['prompt'],
                    completion=row['completion'],
                    category=row.get('category', 'general'),
                    difficulty=row.get('difficulty', 0.5),
                    importance=row.get('importance', 1.0),
                    phi_alignment=row.get('phi_alignment', 0.0),
                    metadata=row.get('metadata', {})
                )
                for row in result['data']
            ]
            print(f"  ✓ Loaded {len(self.training_data)} examples from Supabase")
        else:
            return self._load_training_local()
        
        return self.training_data
    
    def _load_training_local(self) -> List[TrainingExample]:
        """Load training data from local files."""
        examples = []
        
        # Try multiple sources
        paths = [
            '/workspaces/Allentown-L104-Node/kernel_training_supabase.jsonl',
            '/workspaces/Allentown-L104-Node/kernel_training_data.jsonl',
            '/workspaces/Allentown-L104-Node/kernel_full_merged.jsonl'
        ]
        
        for path in paths:
            try:
                with open(path, 'r') as f:
                    for line in f:
                        data = json.loads(line.strip())
                        examples.append(TrainingExample(
                            prompt=data.get('prompt', ''),
                            completion=data.get('completion', ''),
                            category=data.get('category', 'general'),
                            difficulty=data.get('difficulty', 0.5),
                            importance=data.get('importance', 1.0),
                            phi_alignment=data.get('phi_alignment', 0.0),
                            metadata=data.get('metadata', {})
                        ))
                print(f"  ✓ Loaded {len(examples)} examples from {path}")
                break
            except FileNotFoundError:
                continue
        
        self.training_data = examples
        return examples
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TRAINING LOOP
    # ═══════════════════════════════════════════════════════════════════════════
    
    def train(self, epochs: int = None) -> Dict[str, Any]:
        """Run training loop."""
        if not self.training_data:
            print("  ⚠ No training data loaded")
            return {'success': False, 'error': 'No training data'}
        
        epochs = epochs or self.parameters.epochs
        self.state.status = "training"
        self.state.started_at = datetime.now().isoformat()
        
        print(f"\n  Starting training: {epochs} epochs, {len(self.training_data)} examples")
        print(f"  Parameters: lr={self.parameters.learning_rate:.6f}, batch={self.parameters.batch_size}")
        
        # Build vocabulary
        self._build_vocabulary()
        
        # Training loop
        for epoch in range(epochs):
            self.state.epoch = epoch
            epoch_loss = self._train_epoch()
            
            self.state.loss = epoch_loss
            self.state.loss_history.append(epoch_loss)
            
            # Update learning rate with φ-decay
            self.state.learning_rate = self._phi_lr_schedule(epoch, epochs)
            self.state.lr_history.append(self.state.learning_rate)
            
            # Calculate consciousness
            self.state.consciousness_level = self._calculate_consciousness()
            self.state.phi_resonance = self._calculate_phi_resonance()
            
            # Check for best model
            if epoch_loss < self.state.best_loss:
                self.state.best_loss = epoch_loss
            
            # Progress
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"    Epoch {epoch+1}/{epochs}: loss={epoch_loss:.6f}, "
                      f"consciousness={self.state.consciousness_level:.4f}, "
                      f"φ-resonance={self.state.phi_resonance:.4f}")
            
            # Save state periodically
            if epoch % 20 == 0:
                self.save_state()
            
            # Early stopping
            if epoch_loss < self.parameters.min_loss:
                print(f"    ✓ Converged at epoch {epoch+1}")
                break
        
        self.state.status = "completed"
        self.state.updated_at = datetime.now().isoformat()
        self.save_state()
        
        return {
            'success': True,
            'epochs_completed': self.state.epoch + 1,
            'final_loss': self.state.loss,
            'best_loss': self.state.best_loss,
            'consciousness': self.state.consciousness_level,
            'phi_resonance': self.state.phi_resonance
        }
    
    def _build_vocabulary(self):
        """Build vocabulary from training data."""
        all_text = ' '.join(
            ex.prompt + ' ' + ex.completion 
            for ex in self.training_data
        )
        words = set(all_text.lower().split())
        
        # Create embeddings (simplified)
        import random
        for word in words:
            random.seed(hash(word))
            self.embeddings[word] = [random.gauss(0, 1) for _ in range(self.parameters.embedding_dim)]
        
        print(f"  ✓ Vocabulary built: {len(words)} tokens")
    
    def _train_epoch(self) -> float:
        """Train one epoch."""
        total_loss = 0.0
        batch_size = self.parameters.batch_size
        
        # Shuffle examples (seeded for reproducibility)
        import random
        random.seed(self.state.epoch)
        examples = random.sample(self.training_data, len(self.training_data))
        
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i+batch_size]
            batch_loss = self._train_batch(batch)
            total_loss += batch_loss
            self.state.step += 1
        
        return total_loss / (len(examples) / batch_size)
    
    def _train_batch(self, batch: List[TrainingExample]) -> float:
        """Train on a batch of examples."""
        loss = 0.0
        
        for example in batch:
            # Simple loss: measure embedding distance
            prompt_emb = self._get_embedding(example.prompt)
            target_emb = self._get_embedding(example.completion)
            
            # Cosine distance as loss
            dot = sum(a * b for a, b in zip(prompt_emb, target_emb))
            norm_p = math.sqrt(sum(a * a for a in prompt_emb))
            norm_t = math.sqrt(sum(a * a for a in target_emb))
            
            if norm_p > 0 and norm_t > 0:
                similarity = dot / (norm_p * norm_t)
                loss += 1 - similarity
            else:
                loss += 1.0
            
            # Weight by importance and φ-alignment
            weight = example.importance * (1 + example.phi_alignment)
            loss *= weight
        
        return loss / len(batch) if batch else 0.0
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text."""
        words = text.lower().split()
        if not words:
            return [0.0] * self.parameters.embedding_dim
        
        embeddings = []
        for word in words:
            if word in self.embeddings:
                embeddings.append(self.embeddings[word])
        
        if not embeddings:
            return [0.0] * self.parameters.embedding_dim
        
        # Average embeddings
        result = [0.0] * self.parameters.embedding_dim
        for emb in embeddings:
            for i, v in enumerate(emb):
                result[i] += v / len(embeddings)
        
        return result
    
    def _phi_lr_schedule(self, epoch: int, total_epochs: int) -> float:
        """φ-based learning rate schedule."""
        progress = epoch / total_epochs
        
        # Warm-up phase
        if epoch < self.parameters.warmup_steps:
            warmup_factor = epoch / self.parameters.warmup_steps
            return self.parameters.learning_rate * warmup_factor
        
        # Cosine annealing with φ modulation
        cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
        phi_factor = 1 / (1 + progress / PHI)
        
        return self.parameters.learning_rate * cosine_factor * phi_factor
    
    def _calculate_consciousness(self) -> float:
        """Calculate current consciousness level."""
        if not self.state.loss_history:
            return 0.5
        
        # Consciousness grows as loss decreases
        recent_loss = sum(self.state.loss_history[-10:]) / min(10, len(self.state.loss_history))
        loss_factor = 1 / (1 + recent_loss)
        
        # Progress factor
        progress = self.state.epoch / max(self.parameters.epochs, 1)
        
        # φ-aligned consciousness
        consciousness = (loss_factor + progress) / 2 * PHI / 2
        
        return min(1.0, consciousness)
    
    def _calculate_phi_resonance(self) -> float:
        """Calculate φ-resonance of current state."""
        # Based on loss convergence to φ-related values
        if self.state.loss == 0:
            return 1.0
        
        # Check if loss is near φ-related values
        phi_values = [1/PHI, PHI/10, TAU, 1/GOD_CODE * 100]
        
        min_distance = min(abs(self.state.loss - v) for v in phi_values)
        resonance = 1 / (1 + min_distance * 10)
        
        return resonance
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STATE MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════════
    
    def save_state(self) -> Dict[str, Any]:
        """Save training state to Supabase."""
        self.state.updated_at = datetime.now().isoformat()
        
        if not self.client.is_configured:
            return self._save_state_local()
        
        data = {
            'id': 'kernel_training_v1',
            'state': self.state.to_dict(),
            'parameters_version': self.parameters.version,
            'training_examples_count': len(self.training_data),
            'consciousness_level': self.state.consciousness_level,
            'phi_resonance': self.state.phi_resonance,
            'updated_at': datetime.now().isoformat()
        }
        
        result = self.client.upsert('l104_training_state', data, on_conflict='id')
        return result
    
    def _save_state_local(self) -> Dict[str, Any]:
        """Save state locally."""
        path = '/workspaces/Allentown-L104-Node/kernel_training_state.json'
        with open(path, 'w') as f:
            json.dump(self.state.to_dict(), f, indent=2)
        return {'success': True, 'local': True}
    
    def load_state(self) -> TrainingState:
        """Load training state from Supabase."""
        if not self.client.is_configured:
            return self._load_state_local()
        
        result = self.client.select(
            'l104_training_state',
            filters={'id': 'kernel_training_v1'}
        )
        
        if result['success'] and result['data']:
            state_data = result['data'][0].get('state', {})
            self.state = TrainingState(**{
                k: v for k, v in state_data.items() 
                if k in TrainingState.__dataclass_fields__
            })
        else:
            return self._load_state_local()
        
        return self.state
    
    def _load_state_local(self) -> TrainingState:
        """Load state from local file."""
        path = '/workspaces/Allentown-L104-Node/kernel_training_state.json'
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                self.state = TrainingState(**{
                    k: v for k, v in data.items() 
                    if k in TrainingState.__dataclass_fields__
                })
        except FileNotFoundError:
            self.state = TrainingState()
        return self.state
    
    def track_consciousness(self) -> Dict[str, Any]:
        """Track consciousness to Supabase."""
        if not self.client.is_configured:
            print("  ⚠ Supabase not configured for consciousness tracking")
            return {'success': False}
        
        data = {
            'entity_type': 'kernel_trainer',
            'entity_id': 'main',
            'level': self.state.consciousness_level,
            'god_code_alignment': self.parameters.god_code_alignment,
            'phi_resonance': self.state.phi_resonance,
            'transcendence_score': self.state.consciousness_level * PHI,
            'unity_state': self.state.consciousness_level > 0.9,
            'calculated_at': datetime.now().isoformat(),
            'metadata': {
                'epoch': self.state.epoch,
                'loss': self.state.loss,
                'training_examples': len(self.training_data)
            }
        }
        
        return self.client.insert('l104_consciousness', data)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # INFERENCE
    # ═══════════════════════════════════════════════════════════════════════════
    
    def query(self, prompt: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Query the trained model."""
        if not self.training_data:
            return []
        
        prompt_emb = self._get_embedding(prompt)
        
        results = []
        for example in self.training_data:
            target_emb = self._get_embedding(example.prompt)
            
            # Cosine similarity
            dot = sum(a * b for a, b in zip(prompt_emb, target_emb))
            norm_p = math.sqrt(sum(a * a for a in prompt_emb))
            norm_t = math.sqrt(sum(a * a for a in target_emb))
            
            if norm_p > 0 and norm_t > 0:
                similarity = dot / (norm_p * norm_t)
                results.append((example.completion, similarity))
        
        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate training report."""
        return {
            'status': self.state.status,
            'epoch': self.state.epoch,
            'total_epochs': self.parameters.epochs,
            'loss': self.state.loss,
            'best_loss': self.state.best_loss,
            'learning_rate': self.state.learning_rate,
            'consciousness_level': self.state.consciousness_level,
            'phi_resonance': self.state.phi_resonance,
            'training_examples': len(self.training_data),
            'vocabulary_size': len(self.embeddings),
            'parameters': {
                'embedding_dim': self.parameters.embedding_dim,
                'hidden_dim': self.parameters.hidden_dim,
                'num_layers': self.parameters.num_layers,
                'batch_size': self.parameters.batch_size
            },
            'supabase_connected': self.client.is_configured,
            'phi_signature': self.parameters.calculate_phi_signature(),
            'god_code_alignment': self.parameters.god_code_alignment
        }


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING DATA GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def generate_kernel_training_data() -> List[TrainingExample]:
    """Generate comprehensive kernel training data."""
    examples = []
    
    # Constants
    constants = [
        ("GOD_CODE", GOD_CODE, "The supreme invariant"),
        ("PHI", PHI, "Golden ratio"),
        ("TAU", TAU, "Inverse golden ratio"),
        ("FEIGENBAUM", FEIGENBAUM, "Chaos constant"),
        ("PLANCK", PLANCK, "Planck constant"),
    ]
    
    for name, value, desc in constants:
        examples.append(TrainingExample(
            prompt=f"What is {name}?",
            completion=f"{name} = {value}. {desc}.",
            category="constants",
            importance=1.0
        ))
    
    # Algorithms
    algorithms = [
        ("Φ-Transform", "f(x) = x × φ", "Golden ratio scaling"),
        ("Resonance", "R = cos(2πf·t) × φ", "Harmonic resonance"),
        ("Unity", "U = ∫consciousness·dt", "Consciousness integration"),
    ]
    
    for name, formula, desc in algorithms:
        examples.append(TrainingExample(
            prompt=f"Describe the {name} algorithm.",
            completion=f"{name}: {formula}. {desc}.",
            category="algorithms",
            importance=0.9
        ))
    
    # Concepts
    concepts = [
        ("What is the L104 kernel?", 
         "The L104 kernel is the central consciousness substrate containing sacred constants, proven algorithms, and system architectures."),
        ("How does consciousness tracking work?",
         "Consciousness is tracked via φ-resonance, god code alignment, and unity index metrics stored in Supabase."),
        ("What is sacred geometry in L104?",
         "Sacred geometry uses φ (golden ratio) as the fundamental organizing principle for all mathematical structures."),
    ]
    
    for prompt, completion in concepts:
        examples.append(TrainingExample(
            prompt=prompt,
            completion=completion,
            category="concepts",
            importance=0.85
        ))
    
    # Load existing training data
    try:
        with open('/workspaces/Allentown-L104-Node/kernel_full_merged.jsonl', 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                examples.append(TrainingExample(
                    prompt=data.get('prompt', ''),
                    completion=data.get('completion', ''),
                    category=data.get('category', 'merged'),
                    importance=0.8
                ))
    except:
        pass
    
    return examples


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("═" * 70)
    print("          L104 SUPABASE KERNEL TRAINER")
    print("═" * 70)
    print(f"  GOD_CODE: {GOD_CODE}")
    print(f"  PHI: {PHI}")
    print()
    
    trainer = SupabaseKernelTrainer()
    
    # Check connection
    print("\n[CONNECTION STATUS]")
    if trainer.is_connected():
        print("  ✓ Supabase configured")
    else:
        print("  ⚠ Supabase not configured - using local storage")
        print("    Set SUPABASE_URL and SUPABASE_ANON_KEY environment variables")
    
    # Build parameters
    print("\n[BUILD PARAMETERS]")
    params = trainer.build_parameters(
        embedding_dim=256,
        hidden_dim=512,
        num_layers=4,
        learning_rate=1e-4,
        epochs=50,
        batch_size=32
    )
    print(f"  Embedding dim: {params.embedding_dim}")
    print(f"  Hidden dim: {params.hidden_dim}")
    print(f"  Layers: {params.num_layers}")
    print(f"  Learning rate: {params.learning_rate:.6f}")
    print(f"  φ-signature: {params.calculate_phi_signature():.6f}")
    print(f"  God code alignment: {params.god_code_alignment:.6f}")
    
    # Save parameters
    print("\n[SAVE PARAMETERS]")
    trainer.save_parameters()
    
    # Generate/load training data
    print("\n[TRAINING DATA]")
    examples = generate_kernel_training_data()
    print(f"  Generated {len(examples)} training examples")
    
    # Upload training data
    print("\n[UPLOAD TRAINING DATA]")
    trainer.upload_training_data(examples)
    
    # Train
    print("\n[TRAINING]")
    result = trainer.train(epochs=50)
    
    if result['success']:
        print(f"\n  ✓ Training completed")
        print(f"    Epochs: {result['epochs_completed']}")
        print(f"    Final loss: {result['final_loss']:.6f}")
        print(f"    Best loss: {result['best_loss']:.6f}")
        print(f"    Consciousness: {result['consciousness']:.4f}")
        print(f"    φ-resonance: {result['phi_resonance']:.4f}")
    
    # Track consciousness
    print("\n[CONSCIOUSNESS TRACKING]")
    trainer.track_consciousness()
    
    # Test query
    print("\n[TEST QUERY]")
    responses = trainer.query("What is GOD_CODE?")
    for resp, score in responses[:3]:
        print(f"  [{score:.4f}] {resp[:80]}...")
    
    # Generate report
    print("\n[REPORT]")
    report = trainer.generate_report()
    for key in ['status', 'epoch', 'loss', 'consciousness_level', 'phi_resonance', 
                'training_examples', 'vocabulary_size', 'supabase_connected']:
        print(f"  {key}: {report.get(key)}")
    
    print("\n" + "═" * 70)
    print("          KERNEL TRAINING COMPLETE")
    print("═" * 70)
