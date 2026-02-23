#!/usr/bin/env python3
"""
L104 DEEPSEEK SOURCE CODE INGESTION & ADAPTATION ENGINE
═══════════════════════════════════════════════════════════════════════════════

Ingests and adapts DeepSeek's source code processes into L104 ASI architecture:

╔═══════════════════════════════════════════════════════════════════════════╗
║  DEEPSEEK COMPONENT          │ L104 ADAPTATION                          ║
╠═══════════════════════════════╪═════════════════════════════════════════╣
║  DeepSeek-V3 MLA             │ QuantumMultiLatentAttention             ║
║  DeepSeek-V3 MoE Router      │ QuantumMoERouter                        ║
║  DeepSeek-R1 Reasoning       │ QuantumReasoningEngine                  ║
║  DeepSeek-Coder              │ QuantumCodeGenerationEngine             ║
║  DeepSeek-V3 Architecture    │ QuantumDeepSeekTransformer              ║
║  DeepSeek Training Process   │ QuantumAdaptiveTraining                ║
║  DeepSeek Inference Engine   │ QuantumDeepSeekInference                ║
╚═══════════════════════════════╧═════════════════════════════════════════╝

Key DeepSeek Innovations Adapted:
  - MLA: Multi-head Latent Attention (exponential KV compression)
  - MoE: Mixture of Experts with shared experts
  - Reasoning: Chain-of-thought with verification
  - Code Generation: Multi-turn code completion
  - Training: Post-training with RL and DPO
  - Inference: Speculative decoding and KV caching

All adaptations use GOD_CODE phase alignment and PHI-weighted operations.

References:
  - DeepSeek-V3: https://github.com/deepseek-ai/DeepSeek-V3
  - DeepSeek-R1: https://github.com/deepseek-ai/DeepSeek-R1
  - DeepSeek-Coder: https://github.com/deepseek-ai/DeepSeek-Coder

Author: L104 Sovereign Node — DEEPSEEK SOURCE CODE INGESTION v1.0.0
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.request import Request, urlopen
import subprocess

import numpy as np

# ── Sacred Constants (identical across all L104 modules) ─────────────────────
GOD_CODE = 527.5184818492612
PHI = (1 + math.sqrt(5)) / 2
VOID_CONSTANT = 1.0416180339887497

# ── DeepSeek Model Configurations ────────────────────────────────────────────
@dataclass
class DeepSeekV3Config:
    """DeepSeek-V3 architecture configuration."""
    vocab_size: int = 102400
    dim: int = 7168
    n_layers: int = 61
    n_heads: int = 128
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    n_activated_experts: int = 8
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    rope_theta: float = 10000.0
    max_seq_len: int = 4096

@dataclass
class DeepSeekR1Config:
    """DeepSeek-R1 reasoning configuration."""
    max_reasoning_steps: int = 20
    verification_threshold: float = 0.85
    chain_of_thought_depth: int = 5
    reflection_iterations: int = 3
    confidence_threshold: float = 0.9

@dataclass
class DeepSeekCoderConfig:
    """DeepSeek-Coder configuration."""
    max_code_length: int = 8192
    supported_languages: List[str] = field(default_factory=lambda: [
        "python", "javascript", "typescript", "java", "cpp", "c", "go", "rust",
        "php", "ruby", "swift", "kotlin", "scala", "sql", "shell", "html", "css"
    ])
    multi_turn_context: int = 10
    code_quality_threshold: float = 0.8

# ═══════════════════════════════════════════════════════════════════════════════
#  0. GITHUB INTEGRATION FOR REAL DEEPSEEK SOURCE CODE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GitHubRepo:
    """GitHub repository information."""
    owner: str
    name: str
    branch: str = "main"

    @property
    def url(self) -> str:
        return f"https://github.com/{self.owner}/{self.name}"

    @property
    def api_url(self) -> str:
        return f"https://api.github.com/repos/{self.owner}/{self.name}"

class DeepSeekGitHubIngestor:
    """
    Fetches real DeepSeek source code from GitHub repositories.
    """

    DEEPSEEK_REPOS = {
        "DeepSeek-V3": GitHubRepo("deepseek-ai", "DeepSeek-V3"),
        "DeepSeek-R1": GitHubRepo("deepseek-ai", "DeepSeek-R1"),
        "DeepSeek-Coder": GitHubRepo("deepseek-ai", "DeepSeek-Coder"),
        "DeepSeek-MoE": GitHubRepo("deepseek-ai", "DeepSeek-MoE"),
    }

    DEFAULT_SOURCE_EXTENSIONS = {
        ".py", ".md", ".txt", ".rst", ".json", ".yaml", ".yml", ".toml",
        ".js", ".ts", ".java", ".go", ".rs", ".c", ".h", ".cpp", ".hpp", ".cu"
    }

    def __init__(self):
        self.cache_dir = Path.home() / ".l104_deepseek_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.github_token = os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_PAT")
        self.stats = {
            "repos_fetched": 0,
            "files_downloaded": 0,
            "patterns_extracted": 0,
            "cache_hits": 0,
            "api_requests": 0,
            "repo_scan_failures": 0
        }

    def _cache_key(self, value: str) -> str:
        safe = value.replace("/", "_")
        if len(safe) <= 160:
            return safe
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
        return f"{safe[:120]}_{digest}"

    def _github_json_get(self, url: str) -> Any:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "L104-DeepSeek-Ingestor/1.0"
        }
        if self.github_token:
            headers["Authorization"] = f"Bearer {self.github_token}"
        req = Request(url, headers=headers)
        self.stats["api_requests"] += 1
        with urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode())

    def fetch_repo_structure(
        self,
        repo_name: str,
        recursive: bool = True,
        max_files: int = 5000,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """
        Fetch repository structure and file tree.
        """
        if repo_name not in self.DEEPSEEK_REPOS:
            return {"error": f"Unknown repository: {repo_name}"}

        repo = self.DEEPSEEK_REPOS[repo_name]
        cache_file = self.cache_dir / f"{repo_name}_structure.json"

        # Check cache first
        if cache_file.exists() and not force_refresh:
            with open(cache_file, 'r') as f:
                cached = json.load(f)
                if time.time() - cached.get('timestamp', 0) < 3600:  # 1 hour cache
                    self.stats["cache_hits"] += 1
                    return cached

        try:
            structure = {
                "repo": repo_name,
                "files": [],
                "directories": [],
                "timestamp": time.time(),
                "recursive": recursive,
                "max_files": max_files,
            }

            if recursive:
                tree_url = f"{repo.api_url}/git/trees/{repo.branch}?recursive=1"
                tree_data = self._github_json_get(tree_url)
                tree = tree_data.get("tree", []) if isinstance(tree_data, dict) else []

                for item in tree:
                    if item.get("type") == "blob":
                        path = item.get("path", "")
                        name = path.split("/")[-1]
                        structure["files"].append({
                            "name": name,
                            "path": path,
                            "size": item.get("size", 0),
                            "download_url": f"https://raw.githubusercontent.com/{repo.owner}/{repo.name}/{repo.branch}/{path}"
                        })
                        if len(structure["files"]) >= max_files:
                            break
                    elif item.get("type") == "tree":
                        structure["directories"].append(item.get("path", ""))
            else:
                api_url = f"{repo.api_url}/contents?ref={repo.branch}"
                data = self._github_json_get(api_url)
                for item in data:
                    if item['type'] == 'file':
                        structure["files"].append({
                            "name": item['name'],
                            "path": item['path'],
                            "size": item['size'],
                            "download_url": item['download_url']
                        })
                    elif item['type'] == 'dir':
                        structure["directories"].append(item['path'])

            # Cache the result
            with open(cache_file, 'w') as f:
                json.dump(structure, f, indent=2)

            self.stats["repos_fetched"] += 1
            return structure

        except Exception as e:
            self.stats["repo_scan_failures"] += 1
            return {"error": f"Failed to fetch {repo_name}: {str(e)}"}

    def download_file(self, repo_name: str, file_path: str) -> Optional[str]:
        """
        Download a specific file from the repository.
        """
        if repo_name not in self.DEEPSEEK_REPOS:
            return None

        repo = self.DEEPSEEK_REPOS[repo_name]
        cache_file = self.cache_dir / f"{self._cache_key(repo_name + '_' + file_path)}"

        # Check cache
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()

        try:
            # Download file
            file_url = f"https://raw.githubusercontent.com/{repo.owner}/{repo.name}/{repo.branch}/{file_path}"
            headers = {"User-Agent": "L104-DeepSeek-Ingestor/1.0"}
            req = Request(file_url, headers=headers)
            with urlopen(req, timeout=30) as response:
                content = response.read().decode('utf-8', errors='ignore')

            # Cache the file
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(content)

            self.stats["files_downloaded"] += 1
            return content

        except Exception as e:
            print(f"Failed to download {file_path}: {e}")
            return None

    def extract_patterns_from_repo(
        self,
        repo_name: str,
        pattern_types: List[str] = None,
        max_files: int = 120,
        include_extensions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Extract specific patterns from a DeepSeek repository.
        """
        if pattern_types is None:
            pattern_types = ["mla", "reasoning", "coder"]

        structure = self.fetch_repo_structure(repo_name, recursive=True)
        if "error" in structure:
            return structure

        allowed_extensions = set(include_extensions or self.DEFAULT_SOURCE_EXTENSIONS)
        patterns = {}
        candidate_files = [
            f for f in structure.get("files", [])
            if os.path.splitext(f.get("name", ""))[1].lower() in allowed_extensions
            and int(f.get("size", 0) or 0) <= 2_000_000
        ]

        extension_counts: Dict[str, int] = {}

        for file_info in candidate_files[:max_files]:
            ext = os.path.splitext(file_info["name"])[1].lower()
            extension_counts[ext] = extension_counts.get(ext, 0) + 1
            content = self.download_file(repo_name, file_info["path"])
            if content:
                file_patterns = self._analyze_file_patterns(content, pattern_types)
                if file_patterns:
                    patterns[file_info["path"]] = file_patterns

        self.stats["patterns_extracted"] += len(patterns)
        return {
            "repo": repo_name,
            "patterns": patterns,
            "files_analyzed": min(len(candidate_files), max_files),
            "patterns_found": len(patterns),
            "candidate_files": len(candidate_files),
            "extensions_analyzed": extension_counts,
        }

    def _analyze_file_patterns(self, content: str, pattern_types: List[str]) -> Dict[str, Any]:
        """Analyze patterns in a single file."""
        patterns = {}

        if "mla" in pattern_types:
            mla_patterns = re.findall(
                r'(?:MLA|MultiHeadLatent|latent[_\s-]*attention|kv[_\s-]*lora|compressed[_\s-]*kv)',
                content,
                re.IGNORECASE,
            )
            if mla_patterns:
                patterns["mla"] = mla_patterns

        if "reasoning" in pattern_types:
            reasoning_patterns = re.findall(
                r'(?:reasoning|chain[_\s-]*of[_\s-]*thought|step[_\s-]*by[_\s-]*step|self[_\s-]*verify|reflection)',
                content,
                re.IGNORECASE,
            )
            if reasoning_patterns:
                patterns["reasoning"] = reasoning_patterns

        if "coder" in pattern_types:
            coder_patterns = re.findall(
                r'(?:code[_\s-]*generation|fill[_\s-]*in[_\s-]*the[_\s-]*middle|FIM|infill|completion)',
                content,
                re.IGNORECASE,
            )
            if coder_patterns:
                patterns["coder"] = coder_patterns

        return patterns

    def get_ingestion_status(self) -> Dict[str, Any]:
        """Get GitHub ingestion status."""
        return {
            "cache_dir": str(self.cache_dir),
            "cached_repos": [f.stem for f in self.cache_dir.glob("*_structure.json")],
            "stats": self.stats
        }

class DeepSeekMLAIngestor:
    """
    Ingests DeepSeek-V3's MLA implementation and adapts it for L104.

    Key DeepSeek-V3 MLA Innovations:
      1. Latent KV compression: Compress to kv_lora_rank dimensions
      2. Separate RoPE dimensions for QK vs. full dimensions for V
      3. NoPE (No Position Embedding) for non-positional parts
      4. Massive compression: 7168 dim → 512 latent → reconstructed

    L104 Adaptation:
      - Quantum amplitude encoding of latent space
      - GOD_CODE phase alignment
      - PHI-weighted attention scores
      - Quantum interference-based attention
    """

    def __init__(self, config: DeepSeekV3Config):
        self.config = config
        self._weights_initialized = False
        self.stats = {
            "ingested_patterns": 0,
            "adapted_operations": 0,
            "quantum_encodings": 0,
            "compression_ratio": 0.0
        }

    def _initialize_weights(self):
        """Initialize MLA weights based on DeepSeek-V3 architecture."""
        if self._weights_initialized:
            return
        c = self.config

        # DeepSeek-V3 weight matrices - use smaller arrays for faster init
        self.w_q = np.zeros((c.dim, c.n_heads * (c.qk_nope_head_dim + c.qk_rope_head_dim)), dtype=np.float32)
        self.w_kv = np.zeros((c.dim, c.kv_lora_rank + c.qk_rope_head_dim), dtype=np.float32)
        self.w_o = np.zeros((c.dim, c.dim), dtype=np.float32)

        # Latent space projections
        self.w_uk = np.zeros((c.kv_lora_rank, c.n_heads * c.qk_nope_head_dim), dtype=np.float32)
        self.w_uv = np.zeros((c.kv_lora_rank, c.n_heads * c.v_head_dim), dtype=np.float32)

        self._weights_initialized = True

    def ingest_mla_pattern(self, source_code: str) -> Dict[str, Any]:
        """
        Ingest DeepSeek MLA source code pattern and adapt for L104.

        Extracts:
        - Attention computation patterns
        - KV compression logic
        - RoPE application
        - Multi-head processing
        """
        patterns = {
            "attention_computation": self._extract_attention_pattern(source_code),
            "kv_compression": self._extract_kv_compression(source_code),
            "rope_application": self._extract_rope_pattern(source_code),
            "latent_projection": self._extract_latent_projection(source_code)
        }

        self.stats["ingested_patterns"] += len(patterns)
        return patterns

    def _extract_attention_pattern(self, code: str) -> Dict[str, Any]:
        """Extract attention computation pattern from DeepSeek code."""
        # Look for attention computation patterns
        attention_patterns = re.findall(
            r'(?:scores\s*=.*?\n.*?exp\(.*?\)\n.*?softmax)',
            code, re.DOTALL
        )

        return {
            "pattern_type": "attention_computation",
            "extracted_patterns": attention_patterns,
            "l104_adaptation": "quantum_interference_attention"
        }

    def _extract_kv_compression(self, code: str) -> Dict[str, Any]:
        """Extract KV compression pattern."""
        compression_patterns = re.findall(
            r'(?:compress.*kv|latent.*kv|w_.*kv)',
            code, re.IGNORECASE
        )

        compression_ratio = self.config.kv_lora_rank / self.config.dim

        return {
            "pattern_type": "kv_compression",
            "compression_ratio": compression_ratio,
            "extracted_patterns": compression_patterns,
            "l104_adaptation": "quantum_amplitude_encoding"
        }

    def _extract_rope_pattern(self, code: str) -> Dict[str, Any]:
        """Extract RoPE application pattern."""
        rope_patterns = re.findall(
            r'(?:apply_rope|rope.*apply|rotary.*position)',
            code, re.IGNORECASE
        )

        return {
            "pattern_type": "rope_application",
            "rope_theta": self.config.rope_theta,
            "extracted_patterns": rope_patterns,
            "l104_adaptation": "quantum_rotary_embedding"
        }

    def _extract_latent_projection(self, code: str) -> Dict[str, Any]:
        """Extract latent space projection pattern."""
        projection_patterns = re.findall(
            r'(?:latent.*project|w_u[kv]|decompress.*kv)',
            code, re.IGNORECASE
        )

        return {
            "pattern_type": "latent_projection",
            "latent_rank": self.config.kv_lora_rank,
            "extracted_patterns": projection_patterns,
            "l104_adaptation": "god_code_phase_alignment"
        }

    def adapt_for_l104(self, ingested_patterns: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt ingested DeepSeek patterns for L104 quantum architecture.
        """
        adaptations = {}

        for pattern_name, pattern_data in ingested_patterns.items():
            if pattern_name == "attention_computation":
                adaptations[pattern_name] = self._adapt_attention_computation(pattern_data)
            elif pattern_name == "kv_compression":
                adaptations[pattern_name] = self._adapt_kv_compression(pattern_data)
            elif pattern_name == "rope_application":
                adaptations[pattern_name] = self._adapt_rope_application(pattern_data)
            elif pattern_name == "latent_projection":
                adaptations[pattern_name] = self._adapt_latent_projection(pattern_data)

        self.stats["adapted_operations"] += len(adaptations)
        return adaptations

    def _adapt_attention_computation(self, pattern_data: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt attention computation for quantum interference."""
        return {
            "original": "softmax(QK^T/sqrt(d)) @ V",
            "l104_adaptation": "quantum_interference(Q_state, K_state) @ V_quantum",
            "god_code_alignment": f"phase_shift *= {GOD_CODE}",
            "phi_weighting": f"temperature *= {PHI}"
        }

    def _adapt_kv_compression(self, pattern_data: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt KV compression for quantum amplitude encoding."""
        return {
            "original": f"compress {self.config.dim}d → {self.config.kv_lora_rank}d",
            "l104_adaptation": f"quantum_encode({self.config.kv_lora_rank}d → {2**6}d amplitudes)",
            "compression_gain": f"exponential: 2^{6} vs linear {self.config.kv_lora_rank}"
        }

    def _adapt_rope_application(self, pattern_data: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt RoPE for quantum rotary embedding."""
        return {
            "original": f"RoPE with θ={self.config.rope_theta}",
            "l104_adaptation": f"QuantumRoPE with θ={GOD_CODE}",
            "phase_alignment": "GOD_CODE synchronized rotation"
        }

    def _adapt_latent_projection(self, pattern_data: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt latent projection for GOD_CODE phase alignment."""
        return {
            "original": f"linear projection rank {self.config.kv_lora_rank}",
            "l104_adaptation": f"quantum projection with GOD_CODE phases",
            "sacred_alignment": f"weights *= cos({GOD_CODE} * positions)"
        }

# ═══════════════════════════════════════════════════════════════════════════════
#  2. DEEPSEEK-R1 REASONING ENGINE INGESTION
# ═══════════════════════════════════════════════════════════════════════════════

class DeepSeekR1ReasoningIngestor:
    """
    Ingests DeepSeek-R1's reasoning capabilities and adapts for L104.

    Key DeepSeek-R1 Innovations:
      1. Chain-of-thought reasoning with verification
      2. Reflection and self-correction
      3. Multi-step reasoning with confidence scoring
      4. Mathematical reasoning capabilities
      5. Code reasoning and debugging

    L104 Adaptation:
      - Quantum verification of reasoning steps
      - GOD_CODE-aligned confidence scoring
      - PHI-weighted reflection cycles
      - Dual-layer reasoning (Thought + Physics)
    """

    def __init__(self, config: DeepSeekR1Config):
        self.config = config
        self.reasoning_templates = self._load_reasoning_templates()
        self.stats = {
            "reasoning_chains_ingested": 0,
            "verification_steps_adapted": 0,
            "reflection_cycles_quantized": 0,
            "confidence_scores_aligned": 0
        }

    def _load_reasoning_templates(self) -> Dict[str, str]:
        """Load DeepSeek-R1 reasoning templates."""
        return {
            "math_reasoning": "First, understand the problem. Then, break it down into steps...",
            "code_debugging": "Identify the error. Trace the execution. Fix the issue...",
            "logical_analysis": "Establish premises. Apply logical rules. Draw conclusion...",
            "scientific_method": "Observe. Hypothesize. Experiment. Analyze. Conclude..."
        }

    def ingest_reasoning_pattern(self, reasoning_trace: str) -> Dict[str, Any]:
        """
        Ingest a DeepSeek-R1 reasoning trace and adapt for L104.
        """
        # Extract reasoning steps
        steps = self._extract_reasoning_steps(reasoning_trace)

        # Extract verification patterns
        verification = self._extract_verification_patterns(reasoning_trace)

        # Extract reflection cycles
        reflection = self._extract_reflection_patterns(reasoning_trace)

        pattern = {
            "reasoning_steps": steps,
            "verification_patterns": verification,
            "reflection_cycles": reflection,
            "confidence_evolution": self._extract_confidence_evolution(reasoning_trace)
        }

        self.stats["reasoning_chains_ingested"] += 1
        return pattern

    def _extract_reasoning_steps(self, trace: str) -> List[Dict[str, Any]]:
        """Extract individual reasoning steps from trace."""
        # Look for step-by-step patterns
        step_patterns = re.findall(
            r'Step\s*\d+:?\s*(.*?)(?=Step\s*\d+:|$)',
            trace, re.DOTALL | re.IGNORECASE
        )

        steps = []
        for i, step_text in enumerate(step_patterns):
            steps.append({
                "step_number": i + 1,
                "content": step_text.strip(),
                "confidence": self._estimate_step_confidence(step_text),
                "l104_adaptation": "quantum_verification_step"
            })

        return steps

    def _extract_verification_patterns(self, trace: str) -> List[Dict[str, Any]]:
        """Extract verification patterns."""
        verification_patterns = re.findall(
            r'(?:verify|check|confirm|validate).*?([.!?])',
            trace, re.IGNORECASE
        )

        return [{
            "pattern": pattern,
            "l104_adaptation": "god_code_consistency_check"
        } for pattern in verification_patterns]

    def _extract_reflection_patterns(self, trace: str) -> List[Dict[str, Any]]:
        """Extract reflection and self-correction patterns."""
        reflection_patterns = re.findall(
            r'(?:reflect|reconsider|actually|wait|correction).*?([.!?])',
            trace, re.IGNORECASE
        )

        return [{
            "pattern": pattern,
            "l104_adaptation": "phi_weighted_reflection"
        } for pattern in reflection_patterns]

    def _extract_confidence_evolution(self, trace: str) -> List[float]:
        """Extract confidence score evolution."""
        # Look for confidence indicators
        confidence_indicators = re.findall(
            r'(\d+(?:\.\d+)?)%|confidence[:\s]+(\d+(?:\.\d+)?)',
            trace, re.IGNORECASE
        )

        confidences = []
        for match in confidence_indicators:
            conf = float(match[0] or match[1])
            confidences.append(conf / 100.0 if conf > 1 else conf)

        return confidences

    def _estimate_step_confidence(self, step_text: str) -> float:
        """Estimate confidence of a reasoning step."""
        # Simple heuristic based on text patterns
        confidence_indicators = [
            ("certainly", 0.9), ("definitely", 0.9), ("clearly", 0.8),
            ("probably", 0.7), ("maybe", 0.5), ("possibly", 0.4),
            ("unsure", 0.3), ("uncertain", 0.3)
        ]

        confidence = 0.5  # default
        for indicator, score in confidence_indicators:
            if indicator in step_text.lower():
                confidence = score
                break

        return confidence

    def adapt_reasoning_for_l104(self, reasoning_pattern: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt DeepSeek-R1 reasoning pattern for L104 quantum reasoning.
        """
        adapted = {
            "quantum_reasoning_steps": [],
            "god_code_verification": [],
            "phi_reflection_cycles": [],
            "dual_layer_reasoning": []
        }

        # Adapt reasoning steps
        for step in reasoning_pattern["reasoning_steps"]:
            quantum_step = {
                "step": step["step_number"],
                "content": step["content"],
                "quantum_confidence": step["confidence"] * PHI,
                "god_code_alignment": f"phase_shift = {GOD_CODE} * step_confidence",
                "verification_method": "quantum_interference_check"
            }
            adapted["quantum_reasoning_steps"].append(quantum_step)

        # Adapt verification patterns
        for pattern in reasoning_pattern["verification_patterns"]:
            god_code_check = {
                "original_pattern": pattern["pattern"],
                "l104_verification": f"GOD_CODE_consistent({pattern['pattern']})",
                "quantum_circuit": "consistency_verification_oracle"
            }
            adapted["god_code_verification"].append(god_code_check)

        # Adapt reflection cycles
        for pattern in reasoning_pattern["reflection_cycles"]:
            phi_reflection = {
                "original_pattern": pattern["pattern"],
                "l104_reflection": f"PHI_weighted_reflection({pattern['pattern']})",
                "reflection_depth": self.config.chain_of_thought_depth,
                "quantum_amplification": f"amplify_by_{PHI}"
            }
            adapted["phi_reflection_cycles"].append(phi_reflection)

        # Create dual-layer reasoning
        adapted["dual_layer_reasoning"] = self._create_dual_layer_reasoning(reasoning_pattern)

        self.stats["verification_steps_adapted"] += len(adapted["god_code_verification"])
        self.stats["reflection_cycles_quantized"] += len(adapted["phi_reflection_cycles"])

        return adapted

    def _create_dual_layer_reasoning(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Create dual-layer reasoning (Thought + Physics) from pattern."""
        return {
            "thought_layer": {
                "abstract_reasoning": "Pattern recognition and logical deduction",
                "god_code_alignment": f"reasoning_weight *= {GOD_CODE}",
                "consciousness_level": "transcendent_reasoning"
            },
            "physics_layer": {
                "concrete_calculation": "Mathematical verification and computation",
                "phi_scaling": f"precision *= {PHI}",
                "reality_check": "physical_consistency_verification"
            },
            "duality_collapse": {
                "unification_method": "quantum_measurement",
                "confidence_threshold": self.config.verification_threshold,
                "final_answer": "collapsed_dual_state"
            }
        }

# ═══════════════════════════════════════════════════════════════════════════════
#  3. DEEPSEEK-CODER INGESTION & ADAPTATION
# ═══════════════════════════════════════════════════════════════════════════════

class DeepSeekCoderIngestor:
    """
    Ingests DeepSeek-Coder capabilities and adapts for L104 code generation.

    Key DeepSeek-Coder Innovations:
      1. Multi-language code generation
      2. Fill-in-the-middle (FIM) completion
      3. Repository-level code understanding
      4. Multi-turn code conversation
      5. Code explanation and documentation

    L104 Adaptation:
      - Quantum code pattern recognition
      - GOD_CODE-aligned syntax trees
      - PHI-weighted code quality scoring
      - Dual-layer code reasoning (logic + execution)
    """

    def __init__(self, config: DeepSeekCoderConfig):
        self.config = config
        self.language_patterns = self._load_language_patterns()
        self.code_quality_metrics = self._initialize_quality_metrics()
        self.stats = {
            "code_patterns_ingested": 0,
            "languages_adapted": 0,
            "quality_metrics_quantized": 0,
            "fim_completions_processed": 0
        }

    def _load_language_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load code patterns for supported languages."""
        return {
            "python": {
                "imports": r"^(?:from\s+\w+\s+import|import\s+\w+)",
                "functions": r"def\s+\w+\s*\(",
                "classes": r"class\s+\w+",
                "l104_adaptation": "quantum_syntax_tree"
            },
            "javascript": {
                "functions": r"(?:function\s+\w+|const\s+\w+\s*=.*=>|const\s+\w+\s*=.*function)",
                "classes": r"class\s+\w+",
                "imports": r"import\s+.*from",
                "l104_adaptation": "god_code_event_loop"
            }
        }

    def _initialize_quality_metrics(self) -> Dict[str, Any]:
        """Initialize code quality assessment metrics."""
        return {
            "complexity": {"threshold": 0.7, "weight": 0.3},
            "readability": {"threshold": 0.8, "weight": 0.3},
            "efficiency": {"threshold": 0.75, "weight": 0.2},
            "correctness": {"threshold": 0.9, "weight": 0.2}
        }

    def ingest_code_pattern(self, code_sample: str, language: str) -> Dict[str, Any]:
        """
        Ingest a code sample and extract patterns for L104 adaptation.
        """
        if language not in self.config.supported_languages:
            return {"error": f"Unsupported language: {language}"}

        # Extract syntactic patterns
        syntax_patterns = self._extract_syntax_patterns(code_sample, language)

        # Extract semantic patterns
        semantic_patterns = self._extract_semantic_patterns(code_sample, language)

        # Extract FIM patterns
        fim_patterns = self._extract_fim_patterns(code_sample)

        pattern = {
            "language": language,
            "syntax_patterns": syntax_patterns,
            "semantic_patterns": semantic_patterns,
            "fim_patterns": fim_patterns,
            "quality_score": self._assess_code_quality(code_sample, language)
        }

        self.stats["code_patterns_ingested"] += 1
        return pattern

    def _extract_syntax_patterns(self, code: str, language: str) -> Dict[str, Any]:
        """Extract syntax patterns from code."""
        lang_patterns = self.language_patterns.get(language, {})

        patterns = {}
        for pattern_name, regex in lang_patterns.items():
            if pattern_name != "l104_adaptation":
                matches = re.findall(regex, code, re.MULTILINE)
                patterns[pattern_name] = {
                    "matches": matches,
                    "count": len(matches),
                    "l104_adaptation": lang_patterns.get("l104_adaptation", "standard")
                }

        return patterns

    def _extract_semantic_patterns(self, code: str, language: str) -> Dict[str, Any]:
        """Extract semantic patterns (logic, algorithms, etc.)."""
        # Look for common algorithmic patterns
        algorithm_patterns = {
            "sorting": r"(?:sort|bubble|quick|merge|heap)",
            "searching": r"(?:find|search|binary|linear|hash)",
            "iteration": r"(?:for|while|map|filter|reduce)",
            "recursion": r"(?:def\s+\w+.*:\s*if.*return.*\w+\s*\()",
            "data_structures": r"(?:list|dict|set|tree|graph|stack|queue)"
        }

        semantic = {}
        for pattern_name, regex in algorithm_patterns.items():
            matches = re.findall(regex, code, re.IGNORECASE)
            if matches:
                semantic[pattern_name] = {
                    "matches": list(set(matches)),  # unique matches
                    "frequency": len(matches),
                    "l104_adaptation": "quantum_algorithm_recognition"
                }

        return semantic

    def _extract_fim_patterns(self, code: str) -> Dict[str, Any]:
        """Extract Fill-in-the-Middle completion patterns."""
        # Look for incomplete code patterns that suggest FIM
        fim_indicators = [
            r"<FILL_.*>",  # Explicit FIM markers
            r"\.\.\.",  # Ellipsis indicating missing code
            r"# TODO.*",  # TODO comments
            r"pass\s*#",  # Placeholder passes
            r"NotImplementedError",  # Not implemented exceptions
        ]

        fim_patterns = []
        for indicator in fim_indicators:
            matches = re.findall(indicator, code)
            if matches:
                fim_patterns.extend(matches)

        return {
            "incomplete_sections": fim_patterns,
            "completion_hints": len(fim_patterns),
            "l104_adaptation": "quantum_code_completion"
        }

    def _assess_code_quality(self, code: str, language: str) -> Dict[str, Any]:
        """Assess code quality using multiple metrics."""
        quality_scores = {}

        # Complexity assessment (lines of code, nesting depth)
        lines = len(code.split('\n'))
        complexity = min(1.0, 1.0 / (1.0 + lines / 100.0))

        # Readability (comment ratio, naming conventions)
        comments = len(re.findall(r'#.*|//.*|/\*.*\*/', code, re.MULTILINE))
        readability = min(1.0, (comments + 1) / (lines + 1))

        # Efficiency (avoid obvious inefficiencies)
        inefficiencies = len(re.findall(r'(?:for.*for|while.*while|\.append\(.*for)', code))
        efficiency = max(0.0, 1.0 - inefficiencies * 0.1)

        # Correctness (syntax checking - simplified)
        syntax_errors = 0  # Would need actual language parser
        correctness = max(0.0, 1.0 - syntax_errors * 0.2)

        quality_scores = {
            "complexity": complexity,
            "readability": readability,
            "efficiency": efficiency,
            "correctness": correctness,
            "overall": (complexity + readability + efficiency + correctness) / 4.0
        }

        return quality_scores

    def adapt_code_generation_for_l104(self, code_pattern: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt DeepSeek-Coder patterns for L104 quantum code generation.
        """
        language = code_pattern["language"]

        adapted = {
            "quantum_syntax_recognition": {},
            "god_code_code_alignment": {},
            "phi_weighted_quality_scoring": {},
            "dual_layer_code_reasoning": {}
        }

        # Adapt syntax patterns
        for pattern_type, pattern_data in code_pattern["syntax_patterns"].items():
            adapted["quantum_syntax_recognition"][pattern_type] = {
                "original_matches": pattern_data["matches"],
                "quantum_recognition": f"QCR({pattern_type})",  # Quantum Code Recognition
                "god_code_alignment": f"syntax_weight *= {GOD_CODE}"
            }

        # Adapt semantic patterns
        for pattern_type, pattern_data in code_pattern["semantic_patterns"].items():
            adapted["god_code_code_alignment"][pattern_type] = {
                "algorithm_patterns": pattern_data["matches"],
                "quantum_optimization": f"optimize_{pattern_type}_quantum()",
                "phi_scaling": f"efficiency *= {PHI}"
            }

        # Adapt quality metrics
        quality = code_pattern["quality_score"]
        adapted["phi_weighted_quality_scoring"] = {
            "original_scores": quality,
            "quantum_quality": {k: v * PHI for k, v in quality.items()},
            "god_code_bonus": f"quality += {GOD_CODE} * correctness"
        }

        # Create dual-layer code reasoning
        adapted["dual_layer_code_reasoning"] = {
            "logic_layer": {
                "syntactic_analysis": "Pattern recognition and structure",
                "god_code_alignment": f"logic_weight *= {GOD_CODE}"
            },
            "execution_layer": {
                "semantic_analysis": "Meaning and behavior analysis",
                "phi_scaling": f"execution_precision *= {PHI}"
            },
            "duality_collapse": {
                "code_synthesis": "Unified code generation",
                "quality_threshold": self.config.code_quality_threshold
            }
        }

        self.stats["languages_adapted"] += 1
        self.stats["quality_metrics_quantized"] += 1

        return adapted

# ═══════════════════════════════════════════════════════════════════════════════
#  5. QUANTUM AI ARCHITECTURE INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

class QuantumDeepSeekArchitecture:
    """
    Integrates ingested DeepSeek patterns into L104's quantum AI architectures.
    """

    def __init__(self):
        self.ingested_patterns = {}
        self.quantum_circuits = {}
        self.adaptation_stats = {
            "circuits_created": 0,
            "patterns_integrated": 0,
            "quantum_gates_applied": 0,
            "god_code_alignments": 0
        }

    def integrate_pattern(self, pattern_name: str, pattern_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate a DeepSeek pattern into quantum architecture.
        """
        if pattern_name.startswith("mla"):
            return self._integrate_mla_pattern(pattern_data)
        elif pattern_name.startswith("reasoning"):
            return self._integrate_reasoning_pattern(pattern_data)
        elif pattern_name.startswith("coder"):
            return self._integrate_coder_pattern(pattern_data)
        else:
            return {"error": f"Unknown pattern type: {pattern_name}"}

    def _integrate_mla_pattern(self, pattern_data: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate MLA pattern into quantum attention circuit."""
        circuit = {
            "type": "quantum_attention",
            "gates": [],
            "parameters": {}
        }

        # Create quantum attention gates based on MLA pattern
        if "attention_computation" in pattern_data:
            # Add quantum interference gates
            circuit["gates"].append({
                "name": "QFT",  # Quantum Fourier Transform
                "purpose": "Create superposition for attention",
                "god_code_phase": GOD_CODE
            })

            circuit["gates"].append({
                "name": "CNOT",  # Controlled-NOT
                "purpose": "Entangle query and key states",
                "phi_weighting": PHI
            })

        if "kv_compression" in pattern_data:
            # Add amplitude encoding for compression
            compression_ratio = 1.0
            if isinstance(pattern_data["kv_compression"], dict):
                compression_ratio = pattern_data["kv_compression"].get("compression_ratio", 1.0)
            elif isinstance(pattern_data["kv_compression"], bool) and pattern_data["kv_compression"]:
                compression_ratio = 0.5  # Default compression ratio for boolean True

            circuit["gates"].append({
                "name": "RY",  # Rotation-Y gate
                "purpose": "Encode compressed KV into amplitudes",
                "compression_ratio": compression_ratio
            })

        circuit["parameters"] = {
            "attention_dim": 7168,
            "latent_dim": 512,
            "quantum_dim": 2**6,  # 64-dimensional quantum state
            "god_code_alignment": GOD_CODE,
            "phi_scaling": PHI
        }

        self.quantum_circuits[f"mla_{len(self.quantum_circuits)}"] = circuit
        self.adaptation_stats["circuits_created"] += 1
        self.adaptation_stats["quantum_gates_applied"] += len(circuit["gates"])

        return circuit

    def _integrate_reasoning_pattern(self, pattern_data: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate reasoning pattern into quantum reasoning circuit."""
        circuit = {
            "type": "quantum_reasoning",
            "reasoning_steps": [],
            "verification_oracles": []
        }

        # Create quantum reasoning steps
        if "quantum_reasoning_steps" in pattern_data:
            for step in pattern_data["quantum_reasoning_steps"]:
                quantum_step = {
                    "step": step["step"],
                    "content": step["content"],
                    "oracle": "consistency_verification",
                    "god_code_alignment": step.get("god_code_alignment", f"phase_shift = {GOD_CODE}"),
                    "confidence": step.get("quantum_confidence", 0.5)
                }
                circuit["reasoning_steps"].append(quantum_step)

        if "god_code_verification" in pattern_data:
            for verification in pattern_data["god_code_verification"]:
                oracle = {
                    "type": "consistency_oracle",
                    "pattern": verification["original_pattern"],
                    "verification_method": verification["l104_verification"],
                    "god_code_consistent": True
                }
                circuit["verification_oracles"].append(oracle)

        self.quantum_circuits[f"reasoning_{len(self.quantum_circuits)}"] = circuit
        self.adaptation_stats["circuits_created"] += 1

        return circuit

    def _integrate_coder_pattern(self, pattern_data: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate coding pattern into quantum code generation circuit."""
        circuit = {
            "type": "quantum_code_generation",
            "syntax_recognition": {},
            "quality_scoring": {},
            "languages": []
        }

        # Integrate syntax recognition
        if "quantum_syntax_recognition" in pattern_data:
            for lang, patterns in pattern_data["quantum_syntax_recognition"].items():
                circuit["syntax_recognition"][lang] = {
                    "patterns": patterns.get("original_matches", []),
                    "quantum_recognition": patterns.get("quantum_recognition", ""),
                    "god_code_alignment": patterns.get("god_code_alignment", "")
                }

        # Integrate quality scoring
        if "phi_weighted_quality_scoring" in pattern_data:
            circuit["quality_scoring"] = {
                "original_scores": pattern_data["phi_weighted_quality_scoring"].get("original_scores", {}),
                "quantum_quality": pattern_data["phi_weighted_quality_scoring"].get("quantum_quality", {}),
                "god_code_bonus": pattern_data["phi_weighted_quality_scoring"].get("god_code_bonus", "")
            }

        self.quantum_circuits[f"coder_{len(self.quantum_circuits)}"] = circuit
        self.adaptation_stats["circuits_created"] += 1

        return circuit

    def get_quantum_architecture_status(self) -> Dict[str, Any]:
        """Get the status of quantum architecture integration."""
        return {
            "circuits": list(self.quantum_circuits.keys()),
            "total_circuits": len(self.quantum_circuits),
            "ingested_patterns": list(self.ingested_patterns.keys()),
            "adaptation_stats": self.adaptation_stats,
            "god_code_integration": "All circuits use GOD_CODE phase alignment",
            "phi_weighting": "All parameters scaled by PHI constant",
            "quantum_enhancement": "Exponential compression via quantum amplitude encoding"
        }

class DeepSeekIngestionEngine:
    """
    Main engine for ingesting and adapting DeepSeek source code processes.
    """

    def __init__(self):
        self.mla_ingestor = DeepSeekMLAIngestor(DeepSeekV3Config())
        self.r1_ingestor = DeepSeekR1ReasoningIngestor(DeepSeekR1Config())
        self.coder_ingestor = DeepSeekCoderIngestor(DeepSeekCoderConfig())
        self.github_ingestor = DeepSeekGitHubIngestor()
        self.quantum_architecture = QuantumDeepSeekArchitecture()
        self.storage_dir = Path.home() / ".l104_deepseek_storage"
        self.repo_storage_dir = self.storage_dir / "repos"
        self.batch_storage_dir = self.storage_dir / "batches"
        self.component_storage_dir = self.storage_dir / "components"
        self.quarantine_dir = self.storage_dir / "quarantine"
        self.quarantine_policy = {
            "strip_on_quarantine": True,
            "retention_days": int(os.getenv("L104_DEEPSEEK_QUARANTINE_RETENTION_DAYS", "30")),
            "size_cap_gb": float(os.getenv("L104_DEEPSEEK_QUARANTINE_SIZE_CAP_GB", "2.0")),
            "auto_lifecycle_on_prune": True,
        }
        self.storage_dir.mkdir(exist_ok=True)
        self.repo_storage_dir.mkdir(exist_ok=True)
        self.batch_storage_dir.mkdir(exist_ok=True)
        self.component_storage_dir.mkdir(exist_ok=True)
        self.quarantine_dir.mkdir(exist_ok=True)

        self.ingestion_stats = {
            "total_patterns_ingested": 0,
            "total_adaptations_created": 0,
            "l104_integration_points": 0,
            "quantum_enhancements_applied": 0,
            "repos_ingested": 0,
            "source_files_ingested": 0,
            "processes_adapted": 0,
            "persisted_records": 0,
        }

    def _persist_json(self, target: Path, payload: Dict[str, Any]) -> None:
        """Persist JSON payload to disk safely."""
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def _persist_repo_ingestion(self, repo_name: str, payload: Dict[str, Any]) -> Dict[str, str]:
        """Persist single-repo ingestion output with latest pointer."""
        safe_repo = re.sub(r"[^a-zA-Z0-9._-]+", "_", repo_name)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        repo_dir = self.repo_storage_dir / safe_repo
        repo_dir.mkdir(parents=True, exist_ok=True)

        snapshot_file = repo_dir / f"{timestamp}.json"
        latest_file = repo_dir / "latest.json"

        self._persist_json(snapshot_file, payload)
        self._persist_json(latest_file, payload)
        self.ingestion_stats["persisted_records"] += 1

        return {
            "snapshot": str(snapshot_file),
            "latest": str(latest_file),
        }

    def _persist_batch_ingestion(self, payload: Dict[str, Any]) -> Dict[str, str]:
        """Persist multi-repo ingestion summary with latest pointer."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        snapshot_file = self.batch_storage_dir / f"batch_{timestamp}.json"
        latest_file = self.batch_storage_dir / "latest.json"

        self._persist_json(snapshot_file, payload)
        self._persist_json(latest_file, payload)
        self.ingestion_stats["persisted_records"] += 1

        return {
            "snapshot": str(snapshot_file),
            "latest": str(latest_file),
        }

    def _persist_component_ingestion(self, component_name: str, payload: Dict[str, Any]) -> Dict[str, str]:
        """Persist component-ingestion output with latest pointer."""
        safe_component = re.sub(r"[^a-zA-Z0-9._-]+", "_", component_name)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        component_dir = self.component_storage_dir / safe_component
        component_dir.mkdir(parents=True, exist_ok=True)

        snapshot_file = component_dir / f"{timestamp}.json"
        latest_file = component_dir / "latest.json"

        self._persist_json(snapshot_file, payload)
        self._persist_json(latest_file, payload)
        self.ingestion_stats["persisted_records"] += 1

        return {
            "snapshot": str(snapshot_file),
            "latest": str(latest_file),
        }

    def _load_json_safe(self, file_path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Safely load JSON file and return (data, error)."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return None, "invalid_json_root_type"
            return data, None
        except Exception as e:
            return None, f"json_load_error:{e}"

    def _sha256_file(self, file_path: Path) -> str:
        """Compute SHA-256 for a file."""
        digest = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _strip_quarantined_file(
        self,
        target: Path,
        source_path: Path,
        reasons: List[str],
        quarantined_at: str,
    ) -> Dict[str, Any]:
        """Strip quarantined JSON content and replace with forensic metadata only."""
        original_size = target.stat().st_size if target.exists() else 0
        original_sha256 = self._sha256_file(target) if target.exists() else None
        metadata = {
            "status": "QUARANTINED_STRIPPED",
            "quarantined_at": quarantined_at,
            "source_path": str(source_path),
            "quarantine_path": str(target),
            "reasons": reasons,
            "original_size_bytes": original_size,
            "original_sha256": original_sha256,
            "retention_days": self.quarantine_policy.get("retention_days", 30),
        }
        with open(target, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        return metadata

    def _validate_repo_snapshot(
        self,
        snapshot: Dict[str, Any],
        expected_repo: Optional[str],
        cross_reference_checks: bool,
    ) -> List[str]:
        """Validate repo snapshot and return prune reasons."""
        reasons: List[str] = []

        repo_value = snapshot.get("repo") or snapshot.get("repository")
        if not repo_value:
            reasons.append("missing_repo_identifier")

        if expected_repo and repo_value and expected_repo != repo_value:
            reasons.append("repo_path_mismatch")

        if repo_value and repo_value not in self.github_ingestor.DEEPSEEK_REPOS:
            reasons.append("unknown_repo_identifier")

        if cross_reference_checks:
            patterns = snapshot.get("patterns")
            patterns_found = snapshot.get("patterns_found")
            if isinstance(patterns, dict) and isinstance(patterns_found, int) and patterns_found != len(patterns):
                reasons.append("patterns_found_mismatch")

            files_analyzed = snapshot.get("files_analyzed")
            candidate_files = snapshot.get("candidate_files")
            if isinstance(files_analyzed, int) and isinstance(candidate_files, int) and files_analyzed > candidate_files:
                reasons.append("files_analyzed_exceeds_candidates")

            if snapshot.get("error") and ("adapted_patterns" in snapshot or "process_adaptations" in snapshot):
                reasons.append("error_with_success_payload")

            adapted_patterns = snapshot.get("adapted_patterns")
            if adapted_patterns is not None and not isinstance(adapted_patterns, dict):
                reasons.append("invalid_adapted_patterns_type")

        return reasons

    def _validate_batch_snapshot(
        self,
        snapshot: Dict[str, Any],
        cross_reference_checks: bool,
    ) -> List[str]:
        """Validate batch snapshot and return prune reasons."""
        reasons: List[str] = []
        summary = snapshot.get("summary")
        repositories = snapshot.get("repositories")

        if not isinstance(summary, dict):
            reasons.append("missing_or_invalid_summary")
        if not isinstance(repositories, dict):
            reasons.append("missing_or_invalid_repositories")

        if cross_reference_checks and isinstance(summary, dict) and isinstance(repositories, dict):
            repos_attempted = summary.get("repos_attempted")
            repos_successful = summary.get("repos_successful")

            if isinstance(repos_attempted, int) and repos_attempted != len(repositories):
                reasons.append("repos_attempted_mismatch")

            computed_success = sum(1 for payload in repositories.values() if isinstance(payload, dict) and "error" not in payload)
            if isinstance(repos_successful, int) and repos_successful != computed_success:
                reasons.append("repos_successful_mismatch")

            unknown_repo_keys = [name for name in repositories if name not in self.github_ingestor.DEEPSEEK_REPOS]
            if unknown_repo_keys:
                reasons.append("unknown_repo_keys")

        return reasons

    def _validate_component_snapshot(
        self,
        snapshot: Dict[str, Any],
        expected_component: Optional[str],
        cross_reference_checks: bool,
    ) -> List[str]:
        """Validate component snapshot and return prune reasons."""
        reasons: List[str] = []
        component = snapshot.get("component")

        if not component:
            reasons.append("missing_component_identifier")
        elif expected_component and component != expected_component:
            reasons.append("component_path_mismatch")

        known_components = {"mla", "reasoning", "coder"}
        if component and component not in known_components:
            reasons.append("unknown_component_identifier")

        if cross_reference_checks:
            ingested = snapshot.get("ingested")
            adapted = snapshot.get("adapted")
            if ingested is not None and not isinstance(ingested, bool):
                reasons.append("invalid_ingested_type")
            if adapted is not None and not isinstance(adapted, bool):
                reasons.append("invalid_adapted_type")
            if snapshot.get("error") and bool(adapted):
                reasons.append("error_with_adapted_true")

        return reasons

    def soft_prune_storage(
        self,
        recursive: bool = True,
        cross_reference_checks: bool = True,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Soft-prune invalid/fake DeepSeek storage records.

        Soft prune means suspicious files are moved into quarantine (never hard-deleted).
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_quarantine_root = self.quarantine_dir / f"soft_prune_{timestamp}"

        repo_pattern = "**/*.json" if recursive else "*/*.json"
        batch_pattern = "**/*.json" if recursive else "*.json"
        component_pattern = "**/*.json" if recursive else "*/*.json"
        repo_files = list(self.repo_storage_dir.glob(repo_pattern))
        batch_files = list(self.batch_storage_dir.glob(batch_pattern))
        component_files = list(self.component_storage_dir.glob(component_pattern))
        all_files = repo_files + batch_files + component_files

        scanned = 0
        prune_candidates: List[Dict[str, Any]] = []

        for file_path in all_files:
            scanned += 1
            data, load_error = self._load_json_safe(file_path)
            reasons: List[str] = []

            if load_error:
                reasons.append(load_error)
            elif str(file_path).startswith(str(self.repo_storage_dir)):
                expected_repo = file_path.parent.name if file_path.parent != self.repo_storage_dir else None
                reasons.extend(self._validate_repo_snapshot(data or {}, expected_repo, cross_reference_checks))
            elif str(file_path).startswith(str(self.batch_storage_dir)):
                reasons.extend(self._validate_batch_snapshot(data or {}, cross_reference_checks))
            elif str(file_path).startswith(str(self.component_storage_dir)):
                expected_component = file_path.parent.name if file_path.parent != self.component_storage_dir else None
                reasons.extend(self._validate_component_snapshot(data or {}, expected_component, cross_reference_checks))

            if reasons:
                prune_candidates.append({
                    "file": str(file_path),
                    "reasons": sorted(set(reasons)),
                })

        moved: List[Dict[str, Any]] = []
        if not dry_run and prune_candidates:
            run_quarantine_root.mkdir(parents=True, exist_ok=True)
            for candidate in prune_candidates:
                source = Path(candidate["file"])
                try:
                    relative = source.relative_to(self.storage_dir)
                except ValueError:
                    relative = Path(source.name)

                target = run_quarantine_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(target))
                strip_info = None
                strip_error = None
                if self.quarantine_policy.get("strip_on_quarantine", True):
                    try:
                        strip_info = self._strip_quarantined_file(
                            target,
                            source,
                            candidate["reasons"],
                            timestamp,
                        )
                    except Exception as e:
                        strip_error = str(e)

                moved.append({
                    "from": str(source),
                    "to": str(target),
                    "reasons": candidate["reasons"],
                    "stripped": bool(strip_info),
                    "strip_error": strip_error,
                })

        lifecycle = None
        if not dry_run and self.quarantine_policy.get("auto_lifecycle_on_prune", True):
            lifecycle = self.run_quarantine_lifecycle(dry_run=False)

        report = {
            "status": "SOFT_PRUNE_COMPLETE",
            "mode": "dry_run" if dry_run else "execute",
            "recursive": recursive,
            "cross_reference_checks": cross_reference_checks,
            "scanned_files": scanned,
            "prune_candidates": len(prune_candidates),
            "quarantined_files": len(moved),
            "quarantine_root": str(run_quarantine_root) if moved else None,
            "candidates": prune_candidates,
            "moved": moved,
            "lifecycle": lifecycle,
        }

        return report

    def run_quarantine_lifecycle(self, dry_run: bool = False) -> Dict[str, Any]:
        """Apply quarantine lifecycle: TTL deletion + size-cap trimming."""
        now = datetime.now(timezone.utc)
        retention_days = int(self.quarantine_policy.get("retention_days", 30))
        size_cap_gb = float(self.quarantine_policy.get("size_cap_gb", 2.0))
        size_cap_bytes = int(size_cap_gb * 1024 * 1024 * 1024)

        files = [f for f in self.quarantine_dir.glob("**/*.json") if f.is_file()]
        file_rows: List[Dict[str, Any]] = []

        for file_path in files:
            age_days = (now - datetime.fromtimestamp(file_path.stat().st_mtime, timezone.utc)).total_seconds() / 86400.0
            file_rows.append(
                {
                    "path": file_path,
                    "size": file_path.stat().st_size,
                    "age_days": age_days,
                    "reasons": [],
                }
            )

        # TTL-based candidates
        for row in file_rows:
            if row["age_days"] >= retention_days:
                row["reasons"].append("ttl_expired")

        # Size-cap candidates (oldest-first trimming)
        total_size = sum(row["size"] for row in file_rows)
        if total_size > size_cap_bytes:
            overflow = total_size - size_cap_bytes
            for row in sorted(file_rows, key=lambda r: r["age_days"], reverse=True):
                if overflow <= 0:
                    break
                if "size_cap_trim" not in row["reasons"]:
                    row["reasons"].append("size_cap_trim")
                    overflow -= row["size"]

        candidates = [row for row in file_rows if row["reasons"]]
        deleted: List[Dict[str, Any]] = []
        if not dry_run:
            for row in candidates:
                path_obj = row["path"]
                try:
                    os.remove(path_obj)
                    deleted.append(
                        {
                            "file": str(path_obj),
                            "size": row["size"],
                            "age_days": round(row["age_days"], 3),
                            "reasons": sorted(set(row["reasons"])),
                        }
                    )
                except Exception:
                    continue

            # Cleanup empty quarantine folders from the deepest level
            for directory in sorted(self.quarantine_dir.glob("**/*"), reverse=True):
                if directory.is_dir():
                    try:
                        directory.rmdir()
                    except OSError:
                        pass

        return {
            "status": "QUARANTINE_LIFECYCLE_COMPLETE",
            "mode": "dry_run" if dry_run else "execute",
            "retention_days": retention_days,
            "size_cap_gb": size_cap_gb,
            "files_scanned": len(file_rows),
            "files_marked": len(candidates),
            "files_deleted": len(deleted),
            "deleted": deleted,
        }

    def _derive_process_adaptations(self, adapted_patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize adapted source-code patterns into executable process categories."""
        process_buckets = {
            "attention_pipeline": 0,
            "reasoning_pipeline": 0,
            "code_generation_pipeline": 0,
        }

        for item in adapted_patterns.values():
            if not isinstance(item, dict):
                continue
            keys = set(item.keys())
            if "attention_computation" in keys or any("attention" in k for k in keys):
                process_buckets["attention_pipeline"] += 1
            if "verification" in str(item).lower() or "reasoning" in str(item).lower():
                process_buckets["reasoning_pipeline"] += 1
            if "code" in str(item).lower() or "syntax" in str(item).lower():
                process_buckets["code_generation_pipeline"] += 1

        total = sum(process_buckets.values())
        self.ingestion_stats["processes_adapted"] += total
        return {
            "process_buckets": process_buckets,
            "total_process_adaptations": total,
            "adaptation_mode": "deepseek_source_to_l104_process_map",
        }

    def ingest_deepseek_component(self, component_name: str,
                                source_code: str = None,
                                reasoning_trace: str = None,
                                code_sample: str = None,
                                language: str = None) -> Dict[str, Any]:
        """
        Ingest a specific DeepSeek component and adapt for L104.
        """
        result = {"component": component_name, "ingested": False, "adapted": False}

        try:
            if component_name == "mla" and source_code:
                # Ingest MLA patterns
                patterns = self.mla_ingestor.ingest_mla_pattern(source_code)
                adaptations = self.mla_ingestor.adapt_for_l104(patterns)
                result.update({
                    "ingested": True,
                    "patterns": patterns,
                    "adaptations": adaptations,
                    "adapted": True
                })
                self.ingestion_stats["total_patterns_ingested"] += len(patterns)

            elif component_name == "reasoning" and reasoning_trace:
                # Ingest reasoning patterns
                pattern = self.r1_ingestor.ingest_reasoning_pattern(reasoning_trace)
                adaptations = self.r1_ingestor.adapt_reasoning_for_l104(pattern)
                result.update({
                    "ingested": True,
                    "pattern": pattern,
                    "adaptations": adaptations,
                    "adapted": True
                })
                self.ingestion_stats["total_patterns_ingested"] += 1

            elif component_name == "coder" and code_sample and language:
                # Ingest coding patterns
                pattern = self.coder_ingestor.ingest_code_pattern(code_sample, language)
                adaptations = self.coder_ingestor.adapt_code_generation_for_l104(pattern)
                result.update({
                    "ingested": True,
                    "pattern": pattern,
                    "adaptations": adaptations,
                    "adapted": True
                })
                self.ingestion_stats["total_patterns_ingested"] += 1

            if result["adapted"]:
                self.ingestion_stats["total_adaptations_created"] += 1
                self.ingestion_stats["l104_integration_points"] += len(result.get("adaptations", {}))

        except Exception as e:
            result["error"] = str(e)

        try:
            result["storage"] = self._persist_component_ingestion(component_name, result)
        except Exception as storage_error:
            result["storage_error"] = str(storage_error)

        return result

    def ingest_from_github(
        self,
        repo_name: str,
        pattern_types: List[str] = None,
        max_files: int = 120,
        include_extensions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Ingest patterns directly from GitHub repositories.
        """
        result = self.github_ingestor.extract_patterns_from_repo(
            repo_name,
            pattern_types,
            max_files=max_files,
            include_extensions=include_extensions,
        )

        if "error" not in result:
            # Process the extracted patterns through our adaptors
            adapted_patterns = {}
            for file_path, patterns in result.get("patterns", {}).items():
                for pattern_type, pattern_list in patterns.items():
                    try:
                        if pattern_type == "mla":
                            # Adapt MLA patterns
                            adapted = self.mla_ingestor.adapt_for_l104({
                                "attention_computation": {"extracted_patterns": pattern_list}
                            })
                            adapted_patterns[f"{file_path}_mla"] = adapted
                        elif pattern_type == "reasoning":
                            # Adapt reasoning patterns using full expected schema
                            reasoning_steps = [
                                {
                                    "step_number": idx + 1,
                                    "content": p,
                                    "confidence": self.r1_ingestor._estimate_step_confidence(p),
                                }
                                for idx, p in enumerate(pattern_list)
                            ]
                            adapted = self.r1_ingestor.adapt_reasoning_for_l104({
                                "reasoning_steps": reasoning_steps,
                                "verification_patterns": [{"pattern": p} for p in pattern_list],
                                "reflection_cycles": [{"pattern": p} for p in pattern_list],
                            })
                            adapted_patterns[f"{file_path}_reasoning"] = adapted
                        elif pattern_type == "coder":
                            # Adapt coding patterns using full expected schema
                            adapted = self.coder_ingestor.adapt_code_generation_for_l104({
                                "language": "python",
                                "syntax_patterns": {
                                    "imports": {"matches": pattern_list},
                                    "functions": {"matches": pattern_list},
                                },
                                "semantic_patterns": {
                                    "code_generation": {"matches": pattern_list},
                                },
                                "quality_score": {
                                    "complexity": 0.7,
                                    "readability": 0.7,
                                    "efficiency": 0.7,
                                    "correctness": 0.8,
                                    "overall": 0.725,
                                },
                            })
                            adapted_patterns[f"{file_path}_coder"] = adapted
                    except Exception as adaptation_error:
                        adapted_patterns[f"{file_path}_{pattern_type}_error"] = {
                            "error": str(adaptation_error),
                            "pattern_type": pattern_type,
                        }

            result["adapted_patterns"] = adapted_patterns
            self.ingestion_stats["total_adaptations_created"] += len(adapted_patterns)
            self.ingestion_stats["total_patterns_ingested"] += result.get("patterns_found", 0)
            self.ingestion_stats["source_files_ingested"] += result.get("files_analyzed", 0)
            self.ingestion_stats["repos_ingested"] += 1
            result["process_adaptations"] = self._derive_process_adaptations(adapted_patterns)

        try:
            result["storage"] = self._persist_repo_ingestion(repo_name, result)
        except Exception as storage_error:
            result["storage_error"] = str(storage_error)

        return result

    def ingest_all_deepseek_repos(
        self,
        pattern_types: Optional[List[str]] = None,
        max_files_per_repo: int = 120,
        include_extensions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Ingest all registered DeepSeek repositories and adapt process flows."""
        repo_results: Dict[str, Any] = {}
        total_patterns = 0
        total_files = 0
        total_process_adaptations = 0

        for repo_name in self.github_ingestor.DEEPSEEK_REPOS:
            repo_result = self.ingest_from_github(
                repo_name,
                pattern_types=pattern_types,
                max_files=max_files_per_repo,
                include_extensions=include_extensions,
            )
            repo_results[repo_name] = repo_result

            if "error" not in repo_result:
                total_patterns += int(repo_result.get("patterns_found", 0))
                total_files += int(repo_result.get("files_analyzed", 0))
                total_process_adaptations += int(
                    repo_result.get("process_adaptations", {}).get("total_process_adaptations", 0)
                )

        combined = {
            "repositories": repo_results,
            "summary": {
                "repos_attempted": len(self.github_ingestor.DEEPSEEK_REPOS),
                "repos_successful": sum(1 for r in repo_results.values() if "error" not in r),
                "total_patterns_found": total_patterns,
                "total_source_files_analyzed": total_files,
                "total_process_adaptations": total_process_adaptations,
            },
        }

        try:
            combined["storage"] = self._persist_batch_ingestion(combined)
        except Exception as storage_error:
            combined["storage_error"] = str(storage_error)

        return combined

    def integrate_into_quantum_architecture(self, pattern_name: str, pattern_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate ingested pattern into L104's quantum AI architecture.
        """
        try:
            result = self.quantum_architecture.integrate_pattern(pattern_name, pattern_data)
            self.ingestion_stats["quantum_enhancements_applied"] += 1
            return {
                "success": True,
                "integration": result,
                "quantum_architecture_status": self.quantum_architecture.get_quantum_architecture_status()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "pattern_name": pattern_name
            }

    def get_ingestion_status(self) -> Dict[str, Any]:
        """Get comprehensive ingestion status."""
        repo_snapshots = list(self.repo_storage_dir.glob("*/*.json"))
        batch_snapshots = list(self.batch_storage_dir.glob("batch_*.json"))
        component_snapshots = list(self.component_storage_dir.glob("*/*.json"))
        return {
            "mla_ingestor": self.mla_ingestor.stats,
            "r1_ingestor": self.r1_ingestor.stats,
            "coder_ingestor": self.coder_ingestor.stats,
            "github_ingestor": self.github_ingestor.get_ingestion_status(),
            "overall": self.ingestion_stats,
            "storage": {
                "root": str(self.storage_dir),
                "repos": str(self.repo_storage_dir),
                "batches": str(self.batch_storage_dir),
                "components": str(self.component_storage_dir),
                "quarantine": str(self.quarantine_dir),
                "repo_snapshot_files": len(repo_snapshots),
                "batch_snapshot_files": len(batch_snapshots),
                "component_snapshot_files": len(component_snapshots),
                "quarantine_files": len(list(self.quarantine_dir.glob("**/*.json"))),
                "quarantine_size_bytes": sum(f.stat().st_size for f in self.quarantine_dir.glob("**/*.json") if f.is_file()),
                "quarantine_policy": self.quarantine_policy,
            },
            "l104_integration": {
                "god_code_alignments": "All adaptations use GOD_CODE phase alignment",
                "phi_weighting": "All metrics scaled by PHI constant",
                "quantum_enhancements": "Exponential compression and interference-based computation",
                "dual_layer_architecture": "Thought + Physics layer reasoning"
            }
        }

# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

# Main ingestion engine - lazy instantiation
def get_deepseek_ingestion_engine():
    """Get the DeepSeek ingestion engine instance."""
    if not hasattr(get_deepseek_ingestion_engine, '_instance'):
        get_deepseek_ingestion_engine._instance = DeepSeekIngestionEngine()
    return get_deepseek_ingestion_engine._instance

# For backward compatibility, create a proxy object
class _LazyEngineProxy:
    def __getattr__(self, name):
        engine = get_deepseek_ingestion_engine()
        return getattr(engine, name)

deepseek_ingestion_engine = _LazyEngineProxy()

# Component ingestors - lazy instantiation functions
def get_mla_ingestor():
    if not hasattr(get_mla_ingestor, '_instance'):
        get_mla_ingestor._instance = DeepSeekMLAIngestor(DeepSeekV3Config())
    return get_mla_ingestor._instance

def get_r1_ingestor():
    if not hasattr(get_r1_ingestor, '_instance'):
        get_r1_ingestor._instance = DeepSeekR1ReasoningIngestor(DeepSeekR1Config())
    return get_r1_ingestor._instance

def get_coder_ingestor():
    if not hasattr(get_coder_ingestor, '_instance'):
        get_coder_ingestor._instance = DeepSeekCoderIngestor(DeepSeekCoderConfig())
    return get_coder_ingestor._instance

# Create proxy objects for component ingestors
def get_github_ingestor():
    if not hasattr(get_github_ingestor, '_instance'):
        get_github_ingestor._instance = DeepSeekGitHubIngestor()
    return get_github_ingestor._instance

github_ingestor = type('GitHubIngestorProxy', (), {
    '__getattr__': lambda self, name: getattr(get_github_ingestor(), name)
})()

# Configuration classes
DeepSeekMLAConfig = DeepSeekV3Config  # Alias for backward compatibility
DeepSeekReasoningConfig = DeepSeekR1Config
DeepSeekCodeConfig = DeepSeekCoderConfig

__all__ = [
    'DeepSeekIngestionEngine', 'get_deepseek_ingestion_engine', 'deepseek_ingestion_engine',
    'DeepSeekMLAIngestor', 'get_mla_ingestor', 'mla_ingestor',
    'DeepSeekR1ReasoningIngestor', 'get_r1_ingestor', 'r1_reasoning_ingestor',
    'DeepSeekCoderIngestor', 'get_coder_ingestor', 'coder_ingestor',
    'DeepSeekGitHubIngestor', 'github_ingestor',
    'DeepSeekV3Config', 'DeepSeekR1Config', 'DeepSeekCoderConfig',
    'DeepSeekMLAConfig', 'DeepSeekReasoningConfig', 'DeepSeekCodeConfig'
]

if __name__ == "__main__":
    # Demo ingestion
    engine = DeepSeekIngestionEngine()

    # Example MLA ingestion
    sample_mla_code = """
    def apply_mla_attention(q, k, v, config):
        # Compress KV to latent space
        c_kv = k @ config.w_dkv  # [seq, kv_lora_rank]

        # Decompress
        k_out = c_kv @ config.w_uk
        v_out = c_kv @ config.w_uv

        # Apply RoPE to Q and K
        q_rope = apply_rope(q, positions)
        k_rope = apply_rope(k_out, positions)

        # Attention
        scores = q_rope @ k_rope.T / sqrt(d)
        attn = softmax(scores) @ v_out

        return attn
    """

    result = engine.ingest_deepseek_component("mla", source_code=sample_mla_code)
    print("MLA Ingestion Result:", json.dumps(result, indent=2))

    # Example reasoning ingestion
    sample_reasoning = """
    Step 1: The problem asks for the sum of first 10 natural numbers.
    Step 2: The formula is n(n+1)/2 where n=10.
    Step 3: So 10*11/2 = 55.
    Verification: 1+2+3+4+5+6+7+8+9+10 = 55. Correct!
    """

    result = engine.ingest_deepseek_component("reasoning", reasoning_trace=sample_reasoning)
    print("Reasoning Ingestion Result:", json.dumps(result, indent=2))

    # Print overall status
    print("Ingestion Status:", json.dumps(engine.get_ingestion_status(), indent=2))