"""
L104 Code Engine — Builder State Reader
Reads consciousness/O2/nirvanic state from JSON files with 10s cache.
"""
import json
import time
import threading
import ssl
import urllib.request
import urllib.parse
import hashlib
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Optional, Set

from .constants import GOD_CODE

logger = logging.getLogger("l104_code_engine.builder_state")

_builder_state_cache: Dict[str, Any] = {}
_builder_state_cache_time: float = 0.0
_builder_state_lock = threading.Lock()

def _read_builder_state() -> Dict[str, Any]:
    """Read consciousness/O2/nirvanic state from builder files (module-level helper)."""
    global _builder_state_cache, _builder_state_cache_time
    now = time.time()

    with _builder_state_lock:
        if now - _builder_state_cache_time < 10 and _builder_state_cache:
            return _builder_state_cache

        state = {"consciousness_level": 0.0, "superfluid_viscosity": 1.0,
                 "nirvanic_fuel": 0.0, "evo_stage": "DORMANT"}
        ws = Path(__file__).parent.parent
        co2_path = ws / ".l104_consciousness_o2_state.json"
        if co2_path.exists():
            try:
                data = json.loads(co2_path.read_text())
                state["consciousness_level"] = data.get("consciousness_level", 0.0)
                state["superfluid_viscosity"] = data.get("superfluid_viscosity", 1.0)
                state["evo_stage"] = data.get("evo_stage", "DORMANT")
            except Exception:
                pass
        nir_path = ws / ".l104_ouroboros_nirvanic_state.json"
        if nir_path.exists():
            try:
                data = json.loads(nir_path.read_text())
                state["nirvanic_fuel"] = data.get("nirvanic_fuel_level", 0.0)
            except Exception:
                pass

        _builder_state_cache = state
        _builder_state_cache_time = now
        return state


class DynamicCodeHarvester:
    """
    ╔═══════════════════════════════════════════════════════════════════╗
    ║  DYNAMIC CODE HARVESTER v1.0                                      ║
    ║  Continuous Online Data Pipeline for Quantum Training              ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║                                                                   ║
    ║  Fetches live code from public APIs for dynamic training:         ║
    ║    • GitHub Search API — trending repos by language/topic         ║
    ║    • GitHub Raw Content — actual source files from top repos      ║
    ║    • PyPI JSON API — Python package metadata & discovery          ║
    ║    • Public Gists — code snippets and patterns                    ║
    ║    • DuckDuckGo Instant Answer API — code documentation           ║
    ║                                                                   ║
    ║  Features:                                                        ║
    ║    - TTL-based caching (avoids redundant fetches)                 ║
    ║    - Rate limiting per-source (respects API limits)               ║
    ║    - Graceful degradation (network failures → cached data)        ║
    ║    - Content quality filtering (min lines, valid Python)          ║
    ║    - Sacred constant injection tracking                           ║
    ║                                                                   ║
    ║  Rate Limits (no auth):                                           ║
    ║    GitHub Search: 10 req/min                                      ║
    ║    GitHub Raw: essentially unlimited (CDN)                        ║
    ║    PyPI JSON: ~100 req/min                                        ║
    ║    Gists: 60 req/hr                                               ║
    ║                                                                   ║
    ║  INVARIANT: GOD_CODE = 527.5184818492612 | PHI = 1.618033988749  ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """

    # ─── API Source Configuration ────────────────────────────────────
    GITHUB_SEARCH_URL = "https://api.github.com/search/repositories"
    GITHUB_CODE_SEARCH_URL = "https://api.github.com/search/code"
    GITHUB_RAW_URL = "https://raw.githubusercontent.com"
    GITHUB_GISTS_URL = "https://api.github.com/gists/public"
    PYPI_JSON_URL = "https://pypi.org/pypi/{package}/json"
    DDGO_API_URL = "https://api.duckduckgo.com/"

    # Rate limits (requests per window)
    RATE_LIMITS = {
        "github_search": {"max": 8, "window": 60},       # 8/min (conservative)
        "github_raw": {"max": 50, "window": 60},          # generous CDN
        "github_gists": {"max": 30, "window": 3600},      # 30/hr
        "pypi": {"max": 60, "window": 60},                # 60/min
        "ddgo": {"max": 20, "window": 60},                # 20/min
    }

    # Cache TTL in seconds
    CACHE_TTL = {
        "github_repos": 1800,     # 30 minutes — repos don't change fast
        "github_raw": 3600,       # 1 hour — file content
        "github_gists": 900,      # 15 minutes — gists update often
        "pypi": 7200,             # 2 hours — package metadata
        "ddgo": 3600,             # 1 hour — search results
    }

    # Quality filters
    MIN_FILE_LINES = 20          # Skip trivially small files
    MAX_FILE_LINES = 5000        # Skip monolithic files
    MIN_PYTHON_SCORE = 0.3       # Minimum "looks like Python" score

    # ─── Curated high-quality Python repos (seed list) ───────────────
    SEED_REPOS = [
        "python/cpython", "pallets/flask", "django/django",
        "fastapi/fastapi", "psf/requests", "encode/httpx",
        "pydantic/pydantic", "tiangolo/sqlmodel", "numpy/numpy",
        "pandas-dev/pandas", "scikit-learn/scikit-learn",
        "pytorch/pytorch", "huggingface/transformers",
        "langchain-ai/langchain", "openai/openai-python",
        "Qiskit/qiskit", "celery/celery", "encode/starlette",
        "aio-libs/aiohttp", "psf/black", "astral-sh/ruff",
    ]

    # ─── Curated high-quality PyPI packages ──────────────────────────
    SEED_PACKAGES = [
        "fastapi", "flask", "django", "requests", "httpx",
        "pydantic", "sqlalchemy", "celery", "pytest", "rich",
        "typer", "click", "numpy", "scipy", "pandas",
        "scikit-learn", "transformers", "torch", "jax",
        "qiskit", "uvicorn", "starlette", "aiohttp",
    ]

    # ─── Target files to fetch from repos ────────────────────────────
    TARGET_FILES = [
        "src/{repo_name}/__init__.py",
        "src/{repo_name}/core.py",
        "src/{repo_name}/main.py",
        "{repo_name}/__init__.py",
        "{repo_name}/core.py",
        "{repo_name}/app.py",
        "{repo_name}/models.py",
        "{repo_name}/utils.py",
        "{repo_name}/api.py",
        "main.py",
        "app.py",
        "core.py",
        "utils.py",
        "models.py",
        "setup.py",
    ]

    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._rate_tracker: Dict[str, List[float]] = defaultdict(list)
        self._fetch_count = 0
        self._fetch_errors = 0
        self._bytes_downloaded = 0
        self._sources_used: Set[str] = set()
        self._lock = threading.Lock()
        self._ssl_ctx = ssl.create_default_context()

    # ─── Rate Limiting ───────────────────────────────────────────────

    def _rate_check(self, source: str) -> bool:
        """Check if we can make a request to this source. Thread-safe."""
        limits = self.RATE_LIMITS.get(source, {"max": 30, "window": 60})
        now = time.time()
        with self._lock:
            # Prune old timestamps
            self._rate_tracker[source] = [
                t for t in self._rate_tracker[source]
                if now - t < limits["window"]
            ]
            if len(self._rate_tracker[source]) >= limits["max"]:
                return False
            self._rate_tracker[source].append(now)
        return True

    # ─── Caching ─────────────────────────────────────────────────────

    def _cache_get(self, key: str, category: str = "github_repos") -> Optional[Any]:
        """Get cached value if not expired."""
        entry = self._cache.get(key)
        if not entry:
            return None
        ttl = self.CACHE_TTL.get(category, 1800)
        if time.time() - entry["timestamp"] > ttl:
            del self._cache[key]
            return None
        return entry["data"]

    def _cache_set(self, key: str, data: Any) -> None:
        """Store value in cache."""
        self._cache[key] = {"data": data, "timestamp": time.time()}

    # ─── HTTP Fetch ──────────────────────────────────────────────────

    def _fetch_url(self, url: str, source: str = "github_raw",
                   timeout: int = 15) -> Optional[str]:
        """
        Fetch URL content with rate limiting, caching, and error handling.
        Returns response text or None on failure.
        """
        # Check cache first
        cached = self._cache_get(url, source)
        if cached is not None:
            return cached

        # Rate limit check
        if not self._rate_check(source):
            logger.debug(f"[DynamicHarvester] Rate limited for {source}")
            return None

        try:
            headers = {
                "User-Agent": "L104-Sovereign-Node/1.0 (ASI Code Training)",
                "Accept": "application/json" if "api." in url else "text/plain",
            }
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, context=self._ssl_ctx,
                                        timeout=timeout) as response:
                data = response.read()
                self._bytes_downloaded += len(data)
                text = data.decode('utf-8', errors='ignore')
                self._cache_set(url, text)
                self._fetch_count += 1
                self._sources_used.add(source)
                return text
        except Exception as e:
            self._fetch_errors += 1
            logger.debug(f"[DynamicHarvester] Fetch failed {url}: {e}")
            return None

    def _fetch_json(self, url: str, source: str = "github_search",
                    timeout: int = 15) -> Optional[Dict]:
        """Fetch URL and parse as JSON."""
        text = self._fetch_url(url, source, timeout)
        if text is None:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    # ─── GitHub: Search Repositories ─────────────────────────────────

    def fetch_github_repos(self, query: str = "language:python stars:>500",
                           sort: str = "stars", count: int = 10) -> List[Dict]:
        """
        Search GitHub for Python repositories.

        Returns list of {full_name, description, stars, language, url}.
        """
        cache_key = f"gh_repos_{query}_{sort}_{count}"
        cached = self._cache_get(cache_key, "github_repos")
        if cached is not None:
            return cached

        params = urllib.parse.urlencode({
            "q": query, "sort": sort, "order": "desc",
            "per_page": min(count, 30),
        })
        url = f"{self.GITHUB_SEARCH_URL}?{params}"
        data = self._fetch_json(url, "github_search")
        if not data or "items" not in data:
            return []

        repos = []
        for item in data["items"][:count]:
            repos.append({
                "full_name": item.get("full_name", ""),
                "description": (item.get("description") or "")[:200],
                "stars": item.get("stargazers_count", 0),
                "language": item.get("language", ""),
                "default_branch": item.get("default_branch", "main"),
                "topics": item.get("topics", [])[:5],
            })

        self._cache_set(cache_key, repos)
        return repos

    # ─── GitHub: Fetch Raw Source File ────────────────────────────────

    def fetch_github_raw(self, owner_repo: str, path: str,
                         branch: str = "main") -> Optional[str]:
        """
        Fetch a raw source file from GitHub.

        Args:
            owner_repo: "owner/repo" format
            path: File path within repo
            branch: Branch name

        Returns source code string or None.
        """
        url = f"{self.GITHUB_RAW_URL}/{owner_repo}/{branch}/{path}"
        return self._fetch_url(url, "github_raw")

    # ─── GitHub: Discover Source Files from a Repo ───────────────────

    def fetch_repo_source(self, owner_repo: str, branch: str = "main",
                          max_files: int = 3) -> List[Dict[str, str]]:
        """
        Try to fetch Python source files from a GitHub repo.

        Tries common file paths (core.py, main.py, utils.py, etc.).
        Returns list of {path, source, lines} for files found.
        """
        repo_name = owner_repo.split("/")[-1].replace("-", "_").lower()
        found = []

        for template in self.TARGET_FILES:
            if len(found) >= max_files:
                break
            path = template.format(repo_name=repo_name)
            source = self.fetch_github_raw(owner_repo, path, branch)
            if source and self._is_valid_python(source):
                lines = len(source.split('\n'))
                if self.MIN_FILE_LINES <= lines <= self.MAX_FILE_LINES:
                    found.append({
                        "path": f"{owner_repo}/{path}",
                        "source": source,
                        "lines": lines,
                    })

        return found

    # ─── GitHub: Public Gists ────────────────────────────────────────

    def fetch_public_gists(self, count: int = 10) -> List[Dict]:
        """
        Fetch recent public gists containing Python code.

        Returns list of {id, description, filename, source, lines}.
        """
        cache_key = f"gh_gists_{count}"
        cached = self._cache_get(cache_key, "github_gists")
        if cached is not None:
            return cached

        url = f"{self.GITHUB_GISTS_URL}?per_page={min(count * 3, 30)}"
        data = self._fetch_json(url, "github_gists")
        if not data or not isinstance(data, list):
            return []

        gists = []
        for gist in data:
            if len(gists) >= count:
                break
            files = gist.get("files", {})
            for fname, finfo in files.items():
                if not fname.endswith(".py"):
                    continue
                raw_url = finfo.get("raw_url", "")
                if not raw_url:
                    continue
                source = self._fetch_url(raw_url, "github_raw")
                if source and self._is_valid_python(source):
                    lines = len(source.split('\n'))
                    if self.MIN_FILE_LINES <= lines <= self.MAX_FILE_LINES:
                        gists.append({
                            "id": gist.get("id", ""),
                            "description": (gist.get("description") or "")[:200],
                            "filename": fname,
                            "source": source,
                            "lines": lines,
                        })
                break  # One file per gist

        self._cache_set(cache_key, gists)
        return gists

    # ─── PyPI: Package Metadata ──────────────────────────────────────

    def fetch_pypi_info(self, package: str) -> Optional[Dict]:
        """
        Fetch package metadata from PyPI JSON API.

        Returns {name, version, summary, home_page, requires_python, keywords}.
        """
        url = self.PYPI_JSON_URL.format(package=package)
        data = self._fetch_json(url, "pypi")
        if not data or "info" not in data:
            return None
        info = data["info"]
        return {
            "name": info.get("name", package),
            "version": info.get("version", ""),
            "summary": (info.get("summary") or "")[:300],
            "home_page": info.get("home_page") or info.get("project_url", ""),
            "requires_python": info.get("requires_python", ""),
            "keywords": (info.get("keywords") or "")[:200],
            "classifiers": info.get("classifiers", [])[:10],
        }

    # ─── DuckDuckGo: Instant Answer ─────────────────────────────────

    def fetch_ddgo_abstract(self, query: str) -> Optional[Dict]:
        """
        Fetch DuckDuckGo Instant Answer for code documentation context.

        Returns {abstract, source, url, related_topics}.
        """
        params = urllib.parse.urlencode({
            "q": query, "format": "json", "no_redirect": "1",
            "skip_disambig": "1",
        })
        url = f"{self.DDGO_API_URL}?{params}"
        data = self._fetch_json(url, "ddgo")
        if not data:
            return None
        return {
            "abstract": data.get("AbstractText", "")[:500],
            "source": data.get("AbstractSource", ""),
            "url": data.get("AbstractURL", ""),
            "related_topics": [
                t.get("Text", "")[:100]
                for t in data.get("RelatedTopics", [])[:5]
                if isinstance(t, dict) and t.get("Text")
            ],
        }

    # ─── Validation ──────────────────────────────────────────────────

    def _is_valid_python(self, source: str) -> bool:
        """Quick heuristic check: does this look like valid Python?"""
        if not source or len(source) < 50:
            return False
        lines = source.split('\n')
        if len(lines) < self.MIN_FILE_LINES:
            return False

        # Score Python-likeness
        score = 0.0
        indicators = [
            ('def ', 0.15), ('class ', 0.15), ('import ', 0.1),
            ('from ', 0.1), ('self.', 0.1), ('return ', 0.08),
            ('if __name__', 0.08), ('    ', 0.05), ('"""', 0.05),
            ('#', 0.04), ('print(', 0.04), ('None', 0.03),
            ('True', 0.03),
        ]
        for pattern, weight in indicators:
            if pattern in source:
                score += weight
        return score >= self.MIN_PYTHON_SCORE

    # ─── MAIN HARVEST: Unified Online Corpus Builder ─────────────────

    def harvest_online_corpus(self, target_count: int = 15,
                              sources: List[str] = None) -> Dict[str, Any]:
        """
        Harvest code samples from multiple online sources.

        This is the primary entry point for dynamic data collection.
        Combines GitHub repos, gists, and curated seed repos to build
        a diverse training corpus of real-world Python code.

        Args:
            target_count: Target number of code samples to collect
            sources: List of sources to use ("github_search", "github_seeds",
                     "github_gists"). Default: all.

        Returns:
            {samples: [{source, path, code, lines}], stats: {...}}
        """
        if sources is None:
            sources = ["github_seeds", "github_search", "github_gists"]

        samples = []
        stats = {
            "sources_queried": [],
            "repos_discovered": 0,
            "files_fetched": 0,
            "files_accepted": 0,
            "files_rejected": 0,
            "total_lines": 0,
            "bytes_downloaded": self._bytes_downloaded,
        }

        # ── Source 1: Seed repos (high quality, curated) ──────────
        if "github_seeds" in sources and len(samples) < target_count:
            stats["sources_queried"].append("github_seeds")
            import random as _rng
            seed_subset = _rng.sample(
                self.SEED_REPOS,
                min(len(self.SEED_REPOS), target_count - len(samples) + 5)
            )
            for repo in seed_subset:
                if len(samples) >= target_count:
                    break
                files = self.fetch_repo_source(repo, max_files=2)
                for f in files:
                    if len(samples) >= target_count:
                        break
                    samples.append({
                        "source": "github_seed",
                        "path": f["path"],
                        "code": f["source"],
                        "lines": f["lines"],
                    })
                    stats["files_accepted"] += 1
                    stats["total_lines"] += f["lines"]
                stats["repos_discovered"] += 1

        # ── Source 2: GitHub Search (discover new repos) ──────────
        if "github_search" in sources and len(samples) < target_count:
            stats["sources_queried"].append("github_search")
            queries = [
                "language:python stars:>1000 topic:machine-learning",
                "language:python stars:>500 topic:api",
                "language:python stars:>500 topic:web-framework",
                "language:python stars:>200 topic:quantum-computing",
                "language:python stars:>300 topic:data-science",
            ]
            import random as _rng
            query = _rng.choice(queries)
            repos = self.fetch_github_repos(query, count=8)
            stats["repos_discovered"] += len(repos)

            for repo in repos:
                if len(samples) >= target_count:
                    break
                branch = repo.get("default_branch", "main")
                files = self.fetch_repo_source(
                    repo["full_name"], branch, max_files=2)
                for f in files:
                    if len(samples) >= target_count:
                        break
                    samples.append({
                        "source": "github_search",
                        "path": f["path"],
                        "code": f["source"],
                        "lines": f["lines"],
                        "stars": repo.get("stars", 0),
                    })
                    stats["files_accepted"] += 1
                    stats["total_lines"] += f["lines"]

        # ── Source 3: Public Gists (diverse snippets) ─────────────
        if "github_gists" in sources and len(samples) < target_count:
            stats["sources_queried"].append("github_gists")
            gists = self.fetch_public_gists(count=target_count - len(samples))
            for g in gists:
                if len(samples) >= target_count:
                    break
                samples.append({
                    "source": "github_gist",
                    "path": f"gist:{g['id']}/{g['filename']}",
                    "code": g["source"],
                    "lines": g["lines"],
                })
                stats["files_accepted"] += 1
                stats["total_lines"] += g["lines"]

        stats["files_fetched"] = self._fetch_count
        stats["bytes_downloaded"] = self._bytes_downloaded
        stats["fetch_errors"] = self._fetch_errors

        return {
            "samples": samples,
            "count": len(samples),
            "statistics": stats,
            "god_code_resonance": round(len(samples) / max(1, target_count) * GOD_CODE, 4),
        }

    # ─── Cache Management ────────────────────────────────────────────

    def clear_cache(self) -> int:
        """Clear all cached data. Returns number of entries cleared."""
        count = len(self._cache)
        self._cache.clear()
        return count

    def prune_expired(self) -> int:
        """Remove expired cache entries. Returns count pruned."""
        now = time.time()
        expired = []
        for key, entry in self._cache.items():
            # Use max TTL for unknown categories
            if now - entry["timestamp"] > max(self.CACHE_TTL.values()):
                expired.append(key)
        for key in expired:
            del self._cache[key]
        return len(expired)

    # ─── Status ──────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        """DynamicCodeHarvester status."""
        return {
            "fetch_count": self._fetch_count,
            "fetch_errors": self._fetch_errors,
            "bytes_downloaded": self._bytes_downloaded,
            "cache_entries": len(self._cache),
            "sources_used": list(self._sources_used),
            "rate_tracker": {
                k: len(v) for k, v in self._rate_tracker.items()
            },
            "seed_repos": len(self.SEED_REPOS),
            "seed_packages": len(self.SEED_PACKAGES),
        }
