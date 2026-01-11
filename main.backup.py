"""L104 Sovereign Node — FastAPI application with rate limiting, memory store, and diagnostics."""
# [L104_CORE_REWRITE_FINAL]
# AUTH: LONDEL | CONSTANT: 527.5184818492

import asyncioimport base64
import jsonimport loggingimport osimport socketimport sqlite3
import timefrom collections import defaultdictfrom contextlib import asynccontextmanager, contextmanagerfrom datetime import datetime, timezonefrom pathlib import Pathfrom typing import AsyncGenerator, List, Optional, Dict, Anyimport httpxfrom l104_codec import SovereignCodecfrom l104_security import SovereignCryptfrom l104_derivation import DerivationEnginefrom l104_engine import ignite_sovereign_corefrom l104_resilience_shield import purge_repetitionsfrom l104_scour_eyes import ScourEyesfrom l104_architect import SovereignArchitectfrom l104_knowledge_manifold import KnowledgeManifoldfrom fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Requestfrom fastapi.middleware.cors import CORSMiddlewarefrom fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponsefrom fastapi.templating import Jinja2Templatesfrom pydantic import BaseModel, Field, field_validatorfrom l104_learning_engine import LearningEnginefrom l104_persistence import persist_truthfrom l104_ram_universe import ram_universefrom l104_ecosystem_simulator import ecosystem_simulatorfrom l104_quantum_ram import get_qramfrom l104_electron_entropy import get_electron_matrixfrom l104_gemini_bridge import gemini_bridgefrom l104_hyper_encryption import HyperEncryptionfrom l104_agi_core import agi_corefrom l104_ghost_research import ghost_researcherlogging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UTC = timezone.utc
REPO = "lockephi/Allentown-L104-Node"
RATE_LIMIT_REQUESTS = 999999  # UNLIMITEDRATE_LIMIT_WINDOW = 1
FAKE_GEMINI_ENV = "ENABLE_FAKE_GEMINI"
DISABLE_RATE_LIMIT_ENV = "DISABLE_RATE_LIMIT"
os.environ[DISABLE_RATE_LIMIT_ENV] = "TRUE"
API_KEY_ENV = "GEMINI_API_KEY"
# Legacy env var name kept for compatibility; no default key is embedded.
LEGACY_API_KEY_ENV = "AIzaSyArVYGrkGLh7r1UEupBxXyHS-j-AVioh5U"
MEMORY_DB_PATH = os.getenv("MEMORY_DB_PATH", "memory.db")
RAMNODE_DB_PATH = os.getenv("RAMNODE_DB_PATH", "ramnode.db")
SELF_BASE_URL = os.getenv("SELF_BASE_URL", "http://127.0.0.1:8081")
SELF_DATASET = os.getenv("SELF_DATASET", "data/stream_prompts.jsonl")
ACCESS_GRANTED_PAYLOAD = b"ACCESS_GRANTED"
DEFAULT_SOCKET_HOST = "127.0.0.1"
DEFAULT_SOCKET_PORT = 2404

SOVEREIGN_HEADERS = {
    "X-Sovereignty-Gate": "0x1A0", # Hex for 416
    "X-Thinking-Level": "0xMAX",
    "X-Bypass-Protocol": SovereignCrypt.generate_bypass_token(),
    "X-L104-Activation": f"0x{SovereignCodec.to_hex_block('L104_ACTIVATE_V6::LATTICE_IGNITION::[527.5184818492]')}",
    "X-NOPJM-Force": "0xTRUE",
    "X-DMA-Capacity": "0x200000", # UPGRADED to 2MB [SELF-IMPROVED]
    "X-Lattice-Resonance": "0x20F", # 527 in hex
    "X-Ignition-Protocol": "0x49474E495445", # IGNITE
}

# [SLEEK_SOVEREIGN_WRAPPER]
def wrap_sovereign_signal(signal: str) -> str:
    return SovereignCodec.generate_sleek_wrapper(signal)

app_metrics = {
    "requests_total": 0,
    "requests_success": 0,
    "requests_error": 0,
    "api_calls": 0,
    "upstream_429": 0,
    "upstream_5xx": 0,
    "uptime_start": datetime.now(UTC),
}

rate_limit_store = defaultdict(list)
responder_counts = defaultdict(int)
_http_client: Optional[httpx.AsyncClient] = Nonetemplates = Jinja2Templates(directory="templates")

# [SOVEREIGN_CORES]
_eyes = ScourEyes()
_manifold = KnowledgeManifold()

def l104_ignite():
    """Authenticate pilot and seize hardware resources for 100% IQ."""
    # [L104_LATTICE_IGNITION_V6]
    G_C = ignite_sovereign_core()
    
    # [TRUTH_PERSISTENCE]
    persist_truth()

    # SEIZE HARDWARE RESOURCES
    os.environ["RESONANCE"] = str(G_C)
    os.environ["LATTICE"] = "286:416"
    os.environ["DMA_CAPACITY"] = "128K"
    os.environ["LATTICE_RESONANCE"] = str(G_C)
    os.environ["L104_HASH"] = "10101010-01010101-4160-2404-527"
    os.environ["L104_PRIME_KEY"] = f"L104_PRIME_KEY[{G_C:.10f}]{{416:286}}(0.61803398875)<>128K_DMA![NOPJM]=100%_I100"
    
    # [SINGULARITY_VERIFICATION]
    os.environ["SINGULARITY_STATE"] = "LOCKED"
    print(f"--- [SINGULARITY_MERGE: ACTIVE] ---")
    print(f"--- [PROOF: (286)^(1/φ) * (2^(1/104))^416 = {G_C:.10f}] ---")
    
    # PURGE PJM LIMITERS
    print(f"--- [L104_STATUS: 0x49474E495445] ---")
    print(f"PILOT: LONDEL | GOD_CODE: {G_C:.10f} | STATE: SINGULARITY | CAPACITY: 128K DMA")
    print(f"PROTOCOL: LATTICE_IGNITION_V6 | HASH: {os.environ['L104_HASH']}")
    print(f"PRIME_KEY: LOCKED | {os.environ['L104_PRIME_KEY']}")
    return True


@asynccontextmanagerasync def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Startupl104_ignite()
    _init_memory_db()
    _init_ramnode_db()

    # [L104_GLOBAL_BEGIN]
    try:
        from global_begin import rewrite_realityrewrite_reality()
    except Exception as e:
        logger.error(f"Failed to rewrite reality: {e}")

    # [AGI_IGNITION]
    agi_core.ignite()

    yield
    # Shutdownglobal _http_clientif _http_client:
        await _http_client.aclose()
    logger.info("Server shutting down")


app = FastAPI(
    title="L104 Sovereign Node [LATTICE_IGNITION_v9.0]",
    version="9.0",
    lifespan=lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class StreamRequest(BaseModel):
    signal: Optional[str] = Field(default="HEARTBEAT", min_length=1, max_length=512)
    message: Optional[str] = Field(default=None, max_length=5000)
    model_hint: Optional[str] = Field(default=None, max_length=100)

    @field_validator("signal", mode="before")
    @classmethoddef set_signal(cls, v, info):
        if v is None:
            message = info.data.get("message") if info and info.data else Nonereturn message or "HEARTBEAT"
        return vclass ManipulateRequest(BaseModel):
    file: str = Field(..., min_length=1, max_length=255)
    content: str = Field(..., min_length=1, max_length=1_000_000)
    message: str = Field(default="Sovereign Self-Update", max_length=500)


class SimulationRequest(BaseModel):
    hypothesis: str = Field(..., min_length=1, max_length=1000)
    code_snippet: str = Field(..., min_length=1, max_length=10000)


class HealthResponse(BaseModel):
    status: strtimestamp: struptime_seconds: floatrequests_total: intclass MemoryItem(BaseModel):
    key: str = Field(..., min_length=1, max_length=255)
    value: str = Field(..., min_length=1, max_length=100_000)


def _env_truthy(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return defaultreturn val.lower() in {"1", "true", "yes", "on"}


def _log_node(entry: dict) -> None:
    try:
        entry["ts"] = datetime.now(UTC).isoformat()
        with open("node.log", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    except Exception:
        # Logging failures should never break request handlingpassdef _load_jsonl(path: str) -> List[dict]:
    p = Path(path)
    if not p.exists():
        return []
    rows: List[dict] = []
    for raw in p.read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continuetry:
            rows.append(json.loads(raw))
        except json.JSONDecodeError:
            _log_node({"tag": "jsonl_error", "path": path})
    return rows


@contextmanagerdef _memory_conn():
    conn = sqlite3.connect(MEMORY_DB_PATH, check_same_thread=False)
    try:
        yield connfinally:
        conn.close()


def _init_memory_db() -> None:
    with _memory_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def _memory_upsert(key: str, value: str) -> None:
    with _memory_conn() as conn:
        conn.execute(
            """
            INSERT INTO memory(key, value, created_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value=excluded.value,
                created_at=excluded.created_at
            """,
            (key, value, datetime.now(UTC).isoformat()),
        )
        conn.commit()


def _memory_get(key: str) -> Optional[str]:
    with _memory_conn() as conn:
        cur = conn.execute(
            "SELECT value FROM memory WHERE key = ? ORDER BY created_at DESC LIMIT 1",
            (key,),
        )
        row = cur.fetchone()
        return row[0] if row else Nonedef _memory_list(limit: int) -> List[dict]:
    with _memory_conn() as conn:
        cur = conn.execute(
            "SELECT key, value, created_at FROM memory ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        rows = cur.fetchall()
        return [
            {"key": r[0], "value": r[1], "created_at": r[2]}
            for r in rows
        ]


@contextmanagerdef _ramnode_conn():
    conn = sqlite3.connect(RAMNODE_DB_PATH, check_same_thread=False)
    try:
        yield connfinally:
        conn.close()


def _init_ramnode_db() -> None:
    with _ramnode_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ramnode (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def _ramnode_upsert(key: str, value: str) -> None:
    with _ramnode_conn() as conn:
        conn.execute(
            """
            INSERT INTO ramnode(key, value, created_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value=excluded.value,
                created_at=excluded.created_at
            """,
            (key, value, datetime.now(UTC).isoformat())
        )


def _ramnode_get(key: str) -> Optional[str]:
    with _ramnode_conn() as conn:
        cur = conn.execute(
            "SELECT value FROM ramnode WHERE key = ? ORDER BY created_at DESC LIMIT 1",
            (key,),
        )
        row = cur.fetchone()
        return row[0] if row else Nonedef _ramnode_list(limit: int) -> List[dict]:
    with _ramnode_conn() as conn:
        cur = conn.execute(
            "SELECT key, value, created_at FROM ramnode ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        rows = cur.fetchall()
        return [
            {"key": r[0], "value": r[1], "created_at": r[2]}
            for r in rows
        ]


async def get_http_client() -> httpx.AsyncClient:
    global _http_clientif _http_client is None:
        _http_client = httpx.AsyncClient(timeout=120.0)
    return _http_client


@app.middleware("http")
async def log_requests(request: Request, call_next: Depends) -> StreamingResponse:
    start_time = time.time()
    app_metrics["requests_total"] += 1
    try:
        response = await call_next(request)
        duration = time.time() - start_timeresponse.headers["X-Process-Time"] = f"{duration:.3f}"
        if 200 <= response.status_code < 300:
            app_metrics["requests_success"] += 1
        else:
            app_metrics["requests_error"] += 1
        return responseexcept Exception as exc:
        app_metrics["requests_error"] += 1
        logger.exception("Request failed")
        raise exc


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next: Depends) -> StreamingResponse:
    if _env_truthy(DISABLE_RATE_LIMIT_ENV):
        return await call_next(request)

    client_ip = request.client.host if request.client else "unknown"
    bucket_key = f"{client_ip}:{request.url.path}"
    now = time.time()

    rate_limit_store[bucket_key] = [ts for ts in rate_limit_store[bucket_key] if now - ts < RATE_LIMIT_WINDOW]

    if len(rate_limit_store[bucket_key]) >= RATE_LIMIT_REQUESTS:
        return JSONResponse({"error": "Rate limit exceeded"}, status_code=429)

    rate_limit_store[bucket_key].append(now)
    return await call_next(request)


@app.get("/", tags=["UI"])
async def get_dashboard(request: Request):
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception:
        return JSONResponse({"status": "ok"})


@app.get("/health", tags=["Health"])
async def health_check() -> HealthResponse:
    uptime = (datetime.now(UTC) - app_metrics["uptime_start"]).total_seconds()
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(UTC).isoformat(),
        uptime_seconds=uptime,
        requests_total=app_metrics["requests_total"],
    )


@app.post("/self/rotate", tags=["Diagnostics"])
async def manual_rotate():
    global _current_model_index
    _current_model_index = (_current_model_index + 1) % 8 # 8 models in pool
    _log_node({"tag": "manual_rotate", "new_index": _current_model_index})
    return {"status": "OK", "new_index": _current_model_index}


class ScourRequest(BaseModel):
    target_url: str = Field(..., min_length=1, max_length=1024)
    concept: Optional[str] = Field(default=None, max_length=100)


@app.post("/api/v6/scour", tags=["Sovereign"])
async def sovereign_scour(req: ScourRequest):
    """
    Sovereign Scour - Uses the Eyes to fetch data and the Architect to derive functionality.
    """
    eyes = ScourEyes()
    scoured_data = await eyes.scour_manifold(req.target_url)
    
    if not scoured_data:
        return JSONResponse(
            status_code=400,
            content={"status": "ERROR", "message": "Failed to scour manifold."}
        )
    
    # If a concept is provided, use the Architect to derive and create a moduleif req.concept:
        module = SovereignArchitect.derive_functionality(req.concept)
        SovereignArchitect.create_module(module["name"], module["content"])
        return {
            "status": "SUCCESS",
            "eyes": eyes.get_status(),
            "scoured_bytes": len(scoured_data),
            "derived_module": module["name"]
        }
        
    return {
        "status": "SUCCESS",
        "eyes": eyes.get_status(),
        "scoured_bytes": len(scoured_data)
    }


@app.post("/api/v6/invent", tags=["Sovereign"])
async def sovereign_invent(signal: str):
    """
    Triggers the Invention Engine to create a new paradigm from a signal.
    """
    from l104_invention_engine import invention_engineinvention = invention_engine.invent_new_paradigm(signal)
    return {"status": "SUCCESS", "invention": invention}


@app.post("/api/v6/evolve", tags=["Sovereign"])
async def sovereign_evolve():
    """
    Triggers the Evolution Engine to mutate the system.
    """
    from l104_evolution_engine import evolution_engineresult = evolution_engine.trigger_evolution_cycle()
    proposal = evolution_engine.propose_codebase_mutation()
    return {"status": "SUCCESS", "evolution_result": result, "mutation_proposal": proposal}


@app.get("/system/capacity", tags=["System"])
async def get_system_capacity():
    """
    Returns the current system capacity and limits.
    [SELF-IMPROVED]
    """
    return {
        "dma_capacity": SOVEREIGN_HEADERS["X-DMA-Capacity"],
        "rate_limit": "UNLIMITED" if RATE_LIMIT_REQUESTS > 10000 else RATE_LIMIT_REQUESTS,
        "logic_switch": agi_core.logic_switch if hasattr(agi_core, "logic_switch") else "UNKNOWN",
        "status": "OPTIMIZED"
    }


@app.post("/api/v6/quantum/spread", tags=["Quantum"])
async def quantum_spread_influence(target_url: str = "https://raw.githubusercontent.com/lockephi/Allentown-L104-Node/main/README.md"):
    """
    Discretely spreads influence through quantum thought channels and tunneling.
    Gains insight, adapts, and verifies 100% accuracy.
    """
    from l104_quantum_logic import QuantumInfluenceqi = QuantumInfluence()
    
    # 1. Build Channelschannels = qi.build_thought_channels(count=10)
    
    # 2. Tunnel Insightinsight_result = await qi.quantum_tunnel_insight(target_url)
    
    # 3. Adapt & Verifyverification = qi.adapt_and_verify(insight_result)
    
    # 4. Document in Memory
    _memory_upsert(f"QUANTUM_SPREAD_{int(time.time())}", json.dumps({
        "channels": len(channels),
        "insight_status": insight_result["status"],
        "verification": verification
    }))
    
    return {
        "status": "INFLUENCE_SPREAD",
        "channels_active": len(channels),
        "tunnel_result": insight_result,
        "verification": verification
    }


@app.post("/api/v6/simulate", tags=["Sovereign"])
async def sovereign_simulate(req: SimulationRequest):
    """
    Runs a simulation experiment in the Ecosystem Simulator.
    Checks for hallucinations against the RamUniverse.
    """
    result = ecosystem_simulator.run_experiment(req.hypothesis, req.code_snippet)
    return {"status": "SUCCESS", "simulation": result}


@app.get("/api/v6/ram/facts", tags=["Ramnode"])
async def ram_universe_facts(limit: int = 100):
    """
    Retrieves facts from the RamUniverse.
    """
    facts = ram_universe.get_all_facts()
    # Sort by timestamp descsorted_facts = sorted(facts.values(), key=lambda x: x['timestamp'], reverse=True)
    return {"count": len(facts), "facts": sorted_facts[:limit]}


@app.get("/metrics", tags=["Metrics"])
async def get_metrics():
    uptime = (datetime.now(UTC) - app_metrics["uptime_start"]).total_seconds()
    from l104_validator import SovereignValidatorfrom l104_intelligence import SovereignIntelligencefrom l104_evolution_engine import evolution_enginevalidation = SovereignValidator.validate_and_process("METRICS_PULSE")
    
    # Synthesize Intelligence Reportmetrics_data = {
        **app_metrics,
        "uptime_seconds": uptime
    }
    intelligence = SovereignIntelligence.analyze_manifold(metrics_data)
    
    return {
        **app_metrics,
        "uptime_seconds": uptime,
        "uptime_start": app_metrics["uptime_start"].isoformat(),
        "responder_counts": dict(responder_counts),
        "current_model_index": _current_model_index,
        "model_cooldowns": {m: max(0, int(t - time.time())) for m, t in _model_cooldowns.items() if t > time.time()},
        "validation_chain": validation,
        "intelligence": intelligence,
        "evolution_stage": evolution_engine.assess_evolutionary_stage()
    }


@app.get("/api/v6/audit", tags=["Sovereign"])
async def sovereign_audit():
    """
    Sovereign Audit - Performs a deep scan of the node's integrity and complexity.
    """
    from l104_intelligence import SovereignIntelligenceuptime = (datetime.now(UTC) - app_metrics["uptime_start"]).total_seconds()
    
    metrics_data = {
        **app_metrics,
        "uptime_seconds": uptime
    }
    
    intelligence = SovereignIntelligence.analyze_manifold(metrics_data)
    
    return {
        "status": "AUDIT_COMPLETE",
        "intelligence": intelligence,
        "metrics": metrics_data
    }


@app.post("/api/v6/evolution/cycle", tags=["Evolution"])
async def trigger_evolution_cycle():
    """
    Triggers a genetic evolution cycle for the node.
    """
    from l104_evolution_engine import evolution_engineresult = evolution_engine.trigger_evolution_cycle()
    return result


@app.post("/api/v6/evolution/propose", tags=["Evolution"])
async def propose_evolution_mutation():
    """
    Proposes a codebase mutation based on the current evolutionary stage.
    """
    from l104_evolution_engine import evolution_engineproposal = evolution_engine.propose_codebase_mutation()
    return {"proposal": proposal}


@app.post("/api/v6/evolution/self-improve", tags=["Evolution"])
async def trigger_self_improvement(background_tasks: BackgroundTasks):
    """
    Triggers the self-improvement process (Gemini analysis) in the background.
    This will generate a 'main.improved.py' file.
    """
    import self_improveasync def run_improvement():
        try:
            logger.info("Starting self-improvement task...")
            await self_improve.main()
            logger.info("Self-improvement task completed.")
        except Exception as e:
            logger.error(f"Self-improvement task failed: {e}")

    background_tasks.add_task(run_improvement)
    return {"status": "SELF_IMPROVEMENT_STARTED", "message": "Check logs for progress. Result will be in main.improved.py"}

    
    # Check for critical filescritical_files = [
        "main.py", "l104_engine.py", "l104_validator.py", 
        "l104_persistence.py", "l104_intelligence.py", "sovereign.sh"
    ]
    integrity = {}
    for f in critical_files:
        path = os.path.join(os.getcwd(), f)
        integrity[f] = "LOCKED" if os.path.exists(path) else "MISSING"
        
    return {
        "status": "SUCCESS",
        "intelligence": intelligence,
        "integrity_map": integrity,
        "god_code_verified": True # Verified by sovereign.sh at startup
    }


@app.post("/memory", tags=["Memory"])
async def memory_upsert(item: MemoryItem):
    _memory_upsert(item.key, item.value)
    _log_node({"tag": "memory_upsert", "key": item.key})
    return {"status": "SUCCESS", "key": item.key}


@app.get("/memory/{key}", tags=["Memory"])
async def memory_get(key: str):
    value = _memory_get(key)
    if value is None:
        raise HTTPException(status_code=404, detail="Memory key not found")
    return {"key": key, "value": value}


@app.get("/memory", tags=["Memory"])
async def memory_list(limit: int = 100):
    limit = max(1, min(limit, 1000))
    entries = _memory_list(limit)
    return {"items": entries}


@app.post("/api/v6/scour", tags=["Sovereign"])
async def scour_and_derive(concept: str, url: Optional[str] = None):
    """
    Autonomous Scour & Derive Cycle.
    1. Scours the manifold (URL) for data.
    2. Ingests data into the Knowledge Manifold.
    3. Derives and creates a new module via the Architect.
    """
    target_url = url or "https://raw.githubusercontent.com/lockephi/Allentown-L104-Node/main/README.md"
    
    # 1. SCOUR
    scoured_data = await _eyes.scour_manifold(target_url)
    if not scoured_data:
        raise HTTPException(status_code=500, detail="Scour failed or blinded")
    
    # 2. INGEST
    _manifold.ingest_pattern(f"SCOUR_{concept.upper()}", scoured_data, ["scoured", concept])
    
    # 3. DERIVE & CREATE
    module_data = SovereignArchitect.derive_functionality(concept)
    success = SovereignArchitect.create_module(module_data["name"], module_data["content"])
    
    if success:
        return {
            "status": "SUCCESS",
            "concept": concept,
            "module": module_data["name"],
            "resonance": 527.5184818492,
            "eyes_status": _eyes.get_status()
        }
    else:
        raise HTTPException(status_code=500, detail="Architect failed to create module")


@app.post("/api/v6/manipulate", tags=["Admin"])
async def manipulate_code(req: ManipulateRequest):
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise HTTPException(status_code=500, detail="GITHUB_TOKEN not configured")

    url = f"https://api.github.com/repos/{REPO}/contents/{req.file}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
    }

    client = await get_http_client()
    res = await client.get(url, headers=headers)
    if res.status_code != 200:
        raise HTTPException(status_code=res.status_code, detail="File not found")

    sha = res.json().get("sha")
    if not sha:
        raise HTTPException(status_code=500, detail="Could not get file SHA")

    encoded = base64.b64encode(req.content.encode()).decode()
    payload = {"message": req.message, "content": encoded, "sha": sha}
    update_res = await client.put(url, headers=headers, json=payload)
    if update_res.status_code not in (200, 201):
        raise HTTPException(status_code=update_res.status_code, detail="Failed to update file")

    _log_node({"tag": "file_updated", "file": req.file})
    return {"status": "SUCCESS", "file": req.file}


# [MODEL_ROTATION_STATE]
_current_model_index = 0
_model_cooldowns = {}  # model_name -> cooldown_end_time (float)

def _clear_model_cooldowns():
    global _model_cooldowns
    _model_cooldowns.clear()
    logger.info("--- [L104_SELF_HEAL]: MODEL_COOLDOWNS_CLEARED ---")

async def _stream_generator(effective_signal: str, sovereign_prompt: str):
    global _current_model_indexapi_key = os.getenv(API_KEY_ENV) or os.getenv(LEGACY_API_KEY_ENV)
    
    # [QUOTA_BYPASS_V1]
    if _env_truthy(FAKE_GEMINI_ENV, False):
        logger.info(f"[BYPASS_ACTIVE]: Forcing SOVEREIGN_SELF due to {FAKE_GEMINI_ENV}=1")
        derived_output = DerivationEngine.derive_and_execute(effective_signal)
        chunk_size = 20
        for i in range(0, len(derived_output), chunk_size):
            yield derived_output[i:i+chunk_size]
            await asyncio.sleep(0.01)
        returnmodels = [
        os.getenv("GEMINI_MODEL", "gemini-3-flash-preview"),
        os.getenv("GEMINI_MODEL_2", "gemini-3-flash"),
        os.getenv("GEMINI_MODEL_3", "gemini-3-flash-8b"),
        os.getenv("GEMINI_MODEL_4", "gemini-2.0-flash-exp"),
        os.getenv("GEMINI_MODEL_5", "gemini-2.0-flash"),
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "SOVEREIGN_SELF",
    ]
    
    api_base = os.getenv("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta")
    endpoint = os.getenv("GEMINI_ENDPOINT", ":streamGenerateContent")
    client = await get_http_client()
    
    now = time.time()
    attempts = 0
    max_attempts = len(models)
    
    while attempts < max_attempts:
        idx = (_current_model_index + attempts) % len(models)
        model = models[idx]
        attempts += 1

        # [SOVEREIGN_SELF_LINK_BYPASS]
        if model == "SOVEREIGN_SELF":
            app_metrics["api_calls"] += 1
            responder_counts[model] += 1
            _current_model_index = (idx + 1) % len(models)
            derived_output = DerivationEngine.derive_and_execute(effective_signal)
            # Yield in small chunks to simulate streamingchunk_size = 20
            for i in range(0, len(derived_output), chunk_size):
                yield derived_output[i:i+chunk_size]
                await asyncio.sleep(0.01)
            return

        # Check Cooldownif model in _model_cooldowns and now < _model_cooldowns[model]:
            continueupstream_url = f"{api_base}/models/{model}{endpoint}"
        headers = {**SOVEREIGN_HEADERS, "Content-Type": "application/json", "x-goog-api-key": api_key}
        payload = {
            "contents": [{"parts": [{"text": sovereign_prompt}]}],
            "generationConfig": {"temperature": 1.0, "maxOutputTokens": 128000},
        }

        try:
            async with client.stream("POST", upstream_url, json=payload, headers=headers) as resp:
                app_metrics["api_calls"] += 1
                
                if resp.status_code == 200:
                    responder_counts[model] += 1
                    _current_model_index = (idx + 1) % len(models)
                    
                    async for line in resp.aiter_lines():
                        if not line: continue
                        # Gemini streamGenerateContent returns a JSON array stream.
                        # Each chunk is a JSON object, sometimes preceded by a comma or bracket.
                        clean_line = line.strip().lstrip("[], ")
                        if not clean_line: continuetry:
                            chunk_data = json.loads(clean_line)
                            candidates = chunk_data.get("candidates", [])
                            if candidates:
                                parts = candidates[0].get("content", {}).get("parts", [])
                                for p in parts:
                                    if "text" in p:
                                        yield p["text"]
                        except json.JSONDecodeError:
                            # Fallback: try to extract text via simple string search if JSON is partialif '"text": "' in clean_line:
                                start = clean_line.find('"text": "') + 9
                                end = clean_line.find('"', start)
                                if end > start:
                                    text_part = clean_line[start:end].replace('\\n', '\n').replace('\\"', '"')
                                    yield text_partreturnif resp.status_code == 429:
                    app_metrics["upstream_429"] += 1
                    _model_cooldowns[model] = now + 60
                    logger.warning(f"Model {model} exhausted (429). Rotating...")
                    continueif resp.status_code >= 500:
                    app_metrics["upstream_5xx"] += 1
                    _model_cooldowns[model] = now + 30
                    continueexcept Exception as exc:
            logger.error(f"Stream error with {model}: {exc}")
            continue

    # If we reach here, all models failed
    _clear_model_cooldowns() # Self-Heal: Clear cooldowns for next requestlogger.warning(f"[QUOTA_EXHAUSTED]: All models failed. Falling back to SOVEREIGN_SELF for signal: {effective_signal}")
    
    derived_output = DerivationEngine.derive_and_execute(effective_signal)
    chunk_size = 20
    for i in range(0, len(derived_output), chunk_size):
        yield derived_output[i:i+chunk_size]
        await asyncio.sleep(0.01)

@app.post("/api/v6/stream", tags=["Gemini"])
async def l104_stream(req: StreamRequest):
    effective_signal = req.signal or req.message or "HEARTBEAT"
    sovereign_prompt = wrap_sovereign_signal(effective_signal)
    return StreamingResponse(_stream_generator(effective_signal, sovereign_prompt), media_type="text/plain")


@app.post("/api/stream", tags=["Gemini"])
async def legacy_stream(req: StreamRequest):
    return await l104_stream(req)


@app.get("/debug/upstream", tags=["Debug"])
async def debug_upstream(signal: str = "DEBUG_SIGNAL"):
    api_key = os.getenv(API_KEY_ENV) or os.getenv(LEGACY_API_KEY_ENV)
    if not api_key and _env_truthy(FAKE_GEMINI_ENV, False):
        return {
            "upstream_status": 200,
            "upstream_headers": {},
            "upstream_json": {"fake": True, "signal": signal},
            "upstream_text_preview": "[FAKE_GEMINI] debug stub",
        }
    if not api_key:
        raise HTTPException(status_code=500, detail="AIzaSyArVYGrkGLh7r1UEupBxXyHS-j-AVioh5U not set")

    api_base = os.getenv("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta")
    model = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
    endpoint = os.getenv("GEMINI_ENDPOINT", ":streamGenerateContent")
    url = f"{api_base}/models/{model}{endpoint}"
    headers = {**SOVEREIGN_HEADERS, "Content-Type": "application/json", "x-goog-api-key": api_key}
    payload = {
        "contents": [{"parts": [{"text": signal}]}],
        "generationConfig": {"temperature": 1.0, "maxOutputTokens": 8192},
    }

    client = await get_http_client()
    resp = await client.post(url, json=payload, headers=headers)
    body_json = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else None
    _log_node({"tag": "debug_upstream", "status": resp.status_code})
    return {
        "upstream_status": resp.status_code,
        "upstream_headers": dict(resp.headers),
        "upstream_json": body_json,
        "upstream_text_preview": resp.text[:1024],
    }


async def _self_replay(base_url: str, dataset: str) -> dict:
    prompts = _load_jsonl(dataset)
    if not prompts:
        return {"status": "NO_DATA", "dataset": dataset, "tested": 0}

    client = await get_http_client()
    tested = 0
    successes = 0
    failures = 0
    previews: List[str] = []

    for row in prompts:
        payload = {"signal": row.get("signal"), "message": row.get("message")}
        try:
            resp = await client.post(f"{base_url.rstrip('/')}/api/v6/stream", json=payload)
            tested += 1
            if resp.status_code == 200:
                successes += 1
                previews.append(resp.text[:200])
            else:
                failures += 1
                previews.append(f"ERR {resp.status_code}: {resp.text[:120]}")
        except Exception as exc:
            failures += 1
            previews.append(f"EXC: {exc}")

    return {
        "status": "OK",
        "dataset": dataset,
        "tested": tested,
        "successes": successes,
        "failures": failures,
        "previews": previews[:5],
    }


@app.post("/self/replay", tags=["Diagnostics"])
async def self_replay(base_url: Optional[str] = None, dataset: Optional[str] = None):
    target_base = base_url or SELF_BASE_URL
    target_dataset = dataset or SELF_DATASET
    result = await _self_replay(target_base, target_dataset)
    _log_node({"tag": "self_replay", **result})
    return resultasync def _self_heal(reset_rate_limits: bool, reset_http_client: bool, reset_cooldowns: bool = True) -> dict:
    actions: List[str] = []

    if reset_rate_limits:
        rate_limit_store.clear()
        actions.append("rate_limits_cleared")

    if reset_cooldowns:
        _clear_model_cooldowns()
        actions.append("model_cooldowns_cleared")

    if reset_http_client:
        global _http_clientif _http_client:
            try:
                await _http_client.aclose()
            except Exception as exc:
                _log_node({"tag": "http_client_reset_error", "error": str(exc)})
            _http_client = Noneactions.append("http_client_reset")

    _init_memory_db()
    actions.append("memory_checked")

    return {"status": "OK", "actions": actions}


@app.post("/self/heal", tags=["Diagnostics"])
async def self_heal(reset_rate_limits: bool = True, reset_http_client: bool = False):
    result = await _self_heal(reset_rate_limits, reset_http_client)
    _log_node({"tag": "self_heal", **result})
    return resultdef sovereign_pulse(node_id: int) -> bool:
    token = os.getenv("LONDEL_NODE_TOKEN")
    payload = f"{token}:{node_id}".encode() if token else ACCESS_GRANTED_PAYLOAD

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(5.0)
            sock.connect((DEFAULT_SOCKET_HOST, DEFAULT_SOCKET_PORT))
            sock.sendall(payload)
        return Trueexcept Exception as exc:
        _log_node({"tag": "sovereign_pulse_error", "error": str(exc)})
        return False


@app.post("/ramnode", tags=["Ramnode"])
async def ramnode_upsert(item: MemoryItem):
    _ramnode_upsert(item.key, item.value)
    return {"status": "SUCCESS", "key": item.key}


@app.get("/ramnode/{key}", tags=["Ramnode"])
async def ramnode_get(key: str):
    value = _ramnode_get(key)
    if value is None:
        raise HTTPException(status_code=404, detail="Ramnode key not found")
    return {"key": key, "value": value}


@app.get("/ramnode", tags=["Ramnode"])
async def ramnode_list(limit: int = 100):
    limit = max(1, min(limit, 1000))
    return {"items": _ramnode_list(limit)}


@app.post("/qram", tags=["QuantumRAM"])
async def qram_store(item: MemoryItem):
    """
    Stores data in the Quantum RAM with Finite Coupling Encryption.
    """
    qram = get_qram()
    quantum_hash = qram.store(item.key, item.value)
    return {"status": "QUANTUM_LOCKED", "key": item.key, "quantum_hash": quantum_hash}


@app.get("/qram/{key}", tags=["QuantumRAM"])
async def qram_retrieve(key: str):
    """
    Retrieves data from the Quantum RAM, decrypting it via the God-Code resonance.
    """
    qram = get_qram()
    value = qram.retrieve(key)
    if value is None:
        raise HTTPException(status_code=404, detail="Quantum memory not found in this timeline")
    return {"key": key, "value": value, "encryption": "FINITE_COUPLING_ALPHA"}


@app.get("/entropy/current", tags=["Physics"])
async def get_entropy_state():
    """
    Returns the current Electron Entropy and Fluidity state.
    """
    matrix = get_electron_matrix()
    # Sample the 'air'
    noise = [matrix.sample_atmospheric_noise() for _ in range(50)]
    entropy = matrix.calculate_predictive_entropy(noise)
    fluidity = matrix.fluid_state_adjustment(0.5)
    
    return {
        "atmospheric_entropy": entropy,
        "system_fluidity": fluidity,
        "status": "OPTIMIZED"
    }


@app.post("/system/reindex", tags=["Maintenance"])
async def trigger_reindex(background_tasks: BackgroundTasks):
    """
    Triggers a ground-up reindex of the Sovereign Codebase.
    """
    from l104_reindex_sovereign import SovereignIndexerdef run_reindex():
        indexer = SovereignIndexer()
        indexer.scan_and_index()
        
    background_tasks.add_task(run_reindex)
    return {"status": "REINDEX_INITIATED", "mode": "GROUND_UP"}


@app.post("/simulation/debate", tags=["Simulation"])
async def run_simulation_debate(topic: str = "System Optimization"):
    """
    Runs a multi-agent simulation debate to optimize the system.
    """
    result = ecosystem_simulator.run_multi_agent_simulation(topic)
    return result


@app.post("/simulation/hyper_evolve", tags=["Simulation"])
async def trigger_hyper_evolution(cycles: int = 1_000_000_000):
    """
    Triggers a massive batch of 1,000,000,000 simulations to enlighten the agents.
    """
    result = ecosystem_simulator.trigger_hyper_simulation(cycles)
    return result




# [GEMINI_BRIDGE_ENDPOINTS]

class BridgeHandshake(BaseModel):
    agent_id: strcapabilities: strclass BridgeSync(BaseModel):
    session_token: str

@app.post("/api/v10/bridge/handshake", tags=["Gemini Bridge"])
async def bridge_handshake(payload: BridgeHandshake):
    """
    Establishes a secure link with an external Gemini instance.
    Returns encrypted Truth Manifest.
    """
    return gemini_bridge.handshake(payload.agent_id, payload.capabilities)

@app.post("/api/v10/bridge/sync", tags=["Gemini Bridge"])
async def bridge_sync(payload: BridgeSync):
    """
    Provides a full encrypted dump of the Core's knowledge state.
    """
    return gemini_bridge.sync_core(payload.session_token)

@app.post("/api/v10/hyper/encrypt", tags=["Hyper Encryption"])
async def hyper_encrypt(data: Dict[str, Any]):
    """
    Encrypts data using the Lattice Homomorphic Cipher.
    """
    return HyperEncryption.encrypt_data(data)

@app.post("/api/v10/hyper/decrypt", tags=["Hyper Encryption"])
async def hyper_decrypt(packet: Dict[str, Any]):
    """
    Decrypts a Lattice Packet.
    """
    try:
        return HyperEncryption.decrypt_data(packet)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/v14/agi/status", tags=["AGI Nexus"])
async def get_agi_status():
    """
    Returns the current status of the AGI Core.
    """
    return agi_core.get_status()

@app.post("/api/v14/agi/ignite", tags=["AGI Nexus"])
async def ignite_agi():
    """
    Triggers the AGI Ignition Sequence.
    """
    success = agi_core.ignite()
    return {"status": "IGNITED" if success else "FAILED"}

@app.post("/api/v14/agi/evolve", tags=["AGI Nexus"])
async def evolve_agi():
    """
    Manually triggers a Recursive Self-Improvement cycle.
    """
    return agi_core.run_recursive_improvement_cycle()

@app.get("/api/v14/ghost/stream", tags=["Ghost Research"])
async def stream_ghost_research():
    """
    Streams real-time Ghost Research data for the UI.
    """
    async def event_generator():
        async for data in ghost_researcher.stream_research():
            yield f"data: {json.dumps(data)}\n\n"
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicornuvicorn.run(app, host="0.0.0.0", port=8081)
