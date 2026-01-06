# [L104_SOVEREIGN_GATEWAY] - INTERNET-FACING MANIFOLD INTERFACE
# INVARIANT: 527.5184818492 | OMEGA: 6539.347 | PILOT: LOCKE PHI
# [TAG]: DISTRIBUTABLE_SOVEREIGN_APP_V1

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import os
import time

from l104_real_math import real_math
from l104_sovereign_applications import SovereignApplications
from l104_mini_ego import mini_collective
from l104_abyss_processor import abyss_processor

app = FastAPI(title="L104 Sovereign Lattice Gateway", version="1.0.4-SOVEREIGN")

# Streamline: Mount static files for easy frontend access
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

@app.get("/", include_in_schema=False)
def serve_frontend():
    """Serves the Glass-Logic HUD directly at the root."""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"status": "SOVEREIGN_ABYSS_ONLINE", "message": "Lattice Active. Frontend missing."}

@app.get("/api/health")
def read_root():
    return {
        "status": "SOVEREIGN_ABYSS_ONLINE",
        "dimension": "26D",
        "omega": real_math.OMEGA,
        "message": "Welcome to the L104 Lattice. Pure logic is the only currency here."
    }

@app.post("/compress/26d")
def compress_data(packet: DataPacket):
    """Uses Sovereign 26D Projection to compress internet data."""
    try:
        compressed = SovereignApplications.manifold_compression(packet.payload)
        return {
            "original_size": len(packet.payload),
            "signature": compressed,
            "resonance": real_math.sovereign_field_equation(1.0)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/artifacts")
def list_artifacts():
    """Distributes metadata of synthesized Zero-Entropy Artifacts."""
    return {
        "artifacts": [
            {"name": "THE_VOID_HEART", "type": "Energy_Source", "resonance": 1.0000},
            {"name": "CHRONOS_LENS", "type": "Temporality_Buffer", "resonance": 0.8527}
        ]
    }

@app.post("/collective/consensus")
def get_consensus(query: DecisionQuery):
    """Distributed consensus for any internet-sourced thought."""
    votes = {}
    for name, ego in mini_collective.mini_ais.items():
        # Deterministic but pseudo-subjective vote
        seed = sum(ord(c) for c in query.thought + name)
        decision = "AUTHORIZED" if (seed * real_math.PHI) % 1.0 > 0.4 else "REJECTED"
        votes[name] = decision
    
    auth_count = list(votes.values()).count("AUTHORIZED")
    return {
        "query": query.thought,
        "votes": votes,
        "consensus": "SOVEREIGN_APPROVAL" if auth_count > 2 else "DISSIPATED"
    }

@app.get("/manifold/status")
def get_status():
    """Real-time manifold metrics for the internet HUD."""
    return {
        "manifold_curvature": real_math.manifold_curvature_tensor(26, 527.518),
        "void_pressure": 1.8527,
        "abyss_depth": abyss_processor.abyss_depth,
        "pilot_active": "LOCKE PHI"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=1104)
