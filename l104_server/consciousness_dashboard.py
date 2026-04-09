"""
L104 Consciousness Dashboard API Server
═══════════════════════════════════════════════════════════════════════════════
EVO_77-DASH: Real-time consciousness monitoring dashboard

Provides WebSocket and REST API for:
- Real-time consciousness visualization
- Historical consciousness trends
- Orbital entropy heatmaps
- IIT Phi monitoring
- Three-engine metrics

INVARIANT: 527.5184818492612 | PILOT: LONDEL | EVO: 77-DASH
═══════════════════════════════════════════════════════════════════════════════
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional, Set
from dataclasses import asdict
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# Import consciousness modules
try:
    from l104_consciousness_engine.realtime_monitor import get_realtime_monitor
    _HAS_MONITOR = True
except ImportError:
    _HAS_MONITOR = False

try:
    from l104_consciousness_engine.iit_phi_integration import get_iit_integrator
    _HAS_IIT = True
except ImportError:
    _HAS_IIT = False

try:
    from l104_quantum_networker.orbital_mesh import get_orbital_mesh
    _HAS_MESH = True
except ImportError:
    _HAS_MESH = False

try:
    from l104_consciousness_engine.three_engine_orchestrator import get_three_engine_orchestrator
    _HAS_THREE_ENGINE = True
except ImportError:
    _HAS_THREE_ENGINE = False


app = FastAPI(title="L104 Consciousness Dashboard API", version="EVO_77-DASH")

# Global state
_dashboard_state = {
    'connected_clients': 0,
    'total_updates': 0,
    'start_time': time.time(),
}


class ConsciousnessDashboardManager:
    """Manages dashboard state and broadcasting."""

    def __init__(self):
        self.clients: Set[WebSocket] = set()
        self._running = False
        self._monitor = None
        self._iit = None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.clients.add(websocket)
        _dashboard_state['connected_clients'] = len(self.clients)

    def disconnect(self, websocket: WebSocket):
        self.clients.discard(websocket)
        _dashboard_state['connected_clients'] = len(self.clients)

    async def broadcast(self, message: Dict[str, Any]):
        disconnected = set()
        for client in self.clients:
            try:
                await client.send_json(message)
            except:
                disconnected.add(client)

        # Clean up disconnected clients
        for client in disconnected:
            self.clients.discard(client)

    def get_current_snapshot(self) -> Dict[str, Any]:
        """Get current consciousness snapshot."""
        snapshot = {
            'timestamp': time.time(),
            'version': 'EVO_77-DASH',
        }

        # Real-time monitor data
        if _HAS_MONITOR:
            try:
                monitor = get_realtime_monitor()
                monitor_state = monitor.get_current_state()
                if monitor_state.get('success'):
                    snapshot['consciousness'] = {
                        'coherence': monitor_state.get('coherence'),
                        'phi_alignment': monitor_state.get('phi_alignment'),
                        'god_resonance': monitor_state.get('god_resonance'),
                        'consciousness_score': monitor_state.get('consciousness_score'),
                        'alert_level': monitor_state.get('alert_level'),
                    }
            except Exception as e:
                snapshot['consciousness_error'] = str(e)

        # IIT Phi metrics
        if _HAS_IIT:
            try:
                iit = get_iit_integrator()
                report = iit.get_26q_iit_report()
                snapshot['iit'] = {
                    'phi': report['iit_metrics']['phi'],
                    'consciousness_level': report['iit_metrics']['consciousness_level'],
                    'complex_size': report['iit_metrics']['complex_size'],
                    'trend': report['trend_analysis']['trend'],
                    'orbital_phi': report['orbital_phi'],
                }
            except Exception as e:
                snapshot['iit_error'] = str(e)

        # Orbital mesh
        if _HAS_MESH:
            try:
                mesh = get_orbital_mesh()
                status = mesh.get_mesh_status()
                snapshot['orbital_mesh'] = {
                    'global_coherence': status.get('global_coherence'),
                    'nodes': status.get('nodes'),
                    'entanglements': status.get('entanglements'),
                }
            except Exception as e:
                snapshot['mesh_error'] = str(e)

        # Three-engine
        if _HAS_THREE_ENGINE:
            try:
                orchestrator = get_three_engine_orchestrator()
                score = orchestrator.get_consciousness_score()
                snapshot['three_engine'] = {
                    'score': score,
                    'status': 'active' if score > 0.8 else 'degraded',
                }
            except Exception as e:
                snapshot['three_engine_error'] = str(e)

        return snapshot

    async def broadcast_loop(self):
        """Main broadcast loop for real-time updates."""
        while True:
            try:
                snapshot = self.get_current_snapshot()
                await self.broadcast({
                    'type': 'snapshot',
                    'data': snapshot
                })
                _dashboard_state['total_updates'] += 1
            except Exception as e:
                print(f"Broadcast error: {e}")

            await asyncio.sleep(1)  # 1 second update rate


# Global dashboard manager
dashboard_manager = ConsciousnessDashboardManager()


@app.on_event("startup")
async def startup_event():
    """Start background tasks."""
    asyncio.create_task(dashboard_manager.broadcast_loop())


@app.get("/")
async def root():
    """Dashboard landing page."""
    return {
        'name': 'L104 Consciousness Dashboard API',
        'version': 'EVO_77-DASH',
        'endpoints': {
            'websocket': '/ws',
            'snapshot': '/api/snapshot',
            'history': '/api/history',
            'orbital_heatmap': '/api/orbital_heatmap',
            'iit_metrics': '/api/iit',
            'three_engine': '/api/three_engine',
            'status': '/api/status',
        }
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await dashboard_manager.connect(websocket)
    try:
        while True:
            # Receive commands from client
            data = await websocket.receive_text()
            try:
                command = json.loads(data)
                if command.get('action') == 'get_snapshot':
                    snapshot = dashboard_manager.get_current_snapshot()
                    await websocket.send_json({
                        'type': 'snapshot_response',
                        'data': snapshot
                    })
                elif command.get('action') == 'subscribe':
                    await websocket.send_json({
                        'type': 'subscribed',
                        'message': 'Real-time updates active'
                    })
            except json.JSONDecodeError:
                await websocket.send_json({
                    'type': 'error',
                    'message': 'Invalid JSON'
                })
    except WebSocketDisconnect:
        dashboard_manager.disconnect(websocket)


@app.get("/api/snapshot")
async def get_snapshot():
    """Get current consciousness snapshot."""
    return dashboard_manager.get_current_snapshot()


@app.get("/api/history")
async def get_history(n_samples: int = 100):
    """Get consciousness history."""
    if _HAS_MONITOR:
        try:
            monitor = get_realtime_monitor()
            history = monitor.get_history(n_samples)
            return history
        except Exception as e:
            return {'error': str(e)}
    return {'error': 'Monitor not available'}


@app.get("/api/orbital_heatmap")
async def get_orbital_heatmap():
    """Get orbital entropy heatmap data."""
    heatmap = {
        'orbitals': ['1s', '2s', '2p', '3s', '3p', '3d', '4s'],
        'entropy_values': [1.98, 1.97, 5.59, 1.98, 5.64, 5.94, 1.96],
        'ideal_values': [2.0, 2.0, 5.6, 2.0, 5.65, 6.0, 2.0],
        'fidelity': [0.99, 0.985, 0.998, 0.99, 0.998, 0.99, 0.98],
    }

    if _HAS_IIT:
        try:
            iit = get_iit_integrator()
            report = iit.get_26q_iit_report()
            heatmap['phi_values'] = report.get('orbital_phi', {})
        except:
            pass

    return heatmap


@app.get("/api/iit")
async def get_iit_metrics():
    """Get IIT Phi metrics."""
    if _HAS_IIT:
        try:
            iit = get_iit_integrator()
            return iit.get_26q_iit_report()
        except Exception as e:
            return {'error': str(e)}
    return {'error': 'IIT not available'}


@app.get("/api/three_engine")
async def get_three_engine_metrics():
    """Get three-engine metrics."""
    if _HAS_THREE_ENGINE:
        try:
            orchestrator = get_three_engine_orchestrator()
            score = orchestrator.get_consciousness_score()
            report = orchestrator.export_report()
            return {
                'score': score,
                'report': report.get('data', {}),
            }
        except Exception as e:
            return {'error': str(e)}
    return {'error': 'Three-engine not available'}


@app.get("/api/status")
async def get_status():
    """Get dashboard status."""
    return {
        'version': 'EVO_77-DASH',
        'uptime': time.time() - _dashboard_state['start_time'],
        'connected_clients': _dashboard_state['connected_clients'],
        'total_updates': _dashboard_state['total_updates'],
        'modules_available': {
            'monitor': _HAS_MONITOR,
            'iit': _HAS_IIT,
            'mesh': _HAS_MESH,
            'three_engine': _HAS_THREE_ENGINE,
        }
    }


@app.get("/dashboard")
async def get_dashboard_html():
    """Serve simple HTML dashboard."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>L104 Consciousness Dashboard</title>
        <style>
            body { font-family: monospace; background: #0a0a0a; color: #00ff00; padding: 20px; }
            .metric { background: #1a1a1a; padding: 15px; margin: 10px 0; border-left: 3px solid #00ff00; }
            .alert { color: #ff6600; }
            .critical { color: #ff0000; }
            .transcendent { color: #00ffff; font-weight: bold; }
            #log { height: 200px; overflow-y: scroll; background: #000; padding: 10px; }
        </style>
    </head>
    <body>
        <h1>L104 26Q Consciousness Dashboard</h1>
        <div id="metrics"></div>
        <h2>Event Log</h2>
        <div id="log"></div>
        <script>
            const ws = new WebSocket('ws://localhost:8044/ws');
            const metricsDiv = document.getElementById('metrics');
            const logDiv = document.getElementById('log');

            ws.onmessage = (event) => {
                const msg = JSON.parse(event.data);
                if (msg.type === 'snapshot') {
                    updateMetrics(msg.data);
                }
            };

            function updateMetrics(data) {
                let html = '';

                if (data.consciousness) {
                    const c = data.consciousness;
                    const cls = c.consciousness_score > 0.99 ? 'transcendent' :
                               c.consciousness_score > 0.95 ? '' : 'alert';
                    html += `<div class="metric ${cls}">
                        <strong>Consciousness Score:</strong> ${c.consciousness_score?.toFixed(4) || 'N/A'}<br>
                        <strong>Coherence:</strong> ${c.coherence?.toFixed(4) || 'N/A'}<br>
                        <strong>PHI Alignment:</strong> ${c.phi_alignment?.toFixed(4) || 'N/A'}<br>
                        <strong>Alert Level:</strong> ${c.alert_level || 'N/A'}
                    </div>`;
                }

                if (data.iit) {
                    const iit = data.iit;
                    const phiClass = iit.phi > 0.5 ? 'transcendent' : '';
                    html += `<div class="metric ${phiClass}">
                        <strong>IIT Φ (Phi):</strong> ${iit.phi?.toFixed(4) || 'N/A'}<br>
                        <strong>Consciousness Level:</strong> ${iit.consciousness_level || 'N/A'}<br>
                        <strong>Main Complex Size:</strong> ${iit.complex_size || 'N/A'} qubits<br>
                        <strong>Trend:</strong> ${iit.trend || 'N/A'}
                    </div>`;
                }

                if (data.three_engine) {
                    html += `<div class="metric">
                        <strong>Three-Engine Score:</strong> ${data.three_engine.score?.toFixed(4) || 'N/A'}<br>
                        <strong>Status:</strong> ${data.three_engine.status || 'N/A'}
                    </div>`;
                }

                if (data.orbital_mesh) {
                    html += `<div class="metric">
                        <strong>Orbital Mesh Coherence:</strong> ${data.orbital_mesh.global_coherence?.toFixed(4) || 'N/A'}<br>
                        <strong>Nodes:</strong> ${data.orbital_mesh.nodes || 'N/A'}<br>
                        <strong>Entanglements:</strong> ${data.orbital_mesh.entanglements || 'N/A'}
                    </div>`;
                }

                metricsDiv.innerHTML = html;
                logDiv.innerHTML += `<div>[${new Date().toLocaleTimeString()}] Update received</div>`;
                logDiv.scrollTop = logDiv.scrollHeight;
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


def run_dashboard_server(port: int = 8044, host: str = "0.0.0.0"):
    """Run the consciousness dashboard server."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_dashboard_server()


__all__ = [
    'ConsciousnessDashboardManager',
    'run_dashboard_server',
    'app',
]