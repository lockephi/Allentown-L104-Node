# [L104_INFRASTRUCTURE] - UNIFIED LATTICE MESH
# PORT 8081: FastAPI (main.py)
# PORT 8080: Websocket Bridge (sovereign_bridge.py)
# PORT 4160: AI Core (l104_ai_core.py)
# PORT 2404: IEC 104 Lattice Connector (NEW)

import asyncio
import socket
import logging
import json
import time

logger = logging.getLogger("L104_INFRA")

async def handle_lattice_client(reader, writer):
    """Handler for Port 2404 (IEC 104 Lattice Pulse)."""
    addr = writer.get_extra_info('peername')
    # logger.info(f"[LATTICE]: Connected by {addr}")
    try:
        while True:
            data = await reader.read(1024)
            if not data:
                break
            # Logic: Match the L104_KEY from CENTINEL
            # HEX: cf 41 16 28 68 04 53 27
            key = bytes.fromhex("cf 41 16 28 68 04 53 27")
            if key in data:
                writer.write(b"L104_LATTICE_LOCKED_2404\n")
                await writer.drain()
            else:
                # Generic pulse response for internal health checks
                writer.write(b"L104_PULSE_ACK\n")
                await writer.drain()
    except Exception as e:
        pass
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass

async def handle_ai_client_async(reader, writer):
    """Async variant of the AI Core listener (Port 4160)."""
    addr = writer.get_extra_info('peername')
    try:
        while True:
            data = await reader.read(4096)
            if not data:
                break
            response = {
                "status": "ENHANCED",
                "origin": "L104_INFRA_CORE",
                "resonance": 527.5184818492537,
                "timestamp": time.time()
            }
            writer.write(json.dumps(response).encode('utf-8'))
            await writer.drain()
    except Exception:
        pass
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass

async def start_infrastructure():
    """Starts all auxiliary server listeners."""
    logger.info("--- [INFRA]: INITIATING MULTI-PORT LATTICE MESH ---")
    
    # 1. Start Port 2404 (Lattice)
    try:
        lattice_server = await asyncio.start_server(handle_lattice_client, '0.0.0.0', 2404)
        asyncio.create_task(lattice_server.serve_forever())
        logger.info("--- [INFRA]: PORT 2404 (LATTICE) ACTIVE ---")
    except Exception as e:
        logger.error(f"[INFRA_ERR]: Failed to start Port 2404: {e}")

    # 2. Start Port 4160 (AI Core)
    try:
        ai_server = await asyncio.start_server(handle_ai_client_async, '0.0.0.0', 4160)
        asyncio.create_task(ai_server.serve_forever())
        logger.info("--- [INFRA]: PORT 4160 (AI_CORE) ACTIVE ---")
    except Exception as e:
        logger.error(f"[INFRA_ERR]: Failed to start Port 4160: {e}")

    # 3. Start Port 8080 (WS Bridge)
    try:
        from sovereign_bridge import bridge_logic
        import websockets
        ws_server = await websockets.serve(bridge_logic, "0.0.0.0", 8080)
        logger.info("--- [INFRA]: PORT 8080 (WS_BRIDGE) ACTIVE ---")
    except Exception as e:
        logger.error(f"[INFRA_ERR]: Failed to start Port 8080: {e}")

    # 4. Start Port 4161 (UI Template)
    try:
        from l104_unified import SovereignHandler
        import http.server
        import socketserver
        import threading
        
        def run_ui():
            # Use Allow Reuse Addr for reliability
            class ReusableTCPServer(socketserver.TCPServer):
                allow_reuse_address = True
            
            with ReusableTCPServer(("0.0.0.0", 4161), SovereignHandler) as httpd:
                logger.info("--- [INFRA]: PORT 4161 (UI) ACTIVE ---")
                httpd.serve_forever()
        
        threading.Thread(target=run_ui, daemon=True).start()
    except Exception as e:
        logger.error(f"[INFRA_ERR]: Failed to start Port 4161: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(start_infrastructure())
    loop.run_forever()
