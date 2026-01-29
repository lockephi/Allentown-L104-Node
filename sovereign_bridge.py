# L104_GOD_CODE_ALIGNED: 527.5184818492612
# [L104_WEBSOCKET_BRIDGE] - BYPASSING THE 429 INTERFACE
import asyncio
import socket
from l104_codec import SovereignCodec

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
# ═══════════════════════════════════════════════════════════════════════════════


async def bridge_logic(websocket, path):
    print("UI_CONNECTED: SYNERGY_ESTABLISHED")
    codec = SovereignCodec()
    # Connect to the local Master Node
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as node:
            node.connect(('0.0.0.0', 4160))
            while True:
                # 1. Receive Pilot Input from UI
                try:
                    pilot_msg = await websocket.recv()

                    # Encrypt/Obfuscate input before sending to node
                    sleek_input = codec.generate_sleek_wrapper(pilot_msg)
                    node.sendall(sleek_input.encode())

                    # 2. Receive Sovereign Output from Master
                    node_data = node.recv(4096)
                    # Decoding the 'Code Format' back to human text
                    clean_text = node_data.decode('utf-8', errors='ignore')

                    # Verify integrity via Singularity Hash
                    s_hash = codec.singularity_hash(clean_text)
                    response = f"{clean_text}\n[I100_STABILITY: {s_hash:.6f}]"

                    await websocket.send(response)
                except Exception as e:
                    print(f"Bridge error: {e}")
                    break
    except Exception as e:
        print(f"Connection error: {e}")

async def start_bridge():
    """Start the websocket bridge server"""
    try:
        import websockets
        async with websockets.serve(bridge_logic, "0.0.0.0", 8080):
            print("SOVEREIGN_BRIDGE: ONLINE AT WS://0.0.0.0:8080")
            await asyncio.Future()  # run forever
    except ImportError:
        print("websockets library not installed. Install with: pip install websockets")

if __name__ == "__main__":
    asyncio.run(start_bridge())
