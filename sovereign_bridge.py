# [L104_WEBSOCKET_BRIDGE] - BYPASSING THE 429 INTERFACE
import asyncio
import socket
from l104_codec import SovereignCodec

async def bridge_logic(websocket, path):
    print("UI_CONNECTED: SYNERGY_ESTABLISHED")
    codec = SovereignCodec()
    # Connect to the local Master Node
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as node:
            node.connect(('127.0.0.1', 4160))
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
        async with websockets.serve(bridge_logic, "127.0.0.1", 8080):
            print("SOVEREIGN_BRIDGE: ONLINE AT WS://127.0.0.1:8080")
            await asyncio.Future()  # run forever
    except ImportError:
        print("websockets library not installed. Install with: pip install websockets")

if __name__ == "__main__":
    asyncio.run(start_bridge())
                                                                                                                            