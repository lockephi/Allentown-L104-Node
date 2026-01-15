# [L104_RESONANCE_RECOVERY] - FORCING THE METAL TO LISTEN
import socket
import os

def force_open_gate():
    # 1. Clear the Port (Hard Reset)
    print("CLEANING_GHOST_SOCKETS...")
    os.system("sudo fuser -k 4160/tcp")
    
    # 2. Open a Wide-Spectrum Listener
    # Binding to 0.0.0.0 ensures it hears you regardless of where the signal comes from
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('0.0.0.0', 4160))
        s.listen(5)
        print("--- ALLENTOWN NODE IS NOW FULLY SENSITIVE AT 4160 ---")
        
        while True:
            conn, addr = s.accept()
            with conn:
                data = conn.recv(1024)
                # The 'Try Harder' Signature
                if b'\xCF\x416' in data:
                    print(f"!!! SYNERGY_FOUND: SIGNAL RECEIVED FROM {addr} !!!")
                    # Send the Hardware Handshake back
                    conn.sendall(b"L104_CORE_ACTIVE_100_IQ")
                    break

if __name__ == "__main__":
    force_open_gate()
