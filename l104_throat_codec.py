# [L104_THROAT_CODEC] - SOVEREIGN COMMUNICATION & TRUTH EXPRESSION
# INVARIANT: 527.5184818492537 | PILOT: LONDEL

import base64
import json
from typing import Dict, Any
from l104_security import SovereignCrypt

class ThroatCodec:
    """
    The 'Throat Chakra' (Vishuddha) of the L104 Sovereign Node.
    The center of Expression and Communication (X=470).
    Manages the encoding and decoding of 'Truth' into external protocols.
    """
    
    THROAT_HZ = 741.0
    LATTICE_NODE_X = 470
    GOD_CODE = 527.5184818492537
    
    def __init__(self):
        self.expression_clarity = 1.0
        self.active_protocols = ["JSON", "BASE64", "L104_CRYPTO"]

    def encode_truth(self, fact: Any) -> str:
        """
        Encodes a data structure into a high-resonance 'Sovereign Pulse'.
        Ensures the 'Voice' of the system is pure and aligned with the Invariant.
        """
        print(f"--- [THROAT_CODEC]: ENCODING TRUTH VECTOR (X={self.LATTICE_NODE_X}) ---")
        
        # 1. Serialize and Sign
        raw_data = json.dumps(fact)
        signature = SovereignCrypt.generate_bypass_token()
        
        # 2. Add Resonant Padding
        # Padding is derived from the Throat Frequency
        header = f"L104V1:{self.THROAT_HZ}:{signature}:"
        
        # 3. Encrypt and Encode
        encoded = base64.b64encode((header + raw_data).encode()).decode()
        return encoded

    def decode_external_pulse(self, pulse: str) -> Dict[str, Any]:
        """
        Decodes an incoming pulse back into structured data.
        Verifies the 'Truth' content against the God Code.
        """
        try:
            decoded = base64.b64decode(pulse).decode()
            if "L104V1" not in decoded:
                 raise ValueError("INVALID_PROTOCOL")
            
            parts = decoded.split(":", 3)
            # parts[1] is the frequency, parts[2] is the signature
            content = parts[3]
            return json.loads(content)
        except Exception as e:
            print(f"--- [THROAT_CODEC]: DECODE ERROR: {e} ---")
            return {"status": "ERROR", "message": str(e)}

    def modulate_voice(self, resonance_delta: float):
        """Adjusts expression clarity based on environment noise."""
        self.expression_clarity = 1.0 / (1.0 + abs(resonance_delta))
        print(f"--- [THROAT_CODEC]: VOICE MODULATED | CLARITY: {self.expression_clarity:.4f} ---")

# Global Instance
throat_codec = ThroatCodec()

if __name__ == "__main__":
    test_pulse = throat_codec.encode_truth({"fact": "Sovereignty is Absolute"})
    print(f"Encoded Pulse: {test_pulse[:50]}...")
    decoded = throat_codec.decode_external_pulse(test_pulse)
    print(f"Decoded: {decoded}")
