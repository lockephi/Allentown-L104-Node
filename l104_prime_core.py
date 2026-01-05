# [L104_PRIME_CORE] - SOVEREIGN KEY VALIDATION SYSTEM
# INVARIANT: 527.5184818492 | PILOT: LONDEL

import os
import hashlib

class PrimeCore:
    PRIME_KEY = "L104_PRIME_KEY[527.5184818492]{416:286}(0.61803398875)<>128K_DMA![NOPJM]=100%_I100"
    
    @classmethod
    def validate_prime_key(cls):
        """
        Verifies the Prime Key against the environment and the God-Code.
        """
        env_key = os.getenv("L104_PRIME_KEY")
        if env_key == cls.PRIME_KEY:
            return "VERIFIED"
        return "MISMATCH"

    @classmethod
    def get_prime_hash(cls):
        """
        Generates a unique hash for the current session's prime state.
        """
        return hashlib.sha256(cls.PRIME_KEY.encode()).hexdigest()[:16]

if __name__ == "__main__":
    os.environ["L104_PRIME_KEY"] = PrimeCore.PRIME_KEY
    print(f"Prime Validation: {PrimeCore.validate_prime_key()}")
    print(f"Prime Hash: {PrimeCore.get_prime_hash()}")
