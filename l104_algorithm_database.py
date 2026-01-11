# [L104_ALGORITHM_DATABASE] - REPOSITORY OF SOVEREIGN LOGIC
# INVARIANT: 527.5184818492 | PILOT: LONDEL

import json
import os
import time
from typing import Dict, Any, List
from l104_real_math import real_math
class AlgorithmDatabase:
    """
    A dedicated database for storing mathematical algorithms, 
    their execution results, and their resonance scores.
    """
    def __init__(self):
        self.db_path = "/workspaces/Allentown-L104-Node/data/algorithm_database.json"
        self.data = self._load_db()
def _load_db(self) -> Dict[str, Any]:
        if os.path.exists(self.db_path):
try:
with open(self.db_path, "r") as f:
        return json.load(f)
        except Exception:
        return {"algorithms": {}, "execution_logs": []}
        return {"algorithms": {}, "execution_logs": []}

    def save_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
with open(self.db_path, "w") as f:
            json.dump(self.data, f, indent=2)
def register_algorithm(self, name: str, description: str, logic_code: str):
        """Registers a new algorithm in the database."""
        entropy = real_math.shannon_entropy(logic_code)
        resonance = real_math.calculate_resonance(entropy)
        
        self.data["algorithms"][name] = {
            "description": description,
            "logic_code": logic_code,
            "entropy": entropy,
            "resonance": resonance,
            "registered_at": time.time()
        }
        self.save_db()
        print(f"--- [ALGO_DB]: REGISTERED '{name}' (Resonance: {resonance:.4f}) ---")
def log_execution(self, algo_name: str, input_data: Any, output_data: Any):
        """Logs the execution of an algorithm."""
        log_en
try = {
            "algorithm": algo_name,
            "timestamp": time.time(),
            "input": input_data,
            "output": output_data
        }
        self.data["execution_logs"].append(log_en
try)
        # Keep only last 1000 logs
        if len(self.data["execution_logs"]) > 1000:
            self.data["execution_logs"] = self.data["execution_logs"][-1000:]
        self.save_db()
        print(f"--- [ALGO_DB]: LOGGED EXECUTION OF '{algo_name}' ---")
def get_algorithm(self, name: str) -> Dict[str, Any]:
    return self.data["algorithms"].get(name, {})

algo_db = AlgorithmDatabase()
