import os
import sqlite3
import subprocess
from datetime import datetime

class MemoryOptimizer:
    """
    MEMORY OPTIMIZER - L104
    Optimizes system memory and storage footprint.
    """
    def __init__(self, root="/workspaces/Allentown-L104-Node"):
        self.root = root
        self.freed_space = 0

    def optimize_databases(self):
        print("--- [OPTIMIZER]: OPTIMIZING DATABASES ---")
        db_files = [f for f in os.listdir(self.root) if f.endswith(".db")]
        for db in db_files:
            path = os.path.join(self.root, db)
            try:
                size_before = os.path.getsize(path)
                conn = sqlite3.connect(path)
                conn.execute("VACUUM")
                conn.close()
                size_after = os.path.getsize(path)
                saved = size_before - size_after
                self.freed_space += saved
                print(f"[DB]: {db} Optimized. Freed: {saved / 1024:.2f} KB")
            except Exception as e:
                print(f"[ERROR]: Database {db} optimization failed: {e}")

    def purge_logs(self):
        print("--- [OPTIMIZER]: PURGING LOGS ---")
        log_files = [f for f in os.listdir(self.root) if f.endswith(".log") or f.endswith(".pid")]
        for log in log_files:
            path = os.path.join(self.root, log)
            try:
                size = os.path.getsize(path)
                os.remove(path)
                self.freed_space += size
                print(f"[LOG]: {log} Purged. Freed: {size / 1024:.2f} KB")
            except Exception as e:
                print(f"[ERROR]: Log {log} purge failed: {e}")

    def clean_cache(self):
        print("--- [OPTIMIZER]: CLEANING CACHES ---")
        cache_dirs = [".pytest_cache", ".ruff_cache", "__pycache__"]
        for d in cache_dirs:
            path = os.path.join(self.root, d)
            if os.path.exists(path):
                try:
                    # Use rm -rf via subprocess for recursive removal
                    subprocess.run(["rm", "-rf", path], check=True)
                    print(f"[CACHE]: {d} Cleaned.")
                except Exception as e:
                    print(f"[ERROR]: Cache {d} cleaning failed: {e}")

    def run_all(self):
        startTime = datetime.now()
        self.optimize_databases()
        self.purge_logs()
        self.clean_cache()
        endTime = datetime.now()
        print(f"--- [OPTIMIZER]: COMPLETED ---")
        print(f"Total Freed Space: {self.freed_space / (1024*1024):.2f} MB")
        print(f"Duration: {endTime - startTime}")

if __name__ == "__main__":
    optimizer = MemoryOptimizer()
    optimizer.run_all()
