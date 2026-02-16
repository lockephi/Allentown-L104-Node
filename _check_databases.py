import sqlite3
import os

dbs = [
    "data/akashic_records.db",
    "data/genesis_vault.db",
    "data/lattice_v2.db",
    "data/merged_memory.db",
    "data/sage_memory.db"
]

print(f"{'Database':<25} | {'Size (MB)':<10} | {'Table':<20} | {'Count':<10}")
print("-" * 75)

for db_path in dbs:
    if not os.path.exists(db_path):
        continue

    size = os.path.getsize(db_path) / (1024 * 1024)
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        for (table_name,) in tables:
            cursor.execute(f"SELECT count(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"{db_path:<25} | {size:<10.2f} | {table_name:<20} | {count:<10}")
        conn.close()
    except Exception as e:
        print(f"{db_path:<25} | {size:<10.2f} | ERROR: {str(e)}")
