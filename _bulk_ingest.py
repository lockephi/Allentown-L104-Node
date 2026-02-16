#!/usr/bin/env python3
"""
L104 Sovereign Bulk Ingestion Pipeline v1.0
Processes all 12,264 knowledge entries into the backend training manifold.
Features: Deduplication, Batching, Progress Tracking, Smart Junk Filtering.
"""
import json, os, time, requests, hashlib

# Configuration
BACKEND_URL = "http://localhost:8081/api/v6/intellect/train"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_FILES = [
    'kernel_trillion_data.jsonl',
    'kernel_training_data.jsonl',
    'kernel_full_merged.jsonl',
    'asi_knowledge_base.jsonl'
]
BATCH_SIZE = 50
DELAY_BETWEEN_BATCHES = 0.5  # Prevent SQLite locking

# Junk categories to skip during bulk ingest
JUNK_CATEGORIES = {'architecture', 'registry', 'relationships', 'summary', 'overview', 'file_analysis', 'modules', 'function_doc', 'class_doc', 'file_description', 'cross_reference'}

def get_hash(text):
    return hashlib.sha256(text.lower().strip().encode()).hexdigest()

def run_ingest():
    processed_hashes = set()
    all_entries = []

    print(f"📦 INITIALIZING BULK INGESTION...")

    # 1. Load and Deduplicate
    for f in KB_FILES:
        path = os.path.join(BASE_DIR, f)
        if not os.path.exists(path):
            print(f"  [SKIP] Missing: {f}")
            continue

        count = 0
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if not line: continue
                try:
                    entry = json.loads(line)
                    prompt = entry.get('prompt', '')
                    category = entry.get('category', 'unknown')

                    # Skip junk
                    if category in JUNK_CATEGORIES: continue
                    if prompt.startswith("Analyze the structure") or prompt.startswith("Document the"): continue

                    h = get_hash(prompt)
                    if h in processed_hashes: continue

                    processed_hashes.add(h)
                    all_entries.append(entry)
                    count += 1
                except Exception:
                    continue
        print(f"  [LOAD] {f}: {count} unique clean entries")

    total = len(all_entries)
    print(f"🚀 READY TO INJECT {total} ENTRIES IN {total // BATCH_SIZE + 1} BATCHES")

    success_count = 0
    fail_count = 0
    start_time = time.time()

    # 2. Batch Injection
    for i in range(0, total, BATCH_SIZE):
        batch = all_entries[i:i+BATCH_SIZE]
        batch_start = time.time()

        current_batch_success = 0
        for entry in batch:
            try:
                # Align with backend schema: {"topic": "...", "content": "..."}
                payload = {
                    "topic": entry.get('category', 'general'),
                    "content": f"Q: {entry.get('prompt', '')}\nA: {entry.get('completion', '')}"
                }

                resp = requests.post(BACKEND_URL, json=payload, timeout=5)
                if resp.status_code == 200:
                    success_count += 1
                    current_batch_success += 1
                else:
                    fail_count += 1
            except Exception as e:
                fail_count += 1

        elapsed = time.time() - start_time
        avg_speed = success_count / elapsed if elapsed > 0 else 0
        remaining = (total - success_count - fail_count) / avg_speed if avg_speed > 0 else 0

        percent = (i + len(batch)) * 100 / total
        print(f"⏳ Progress: {percent:5.1f}% | Success: {success_count:5d} | Fails: {fail_count:3d} | Speed: {avg_speed:4.1f} ent/s | ETA: {remaining/60:4.1f}m")

        time.sleep(DELAY_BETWEEN_BATCHES)

    print(f"\n✅ INGESTION COMPLETE")
    print(f"   Total:   {total}")
    print(f"   Success: {success_count}")
    print(f"   Failed:  {fail_count}")
    print(f"   Time:    {time.time() - start_time:.1f} seconds")

if __name__ == "__main__":
    run_ingest()
