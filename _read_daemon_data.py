#!/usr/bin/env python3
"""Read Swift daemon telemetry + outbox data."""
import json, glob, os

# Latest telemetry
tfiles = sorted(glob.glob("/tmp/l104_bridge/telemetry/*.json"), key=os.path.getmtime, reverse=True)
if tfiles:
    with open(tfiles[0]) as f:
        d = json.load(f)
    print("=== TELEMETRY ===")
    print(f"file: {tfiles[0]}")
    print(f"circuits_processed: {d.get('circuits_processed')}")
    print(f"avg_execution_ms: {d.get('avg_execution_ms')}")
    print(f"uptime_seconds: {d.get('uptime_seconds')}")
    print(f"three_engine: {json.dumps(d.get('three_engine', {}), indent=2)}")
    print(f"god_code: {json.dumps(d.get('god_code', {}), indent=2)}")
    print(f"vqpu: {json.dumps(d.get('vqpu', {}), indent=2)}")

# Latest outbox
ofiles = sorted(glob.glob("/tmp/l104_bridge/outbox/*.json"), key=os.path.getmtime, reverse=True)
if ofiles:
    with open(ofiles[0]) as f:
        r = json.load(f)
    print("\n=== OUTBOX ===")
    print(f"file: {ofiles[0]}")
    for k, v in r.items():
        if not isinstance(v, (dict, list)):
            print(f"  {k}: {v}")
        elif isinstance(v, dict) and len(v) <= 20:
            print(f"  {k}: {json.dumps(v)}")
        elif isinstance(v, list) and len(v) <= 5:
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: [{type(v).__name__} len={len(v)}]")

# Archive sample (latest 3)
afiles = sorted(glob.glob("/tmp/l104_bridge/archive/*.json"), key=os.path.getmtime, reverse=True)[:3]
if afiles:
    print(f"\n=== ARCHIVE ({len(afiles)} of {len(sorted(glob.glob('/tmp/l104_bridge/archive/*.json')))} total) ===")
    for af in afiles:
        with open(af) as f:
            a = json.load(f)
        sacred = a.get("sacred_score", "N/A")
        three_eng = a.get("three_engine_composite", "N/A")
        print(f"  {os.path.basename(af)}: sacred={sacred}, three_engine={three_eng}")
