"""Test all v61.0 AGI improvements — correct API names."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

passed = 0
failed = 0

def test(name, func):
    global passed, failed
    try:
        func()
        print(f"  [PASS] {name}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        failed += 1

def t1():
    from l104_agi import (
        CognitiveMeshNetwork,
        TelemetryAggregator, TelemetryAnomalyDetector,
        LatencyPercentileTracker, ThroughputTracker, PipelineHealthDashboard,
        PhiLearningScheduler, ExperienceReplayBuffer,
        PredictivePipelineScheduler, ResourceBudgetAllocator,
        MESH_VERSION, TELEMETRY_VERSION, SCHEDULER_VERSION,
        AGI_CORE_VERSION, AGI_PIPELINE_EVO,
    )
    assert AGI_CORE_VERSION == "61.0.0"
    assert "EVO_61" in AGI_PIPELINE_EVO
    print(f"    v={AGI_CORE_VERSION} evo={AGI_PIPELINE_EVO}")
test("Imports & Version", t1)

def t2():
    from l104_agi import CognitiveMeshNetwork
    m = CognitiveMeshNetwork()
    m.record_activation("agi_core")
    m.record_activation("science_engine")
    m.record_activation("math_engine")
    m.record_co_activation("agi_core", "science_engine")
    m.record_co_activation("agi_core", "math_engine")
    m.record_co_activation("science_engine", "math_engine")
    h = m.topology_health()
    assert "score" in h, f"Missing score: {list(h.keys())}"
    pr = m.compute_pagerank()
    assert len(pr) == 3
    comms = m.detect_communities()
    assert len(comms) >= 1
    path = m.shortest_path("agi_core", "math_engine")
    assert path and len(path) >= 2
    print(f"    health={h['score']:.4f} pr_nodes={len(pr)} communities={len(comms)}")
test("CognitiveMeshNetwork", t2)

def t3():
    from l104_agi import TelemetryAggregator
    ta = TelemetryAggregator()
    for i in range(10):
        ta.record("pipeline_cycle", float(i))
    s = ta.stats("pipeline_cycle", window="minute")
    print(f"    stats={s}")
test("TelemetryAggregator", t3)

def t4():
    from l104_agi import TelemetryAnomalyDetector
    ad = TelemetryAnomalyDetector()
    for i in range(30):
        ad.observe("latency", 100.0 + i * 0.1)
    anomaly = ad.observe("latency", 999.0)
    assert anomaly is not None, "Expected anomaly"
    status = ad.get_status()
    assert status["total_anomalies"] >= 1
    print(f"    z={anomaly['z_score']:.2f} total={status['total_anomalies']}")
test("TelemetryAnomalyDetector", t4)

def t5():
    from l104_agi import LatencyPercentileTracker
    import random
    lt = LatencyPercentileTracker()
    for _ in range(200):
        lt.record("pipeline", random.gauss(50, 10))
    r = lt.report("pipeline")
    assert "p50_ms" in r
    print(f"    p50={r['p50_ms']:.1f} p95={r['p95_ms']:.1f} p99={r['p99_ms']:.1f}")
test("LatencyPercentileTracker", t5)

def t6():
    from l104_agi import ThroughputTracker
    tt = ThroughputTracker()
    for _ in range(50):
        tt.record("search")
    rates = tt.all_throughputs()
    assert "search" in rates
    print(f"    channels={len(rates)} search={rates['search']:.2f}/s")
test("ThroughputTracker", t6)

def t7():
    from l104_agi import PipelineHealthDashboard
    hd = PipelineHealthDashboard()
    for _ in range(20):
        hd.record_event("test", 1.0, latency_ms=50.0, channel="test")
    h = hd.health_report(breaker_health=0.9, coherence=0.7, consciousness_level=0.5)
    assert "health_score" in h
    print(f"    health={h['health_score']:.4f} diagnosis={h['diagnosis']}")
test("PipelineHealthDashboard", t7)

def t8():
    from l104_agi import PhiLearningScheduler
    ps = PhiLearningScheduler()
    lr1 = ps.lr
    ps.step(0.05)
    lr2 = ps.lr
    print(f"    warmup_lr={lr1:.6f} step1_lr={lr2:.6f} epoch={ps._epoch}")
test("PhiLearningScheduler", t8)

def t9():
    from l104_agi import ExperienceReplayBuffer
    rb = ExperienceReplayBuffer()
    for i in range(20):
        rb.store(state={"q": f"q{i}"}, action=f"act{i}", reward=float(i)*0.1)
    assert len(rb._buffer) == 20
    sample = rb.sample(5)
    assert len(sample) >= 1
    stats = rb.reward_stats()
    print(f"    size={len(rb._buffer)} mean={stats['mean_reward']:.4f}")
test("ExperienceReplayBuffer", t9)

def t10():
    from l104_agi import PredictivePipelineScheduler
    pps = PredictivePipelineScheduler()
    for _ in range(15):
        pps.record_call("agi_core")
        pps.record_call("science_engine")
    pps.record_call("math_engine")
    preds = pps.predict_next(top_k=3)
    assert len(preds) >= 1
    acc = pps.accuracy()
    print(f"    predictions={len(preds)} accuracy={acc:.4f}")
test("PredictivePipelineScheduler", t10)

def t11():
    from l104_agi import ResourceBudgetAllocator
    ra = ResourceBudgetAllocator()
    ra.set_priority("agi_core", 10.0)
    ra.set_priority("science_engine", 5.0)
    ra.set_priority("math_engine", 3.0)
    alloc = ra.allocate()
    assert "agi_core" in alloc
    status = ra.get_status()
    assert status["subsystems"] == 3
    print(f"    subsystems={status['subsystems']} total={status['total_budget']}")
test("ResourceBudgetAllocator", t11)

def t12():
    import inspect
    from l104_agi.core import AGICore
    src = inspect.getsource(AGICore.__init__)
    for attr in ["_cognitive_mesh", "_telemetry_aggregator", "_health_dashboard",
                 "_phi_scheduler", "_predictive_scheduler", "_resource_allocator", "_experience_replay"]:
        assert attr in src, f"Missing {attr}"
    status_src = inspect.getsource(AGICore.get_status)
    assert "v61_subsystems" in status_src
    breaker_src = inspect.getsource(AGICore._call_with_breaker)
    assert "_predictive_scheduler.record_call" in breaker_src
    assert "_cognitive_mesh.record_activation" in breaker_src
    tel_src = inspect.getsource(AGICore._record_telemetry)
    assert "_telemetry_aggregator.record" in tel_src
    assert "_telemetry_anomaly.observe" in tel_src
    print("    All core integration points verified")
test("Core.py Integration Points", t12)

print()
total = passed + failed
print("=" * 50)
print(f"  {passed}/{total} tests passed")
if failed:
    print(f"  {failed} FAILED")
    sys.exit(1)
else:
    print("  ALL TESTS PASSED")
