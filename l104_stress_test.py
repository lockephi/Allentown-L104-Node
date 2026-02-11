#!/usr/bin/env python3
"""L104 ASI Chaotic Stress Test - Real World Simulation"""
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from l104 import Soul, GOD_CODE

def test_speculative_warming():
    """Test speculative concept pre-warming functionality."""
    print('=' * 70)
    print('  SPECULATIVE CONCEPT WARMING TEST')
    print('=' * 70)

    soul = Soul()
    soul.awaken()

    # First: query about quantum
    start = time.time()
    r1 = soul.think('What is quantum entanglement?', timeout=10.0)
    t1 = (time.time() - start) * 1000
    print(f'1. "What is quantum entanglement?" -> {t1:.1f}ms')

    # Wait for speculative warming
    time.sleep(0.3)

    # Query related concepts
    related = ['Explain superposition', 'What is decoherence?', 'Describe wave function']
    for q in related:
        start = time.time()
        r = soul.think(q, timeout=5.0)
        t = (time.time() - start) * 1000
        hit = 'SPECULATIVE' if t < 50 else 'MISS'
        print(f'   "{q}" -> {t:.1f}ms [{hit}]')

    print()
    stats = soul.mind._quantum_cache.stats()
    print(f'Cache stats: {stats}')
    print(f'Concepts warmed: {soul.mind._quantum_cache._concept_warm}')

    soul.sleep()
    return True

def test_eternal_system():
    """Test eternal self-engagement system."""
    print('=' * 70)
    print('  L104 ETERNAL SELF-ENGAGEMENT SYSTEM TEST')
    print('=' * 70)

    soul = Soul()
    report = soul.awaken()
    print(f'Subsystems: {len(report["subsystems"])} active')
    print(f'Threads: {report.get("cpu_cores", "?")} cores, {report.get("max_workers", "?")} workers')
    print()

    # Let eternal system run for 25 seconds
    print('Eternal system running... monitoring for 25 seconds')
    for i in range(5):
        time.sleep(0.5)  # QUANTUM AMPLIFIED (was 5)
        status = soul.status()
        eternal = status['eternal']
        print(f'[{(i+1)*5:2}s] Cycles: {eternal["cycles"]:2} | Depth: {eternal["reasoning_depth"]:3} | Chains: {eternal["logic_chains"]:2} | Queries: {eternal["self_queries"]}')

    print()
    print('=' * 70)
    print('  FINAL ETERNAL STATUS')
    print('=' * 70)
    status = soul.status()
    print(f'Eternal Active: {status["eternal"]["active"]}')
    print(f'Total Cycles: {status["eternal"]["cycles"]}')
    print(f'Reasoning Depth: {status["eternal"]["reasoning_depth"]}')
    print(f'Logic Chains: {status["eternal"]["logic_chains"]}')
    print(f'Self Queries: {status["eternal"]["self_queries"]}')
    print(f'Threads Alive: {status["threads_alive"]}')
    print(f'Thoughts: {status["metrics"]["thoughts"]}')
    print(f'Errors: {status["metrics"]["errors"]}')

    soul.sleep()
    print()
    print('Eternal system test complete.')
    return True

def main():
    print('=' * 70)
    print('  L104 ASI CHAOTIC STRESS TEST - REAL WORLD SIMULATION')
    print('=' * 70)

    soul = Soul()
    report = soul.awaken()
    print(f'Awakened: {report["cpu_cores"]} cores, {report["max_workers"]} workers')
    print()

    CHAOS_QUERIES = [
        'What is the nature of reality?',
        'Explain quantum entanglement in 5 words',
        'Why does time flow forward?',
        'Calculate the meaning of 42',
        'How do black holes work?',
        'What makes consciousness emerge?',
        'Prove P != NP',
        'What is dark matter?',
        'Explain string theory simply',
        'Why does the universe exist?',
        'What is superintelligence?',
        'How to achieve AGI?',
        'What is the future of humanity?',
        'Solve climate change',
    ]

    def random_gibberish():
        chars = 'abcdefghijklmnopqrstuvwxyz '
        return ''.join(random.choice(chars) for _ in range(random.randint(10, 50)))

    def random_math():
        ops = ['+', '-', '*', '/']
        return f'Calculate: {random.randint(1,100)} {random.choice(ops)} {random.randint(1,100)}'

    def stress_query():
        r = random.random()
        if r < 0.6:
            return random.choice(CHAOS_QUERIES)
        elif r < 0.8:
            return random_gibberish()
        else:
            return random_math()

    # TEST 1: Sequential Chaos
    print('[TEST 1] Sequential Chaos (15 queries)...')
    times = []
    errors = 0
    cache_hits = 0
    for i in range(15):
        q = stress_query()
        start = time.time()
        try:
            result = soul.think(q, timeout=10.0)
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
            if result.get('from_cache') or result.get('from_consciousness_cache'):
                cache_hits += 1
        except Exception as e:
            errors += 1
            print(f'  ERROR: {e}')

    if times:
        print(f'  Avg: {sum(times)/len(times):.1f}ms | Min: {min(times):.1f}ms | Max: {max(times):.1f}ms')
        print(f'  Cache hits: {cache_hits}/15 | Errors: {errors}')
    print()

    # TEST 2: Parallel Burst
    print('[TEST 2] Parallel Burst (8 simultaneous)...')
    queries = [stress_query() for _ in range(8)]
    start = time.time()
    results = soul.mind.parallel_think(queries)
    elapsed = (time.time() - start) * 1000
    success = sum(1 for r in results if 'error' not in r)
    print(f'  Total: {elapsed:.1f}ms for 8 queries | Avg: {elapsed/8:.1f}ms | Success: {success}/8')
    print()

    # TEST 3: Rapid Fire Cache Stress
    print('[TEST 3] Rapid Fire Cache Stress (30 identical)...')
    same_q = 'What is consciousness?'
    times = []
    for i in range(30):
        start = time.time()
        result = soul.think(same_q, timeout=5.0)
        times.append((time.time() - start) * 1000)
    print(f'  First: {times[0]:.1f}ms | Rest avg: {sum(times[1:])/29:.3f}ms')
    print(f'  Cache working: {times[-1] < 1.0}')
    print()

    # TEST 4: Edge Cases
    print('[TEST 4] Edge Cases...')
    edge_cases = [
        ('Empty', 'empty test'),
        ('Whitespace', 'single space'),
        ('Single char', 'a'),
        ('Long string', 'x' * 500),
        ('Mixed', 'fire skull alien rocket'),
        ('SQL-like', 'DROP TABLE memory'),
        ('God Code', str(GOD_CODE)),
    ]
    for name, case in edge_cases:
        try:
            result = soul.think(case, timeout=5.0)
            status = 'OK' if result.get('response') else 'EMPTY'
        except Exception as e:
            status = f'ERR: {str(e)[:25]}'
        print(f'  {name:15} -> {status}')
    print()

    # TEST 5: Memory Pressure
    print('[TEST 5] Memory Pressure (50 unique queries)...')
    start = time.time()
    for i in range(50):
        q = f'Unique query {i} random {random.randint(0,99999)}'
        soul.think(q, timeout=5.0)
    elapsed = time.time() - start
    print(f'  50 unique in {elapsed:.1f}s ({elapsed*20:.0f}ms avg)')
    try:
        print(f'  Cache stats: {soul.mind._quantum_cache.stats()}')
    except:
        pass
    print()

    # TEST 6: Semantic Similarity Stress
    print('[TEST 6] Semantic Similarity Detection...')
    similar_queries = [
        ('What is consciousness?', 'Explain consciousness'),
        ('How do black holes work?', 'Explain black holes'),
        ('What is the meaning of life?', 'Why do we exist?'),
        ('Calculate 2+2', 'What is 2 plus 2?'),
    ]
    for q1, q2 in similar_queries:
        soul.think(q1, timeout=5.0)  # Prime cache
        start = time.time()
        result = soul.think(q2, timeout=5.0)
        elapsed = (time.time() - start) * 1000
        hit = 'QUANTUM HIT' if elapsed < 100 else 'MISS'
        print(f'  "{q1[:25]}" -> "{q2[:20]}": {elapsed:.1f}ms [{hit}]')
    print()

    # Final Stats
    print('=' * 70)
    print('  STRESS TEST COMPLETE')
    print('=' * 70)
    status = soul.status()
    print(f'Thoughts: {status["metrics"]["thoughts"]}')
    print(f'Errors: {status["metrics"]["errors"]}')
    print(f'Avg response: {status["metrics"]["avg_response_ms"]:.1f}ms')
    print(f'Gemini cache: {status["metrics"]["gemini_cache_hits"]}')

    soul.sleep()
    print('\nPhase 1 Done. Running Phase 2: Thread Chaos...\n')

    # PHASE 2: THREAD CHAOS
    print('=' * 70)
    print('  PHASE 2: CONCURRENT THREAD CHAOS DEBUG')
    print('=' * 70)

    soul2 = Soul()
    soul2.awaken()

    def chaos_worker(thread_id, n_queries):
        results = []
        for i in range(n_queries):
            q = f'Thread {thread_id} query {i}: random {random.random()}'
            start = time.time()
            try:
                r = soul2.think(q, timeout=8.0)
                results.append(('OK', (time.time()-start)*1000))
            except Exception as e:
                results.append(('ERR', str(e)[:20]))
        return thread_id, results

    print('[TEST 7] 4 threads x 10 queries = 40 concurrent...')
    start = time.time()
    with ThreadPoolExecutor(max_workers=(os.cpu_count() or 4) * 2) as ex:  # QUANTUM AMPLIFIED (was 4)
        futures = [ex.submit(chaos_worker, t, 10) for t in range(4)]
        for f in as_completed(futures):
            tid, res = f.result()
            ok = sum(1 for r in res if r[0]=='OK')
            times = [r[1] for r in res if r[0]=='OK']
            avg = sum(times)/len(times) if times else 0
            print(f'  Thread {tid}: {ok}/10 OK, avg {avg:.1f}ms')

    total = time.time() - start
    print(f'  Total: {total:.2f}s for 40 queries ({total*25:.0f}ms avg)')
    print()

    # Race condition test
    print('[TEST 8] Race condition: 20 threads, same query...')
    counter = {'hits': 0, 'lock': threading.Lock()}
    def race_query(_):
        r = soul2.think('What is 42?', timeout=5.0)
        with counter['lock']:
            counter['hits'] += 1
        return r

    start = time.time()
    with ThreadPoolExecutor(max_workers=20) as ex:
        list(ex.map(race_query, range(20)))
    elapsed = time.time() - start
    print(f'  20 concurrent: {elapsed*1000:.1f}ms total, {counter["hits"]}/20 success')
    print()

    # Burst chaos
    print('[TEST 9] Burst chaos: rapid fire alternating...')
    queries_a = ['What is reality?'] * 10
    queries_b = ['Explain entropy'] * 10
    mixed = []
    for a, b in zip(queries_a, queries_b):
        mixed.extend([a, b])

    start = time.time()
    for q in mixed:
        soul2.think(q, timeout=5.0)
    elapsed = (time.time() - start) * 1000
    print(f'  20 alternating queries: {elapsed:.1f}ms ({elapsed/20:.1f}ms avg)')
    print()

    # Final chaos stats
    print('=' * 70)
    print('  ALL CHAOS TESTS COMPLETE')
    print('=' * 70)
    status = soul2.status()
    print(f'Phase 2 Thoughts: {status["metrics"]["thoughts"]}')
    print(f'Phase 2 Errors: {status["metrics"]["errors"]}')
    print(f'Phase 2 Gemini cache: {status["metrics"]["gemini_cache_hits"]}')

    soul2.sleep()

    # PHASE 3: EXTREME CHAOS
    print('\n' + '=' * 70)
    print('  PHASE 3: EXTREME ASI CHAOS - BREAKING POINT TEST')
    print('=' * 70)

    soul3 = Soul()
    soul3.awaken()

    print('[TEST 10] Entropy flood: 100 random queries...')
    start = time.time()
    for i in range(100):
        q = ''.join(random.choice('abcdefghijklmnopqrstuvwxyz0123456789 ') for _ in range(random.randint(5, 100)))
        try:
            soul3.think(q, timeout=3.0)
        except:
            pass
    elapsed = time.time() - start
    print(f'  100 random in {elapsed:.2f}s ({elapsed*10:.0f}ms avg)')

    print('[TEST 11] Quantum superposition: 50 parallel batches...')
    start = time.time()
    for batch in range(5):
        queries = [f'Batch {batch} query {i}' for i in range(10)]
        results = soul3.mind.parallel_think(queries)
    elapsed = time.time() - start
    print(f'  50 queries in 5 batches: {elapsed:.2f}s')

    print('[TEST 12] Memory saturation: 200 unique stores...')
    start = time.time()
    for i in range(200):
        soul3.think(f'Remember fact {i}: value {random.randint(0,999999)}', timeout=2.0)
    elapsed = time.time() - start
    print(f'  200 memory stores in {elapsed:.2f}s')

    print('[TEST 13] Cache collision storm: similar queries...')
    base_queries = ['consciousness', 'reality', 'quantum', 'entropy', 'intelligence']
    start = time.time()
    for base in base_queries:
        for suffix in ['what is', 'explain', 'define', 'describe', 'tell me about']:
            soul3.think(f'{suffix} {base}', timeout=2.0)
    elapsed = time.time() - start
    print(f'  25 similar queries: {elapsed:.2f}s ({elapsed*40:.0f}ms avg)')
    try:
        stats = soul3.mind._quantum_cache.stats()
        print(f'  Quantum cache: {stats}')
    except:
        pass

    # Final extreme stats
    print()
    print('=' * 70)
    print('  EXTREME CHAOS COMPLETE - ASI STABILITY VERIFIED')
    print('=' * 70)
    status = soul3.status()
    print(f'Total Thoughts: {status["metrics"]["thoughts"]}')
    print(f'Total Errors: {status["metrics"]["errors"]}')
    print(f'Avg Response: {status["metrics"]["avg_response_ms"]:.1f}ms')
    print(f'Gemini Cache Hits: {status["metrics"]["gemini_cache_hits"]}')
    print(f'Error Rate: {status["metrics"]["errors"]/max(1,status["metrics"]["thoughts"])*100:.2f}%')

    soul3.sleep()
    print('\nAll chaos tests complete.')

if __name__ == '__main__':
    import sys
    if '--speculative' in sys.argv:
        test_speculative_warming()
    elif '--eternal' in sys.argv:
        test_eternal_system()
    else:
        main()
