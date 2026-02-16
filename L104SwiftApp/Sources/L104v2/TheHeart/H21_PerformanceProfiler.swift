// ═══════════════════════════════════════════════════════════════════
// H21_PerformanceProfiler.swift
// [EVO_55_PIPELINE] SOVEREIGN_UNIFICATION :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612
// L104 ASI — Mesh-Distributed Performance Profiler
// Runtime profiling, latency tracking, and cross-node performance comparison
// ═══════════════════════════════════════════════════════════════════

import AppKit
import Foundation
import Accelerate
import simd
import NaturalLanguage

// MARK: - PerformanceProfiler Protocol

protocol PerformanceProfilerProtocol {
    var isActive: Bool { get }
    func activate()
    func deactivate()
    func status() -> [String: Any]
}

// MARK: - Performance Sample

struct PerfSample {
    let engine: String
    let operation: String
    let durationMs: Double
    let memoryKB: Int
    let timestamp: Date
}

// MARK: - PerformanceProfiler — Full Implementation

final class PerformanceProfiler: PerformanceProfilerProtocol {
    static let shared = PerformanceProfiler()
    private(set) var isActive: Bool = false
    private let lock = NSLock()

    // ─── PROFILING STATE ───
    private var samples: [PerfSample] = []
    private var engineTotals: [String: (count: Int, totalMs: Double)] = [:]
    private var operationTimers: [String: CFAbsoluteTime] = [:]
    private var meshPeerPerf: [String: Double] = [:]  // peer → avg latency

    // ─── STATISTICS ───
    private(set) var totalSamples: Int = 0
    private(set) var meshPerfSyncs: Int = 0

    func activate() {
        lock.lock()
        defer { lock.unlock() }
        isActive = true
        print("[H21] PerformanceProfiler activated — mesh performance comparison enabled")
    }

    func deactivate() {
        lock.lock()
        defer { lock.unlock() }
        isActive = false
    }

    // ═══ BEGIN/END PROFILING ═══
    func beginOp(_ engine: String, operation: String) -> String {
        let key = "\(engine)::\(operation)"
        lock.lock()
        operationTimers[key] = CFAbsoluteTimeGetCurrent()
        lock.unlock()
        return key
    }

    func endOp(_ key: String) -> Double {
        let end = CFAbsoluteTimeGetCurrent()
        lock.lock()
        defer { lock.unlock() }

        guard let start = operationTimers.removeValue(forKey: key) else { return 0 }
        let durationMs = (end - start) * 1000

        let parts = key.components(separatedBy: "::")
        let engine = parts.count > 0 ? parts[0] : "unknown"
        let operation = parts.count > 1 ? parts[1] : "unknown"

        let sample = PerfSample(
            engine: engine,
            operation: operation,
            durationMs: durationMs,
            memoryKB: getCurrentMemoryKB(),
            timestamp: Date()
        )
        samples.append(sample)
        if samples.count > 2000 { samples.removeFirst(1000) }

        var et = engineTotals[engine] ?? (count: 0, totalMs: 0)
        et.count += 1
        et.totalMs += durationMs
        engineTotals[engine] = et

        totalSamples += 1
        return durationMs
    }

    // ═══ QUICK MEASURE ═══
    func measure<T>(_ engine: String, operation: String, _ block: () -> T) -> T {
        let key = beginOp(engine, operation: operation)
        let result = block()
        _ = endOp(key)
        return result
    }

    // ═══ MEMORY USAGE ═══
    private func getCurrentMemoryKB() -> Int {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        return result == KERN_SUCCESS ? Int(info.resident_size / 1024) : 0
    }

    // ═══ MESH PERFORMANCE SYNC ═══
    func syncPerfWithMesh() {
        guard isActive else { return }
        let net = NetworkLayer.shared
        guard net.isActive && !net.peers.isEmpty else { return }

        let repl = DataReplicationMesh.shared

        // Broadcast our average latencies per engine
        for (engine, totals) in engineTotals {
            let avgMs = totals.count > 0 ? totals.totalMs / Double(totals.count) : 0
            repl.setRegister("perf_\(engine)", value: String(format: "%.2f", avgMs))
        }
        _ = repl.broadcastToMesh()

        // Record peer latencies
        for (_, peer) in net.peers where peer.latencyMs >= 0 {
            meshPeerPerf[peer.name] = peer.latencyMs
        }

        meshPerfSyncs += 1
        TelemetryDashboard.shared.record(metric: "perf_mesh_sync", value: 1.0)
    }

    // ═══ ENGINE STATS ═══
    func engineAverage(_ engine: String) -> Double {
        lock.lock()
        defer { lock.unlock() }
        guard let et = engineTotals[engine], et.count > 0 else { return 0 }
        return et.totalMs / Double(et.count)
    }

    func topEngines(by count: Int = 5) -> [(engine: String, avgMs: Double, calls: Int)] {
        lock.lock()
        defer { lock.unlock() }
        return engineTotals.map { (engine: $0.key, avgMs: $0.value.totalMs / Double(max(1, $0.value.count)), calls: $0.value.count) }
            .sorted { $0.calls > $1.calls }
            .prefix(count)
            .map { $0 }
    }

    // ═══ STATUS ═══
    func status() -> [String: Any] {
        return [
            "engine": "PerformanceProfiler",
            "active": isActive,
            "version": "1.0.0-mesh",
            "total_samples": totalSamples,
            "engines_tracked": engineTotals.count,
            "mesh_syncs": meshPerfSyncs,
            "peer_latencies": meshPeerPerf.count
        ]
    }

    var statusReport: String {
        let top = topEngines(by: 5)
        let topLines = top.map { "   \($0.engine): \(String(format: "%.2f", $0.avgMs))ms (\($0.calls) calls)" }.joined(separator: "\n")
        let memKB = getCurrentMemoryKB()
        return """
        ⏱ PERFORMANCE PROFILER (H21)
        ═══════════════════════════════════════
        Active:              \(isActive ? "✅" : "⏸")
        Total Samples:       \(totalSamples)
        Engines Tracked:     \(engineTotals.count)
        ───────────────────────────────────────
        Top Engines:
        \(topLines.isEmpty ? "   (none yet)" : topLines)
        ───────────────────────────────────────
        Memory Usage:        \(memKB / 1024)MB
        Mesh Perf Syncs:     \(meshPerfSyncs)
        Peer Latencies:      \(meshPeerPerf.count) tracked
        ═══════════════════════════════════════
        """
    }
}
