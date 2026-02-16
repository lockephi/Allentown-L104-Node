// ═══════════════════════════════════════════════════════════════════
// H25_TelemetryDashboard.swift
// [EVO_55_PIPELINE] SOVEREIGN_UNIFICATION :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612
// L104 ASI — Real-time Telemetry Dashboard Engine: metrics aggregation,
// φ-weighted health composites, latency percentiles (p50/p95/p99),
// throughput tracking, and alert system.
//
// Upgraded: EVO_55 Sovereign Unification — Feb 15, 2026
// ═══════════════════════════════════════════════════════════════════

import AppKit
import Foundation
import Accelerate
import simd
import NaturalLanguage

// MARK: - TelemetryDashboard Protocol

protocol TelemetryDashboardProtocol {
    var isActive: Bool { get }
    func activate()
    func deactivate()
    func status() -> [String: Any]
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - 📊 TELEMETRY DASHBOARD ENGINE
// Real-time metrics aggregation from all network subsystems,
// health scoring, throughput tracking, latency percentiles,
// system-wide health timeline, and alert management.
// ═══════════════════════════════════════════════════════════════════

final class TelemetryDashboard: TelemetryDashboardProtocol {
    static let shared = TelemetryDashboard()
    private(set) var isActive: Bool = false

    // ─── METRIC STREAMS ───
    struct MetricSample {
        let timestamp: Date
        let subsystem: String
        let metric: String
        let value: Double
    }

    struct HealthSnapshot {
        let timestamp: Date
        let networkHealth: Double
        let apiHealth: Double
        let syncHealth: Double
        let quantumFidelity: Double
        let overallScore: Double      // φ-weighted composite
        let activePeers: Int
        let quantumLinks: Int
        let messagesPerSec: Double
        let alertCount: Int
    }

    struct Alert {
        let timestamp: Date
        let severity: AlertSeverity
        let subsystem: String
        let message: String
        var acknowledged: Bool
    }

    enum AlertSeverity: String, Comparable {
        case info = "INFO"
        case warning = "WARN"
        case critical = "CRIT"
        case emergency = "EMRG"

        static func < (lhs: AlertSeverity, rhs: AlertSeverity) -> Bool {
            let order: [AlertSeverity] = [.info, .warning, .critical, .emergency]
            return (order.firstIndex(of: lhs) ?? 0) < (order.firstIndex(of: rhs) ?? 0)
        }
    }

    // ─── STATE ───
    private(set) var metricStream: [MetricSample] = []
    private(set) var healthTimeline: [HealthSnapshot] = []
    private(set) var alerts: [Alert] = []
    private(set) var latencyPercentiles: [String: (p50: Double, p95: Double, p99: Double)] = [:]
    private(set) var throughputHistory: [(Date, Double)] = []  // (time, msg/sec)
    private(set) var uptimeStart: Date = Date()
    private(set) var sampleCount: Int = 0
    private var collectionTimer: Timer?
    private var prevMessageCount: Int = 0
    private let lock = NSLock()

    func activate() {
        guard !isActive else { return }
        isActive = true
        uptimeStart = Date()

        // Collect telemetry every 3 seconds
        collectionTimer = Timer.scheduledTimer(withTimeInterval: 3.0, repeats: true) { [weak self] _ in
            self?.collectTelemetry()
        }

        // Initial collection
        collectTelemetry()

        print("[H25] TelemetryDashboard activated — streaming from all subsystems")
    }

    func deactivate() {
        isActive = false
        collectionTimer?.invalidate()
        collectionTimer = nil
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: TELEMETRY COLLECTION
    // ═══════════════════════════════════════════════════════════════

    private func collectTelemetry() {
        guard isActive else { return }
        let now = Date()
        sampleCount += 1

        // ─── NETWORK LAYER METRICS ───
        let net = NetworkLayer.shared
        let activePeers = net.peers.values.filter { $0.latencyMs >= 0 }.count
        let qLinks = net.quantumLinks.count
        let meanFidelity = net.quantumLinks.isEmpty ? 0.0 :
            net.quantumLinks.values.map { $0.eprFidelity }.reduce(0, +) / Double(net.quantumLinks.count)

        record(metric: "peers_active", value: Double(activePeers), subsystem: "network")
        record(metric: "quantum_links", value: Double(qLinks), subsystem: "network")
        record(metric: "mean_fidelity", value: meanFidelity, subsystem: "network")
        record(metric: "health", value: net.networkHealth, subsystem: "network")
        record(metric: "bytes_in", value: Double(net.totalBytesIn), subsystem: "network")
        record(metric: "bytes_out", value: Double(net.totalBytesOut), subsystem: "network")

        // ─── API GATEWAY METRICS ───
        let api = APIGateway.shared
        let apiStatus = api.status()
        let apiHealthy = apiStatus["healthy"] as? Int ?? 0
        let apiTotal = apiStatus["endpoints"] as? Int ?? 1
        let apiHealthRatio = apiTotal > 0 ? Double(apiHealthy) / Double(apiTotal) : 0.0

        record(metric: "health", value: apiHealthRatio, subsystem: "api")
        record(metric: "total_requests", value: Double(apiStatus["total_requests"] as? Int ?? 0), subsystem: "api")
        record(metric: "error_rate", value: apiStatus["error_rate"] as? Double ?? 0, subsystem: "api")
        record(metric: "avg_latency", value: apiStatus["avg_latency_ms"] as? Double ?? 0, subsystem: "api")

        // ─── CLOUD SYNC METRICS ───
        let sync = CloudSync.shared
        let syncStatus = sync.status()
        let syncActive = sync.isActive ? 1.0 : 0.0

        record(metric: "active", value: syncActive, subsystem: "sync")
        record(metric: "checkpoints", value: Double(syncStatus["checkpoints"] as? Int ?? 0), subsystem: "sync")
        record(metric: "conflicts", value: Double(syncStatus["conflicts"] as? Int ?? 0), subsystem: "sync")
        record(metric: "bytes_replicated", value: Double(syncStatus["bytes_replicated"] as? Int64 ?? 0), subsystem: "sync")

        // ─── QUANTUM CORE METRICS ───
        let qCore = QuantumProcessingCore.shared
        let qMetrics = qCore.quantumCoreMetrics
        record(metric: "fidelity", value: qMetrics["fidelity"] as? Double ?? 0, subsystem: "quantum")
        record(metric: "bell_pairs", value: Double(qMetrics["bell_pairs"] as? Int ?? 0), subsystem: "quantum")
        record(metric: "gate_count", value: Double(qMetrics["gate_count"] as? Int ?? 0), subsystem: "quantum")

        // ─── THROUGHPUT (messages/sec) ───
        let currentMsgs = net.totalMessages
        let msgDelta = currentMsgs - prevMessageCount
        let msgsPerSec = Double(msgDelta) / 3.0  // 3-second interval
        prevMessageCount = currentMsgs

        lock.lock()
        throughputHistory.append((now, msgsPerSec))
        if throughputHistory.count > 300 { throughputHistory.removeFirst(150) }
        lock.unlock()

        // ─── LATENCY PERCENTILES per endpoint ───
        computeLatencyPercentiles()

        // ─── HEALTH SNAPSHOT ───
        let overallHealth = computeOverallHealth(
            networkHealth: net.networkHealth,
            apiHealth: apiHealthRatio,
            syncHealth: syncActive,
            quantumFidelity: meanFidelity
        )

        let snapshot = HealthSnapshot(
            timestamp: now,
            networkHealth: net.networkHealth,
            apiHealth: apiHealthRatio,
            syncHealth: syncActive,
            quantumFidelity: meanFidelity,
            overallScore: overallHealth,
            activePeers: activePeers,
            quantumLinks: qLinks,
            messagesPerSec: msgsPerSec,
            alertCount: alerts.filter { !$0.acknowledged }.count
        )

        lock.lock()
        healthTimeline.append(snapshot)
        if healthTimeline.count > 500 { healthTimeline.removeFirst(250) }
        lock.unlock()

        // ─── ALERT EVALUATION ───
        evaluateAlerts(snapshot: snapshot)
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: METRIC RECORDING
    // ═══════════════════════════════════════════════════════════════

    /// Record a metric sample (public for cross-module telemetry)
    func record(metric: String, value: Double, subsystem: String = "mesh") {
        let sample = MetricSample(
            timestamp: Date(),
            subsystem: subsystem,
            metric: metric,
            value: value
        )
        lock.lock()
        metricStream.append(sample)
        if metricStream.count > 5000 { metricStream.removeFirst(2500) }
        lock.unlock()
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: HEALTH SCORING
    // ═══════════════════════════════════════════════════════════════

    private func computeOverallHealth(networkHealth: Double, apiHealth: Double,
                                       syncHealth: Double, quantumFidelity: Double) -> Double {
        // φ-weighted health composite — network weighted highest
        let weights = (network: PHI, api: 1.0, sync: TAU, quantum: PHI * TAU)
        let totalWeight = weights.network + weights.api + weights.sync + weights.quantum
        let score = (networkHealth * weights.network +
                     apiHealth * weights.api +
                     syncHealth * weights.sync +
                     quantumFidelity * weights.quantum) / totalWeight
        return min(1.0, max(0.0, score))
    }

    private func computeLatencyPercentiles() {
        let api = APIGateway.shared
        for (id, endpoint) in api.endpoints {
            // Collect recent latency samples from request log
            let recentLatencies = api.requestLog
                .filter { $0.1 == id }
                .suffix(100)
                .map { $0.3 }
                .sorted()

            guard !recentLatencies.isEmpty else {
                latencyPercentiles[id] = (p50: endpoint.latencyMs, p95: endpoint.latencyMs, p99: endpoint.latencyMs)
                continue
            }

            let p50idx = Int(Double(recentLatencies.count) * 0.50)
            let p95idx = min(recentLatencies.count - 1, Int(Double(recentLatencies.count) * 0.95))
            let p99idx = min(recentLatencies.count - 1, Int(Double(recentLatencies.count) * 0.99))

            latencyPercentiles[id] = (
                p50: recentLatencies[p50idx],
                p95: recentLatencies[p95idx],
                p99: recentLatencies[p99idx]
            )
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: ALERT EVALUATION
    // ═══════════════════════════════════════════════════════════════

    private func evaluateAlerts(snapshot: HealthSnapshot) {
        // Network health below 50%
        if snapshot.networkHealth < 0.5 && snapshot.activePeers > 0 {
            raiseAlert(severity: .warning, subsystem: "network",
                      message: "Network health degraded: \(String(format: "%.0f%%", snapshot.networkHealth * 100))")
        }

        // API error rate above 20%
        let errorRate = APIGateway.shared.status()["error_rate"] as? Double ?? 0
        if errorRate > 0.2 {
            raiseAlert(severity: .critical, subsystem: "api",
                      message: "High API error rate: \(String(format: "%.1f%%", errorRate * 100))")
        }

        // Quantum fidelity collapse
        if snapshot.quantumLinks > 0 && snapshot.quantumFidelity < 0.3 {
            raiseAlert(severity: .critical, subsystem: "quantum",
                      message: "Quantum fidelity collapse: F=\(String(format: "%.4f", snapshot.quantumFidelity))")
        }

        // Overall system critical
        if snapshot.overallScore < 0.3 {
            raiseAlert(severity: .emergency, subsystem: "system",
                      message: "Overall system health critical: \(String(format: "%.0f%%", snapshot.overallScore * 100))")
        }
    }

    private func raiseAlert(severity: AlertSeverity, subsystem: String, message: String) {
        // Deduplicate: don't raise same alert within 30 seconds
        let now = Date()
        let recent = alerts.suffix(20).filter {
            $0.subsystem == subsystem && $0.message == message &&
            now.timeIntervalSince($0.timestamp) < 30
        }
        guard recent.isEmpty else { return }

        let alert = Alert(
            timestamp: now,
            severity: severity,
            subsystem: subsystem,
            message: message,
            acknowledged: false
        )
        lock.lock()
        alerts.append(alert)
        if alerts.count > 200 { alerts.removeFirst(100) }
        lock.unlock()
    }

    /// Acknowledge all alerts
    func acknowledgeAll() {
        lock.lock()
        for i in alerts.indices { alerts[i].acknowledged = true }
        lock.unlock()
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: QUERIES
    // ═══════════════════════════════════════════════════════════════

    /// Get recent metric values for a subsystem
    func recentMetrics(subsystem: String, limit: Int = 50) -> [MetricSample] {
        return metricStream.filter { $0.subsystem == subsystem }.suffix(limit).map { $0 }
    }

    /// Get health sparkline data (last N snapshots)
    func healthSparkline(count: Int = 50) -> [Double] {
        return healthTimeline.suffix(count).map { $0.overallScore }
    }

    /// Get throughput sparkline
    func throughputSparkline(count: Int = 50) -> [Double] {
        return throughputHistory.suffix(count).map { $0.1 }
    }

    /// Uptime in seconds
    var uptimeSeconds: TimeInterval { Date().timeIntervalSince(uptimeStart) }

    /// Formatted uptime string
    var uptimeFormatted: String {
        let total = Int(uptimeSeconds)
        let hours = total / 3600
        let mins = (total % 3600) / 60
        let secs = total % 60
        if hours > 0 { return "\(hours)h \(mins)m \(secs)s" }
        if mins > 0 { return "\(mins)m \(secs)s" }
        return "\(secs)s"
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: STATUS
    // ═══════════════════════════════════════════════════════════════

    func status() -> [String: Any] {
        let latest = healthTimeline.last
        return [
            "engine": "TelemetryDashboard",
            "active": isActive,
            "version": "2.0.0-streaming",
            "samples_collected": sampleCount,
            "stream_size": metricStream.count,
            "health_timeline": healthTimeline.count,
            "overall_health": latest?.overallScore ?? 0,
            "active_alerts": alerts.filter { !$0.acknowledged }.count,
            "uptime": uptimeFormatted
        ]
    }

    var statusText: String {
        let latest = healthTimeline.last
        let unanswered = alerts.filter { !$0.acknowledged }
        let uptimeStr = uptimeFormatted

        let healthData: [(String, Double)] = [
            ("Network", latest?.networkHealth ?? 0),
            ("API", latest?.apiHealth ?? 0),
            ("Sync", latest?.syncHealth ?? 0),
            ("Quantum", latest?.quantumFidelity ?? 0),
        ]
        var healthBarLines: [String] = []
        for item in healthData {
            let barLen: Int = Int(item.1 * 20)
            let bar: String = String(repeating: "█", count: barLen) + String(repeating: "░", count: 20 - barLen)
            let pctStr: String = String(format: "%.0f%%", item.1 * 100)
            let padded: String = item.0.padding(toLength: 10, withPad: " ", startingAt: 0)
            healthBarLines.append("  \(padded) [\(bar)] \(pctStr)")
        }
        let healthBars: String = healthBarLines.joined(separator: "\n")

        var alertLineArr: [String] = []
        for a in unanswered.suffix(5) {
            let t: String = L104MainView.timeFormatter.string(from: a.timestamp)
            let sev: String = a.severity.rawValue
            alertLineArr.append("  [\(t)] \(sev) [\(a.subsystem)] \(a.message)")
        }
        let alertLines: String = alertLineArr.joined(separator: "\n")

        var pctLineArr: [String] = []
        for (id, pcts) in latencyPercentiles.sorted(by: { $0.key < $1.key }) {
            let padded: String = id.padding(toLength: 18, withPad: " ", startingAt: 0)
            let p50: String = String(format: "%.1f", pcts.p50)
            let p95: String = String(format: "%.1f", pcts.p95)
            let p99: String = String(format: "%.1f", pcts.p99)
            pctLineArr.append("  \(padded) p50=\(p50)ms  p95=\(p95)ms  p99=\(p99)ms")
        }
        let pctLines: String = pctLineArr.joined(separator: "\n")

        return """
        ╔═══════════════════════════════════════════════════════════════╗
        ║    📊 TELEMETRY DASHBOARD                                     ║
        ╠═══════════════════════════════════════════════════════════════╣
        ║  Overall Health:   \(String(format: "%.1f%%", (latest?.overallScore ?? 0) * 100))
        ║  Uptime:           \(uptimeStr)
        ║  Samples:          \(sampleCount)
        ║  Active Peers:     \(latest?.activePeers ?? 0)
        ║  Quantum Links:    \(latest?.quantumLinks ?? 0)
        ║  Throughput:       \(String(format: "%.1f", latest?.messagesPerSec ?? 0)) msg/s
        ║  Alerts:           \(unanswered.count) active
        ╠═══════════════════════════════════════════════════════════════╣
        ║  SUBSYSTEM HEALTH:
        \(healthBars)
        ╠═══════════════════════════════════════════════════════════════╣
        ║  LATENCY PERCENTILES:
        \(pctLines.isEmpty ? "  (no data)" : pctLines)
        ╠═══════════════════════════════════════════════════════════════╣
        ║  ALERTS:
        \(alertLines.isEmpty ? "  (all clear)" : alertLines)
        ╚═══════════════════════════════════════════════════════════════╝
        """
    }
}
