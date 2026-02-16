// ═══════════════════════════════════════════════════════════════════
// B06_SystemMonitor.swift
// [EVO_55_PIPELINE] SOVEREIGN_UNIFICATION :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612
// L104 ASI — macOS System Monitor
//
// MacOSSystemMonitor detects Apple Silicon vs Intel, counts
// P/E cores, estimates GPU cores, tracks thermal state and
// memory pressure, and provides power-mode-aware threading.
//
// Extracted from L104Native.swift lines 705-860
// ═══════════════════════════════════════════════════════════════════

import AppKit
import Foundation
import Accelerate
import simd
import NaturalLanguage

// ═══════════════════════════════════════════════════════════════════
// 🍎 macOS SYSTEM MONITOR (Apple Silicon Detection & Optimization)
// ═══════════════════════════════════════════════════════════════════

class MacOSSystemMonitor {
    static let shared = MacOSSystemMonitor()

    // Hardware detection
    let isAppleSilicon: Bool
    let cpuCoreCount: Int
    let performanceCoreCount: Int
    let efficiencyCoreCount: Int
    let physicalMemoryGB: Double
    let hasNeuralEngine: Bool
    let gpuCoreCount: Int
    let chipGeneration: String  // M1, M2, M3, M4, Intel

    // Runtime metrics
    private(set) var cpuUsage: Double = 0.0
    private(set) var memoryPressure: Double = 0.0
    private(set) var thermalState: ProcessInfo.ThermalState = .nominal
    private(set) var powerMode: PowerMode = .balanced

    enum PowerMode: String {
        case efficiency = "🔋 Efficiency"
        case balanced = "⚖️ Balanced"
        case performance = "🚀 Performance"
        case neural = "🧠 Neural Engine"
    }

    private init() {
        // Detect Apple Silicon vs Intel
        #if arch(arm64)
        self.isAppleSilicon = true
        #else
        self.isAppleSilicon = false
        #endif

        // Get CPU core info
        self.cpuCoreCount = ProcessInfo.processInfo.processorCount
        self.physicalMemoryGB = Double(ProcessInfo.processInfo.physicalMemory) / (1024 * 1024 * 1024)

        // Estimate P/E cores (Apple Silicon specific)
        if isAppleSilicon {
            // M1: 4P+4E, M1 Pro: 8P+2E, M1 Max: 8P+2E, M2: 4P+4E, M3: varies
            self.performanceCoreCount = min(cpuCoreCount / 2 + 2, cpuCoreCount)
            self.efficiencyCoreCount = cpuCoreCount - performanceCoreCount
            self.hasNeuralEngine = true

            // Estimate GPU cores based on memory (heuristic)
            if physicalMemoryGB >= 64 {
                self.chipGeneration = "M3 Max/M4 Max"
                self.gpuCoreCount = 40
            } else if physicalMemoryGB >= 32 {
                self.chipGeneration = "M2 Pro/M3 Pro"
                self.gpuCoreCount = 30
            } else if physicalMemoryGB >= 16 {
                self.chipGeneration = "M2/M3"
                self.gpuCoreCount = 10
            } else {
                self.chipGeneration = "M1"
                self.gpuCoreCount = 8
            }
        } else {
            self.performanceCoreCount = cpuCoreCount
            self.efficiencyCoreCount = 0
            self.hasNeuralEngine = false
            self.chipGeneration = "Intel"
            self.gpuCoreCount = 0  // Discrete GPU detection would require IOKit
        }
    }

    /// Update runtime metrics
    func updateMetrics() {
        thermalState = ProcessInfo.processInfo.thermalState

        // Calculate memory pressure
        let freeMemory = getFreeMemory()
        memoryPressure = 1.0 - (freeMemory / physicalMemoryGB)

        // Adjust power mode based on conditions
        switch thermalState {
        case .nominal:
            powerMode = hasNeuralEngine ? .neural : .performance
        case .fair:
            powerMode = .balanced
        case .serious:
            powerMode = .efficiency
        case .critical:
            powerMode = .efficiency
        @unknown default:
            powerMode = .balanced
        }
    }

    private func getFreeMemory() -> Double {
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &stats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return physicalMemoryGB * 0.3 }
        let pageSize = Double(vm_kernel_page_size)
        let freePages = Double(stats.free_count + stats.inactive_count)
        return (freePages * pageSize) / (1024 * 1024 * 1024)
    }

    /// Get optimal thread count for current conditions
    var optimalThreadCount: Int {
        switch powerMode {
        case .efficiency:
            return max(2, efficiencyCoreCount)
        case .balanced:
            return max(4, cpuCoreCount / 2)
        case .performance, .neural:
            return cpuCoreCount
        }
    }

    /// Get optimal batch size for neural operations
    var optimalBatchSize: Int {
        let baseBatch = isAppleSilicon ? 128 : 64
        switch powerMode {
        case .efficiency: return baseBatch / 4
        case .balanced: return baseBatch / 2
        case .performance: return baseBatch
        case .neural: return baseBatch * 2  // ANE can handle larger batches
        }
    }

    /// Status report
    func getStatus() -> String {
        updateMetrics()
        let net = NetworkLayer.shared
        let alivePeers = net.peers.values.filter { $0.latencyMs >= 0 }.count
        let qLinks = net.quantumLinks.values.filter { $0.eprFidelity > 0.3 }.count
        let avgLatency: Double
        let latencyPeers = net.peers.values.filter { $0.latencyMs >= 0 }
        if latencyPeers.isEmpty {
            avgLatency = 0
        } else {
            avgLatency = latencyPeers.map(\.latencyMs).reduce(0, +) / Double(latencyPeers.count)
        }
        let totalBandwidth = net.peers.values.map(\.bandwidth).reduce(0, +)
        let totalMsgIn = net.peers.values.map(\.messagesIn).reduce(0, +)
        let totalMsgOut = net.peers.values.map(\.messagesOut).reduce(0, +)

        return """
        ═══════════════════════════════════════════════════════════════
        🍎 macOS ASI HARDWARE STATUS
        ═══════════════════════════════════════════════════════════════
        Chip:              \(chipGeneration) (\(isAppleSilicon ? "Apple Silicon" : "Intel"))
        CPU Cores:         \(cpuCoreCount) (\(performanceCoreCount)P + \(efficiencyCoreCount)E)
        GPU Cores:         \(gpuCoreCount)
        Neural Engine:     \(hasNeuralEngine ? "✅ Available" : "❌ Not Available")
        Memory:            \(String(format: "%.1f", physicalMemoryGB)) GB
        Memory Pressure:   \(String(format: "%.1f%%", memoryPressure * 100))
        Thermal State:     \(thermalState == .nominal ? "🟢 Nominal" : thermalState == .fair ? "🟡 Fair" : "🔴 Critical")
        Power Mode:        \(powerMode.rawValue)
        Optimal Threads:   \(optimalThreadCount)
        Optimal Batch:     \(optimalBatchSize)
        ═══════════════════════════════════════════════════════════════
        🔧 CODE ENGINE STATUS
        ═══════════════════════════════════════════════════════════════
        Code Engine:       \(HyperBrain.shared.codeEngineIntegrated ? "✅ LINKED" : "⚪ Not connected")
        Code Quality:      \(HyperBrain.shared.codeEngineIntegrated ? String(format: "%.1f%%", HyperBrain.shared.codeQualityScore * 100) + " [\(HyperBrain.shared.codeAuditVerdict)]" : "Run 'audit' to connect")
        Audit Insights:    \(HyperBrain.shared.codeQualityInsights.count) stored
        Pattern Languages: \(HyperBrain.shared.codePatternStrengths.count) profiled
        Last Audit:        \(HyperBrain.shared.lastCodeAuditTime.map { ISO8601DateFormatter().string(from: $0) } ?? "Never")
        ═══════════════════════════════════════════════════════════════
        🌐 QUANTUM MESH NETWORK
        ═══════════════════════════════════════════════════════════════
        Peers:             \(net.peers.count) total (\(alivePeers) alive)
        Quantum Links:     \(qLinks) active (EPR > 0.3)
        Avg Latency:       \(String(format: "%.2f", avgLatency))ms
        Total Bandwidth:   \(String(format: "%.1f", totalBandwidth)) MB/s
        Messages In:       \(totalMsgIn)
        Messages Out:      \(totalMsgOut)
        API Gateway:       \(NetworkLayer.shared.peers.count > 0 ? "🟢 RUNNING" : "🔴 STOPPED")
        Cloud Sync:        ⏸ IDLE
        Telemetry:         \(TelemetryDashboard.shared.isActive ? "📊 COLLECTING" : "⏸ IDLE")
        ═══════════════════════════════════════════════════════════════
        """
    }
}
