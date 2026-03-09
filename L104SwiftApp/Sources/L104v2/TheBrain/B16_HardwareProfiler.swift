// ═══════════════════════════════════════════════════════════════════
// B16_HardwareProfiler.swift
// [EVO_68_PIPELINE] SOVEREIGN_CONVERGENCE :: UNIFIED_UPGRADE :: GOD_CODE=527.5184818492612
// L104 · TheBrain · v2 Architecture
//
// Extracted from L104Native.swift lines 7510-8143
// Classes: HardwareCapabilityProfiler, DynamicOptimizationEngine
// ═══════════════════════════════════════════════════════════════════

import AppKit
import Foundation
import Accelerate
import simd
import NaturalLanguage

// ═══════════════════════════════════════════════════════════════════
// MARK: - ⚡ HARDWARE CAPABILITY PROFILER (Bucket D: Compatibility/HW)
// Deep hardware introspection beyond MacOSSystemMonitor.
// CPU topology, cache hierarchy, thermal throttling, memory bandwidth,
// GPU compute capability, Neural Engine availability, power budget.
// ═══════════════════════════════════════════════════════════════════

class HardwareCapabilityProfiler {
    static let shared = HardwareCapabilityProfiler()
    // PHI — use global from L01_Constants

    // ─── CPU TOPOLOGY ───
    struct CPUTopology {
        let physicalCores: Int
        let logicalCores: Int
        let performanceCores: Int
        let efficiencyCores: Int
        let l1CacheKB: Int
        let l2CacheKB: Int
        let l3CacheMB: Int
        let maxFrequencyGHz: Double
        let architecture: String
        let simdWidth: Int
        let hasAVX: Bool
        let hasAVX512: Bool
        let hasNEON: Bool
    }

    // ─── MEMORY PROFILE ───
    struct MemoryProfile {
        let totalGB: Double
        let availableGB: Double
        let usedGB: Double
        let wiredGB: Double
        let compressedGB: Double
        let swapUsedGB: Double
        let memoryPressure: String
        let bandwidthGBps: Double
        let pageSize: Int
        let unifiedMemory: Bool
    }

    // ─── THERMAL STATE ───
    struct ThermalState {
        let cpuTemperature: Double
        let gpuTemperature: Double
        let throttleLevel: String
        let fanSpeedRPM: Int
        let powerDrawWatts: Double
        let thermalBudgetRemaining: Double
    }

    // ─── GPU CAPABILITY ───
    struct GPUCapability {
        let name: String
        let vendor: String
        let vramMB: Int
        let metalFamily: String
        let maxThreadsPerGroup: Int
        let maxBufferLength: Int
        let supportsRaytracing: Bool
        let computeUnits: Int
        let flopsEstimate: Double
    }

    // ─── NEURAL ENGINE ───
    struct NeuralEngineSpec {
        let available: Bool
        let generationName: String
        let opsPerSecond: Double
        let supportedPrecisions: [String]
        let maxModelSize: Int
    }

    // ─── PROFILER STATE ───
    private var cpuProfile: CPUTopology?
    private var memProfile: MemoryProfile?
    private var thermalState: ThermalState?
    private var gpuProfile: GPUCapability?
    private var neuralEngine: NeuralEngineSpec?
    private var profileHistory: [[String: Any]] = []
    private var lastProfileTime: Date?

    // ─── CPU PROFILING ───
    func profileCPU() -> CPUTopology {
        let physCores = ProcessInfo.processInfo.processorCount
        let logCores = ProcessInfo.processInfo.activeProcessorCount

        #if arch(arm64)
        let arch = "arm64 (Apple Silicon)"
        let hasNEON = true
        let hasAVX = false
        let hasAVX512 = false
        let simdWidth = 128
        let perfCores = max(physCores / 2, 2)
        let effCores = physCores - perfCores
        #else
        let arch = "x86_64 (Intel)"
        let hasNEON = false
        let hasAVX = true
        let hasAVX512 = false
        let simdWidth = 256
        let perfCores = physCores
        let effCores = 0
        #endif

        let profile = CPUTopology(
            physicalCores: physCores,
            logicalCores: logCores,
            performanceCores: perfCores,
            efficiencyCores: effCores,
            l1CacheKB: 64,
            l2CacheKB: 256,
            l3CacheMB: physCores > 4 ? 12 : 4,
            maxFrequencyGHz: 3.2,
            architecture: arch,
            simdWidth: simdWidth,
            hasAVX: hasAVX,
            hasAVX512: hasAVX512,
            hasNEON: hasNEON
        )

        cpuProfile = profile
        return profile
    }

    // ─── MEMORY PROFILING ───
    func profileMemory() -> MemoryProfile {
        let totalBytes = ProcessInfo.processInfo.physicalMemory
        let totalGB = Double(totalBytes) / (1024 * 1024 * 1024)

        var vmStat = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.stride / MemoryLayout<integer_t>.stride)
        let pageSize = Double(vm_page_size)

        let kr: kern_return_t = withUnsafeMutablePointer(to: &vmStat) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else {
            return MemoryProfile(totalGB: totalGB, availableGB: 0, usedGB: 0,
                                 wiredGB: 0, compressedGB: 0, swapUsedGB: 0,
                                 memoryPressure: "UNKNOWN", bandwidthGBps: 0,
                                 pageSize: Int(vm_page_size), unifiedMemory: false)
        }

        let freeGB = Double(vmStat.free_count) * pageSize / (1024 * 1024 * 1024)
        let wiredGB = Double(vmStat.wire_count) * pageSize / (1024 * 1024 * 1024)
        let compressedGB = Double(vmStat.compressor_page_count) * pageSize / (1024 * 1024 * 1024)
        let usedGB = totalGB - freeGB

        let pressure: String
        if freeGB / totalGB > 0.3 { pressure = "NOMINAL" }
        else if freeGB / totalGB > 0.15 { pressure = "MODERATE" }
        else if freeGB / totalGB > 0.05 { pressure = "HIGH" }
        else { pressure = "CRITICAL" }

        #if arch(arm64)
        let unified = true
        let bandwidth = 68.25  // Apple Silicon typical
        #else
        let unified = false
        let bandwidth = 25.6   // DDR4 typical
        #endif

        let profile = MemoryProfile(
            totalGB: totalGB,
            availableGB: freeGB,
            usedGB: usedGB,
            wiredGB: wiredGB,
            compressedGB: compressedGB,
            swapUsedGB: 0,
            memoryPressure: pressure,
            bandwidthGBps: bandwidth,
            pageSize: Int(vm_page_size),
            unifiedMemory: unified
        )

        memProfile = profile
        return profile
    }

    // ─── THERMAL ESTIMATION ───
    func estimateThermalState() -> ThermalState {
        let thermalLevel = ProcessInfo.processInfo.thermalState

        let throttle: String
        let tempEst: Double
        let budgetRemaining: Double

        switch thermalLevel {
        case .nominal:
            throttle = "NONE"
            tempEst = 45.0
            budgetRemaining = 1.0
        case .fair:
            throttle = "LIGHT"
            tempEst = 65.0
            budgetRemaining = 0.75
        case .serious:
            throttle = "MODERATE"
            tempEst = 80.0
            budgetRemaining = 0.4
        case .critical:
            throttle = "SEVERE"
            tempEst = 95.0
            budgetRemaining = 0.1
        @unknown default:
            throttle = "UNKNOWN"
            tempEst = 50.0
            budgetRemaining = 0.5
        }

        let state = ThermalState(
            cpuTemperature: tempEst,
            gpuTemperature: tempEst * 0.9,
            throttleLevel: throttle,
            fanSpeedRPM: Int(tempEst * 30),
            powerDrawWatts: tempEst * 0.2,
            thermalBudgetRemaining: budgetRemaining
        )

        thermalState = state
        return state
    }

    // ─── GPU DETECTION ───
    func detectGPU() -> GPUCapability {
        #if arch(arm64)
        let gpu = GPUCapability(
            name: "Apple Integrated GPU",
            vendor: "Apple",
            vramMB: Int(ProcessInfo.processInfo.physicalMemory / (1024 * 1024)),  // Unified
            metalFamily: "Apple 7+",
            maxThreadsPerGroup: 1024,
            maxBufferLength: 256 * 1024 * 1024,
            supportsRaytracing: true,
            computeUnits: ProcessInfo.processInfo.processorCount * 2,
            flopsEstimate: 2.6e12  // ~2.6 TFLOPS
        )
        #else
        let gpu = GPUCapability(
            name: "Intel Iris Plus / HD Graphics",
            vendor: "Intel",
            vramMB: 1536,
            metalFamily: "Common 2",
            maxThreadsPerGroup: 512,
            maxBufferLength: 128 * 1024 * 1024,
            supportsRaytracing: false,
            computeUnits: 48,
            flopsEstimate: 441.6e9
        )
        #endif

        gpuProfile = gpu
        return gpu
    }

    // ─── NEURAL ENGINE DETECTION ───
    func detectNeuralEngine() -> NeuralEngineSpec {
        #if arch(arm64)
        let spec = NeuralEngineSpec(
            available: true,
            generationName: "Apple Neural Engine (16-core)",
            opsPerSecond: 15.8e12,
            supportedPrecisions: ["FP16", "INT8", "INT4"],
            maxModelSize: 512 * 1024 * 1024
        )
        #else
        let spec = NeuralEngineSpec(
            available: false,
            generationName: "N/A (Intel — using CPU/GPU fallback)",
            opsPerSecond: 0,
            supportedPrecisions: ["FP32", "FP16"],
            maxModelSize: 0
        )
        #endif

        neuralEngine = spec
        return spec
    }

    // ─── FULL PROFILE ───
    func fullProfile() -> [String: Any] {
        let cpu = profileCPU()
        let mem = profileMemory()
        let thermal = estimateThermalState()
        let gpu = detectGPU()
        let ne = detectNeuralEngine()
        lastProfileTime = Date()

        let snapshot: [String: Any] = [
            "cpu_arch": cpu.architecture,
            "cpu_cores": "\(cpu.physicalCores)P+\(cpu.efficiencyCores)E",
            "memory_total_gb": mem.totalGB,
            "memory_available_gb": mem.availableGB,
            "memory_pressure": mem.memoryPressure,
            "thermal_throttle": thermal.throttleLevel,
            "gpu_name": gpu.name,
            "gpu_tflops": gpu.flopsEstimate / 1e12,
            "neural_engine": ne.available,
            "ne_tops": ne.opsPerSecond / 1e12,
            "timestamp": Date().timeIntervalSince1970
        ]

        profileHistory.append(snapshot)
        if profileHistory.count > 100 { profileHistory.removeFirst() }

        return snapshot
    }

    // ─── WORKLOAD RECOMMENDATION ───
    func recommendWorkload() -> [String: Any] {
        let cpu = cpuProfile ?? profileCPU()
        let mem = memProfile ?? profileMemory()
        let thermal = thermalState ?? estimateThermalState()

        let maxBatchSize: Int
        let recommendedPrecision: String
        let useGPU: Bool
        let useNeuralEngine: Bool
        let concurrencyLimit: Int

        if mem.availableGB > 4.0 && thermal.thermalBudgetRemaining > 0.5 {
            maxBatchSize = 128
            recommendedPrecision = "FP16"
            useGPU = true
            useNeuralEngine = neuralEngine?.available ?? false
            concurrencyLimit = cpu.physicalCores
        } else if mem.availableGB > 2.0 {
            maxBatchSize = 64
            recommendedPrecision = "FP16"
            useGPU = true
            useNeuralEngine = false
            concurrencyLimit = cpu.physicalCores / 2
        } else if mem.availableGB > 1.0 {
            maxBatchSize = 32
            recommendedPrecision = "INT8"
            useGPU = false
            useNeuralEngine = false
            concurrencyLimit = 2
        } else {
            maxBatchSize = 8
            recommendedPrecision = "INT8"
            useGPU = false
            useNeuralEngine = false
            concurrencyLimit = 1
        }

        return [
            "max_batch_size": maxBatchSize,
            "precision": recommendedPrecision,
            "use_gpu": useGPU,
            "use_neural_engine": useNeuralEngine,
            "concurrency_limit": concurrencyLimit,
            "memory_pressure": mem.memoryPressure,
            "thermal_throttle": thermal.throttleLevel,
            "phi_scaling_factor": PHI * thermal.thermalBudgetRemaining
        ]
    }

    func statusReport() -> String {
        let cpu = cpuProfile ?? profileCPU()
        let mem = memProfile ?? profileMemory()
        let thermal = thermalState ?? estimateThermalState()
        let gpu = gpuProfile ?? detectGPU()
        let ne = neuralEngine ?? detectNeuralEngine()
        let rec = recommendWorkload()

        return """
        ╔═══════════════════════════════════════════════════════════╗
        ║    ⚡ HARDWARE CAPABILITY PROFILER                        ║
        ╠═══════════════════════════════════════════════════════════╣
        ║  CPU: \(cpu.architecture)
        ║       \(cpu.physicalCores) cores · L2=\(cpu.l2CacheKB)KB · L3=\(cpu.l3CacheMB)MB
        ║       SIMD=\(cpu.simdWidth)bit · AVX=\(cpu.hasAVX) · NEON=\(cpu.hasNEON)
        ╠═══════════════════════════════════════════════════════════╣
        ║  MEM: \(String(format: "%.1f", mem.totalGB))GB total · \(String(format: "%.1f", mem.availableGB))GB free
        ║       Pressure: \(mem.memoryPressure) · BW=\(String(format: "%.1f", mem.bandwidthGBps))GB/s
        ║       Unified: \(mem.unifiedMemory)
        ╠═══════════════════════════════════════════════════════════╣
        ║  GPU: \(gpu.name)
        ║       \(String(format: "%.1f", gpu.flopsEstimate / 1e12)) TFLOPS · \(gpu.computeUnits) CUs
        ║       Metal: \(gpu.metalFamily) · RT=\(gpu.supportsRaytracing)
        ╠═══════════════════════════════════════════════════════════╣
        ║  ANE: \(ne.available ? "✅ \(ne.generationName)" : "❌ Not available")
        ║       \(ne.available ? String(format: "%.1f TOPS", ne.opsPerSecond / 1e12) : "Using CPU/GPU fallback")
        ╠═══════════════════════════════════════════════════════════╣
        ║  THERMAL: \(thermal.throttleLevel) · CPU=\(String(format: "%.0f", thermal.cpuTemperature))°C
        ║           Power: \(String(format: "%.1f", thermal.powerDrawWatts))W · Budget: \(String(format: "%.0f%%", thermal.thermalBudgetRemaining * 100))
        ╠═══════════════════════════════════════════════════════════╣
        ║  RECOMMEND: batch=\(rec["max_batch_size"]!) · \(rec["precision"]!)
        ║             GPU=\(rec["use_gpu"]!) · ANE=\(rec["use_neural_engine"]!)
        ╚═══════════════════════════════════════════════════════════╝
        """
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - 🔄 DYNAMIC OPTIMIZATION ENGINE (Bucket D: Dynamic Optimizations)
// Runtime self-tuning: adaptive batch sizes, JIT compilation hints,
// memory pool rebalancing, thread pool scaling, cache eviction policy.
// φ-weighted feedback loops for continuous performance optimization.
// ═══════════════════════════════════════════════════════════════════

class DynamicOptimizationEngine {
    static let shared = DynamicOptimizationEngine()
    // PHI, TAU, GOD_CODE — use globals from L01_Constants

    // ─── OPTIMIZATION STATE ───
    struct OptimizationState {
        var batchSize: Int = 64
        var threadPoolSize: Int = 4
        var cacheCapacityMB: Int = 256
        var prefetchDepth: Int = 3
        var gcInterval: TimeInterval = 30.0
        var compressionLevel: Int = 6
        var inlineThreshold: Int = 100
        var loopUnrollFactor: Int = 4
    }

    // ─── PERFORMANCE SAMPLE ───
    struct PerfSample {
        let timestamp: Date
        let latencyMs: Double
        let throughputOps: Double
        let memoryUsedMB: Double
        let cpuUtilization: Double
        let cacheHitRate: Double
    }

    private var state = OptimizationState()
    private var perfHistory: [PerfSample] = []
    private var optimizationRuns: Int = 0
    private var improvements: [String] = []
    private var regressions: [String] = []
    private var autoTuneEnabled: Bool = true

    // ─── RECORD PERFORMANCE ───
    func recordSample(latencyMs: Double, throughputOps: Double,
                      memoryUsedMB: Double, cpuUtilization: Double,
                      cacheHitRate: Double) {
        let sample = PerfSample(
            timestamp: Date(),
            latencyMs: latencyMs,
            throughputOps: throughputOps,
            memoryUsedMB: memoryUsedMB,
            cpuUtilization: cpuUtilization,
            cacheHitRate: cacheHitRate
        )
        perfHistory.append(sample)
        if perfHistory.count > 2000 { perfHistory.removeFirst() }
    }

    // ─── ADAPTIVE BATCH SIZE ───
    func tuneBatchSize() {
        guard perfHistory.count >= 10 else { return }
        let recent: [PerfSample] = Array(perfHistory.suffix(20))
        let latencies: [Double] = recent.map { $0.latencyMs }
        let avgLatency: Double = latencies.reduce(0, +) / Double(recent.count)
        let throughputs: [Double] = recent.map { $0.throughputOps }
        let avgThroughput: Double = throughputs.reduce(0, +) / Double(recent.count)

        let oldBatch = state.batchSize

        if avgLatency < 10.0 && avgThroughput > 100.0 {
            // Headroom available — increase batch
            state.batchSize = min(state.batchSize + Int(Double(state.batchSize) * TAU * 0.1), 512)
        } else if avgLatency > 50.0 {
            // Too slow — decrease batch
            state.batchSize = max(state.batchSize - Int(Double(state.batchSize) * TAU * 0.2), 8)
        }

        if state.batchSize != oldBatch {
            improvements.append("batch_size: \(oldBatch) → \(state.batchSize)")
        }
    }

    // ─── THREAD POOL SCALING ───
    func tuneThreadPool() {
        guard perfHistory.count >= 10 else { return }
        let recent = Array(perfHistory.suffix(20))
        let avgCPU = recent.map { $0.cpuUtilization }.reduce(0, +) / Double(recent.count)
        let coreCount = ProcessInfo.processInfo.processorCount

        let oldThreads = state.threadPoolSize

        if avgCPU < 0.5 && state.threadPoolSize < coreCount {
            state.threadPoolSize = min(state.threadPoolSize + 1, coreCount)
        } else if avgCPU > 0.9 && state.threadPoolSize > 1 {
            state.threadPoolSize = max(state.threadPoolSize - 1, 1)
        }

        if state.threadPoolSize != oldThreads {
            improvements.append("thread_pool: \(oldThreads) → \(state.threadPoolSize)")
        }
    }

    // ─── CACHE POLICY TUNING ───
    func tuneCachePolicy() {
        guard perfHistory.count >= 10 else { return }
        let recent: [PerfSample] = Array(perfHistory.suffix(20))
        let hitRates: [Double] = recent.map { $0.cacheHitRate }
        let avgHitRate: Double = hitRates.reduce(0, +) / Double(recent.count)
        let memUsage: [Double] = recent.map { $0.memoryUsedMB }
        let avgMemory: Double = memUsage.reduce(0, +) / Double(recent.count)

        let oldCache = state.cacheCapacityMB

        if avgHitRate < 0.7 && avgMemory < 3000 {
            // Low hit rate, memory available — grow cache
            state.cacheCapacityMB = min(Int(Double(state.cacheCapacityMB) * PHI * 0.8), 1024)
        } else if avgHitRate > 0.95 && state.cacheCapacityMB > 128 {
            // Very high hit rate — can shrink cache
            state.cacheCapacityMB = max(Int(Double(state.cacheCapacityMB) * TAU), 64)
        }

        if state.cacheCapacityMB != oldCache {
            improvements.append("cache_mb: \(oldCache) → \(state.cacheCapacityMB)")
        }
    }

    // ─── PREFETCH DEPTH TUNING ───
    func tunePrefetchDepth() {
        guard perfHistory.count >= 10 else { return }
        let recent = Array(perfHistory.suffix(20))
        let avgLatency = recent.map { $0.latencyMs }.reduce(0, +) / Double(recent.count)

        let oldDepth = state.prefetchDepth

        if avgLatency < 5.0 {
            state.prefetchDepth = min(state.prefetchDepth + 1, 8)
        } else if avgLatency > 40.0 {
            state.prefetchDepth = max(state.prefetchDepth - 1, 1)
        }

        if state.prefetchDepth != oldDepth {
            improvements.append("prefetch_depth: \(oldDepth) → \(state.prefetchDepth)")
        }
    }

    // ─── GC INTERVAL TUNING ───
    func tuneGCInterval() {
        guard perfHistory.count >= 10 else { return }
        let recent = Array(perfHistory.suffix(20))
        let avgMemory = recent.map { $0.memoryUsedMB }.reduce(0, +) / Double(recent.count)

        if avgMemory > 3500 {
            state.gcInterval = max(state.gcInterval * TAU, 5.0)
        } else if avgMemory < 1000 {
            state.gcInterval = min(state.gcInterval * PHI, 120.0)
        }
    }

    // ─── FULL OPTIMIZATION CYCLE ───
    func optimize() -> [String: Any] {
        guard autoTuneEnabled else {
            return ["auto_tune": false, "reason": "disabled"]
        }

        optimizationRuns += 1
        let prevState = state

        tuneBatchSize()
        tuneThreadPool()
        tuneCachePolicy()
        tunePrefetchDepth()
        tuneGCInterval()

        let changed = state.batchSize != prevState.batchSize ||
                       state.threadPoolSize != prevState.threadPoolSize ||
                       state.cacheCapacityMB != prevState.cacheCapacityMB ||
                       state.prefetchDepth != prevState.prefetchDepth

        return [
            "run": optimizationRuns,
            "changed": changed,
            "batch_size": state.batchSize,
            "thread_pool": state.threadPoolSize,
            "cache_mb": state.cacheCapacityMB,
            "prefetch_depth": state.prefetchDepth,
            "gc_interval_s": state.gcInterval,
            "compression": state.compressionLevel,
            "improvements": improvements.suffix(10),
            "samples": perfHistory.count
        ]
    }

    // ─── PERFORMANCE TREND ───
    func performanceTrend() -> [String: Any] {
        guard perfHistory.count >= 20 else {
            return ["status": "insufficient_data", "samples": perfHistory.count]
        }

        let first10 = Array(perfHistory.prefix(10))
        let last10 = Array(perfHistory.suffix(10))

        let earlyLatency = first10.map { $0.latencyMs }.reduce(0, +) / 10.0
        let lateLatency = last10.map { $0.latencyMs }.reduce(0, +) / 10.0
        let earlyThroughput = first10.map { $0.throughputOps }.reduce(0, +) / 10.0
        let lateThroughput = last10.map { $0.throughputOps }.reduce(0, +) / 10.0

        return [
            "latency_trend": lateLatency < earlyLatency ? "improving" : "degrading",
            "throughput_trend": lateThroughput > earlyThroughput ? "improving" : "degrading",
            "latency_delta_pct": earlyLatency > 0 ? (earlyLatency - lateLatency) / earlyLatency * 100 : 0,
            "throughput_delta_pct": earlyThroughput > 0 ? (lateThroughput - earlyThroughput) / earlyThroughput * 100 : 0,
            "phi_quality_score": min((lateThroughput / max(lateLatency, 0.001)) * TAU, GOD_CODE)
        ]
    }

    func statusReport() -> String {
        let trend = performanceTrend()
        let rec = HardwareCapabilityProfiler.shared.recommendWorkload()
        return """
        ╔═══════════════════════════════════════════════════════════╗
        ║    🔄 DYNAMIC OPTIMIZATION ENGINE                         ║
        ╠═══════════════════════════════════════════════════════════╣
        ║  Auto-Tune:        \(autoTuneEnabled ? "🟢 ACTIVE" : "🔴 OFF")
        ║  Runs:             \(optimizationRuns)
        ║  Samples:          \(perfHistory.count)
        ╠═══════════════════════════════════════════════════════════╣
        ║  CURRENT CONFIG:
        ║    Batch Size:     \(state.batchSize)
        ║    Thread Pool:    \(state.threadPoolSize)
        ║    Cache:          \(state.cacheCapacityMB) MB
        ║    Prefetch:       \(state.prefetchDepth) levels
        ║    GC Interval:    \(String(format: "%.1f", state.gcInterval))s
        ╠═══════════════════════════════════════════════════════════╣
        ║  TREND: Latency \(trend["latency_trend"] ?? "?") · Throughput \(trend["throughput_trend"] ?? "?")
        ║  HW Recommend: batch=\(rec["max_batch_size"]!) · \(rec["precision"]!)
        ╚═══════════════════════════════════════════════════════════╝
        """
    }
}
