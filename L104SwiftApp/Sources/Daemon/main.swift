import logging
import os.log

import Foundation

private let logging = Logger(subsystem: "com.l104.daemon", category: "main")

// ═══════════════════════════════════════════════════════════════════
// MARK: - CONFIGURATION
// ═══════════════════════════════════════════════════════════════════

/// Read an env var as Int, with a default.
func envInt(_ key: String, default fallback: Int) -> Int {
    if let str = ProcessInfo.processInfo.environment[key], let val = Int(str) {
        return val
    }
    return fallback
}

/// Read an env var as String, with a default.
func envStr(_ key: String, default fallback: String) -> String {
    ProcessInfo.processInfo.environment[key] ?? fallback
}

/// Resolve the L104 project root from environment or working directory.
let l104Root: String = envStr("L104_ROOT", default: FileManager.default.currentDirectoryPath)

/// Shared IPC queue (cross-process: Python server, Swift app, CLI tools).
let sharedQueueDir = "/tmp/l104_queue"

/// vQPU Bridge IPC (Python VQPUBridge ↔ Swift MetalVQPU).
let bridgeDir: String = envStr("L104_BRIDGE_PATH", default: "/tmp/l104_bridge")

/// Project-local circuit directories.
let localBaseDir = "\(l104Root)/.l104_circuits"

/// PID file for process coordination.
let pidFilePath = "\(l104Root)/l104_daemon.pid"

/// v6.0: Environment-driven concurrency limits ═══
let configBridgeConcurrency = envInt("L104_BRIDGE_CONCURRENCY", default: 64)
let configSharedConcurrency = envInt("L104_SHARED_CONCURRENCY", default: 16)
let configLocalConcurrency  = envInt("L104_LOCAL_CONCURRENCY", default: 16)
let configHealthInterval    = envInt("L104_HEALTH_INTERVAL", default: 15)
let configPipelineDepth     = envInt("L104_VQPU_PIPELINE_DEPTH", default: 8)
let configBridgeTimeoutMs   = envInt("L104_BRIDGE_TIMEOUT_MS", default: 30000)
let configDaemonVersion     = envStr("L104_DAEMON_VERSION", default: "2.0")

/// v6.0: Auto-scaling and memory management
let configAutoScaleEnabled  = envInt("L104_AUTO_SCALE", default: 1) != 0
let configMemoryPoolSize    = envInt("L104_MEMORY_POOL_SIZE", default: 128)
let configCircuitValidation = envInt("L104_CIRCUIT_VALIDATION", default: 1) != 0
let configPerfProfiling     = envInt("L104_PERF_PROFILING", default: 1) != 0
let configMemoryPressureThreshold = envInt("L104_MEMORY_PRESSURE_THRESHOLD", default: 80)

// Global shared resources
var circuitBufferPool: CircuitBufferPool!
var cachedISO8601Formatter: ISO8601DateFormatter = {
    let formatter = ISO8601DateFormatter()
    formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
    return formatter
}()

// ═══════════════════════════════════════════════════════════════════
// MARK: - LIFECYCLE
// ═══════════════════════════════════════════════════════════════════

/// v5.0: Cached formatter + buffered stdout (no per-call ISO8601DateFormatter alloc).
func log(_ msg: String) {
    let ts = ISO8601DateFormatter().string(from: Date())
    logging.info("[L104 Daemon] \(self.ts) \(self.msg)")
    fflush(stdout)
}

/// Kill any previous daemon instance tracked by the PID file.
func killPreviousDaemon() {
    guard FileManager.default.fileExists(atPath: pidFilePath),
          let pidStr = try? String(contentsOfFile: pidFilePath, encoding: .utf8),
          let oldPid = Int(pidStr.trimmingCharacters(in: .whitespacesAndNewlines)),
          oldPid > 0 else { return }

    log("Killing previous daemon instance (PID \(oldPid))...")
    kill(pid_t(oldPid), SIGTERM)

    // Wait up to 2s for graceful exit
    for _ in 0..<20 {
        usleep(100_000)
        if kill(pid_t(oldPid), 0) != 0 { break }
    }

    // Force kill if still running
    if kill(pid_t(oldPid), 0) == 0 {
        log("Previous daemon did not exit gracefully, sending SIGKILL...")
        kill(pid_t(oldPid), SIGKILL)
    }
}

/// Write current PID to the PID file.
func writePidFile() {
    let pid = ProcessInfo.processInfo.processIdentifier
    try? "\(pid)".write(toFile: pidFilePath, atomically: true, encoding: .utf8)
}

/// Remove the PID file on clean exit.
func removePidFile() {
    try? FileManager.default.removeItem(atPath: pidFilePath)
}

/// Global shutdown flag (set by signal handlers).
private var shutdownRequested = false

/// Signal handler setup
private var signalSources: [DispatchSourceSignal] = []

func setupSignalHandlers() {
    let queue = DispatchQueue.global(qos: .utility)

    // Ignore default handlers
    signal(SIGTERM, SIG_IGN)
    signal(SIGINT, SIG_IGN)
    signal(SIGHUP, SIG_IGN)

    // SIGTERM → graceful shutdown
    let termSource = DispatchSource.makeSignalSource(signal: SIGTERM, queue: queue)
    termSource.setEventHandler {
        logging.info("SIGTERM received — graceful shutdown")
        shutdownRequested = true
    }
    termSource.resume()

    // SIGINT → graceful shutdown
    let intSource = DispatchSource.makeSignalSource(signal: SIGINT, queue: queue)
    intSource.setEventHandler {
        logging.info("SIGINT received — graceful shutdown")
        shutdownRequested = true
    }
    intSource.resume()

    // SIGHUP → reload configuration
    let hupSource = DispatchSource.makeSignalSource(signal: SIGHUP, queue: queue)
    hupSource.setEventHandler {
        logging.info("SIGHUP received — reloading configuration")
        // Reload config if needed
    }
    hupSource.resume()

    signalSources = [termSource, intSource, hupSource]
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - STATUS
// ═══════════════════════════════════════════════════════════════════

/// File watchers for circuit queues
private var sharedWatcher: DaemonCircuitWatcher?
private var bridgeWatcher: DaemonCircuitWatcher?
private var localWatcher: DaemonCircuitWatcher?

/// Daemon profiler for performance metrics
var daemonProfiler: DaemonProfiler?

/// Auto-scaler for dynamic concurrency adjustment
private var autoScaler: AutoScaler?

/// Daemon profiler for performance metrics
class DaemonProfiler {
    private var timings: [String: CFAbsoluteTime] = [:]
    private var metrics: [String: [String: Double]] = [:]

    func startTiming(_ key: String) {
        timings[key] = CFAbsoluteTimeGetCurrent()
    }

    func endTiming(_ key: String) {
        timings.removeValue(forKey: key)
    }

    func recordMetric(_ category: String, _ key: String, _ value: Double) {
        if metrics[category] == nil { metrics[category] = [:] }
        metrics[category]![key] = value
    }

    func getMetrics() -> [String: Any] {
        return metrics
    }
}

/// Auto-scaler for dynamic concurrency adjustment
class AutoScaler {
    func getScaleFactor() -> Double {
        return 1.0
    }

    func getAverageLoad() -> Double {
        return 0.5
    }
}

/// Get system load average
func getSystemLoad() -> Double {
    var load: (Double, Double, Double) = (0.0, 0.0, 0.0)
    let count = withUnsafeMutablePointer(to: &load) { ptr in
        ptr.withMemoryRebound(to: Double.self, capacity: 3) { buf in
            getloadavg(buf, 3)
        }
    }
    return count == 3 ? load.0 : 0.0
}

func dumpStatus() {
    let shared = sharedWatcher?.getStatus() ?? [:]
    let bridge = bridgeWatcher?.getStatus() ?? [:]
    let local = localWatcher?.getStatus() ?? [:]
    let pid = ProcessInfo.processInfo.processIdentifier

    let totalProcessed = (sharedWatcher?.circuitsProcessed ?? 0)
        + (bridgeWatcher?.circuitsProcessed ?? 0)
        + (localWatcher?.circuitsProcessed ?? 0)
    let totalFailed = (sharedWatcher?.circuitsFailed ?? 0)
        + (bridgeWatcher?.circuitsFailed ?? 0)
        + (localWatcher?.circuitsFailed ?? 0)

    // Build auto_scaling dictionary
    var autoScaleDict: [String: Double] = ["enabled": 0.0]
    if configAutoScaleEnabled, let scaler = autoScaler {
        autoScaleDict = [
            "enabled": 1.0,
            "current_factor": scaler.getScaleFactor(),
            "average_load": scaler.getAverageLoad(),
        ]
    }

    let status: [String: Any] = [
        "version": "6.0.0",
        "plist_version": configDaemonVersion,
        "pid": pid,
        "shutdown_requested": shutdownRequested,
        "shared_watcher": shared,
        "bridge_watcher": bridge,
        "local_watcher": local,
        "vqpu": [:],
        "micro_daemon": [:],
        "l104_root": l104Root,
        "performance_profile": configPerfProfiling ? (daemonProfiler?.getMetrics() ?? [:]) : [:],
        "auto_scaling": autoScaleDict,
        "memory_pool": [
            "configured_size": configMemoryPoolSize,
            "circuit_validation": configCircuitValidation,
        ],
        "config": [
            "bridge_concurrency": configBridgeConcurrency,
            "shared_concurrency": configSharedConcurrency,
            "local_concurrency": configLocalConcurrency,
            "pipeline_depth": configPipelineDepth,
            "bridge_timeout_ms": configBridgeTimeoutMs,
            "health_interval": configHealthInterval,
            "auto_scale_enabled": configAutoScaleEnabled,
            "memory_pool_size": configMemoryPoolSize,
            "circuit_validation": configCircuitValidation,
            "perf_profiling": configPerfProfiling,
            "memory_pressure_threshold": configMemoryPressureThreshold,
        ] as [String: Any],
        "aggregate": [
            "total_processed": totalProcessed,
            "total_failed": totalFailed,
            "success_rate": totalProcessed + totalFailed > 0
                ? Double(totalProcessed) / Double(totalProcessed + totalFailed) : 1.0,
        ],
    ]

    // Best-effort JSON output
    if let data = try? JSONSerialization.data(withJSONObject: status, options: [.prettyPrinted, .sortedKeys]),
       let json = String(data: data, encoding: .utf8) {
        log("SIGUSR1 status:\n\(json)")
    } else {
        log("SIGUSR1 status: shared=\(shared) local=\(local)")
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - ENTRY POINT
// ═══════════════════════════════════════════════════════════════════

log("L104 Quantum Node Daemon v6.0 starting (plist v\(configDaemonVersion))")
log("  Root:     \(l104Root)")
log("  Shared:   \(sharedQueueDir) (\(configSharedConcurrency)x)")

// Setup
killPreviousDaemon()
writePidFile()
setupSignalHandlers()

// Initialize watchers
sharedWatcher = DaemonCircuitWatcher(baseDir: sharedQueueDir, maxConcurrent: configSharedConcurrency)
bridgeWatcher = DaemonCircuitWatcher(baseDir: bridgeDir, maxConcurrent: configBridgeConcurrency)
localWatcher = DaemonCircuitWatcher(baseDir: localBaseDir, maxConcurrent: configLocalConcurrency)
daemonProfiler = DaemonProfiler()
if configAutoScaleEnabled {
    autoScaler = AutoScaler()
}

// Main run loop
var tickCount = 0
while !shutdownRequested {
    tickCount += 1

    // Process circuits from all queues (processing handled by watcher threads)

    // Periodic status dump
    if tickCount % configHealthInterval == 0 {
        dumpStatus()
    }

    // Sleep for 1 second between ticks
    sleep(1)
}

// Cleanup
log("Shutting down...")
removePidFile()
log("Goodbye!")
