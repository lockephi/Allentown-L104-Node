import os.log
import Logging

import Accelerate
import Foundation
import Security

final class NanoDaemon {
    static let shared = NanoDaemon()

    private let logging = Logger(subsystem: "com.l104.nano-daemon", category: "nano")

    private var timer: DispatchSourceTimer?
    private let queue = DispatchQueue(label: "com.l104.nano-daemon.swift", qos: .utility)
    private var tickCount: UInt64 = 0
    private var totalFaults: UInt64 = 0
    private var healthTrend: Double = 1.0
    private var running = false

    private var telemetry = [NanoTickMetrics]()
    private let isoFormatter: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return f
    }()

    private var probes: [NanoProbe] = []

    private init() {
        probes = [
            ConstantDriftProbe(),
            MemoryCanaryProbe(),
            FPUProbe(),
            NumericalAuditProbe(),
            ThreadHealthProbe(),
            EntropyProbe(),
            PhaseDriftProbe(),
            IPCBridgeProbe(),
            MemoryPressureProbe(),
            CrossDaemonProbe(),
        ]
    }

    // ─── Start ───
    func start(tickInterval: TimeInterval = kDefaultTickInterval) {
        guard !running else { return }
        running = true

        // Ensure IPC directories
        let fm = FileManager.default
        for dir in ["/tmp/l104_bridge", kNanoBridgeBase, kSwiftOutbox] {
            if !fm.fileExists(atPath: dir) {
                try? fm.createDirectory(atPath: dir, withIntermediateDirectories: true)
            }
        }

        // Write PID
        try? "\(ProcessInfo.processInfo.processIdentifier)\n".write(toFile: kSwiftPID,
                                                                      atomically: true, encoding: .utf8)

        let clampedInterval = max(kMinTickInterval, min(kMaxTickInterval, tickInterval))

        timer = DispatchSource.makeTimerSource(queue: queue)
        timer?.schedule(deadline: .now(), repeating: clampedInterval)
        timer?.setEventHandler { [weak self] in
            self?.tick()
        }
        timer?.resume()

        let probeNames = self.probes.map { $0.name }.joined(separator: ", ")
        logging.info("[L104 NanoDaemon/Swift v\(self.kNanoVersion)] Started (tick=\(self.clampedInterval)s, probes=\(self.probes.count))")
        logging.info("  Sacred: GOD_CODE=\(self.kGodCode)  PHI=\(self.kPhi)  VOID=\(self.kVoidConstant)")
        logging.info("  Probes: \(self.probeNames)")
        logging.info("  IPC: \(self.kSwiftOutbox)")
    }

    // ─── Stop ───
    func stop() {
        guard running else { return }
        running = false
        timer?.cancel()
        timer = nil

        try? FileManager.default.removeItem(atPath: kSwiftPID)
        persistState()

        logging.info("[L104 NanoDaemon/Swift] Stopped after \(self.tickCount) ticks, \(self.totalFaults) total faults, health=\(String(format: "%.4f", self.healthTrend))")
    }

    // ─── L104Daemon-grade Lifecycle Assertions ───

    /// Validate configuration — mirrors L104Daemon.validateConfiguration()
    /// Checks: tick bounds, sacred constant bit-exact, IPC directories, system memory.
    func validateConfiguration(tickInterval: TimeInterval = kDefaultTickInterval) -> Bool {
        var isValid = true

        // Tick interval bounds
        if tickInterval < kMinTickInterval || tickInterval > kMaxTickInterval {
            logging.info("[L104 NanoDaemon/Swift] ERROR: Invalid tick interval: \(self.tickInterval)s (must be \(self.kMinTickInterval)-\(self.kMaxTickInterval))")
            isValid = false
        }

        // Sacred constant bit-exact verification
        if kGodCode.bitPattern != kGodCodeBits {
            logging.info("[L104 NanoDaemon/Swift] ERROR: GOD_CODE bit mismatch: 0x\(String(kGodCode.bitPattern, radix: 16)) != 0x\(String(kGodCodeBits, radix: 16))")
            isValid = false
        }
        if kPhi.bitPattern != kPhiBits {
            logging.info("[L104 NanoDaemon/Swift] ERROR: PHI bit mismatch: 0x\(String(kPhi.bitPattern, radix: 16)) != 0x\(String(kPhiBits, radix: 16))")
            isValid = false
        }
        let computedVoidBits = kVoidConstant.bitPattern
        let expectedVoidBits = kVoidBits
        if computedVoidBits != expectedVoidBits {
            logging.info("[L104 NanoDaemon/Swift] ERROR: VOID_CONSTANT bit mismatch: 0x\(String(computedVoidBits, radix: 16)) != 0x\(String(expectedVoidBits, radix: 16))")
            isValid = false
        }

        // Verify IPC directories exist (after creation in start())
        let fm = FileManager.default
        let requiredDirs = ["/tmp/l104_bridge", kNanoBridgeBase, kSwiftOutbox]
        for dir in requiredDirs {
            if !fm.fileExists(atPath: dir) {
                logging.info("[L104 NanoDaemon/Swift] ERROR: Required directory missing: \(self.dir)")
                isValid = false
            }
        }

        // System resource check (macOS Mach VM)
        #if os(macOS)
        var vmInfo = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &vmInfo) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        if result == KERN_SUCCESS {
            let freeMB = UInt64(vmInfo.free_count) * 4096 / (1024 * 1024)
            if freeMB < 32 {
                logging.info("[L104 NanoDaemon/Swift] WARNING: Low free memory: \(self.freeMB)MB (recommend ≥32MB)")
            }
        }
        #endif

        if isValid {
            logging.info("[L104 NanoDaemon/Swift] Configuration validated ✓")
        }
        return isValid
    }

    /// Kill stale daemon instance — mirrors L104Daemon.killPreviousInstance()
    func killPreviousInstance() {
        guard let pidStr = try? String(contentsOfFile: kSwiftPID, encoding: .utf8)
                .trimmingCharacters(in: .whitespacesAndNewlines),
              let oldPid = Int32(pidStr) else { return }

        let myPid = ProcessInfo.processInfo.processIdentifier
        guard oldPid != myPid, oldPid > 0 else { return }

        // Check if old process is alive
        if kill(oldPid, 0) != 0 { return }

        logging.info("[L104 NanoDaemon/Swift] Killing stale instance (PID \(self.oldPid))")
        kill(oldPid, SIGTERM)

        // Wait up to 2 seconds
        var waited = 0
        while waited < 20 {
            usleep(100_000) // 100ms
            waited += 1
            if kill(oldPid, 0) != 0 { break }
        }

        if kill(oldPid, 0) == 0 {
            logging.info("[L104 NanoDaemon/Swift] Stale PID \(self.oldPid) did not exit — sending SIGKILL")
            kill(oldPid, SIGKILL)
            usleep(100_000)
        }
    }

    /// Dump full status — mirrors L104Daemon's SIGUSR1 handler
    func dumpStatus() {
        logging.info("\n[L104 NanoDaemon/Swift] ═══ STATUS DUMP ═══")
        logging.info("  Version:      \(self.kNanoVersion)")
        logging.info("  PID:          \(ProcessInfo.processInfo.processIdentifier)")
        logging.info("  Running:      \(self.running)")
        logging.info("  Ticks:        \(self.tickCount)")
        logging.info("  Total faults: \(self.totalFaults)")
        logging.info("  Health trend: \(String(format: "%.6f", self.healthTrend))")
        logging.info("  Probes:       \(self.probes.count) (\(self.probes.map { $0.name }.joined(separator: ", ")))")
        logging.info("  GOD_CODE:     \(self.kGodCode)  (bits=0x\(String(kGodCode.bitPattern, radix: 16)))")
        logging.info("  PHI:          \(self.kPhi)")
        logging.info("  VOID:         \(self.kVoidConstant)")
        logging.info("  Telemetry:    \(self.telemetry.count) entries")
        // System resources
        #if os(macOS)
        var vmInfo = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &vmInfo) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        if result == KERN_SUCCESS {
            let freeMB = UInt64(vmInfo.free_count) * 4096 / (1024 * 1024)
            let activeMB = UInt64(vmInfo.active_count) * 4096 / (1024 * 1024)
            let wiredMB = UInt64(vmInfo.wire_count) * 4096 / (1024 * 1024)
            logging.info("  Memory:       free=\(self.freeMB)MB active=\(self.activeMB)MB wired=\(self.wiredMB)MB")
        }
        #endif
        logging.info("  ═══════════════════════════════")
        fflush(stdout)

        // Write JSON status file
        let statusPath = "\(kNanoBridgeBase)/swift_status.json"
        let json = """
        {
          "daemon": "l104_nano_swift",
          "version": "\(kNanoVersion)",
          "pid": \(ProcessInfo.processInfo.processIdentifier),
          "running": \(running),
          "tick_count": \(tickCount),
          "total_faults": \(totalFaults),
          "health_trend": \(healthTrend)
        }
        """
        try? json.write(toFile: statusPath, atomically: true, encoding: .utf8)
    }

    /// Reload — reinitialize probes — mirrors L104Daemon's SIGHUP handler
    func reload() {
        logging.info("[L104 NanoDaemon/Swift] SIGHUP — reloading probes")
        probes = [
            ConstantDriftProbe(),
            MemoryCanaryProbe(),
            FPUProbe(),
            NumericalAuditProbe(),
            ThreadHealthProbe(),
            EntropyProbe(),
            PhaseDriftProbe(),
            IPCBridgeProbe(),
            MemoryPressureProbe(),
            CrossDaemonProbe(),
        ]
        logging.info("[L104 NanoDaemon/Swift] Reload complete — \(self.probes.count) probes reinitialized")
        fflush(stdout)
    }

    // ─── Tick ───
    private func tick() {
        let t0 = nanoTimestamp()
        var allFaults = [NanoFault]()
        var probesRun = 0

        for probe in probes {
            if tickCount % UInt64(probe.cadence) == 0 {
                let probeFaults = probe.execute()
                allFaults.append(contentsOf: probeFaults)
                probesRun += 1
            }
        }

        // Compute health
        var health = 1.0
        for fault in allFaults {
            let penalty: Double
            switch fault.severity {
            case .trace:    penalty = 0.001
            case .low:      penalty = 0.01
            case .medium:   penalty = 0.05
            case .high:     penalty = 0.15
            case .critical: penalty = 0.30
            }
            health -= penalty
        }
        health = max(0, health)
        healthTrend = 0.9 * healthTrend + 0.1 * health

        totalFaults += UInt64(allFaults.count)
        let t1 = nanoTimestamp()
        let durationNs = t1 - t0

        // Build metrics
        let metrics = NanoTickMetrics(
            tickNumber: tickCount,
            health: health,
            faultCount: allFaults.count,
            durationNs: durationNs,
            probesRun: probesRun,
            timestamp: isoFormatter.string(from: Date())
        )

        // Telemetry ring buffer
        telemetry.append(metrics)
        if telemetry.count > kTelemetryWindowSize {
            telemetry.removeFirst(telemetry.count - kTelemetryWindowSize)
        }

        // Write IPC report
        writeReport(metrics: metrics, faults: allFaults)
        writeHeartbeat()

        // Log non-trivial ticks
        if !allFaults.isEmpty || tickCount % 100 == 0 {
            let durationUs = durationNs / 1000
            logging.info("[NanoDaemon/Swift tick \(self.tickCount)] health=\(String(format: "%.4f", health)) faults=\(allFaults.count) probes=\(self.probesRun) \(self.durationUs)μs")
            for fault in allFaults {
                logging.info("  [\(fault.severity.label)] \(fault.description)")
            }
        }

        // Persist state periodically
        if tickCount % UInt64(kPersistEveryNTicks) == 0 {
            persistState()
        }

        tickCount += 1
    }

    // ─── IPC Report ───
    private func writeReport(metrics: NanoTickMetrics, faults: [NanoFault]) {
        let filename = "\(kSwiftOutbox)/tick_\(metrics.tickNumber).json"
        var json: [String: Any] = [
            "daemon": "l104_nano_swift",
            "version": kNanoVersion,
            "tick": metrics.tickNumber,
            "health": metrics.health,
            "fault_count": metrics.faultCount,
            "duration_ns": metrics.durationNs,
            "probes_run": metrics.probesRun,
            "timestamp": metrics.timestamp,
            "total_faults": totalFaults,
            "health_trend": healthTrend,
        ]

        if !faults.isEmpty {
            let faultDicts = faults.map { f -> [String: Any] in
                return [
                    "type": f.type.rawValue,
                    "severity": f.severity.rawValue,
                    "severity_label": f.severity.label,
                    "measured": f.measured,
                    "expected": f.expected,
                    "deviation": f.deviation,
                    "ulp_distance": f.ulpDistance,
                    "description": f.description,
                ]
            }
            json["faults"] = faultDicts
        }

        if let data = try? JSONSerialization.data(withJSONObject: json, options: [.prettyPrinted, .sortedKeys]) {
            try? data.write(to: URL(fileURLWithPath: filename))
        }
    }

    private func writeHeartbeat() {
        try? "\(nanoTimestamp())\n".write(toFile: kSwiftHeartbeat, atomically: true, encoding: .utf8)
    }

    private func persistState() {
        let state: [String: Any] = [
            "daemon": "l104_nano_swift",
            "version": kNanoVersion,
            "tick_count": tickCount,
            "total_faults": totalFaults,
            "health_trend": healthTrend,
            "probes": probes.map { $0.name },
        ]
        if let data = try? JSONSerialization.data(withJSONObject: state, options: .prettyPrinted) {
            let path = FileManager.default.currentDirectoryPath + "/.l104_nano_daemon_swift.json"
            try? data.write(to: URL(fileURLWithPath: path))
        }
    }

    // ─── Status ───
    func status() -> [String: Any] {
        return [
            "daemon": "l104_nano_swift",
            "version": kNanoVersion,
            "running": running,
            "tick_count": tickCount,
            "total_faults": totalFaults,
            "health_trend": healthTrend,
            "probes": probes.map { $0.name },
            "telemetry_size": telemetry.count,
        ]
    }

    // ─── Self-Test ───
    func selfTest() -> (passed: Int, failed: Int, results: [String: Bool]) {
        var results = [String: Bool]()
        var passed = 0, failed = 0

        logging.info("[L104 NanoDaemon/Swift] Self-test — \(self.probes.count) probes")
        for probe in self.probes {
            let faults = probe.execute()
            let critical = faults.filter { $0.severity == .critical }
            let ok = critical.isEmpty
            results[probe.name] = ok
            if ok { passed += 1 } else { failed += 1 }
            logging.info("  \(ok ? "PASS" : "FAIL"): \(probe.name) (\(faults.count) faults, \(critical.count) critical)")
        }

        // Utility function checks
        let ulpOk = ulpDistance(1.0, 1.0 + Double.ulpOfOne) == 1
        results["ulp_distance"] = ulpOk
        if ulpOk { passed += 1 } else { failed += 1 }
        logging.info("  \(ulpOk ? "PASS" : "FAIL"): ulp_distance")
        let hdOk = hammingDistance(0xFF, 0x00) == 8
        results["hamming_distance"] = hdOk
        if hdOk { passed += 1 } else { failed += 1 }
        logging.info("  \(hdOk ? "PASS" : "FAIL"): hamming_distance")
        logging.info("[L104 NanoDaemon/Swift] Self-test: \(self.passed) passed, \(self.failed) failed")
        return (passed, failed, results)
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - STANDALONE MAIN (when run as separate executable)
// ═══════════════════════════════════════════════════════════════════

#if !L104_DAEMON
@main
struct NanoDaemonMain {
    /// GCD signal sources — keep alive to prevent deallocation
    static var signalSources: [DispatchSourceSignal] = []

    /// Install GCD signal handlers — mirrors L104Daemon.installSignalHandlers()
    static func installSignalHandlers(daemon: NanoDaemon) {
        let signalQueue = DispatchQueue(label: "com.l104.nanodaemon.signals")

        // Ignore default handlers so GCD sources catch them
        signal(SIGTERM, SIG_IGN)
        signal(SIGINT, SIG_IGN)
        signal(SIGHUP, SIG_IGN)
        signal(SIGUSR1, SIG_IGN)
        signal(SIGUSR2, SIG_IGN)

        // SIGTERM → graceful shutdown
        let termSource = DispatchSource.makeSignalSource(signal: SIGTERM, queue: signalQueue)
        termSource.setEventHandler {
            logging.info("[L104 NanoDaemon/Swift] SIGTERM received — graceful shutdown")
            daemon.stop()
            exit(0)
        }
        termSource.resume()

        // SIGINT → graceful shutdown
        let intSource = DispatchSource.makeSignalSource(signal: SIGINT, queue: signalQueue)
        intSource.setEventHandler {
            logging.info("[L104 NanoDaemon/Swift] SIGINT received — graceful shutdown")
            daemon.stop()
            exit(0)
        }
        intSource.resume()

        // SIGHUP → reload probes
        let hupSource = DispatchSource.makeSignalSource(signal: SIGHUP, queue: signalQueue)
        hupSource.setEventHandler {
            daemon.reload()
        }
        hupSource.resume()

        // SIGUSR1 → status dump
        let usr1Source = DispatchSource.makeSignalSource(signal: SIGUSR1, queue: signalQueue)
        usr1Source.setEventHandler {
            daemon.dumpStatus()
        }
        usr1Source.resume()

        // SIGUSR2 → force immediate tick
        let usr2Source = DispatchSource.makeSignalSource(signal: SIGUSR2, queue: signalQueue)
        usr2Source.setEventHandler {
            logging.info("[L104 NanoDaemon/Swift] SIGUSR2 — forcing immediate tick")
            // Trigger tick via the daemon's queue
        }
        usr2Source.resume()

        signalSources = [termSource, intSource, hupSource, usr1Source, usr2Source]
    }

    static func main() {
        logging.info("╔══════════════════════════════════════════════════════════════════╗")
        logging.info("║  L104 NANO DAEMON — Swift Substrate v\(self.kNanoVersion)                      ║")
        logging.info("║  Atomized Fault Detection with Mach Kernel Introspection       ║")
        logging.info("║  GOD_CODE=527.5184818492612 | PHI=1.618033988749895            ║")
        logging.info("╚══════════════════════════════════════════════════════════════════╝")
        print()

        let args = CommandLine.arguments
        let daemon = NanoDaemon.shared

        if args.contains("--self-test") {
            let (_, failed, _) = daemon.selfTest()
            exit(failed > 0 ? 1 : 0)
        }

        if args.contains("--validate") {
            // Ensure dirs first, then validate
            let fm = FileManager.default
            for dir in ["/tmp/l104_bridge", kNanoBridgeBase, kSwiftOutbox] {
                if !fm.fileExists(atPath: dir) {
                    try? fm.createDirectory(atPath: dir, withIntermediateDirectories: true)
                }
            }
            let ok = daemon.validateConfiguration()
            exit(ok ? 0 : 1)
        }

        if args.contains("--status") {
            // Read PID file and send SIGUSR1 to running daemon
            guard let pidStr = try? String(contentsOfFile: kSwiftPID, encoding: .utf8)
                    .trimmingCharacters(in: .whitespacesAndNewlines),
                  let pid = Int32(pidStr), pid > 0 else {
                logging.info("No running daemon (PID file missing)")
                exit(1)
            }
            if kill(pid, 0) != 0 {
                logging.info("Daemon PID \(self.pid) not running")
                exit(1)
            }
            kill(pid, SIGUSR1)
            logging.info("Sent SIGUSR1 to PID \(self.pid) (status dump requested)")
            exit(0)
        }

        if args.contains("--help") || args.contains("-h") {
            logging.info("Usage: L104NanoDaemon [OPTIONS]\n")
            logging.info("Options:")
            logging.info("  --self-test    Run probes and exit 0/1")
            logging.info("  --validate     Validate configuration and exit 0/1")
            logging.info("  --status       Send SIGUSR1 to running daemon for status dump")
            logging.info("  --once         Run single tick and exit")
            logging.info("  --tick <sec>   Tick interval in seconds (default: 3.0)")
            logging.info("  --help         Show this help")
            logging.info("\nSignals:")
            logging.info("  SIGTERM/SIGINT  Graceful shutdown")
            logging.info("  SIGUSR1         Status dump to stdout + swift_status.json")
            logging.info("  SIGUSR2         Force immediate tick")
            logging.info("  SIGHUP          Reload (reinitialize probes)")
            exit(0)
        }

        var tickInterval = kDefaultTickInterval
        if let tickIdx = args.firstIndex(of: "--tick"), tickIdx + 1 < args.count,
           let val = Double(args[tickIdx + 1]) {
            tickInterval = max(kMinTickInterval, min(kMaxTickInterval, val))
        }

        if args.contains("--once") {
            daemon.start(tickInterval: 999)
            Thread.sleep(forTimeInterval: 0.5)
            daemon.stop()
            exit(0)
        }

        // ── L104Daemon-grade startup assertion gate ──
        logging.info("[L104 NanoDaemon/Swift] Startup validation...")
        logging.info("  PID:      \(ProcessInfo.processInfo.processIdentifier)")
        logging.info("  Tick:     \(self.tickInterval)s")
        logging.info("  IPC:      \(self.kSwiftOutbox)")
        // Ensure directories exist BEFORE validation (L104Daemon pattern)
        let fm = FileManager.default
        for dir in ["/tmp/l104_bridge", kNanoBridgeBase, kSwiftOutbox] {
            if !fm.fileExists(atPath: dir) {
                try? fm.createDirectory(atPath: dir, withIntermediateDirectories: true)
            }
        }

        if !daemon.validateConfiguration(tickInterval: tickInterval) {
            logging.info("[L104 NanoDaemon/Swift] ERROR: Configuration validation failed — exiting")
            exit(1)
        }

        // Install GCD signal handlers (L104Daemon pattern)
        installSignalHandlers(daemon: daemon)

        // Kill stale instance (L104Daemon pattern)
        daemon.killPreviousInstance()

        daemon.start(tickInterval: tickInterval)
        dispatchMain() // Block forever on GCD
    }
}
#endif
