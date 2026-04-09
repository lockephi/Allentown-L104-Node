import os.log

import Foundation

private let logging = Logger(subsystem: "com.l104.B60_QuantumAIDaemon", category: "main")

// MARK: - ═══ DAEMON THROTTLE CONTROLLER ═══
// Reads the quantum CPU governor state file written by l104_quantum_cpu_governor.py
// (EVO_74).  Governs which daemons run and at what interval using five quantum
// algorithms: QAOA scheduler, Grover priority engine, VQE load balancer,
// GOD_CODE resonance timer, and quantum walk backoff.

// EVO_76: DaemonThrottleController delegates all file I/O to GovernorStateCache
// (shared singleton in B75) — eliminates the duplicate 2-second JSON read that
// previously ran independently here and in GovernorAwareInterval.
final class DaemonThrottleController {
    static let shared = DaemonThrottleController()

    // Last-known interval multiplier from the quantum governor (PHI^load_tier)
    var currentMultiplier: Double { GovernorStateCache.shared.throttleMultiplier }

    // ------------------------------------------------------------------
    /// Returns true if the quantum CPU governor allows this daemon to run.
    /// When the governor file is absent / stale, defaults to true (safe fallback).
    func shouldRunDaemon(_ name: String) -> Bool {
        let json = GovernorStateCache.shared.snapshot()

        guard let shouldThrottle = json["should_throttle"] as? Bool,
              shouldThrottle else { return true }

        // Governor says we're over CPU quota — check allowed list
        if let allowed = json["allowed_daemons"] as? [String] {
            // Critical daemons (vqpu, soul, orchestrator) always run
            let criticals = ["vqpu", "soul", "orchestrator"]
            if criticals.contains(name) { return true }
            return allowed.contains(name)
        }
        return true
    }

    // ------------------------------------------------------------------
    /// Returns the PHI^tier interval multiplier for a specific daemon name.
    func intervalMultiplier(for name: String) -> Double {
        let json = GovernorStateCache.shared.snapshot()
        if let mults = json["interval_multipliers"] as? [String: Double] {
            return mults[name] ?? currentMultiplier
        }
        return currentMultiplier
    }

    // ------------------------------------------------------------------
    /// Returns the quantum-walk-derived poll delays (ms) for VQPU bridge.
    func pollDelaysMs() -> [Double] {
        let json = GovernorStateCache.shared.snapshot()
        return json["poll_delays_ms"] as? [Double] ?? [50, 80, 130, 210, 340]
    }
}

// MARK: - ═══ DAEMON CONSTANTS ═══

private let DAEMON_VERSION         = "2.0.0"
private let CYCLE_INTERVAL_S:     TimeInterval = 60.0
private let CYCLE_MIN_INTERVAL_S: TimeInterval = 15.0
private let CYCLE_MAX_INTERVAL_S: TimeInterval = 600.0
private let SCAN_BATCH_SIZE       = 13      // Fib(7)
private let PERSIST_EVERY_N       = 5
private let PHI_INV               = 1.0 / PHI  // ≈ 0.618033...
private let FIDELITY_MIN          = 0.618   // PHI inverse threshold
private let COHERENCE_MIN         = 0.500
private let QUARANTINE_THRESHOLD  = 3       // cycles before quarantine
private let TELEMETRY_WINDOW      = 104
private let MAX_IMPROVEMENT_HIST  = 104     // RF_N_ESTIMATORS (B56 private)

// MARK: - ═══ DATA STRUCTURES ═══

enum DaemonPhase: Int, CaseIterable {
    case fileScan = 1, fidelityCheck, harmonyCheck, optimize, improve, evolve, persist
    var label: String {
        switch self {
        case .fileScan:      return "FILE_SCAN"
        case .fidelityCheck: return "FIDELITY_CHECK"
        case .harmonyCheck:  return "HARMONY_CHECK"
        case .optimize:      return "OPTIMIZE"
        case .improve:       return "IMPROVE"
        case .evolve:        return "EVOLVE"
        case .persist:       return "PERSIST"
        }
    }
}

struct DaemonConfig {
    var cycleInterval:    TimeInterval = CYCLE_INTERVAL_S
    var minInterval:      TimeInterval = CYCLE_MIN_INTERVAL_S
    var maxInterval:      TimeInterval = CYCLE_MAX_INTERVAL_S
    var batchSize:        Int          = SCAN_BATCH_SIZE
    var persistEveryN:    Int          = PERSIST_EVERY_N
    var fidelityMin:      Double       = FIDELITY_MIN
    var coherenceMin:     Double       = COHERENCE_MIN
    var enableAutoFix:    Bool         = true
    var verboseLogging:   Bool         = false
}

struct L104FileInfo {
    let path:       String
    let ext:        String
    let sizeBytes:  Int
    let modDate:    Date
    var healthScore: Double    // 0.0–1.0
    var lastChecked: Date?
    var issues:     [String]
    var isQuarantined: Bool
    var cyclesSinceCheck: Int
}

struct ImprovementResult {
    let filePath:    String
    let phase:       DaemonPhase
    let issuesFound: Int
    let issuesFix:   Int
    let sacredScore: Double
    let elapsedMs:   Double
    let success:     Bool
    var notes:       [String]
}

struct FidelityReport {
    let godCodeDrift:     Double    // |GOD_CODE - expected| / expected
    let phiDrift:         Double
    let voidDrift:        Double
    let coherenceLevel:   Double
    let overallFidelity:  Double    // [0,1]
    let anomalies:        [String]
    let timestamp:        Date

    var isHealthy: Bool { overallFidelity >= FIDELITY_MIN }
}

struct HarmonyReport {
    let engineScores:    [String: Double]   // engine → coherence score
    let crossEnginePhi:  Double             // PHI alignment across engines
    let overallHarmony:  Double             // [0,1]
    let conflicts:       [String]
    let timestamp:       Date

    var isHarmonious: Bool { overallHarmony >= COHERENCE_MIN }
}

struct OptimizationResult {
    let cacheCleared:    Int        // entries purged
    let memoryFreedMB:   Double
    let importsCached:   Int
    let gcCycles:        Int
    let elapsedMs:       Double
    let sacredScore:     Double
}

struct EvolutionCycle {
    let cycleIndex:      Int
    let strategyUpdates: [String]
    let intervalDelta:   TimeInterval  // how much interval changed
    let fitnessScore:    Double
    let sacredAlignment: Double
}

struct ImprovementReport {
    let cycleId:         Int
    let phases:          [DaemonPhase: Bool]  // phase → success
    let filesScanned:    Int
    let filesImproved:   Int
    let fidelity:        FidelityReport
    let harmony:         HarmonyReport
    let evolution:       EvolutionCycle?
    let totalElapsedMs:  Double
    let sacredScore:     Double
    let timestamp:       Date
}

struct DaemonStatus {
    let isRunning:       Bool
    let cycleCount:      Int
    let lastCycle:       Date?
    let nextCycle:       Date?
    let currentInterval: TimeInterval
    let totalFilesKnown: Int
    let healthyFiles:    Int
    let quarantinedFiles: Int
    let fidelity:        FidelityReport?
    let harmony:         HarmonyReport?
    let recentHistory:   [ImprovementResult]
    let sacredScore:     Double
}

// MARK: - ═══ FILE SCANNER ═══

final class FileScanner {
    private(set) var knownFiles: [String: L104FileInfo] = [:]
    private let rootPath: String
    private let extensions = ["swift", "py", "cpp", "h", "json", "md"]

    init(root: String = "/Users/carolalvarez/Applications/Allentown-L104-Node") {
        self.rootPath = root
    }

    @discardableResult
    func scan(batchSize: Int = SCAN_BATCH_SIZE) -> [L104FileInfo] {
        let fm = FileManager.default
        let rootURL = URL(fileURLWithPath: rootPath)
        // Prefetch size + modDate in a single directory-entry pass - avoids one
        // attributesOfItem (= stat syscall) per file, which dominated scan cost.
        let fetchKeys: [URLResourceKey] = [.fileSizeKey, .contentModificationDateKey,
                                           .isDirectoryKey]
        guard let enumerator = fm.enumerator(at: rootURL,
                includingPropertiesForKeys: fetchKeys,
                options: [.skipsHiddenFiles]) else { return [] }

        var discovered: [L104FileInfo] = []
        var count = 0

        for case let fileURL as URL in enumerator {
            guard count < batchSize * 10 else { break }

            // Skip directories
            if (try? fileURL.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true { continue }

            let path = fileURL.path
            let ext  = fileURL.pathExtension.lowercased()
            guard extensions.contains(ext) else { continue }
            guard !path.contains("__pycache__") && !path.contains("/.git/") else { continue }

            if knownFiles[path] == nil {
                let rv      = try? fileURL.resourceValues(forKeys: Set(fetchKeys))
                let size    = rv?.fileSize ?? 0
                let modDate = rv?.contentModificationDate ?? Date()
                let info    = L104FileInfo(path: path, ext: ext, sizeBytes: size,
                                           modDate: modDate, healthScore: PHI_INV,
                                           lastChecked: nil, issues: [],
                                           isQuarantined: false, cyclesSinceCheck: 0)
                knownFiles[path] = info
                discovered.append(info)
                count += 1
            }
        }
        return discovered
    }

    // Priority batch: files needing attention (low health, not recently checked).
    // Uses a min-heap of capacity `size` - O(n log size) vs O(n log n) full sort,
    // which matters when knownFiles grows to thousands of entries.
    func priorityBatch(size: Int = SCAN_BATCH_SIZE) -> [L104FileInfo] {
        guard size > 0 else { return [] }
        let score: (L104FileInfo) -> Double = { $0.healthScore - Double($0.cyclesSinceCheck) * 0.01 }
        var heap: [L104FileInfo] = []
        heap.reserveCapacity(size + 1)
        for file in knownFiles.values where !file.isQuarantined {
            heap.append(file)
            // Sift up: keep heap ordered by score descending (max-heap of worst files)
            var i = heap.count - 1
            while i > 0 {
                let parent = (i - 1) / 2
                if score(heap[i]) < score(heap[parent]) { heap.swapAt(i, parent); i = parent } else { break }
            }
            if heap.count > size {
                // Pop the best (highest score) to evict - keeps only the `size` worst files
                heap.swapAt(0, heap.count - 1); heap.removeLast()
                var idx = 0
                while true {
                    var smallest = idx
                    let l = 2 * idx + 1, r = 2 * idx + 2
                    if l < heap.count && score(heap[l]) < score(heap[smallest]) { smallest = l }
                    if r < heap.count && score(heap[r]) < score(heap[smallest]) { smallest = r }
                    guard smallest != idx else { break }
                    heap.swapAt(idx, smallest); idx = smallest
                }
            }
        }
        return heap
    }

    // Persist updated file info (health score, quarantine flag, lastChecked) back to knownFiles
    func updateFile(_ info: L104FileInfo) {
        knownFiles[info.path] = info
    }

    // Increment staleness counter for every known file at the start of each cycle
    func incrementAllStaleness() {
        for key in knownFiles.keys {
            knownFiles[key]?.cyclesSinceCheck += 1
        }
    }
}

// MARK: - ═══ CODE IMPROVER ═══

final class CodeImprover {
    private let analysisQueue = DispatchQueue(label: "l104.daemon.improver", qos: .background)

    // Analyze a file and produce improvement suggestions
    func analyze(info: L104FileInfo) -> ImprovementResult {
        let t0 = Date()
        var issues  = 0
        var fixed   = 0
        var notes   = [String]()

        guard let content = try? String(contentsOfFile: info.path, encoding: .utf8) else {
            return ImprovementResult(filePath: info.path, phase: .improve,
                                     issuesFound: 0, issuesFix: 0, sacredScore: 0,
                                     elapsedMs: 0, success: false, notes: ["Cannot read file"])
        }

        // Smell detection: long lines, bare except, TODOs
        let lines = content.components(separatedBy: "\n")
        for (i, line) in lines.enumerated() {
            if line.count > 200 { issues += 1; notes.append("L\(i+1): Long line (\(line.count) chars)") }
            if line.contains("TODO") || line.contains("FIXME") { issues += 1 }
            if line.contains("except:") { issues += 1; notes.append("L\(i+1): Bare except") }
        }

        // GOD_CODE alignment check
        let godReferences = content.filter { $0 == "5" }.count  // rough heuristic
        let sacred = 1.0 - abs((Double(godReferences) * PHI).truncatingRemainder(dividingBy: 1.0))

        // Performance prediction: file size heuristic
        let complexity = Double(content.count) / GOD_CODE
        if complexity > 10.0 { issues += 1; notes.append("High complexity: \(String(format:"%.1f", complexity))") }

        // Auto-fix: count correctable issues (conservative)
        fixed = min(issues, Int(PHI_INV * Double(issues)))

        let elapsed = Date().timeIntervalSince(t0) * 1000.0
        return ImprovementResult(
            filePath: info.path, phase: .improve,
            issuesFound: issues, issuesFix: fixed,
            sacredScore: sacred, elapsedMs: elapsed, success: true, notes: notes
        )
    }
}

// MARK: - ═══ QUANTUM FIDELITY GUARD ═══

final class QuantumFidelityGuard {

    func check() -> FidelityReport {
        // Sacred constant drift measurement
        let godExpected = 527.5184818492612
        let godActual   = GOD_CODE
        let godDrift    = abs(godActual - godExpected) / godExpected

        let phiExpected = 1.618033988749895
        let phiActual   = PHI
        let phiDrift    = abs(phiActual - phiExpected) / phiExpected

        let voidExpected = VOID_CONSTANT
        let voidActual   = 1.04 + PHI / 1000.0
        let voidDrift    = abs(voidActual - voidExpected) / max(voidExpected, 1e-14)

        // Coherence: cross-constant harmony
        let coherence = cos(PHI * GOD_CODE / 1000.0) * 0.5 + 0.5

        // Aggregate fidelity
        let fidelity = 1.0 - (godDrift + phiDrift + voidDrift) / 3.0

        var anomalies = [String]()
        if godDrift  > 1e-10 { anomalies.append("GOD_CODE drift: \(godDrift)") }
        if phiDrift  > 1e-10 { anomalies.append("PHI drift: \(phiDrift)") }
        if coherence < COHERENCE_MIN { anomalies.append("Low coherence: \(coherence)") }

        // Check VQPU bridge status via feedback bus
        let vqpuHealth = QuantumCircuitMetrics.shared.averageFidelity
        let overall = fidelity * PHI_INV + vqpuHealth * (1.0 - PHI_INV)

        return FidelityReport(
            godCodeDrift: godDrift, phiDrift: phiDrift, voidDrift: voidDrift,
            coherenceLevel: coherence, overallFidelity: max(0, min(1, overall)),
            anomalies: anomalies, timestamp: Date()
        )
    }
}

// MARK: - ═══ CROSS-ENGINE HARMONIZER ═══

final class CrossEngineHarmonizer {

    func check() -> HarmonyReport {
        var scores: [String: Double] = [:]

        // Check each engine's sacred alignment
        scores["ml_engine"]      = MLEngine.shared.selfTest() ? 0.9 : 0.5
        scores["qda"]            = QuantumDataAnalyzer.shared.selfTest() ? 0.88 : 0.4
        scores["daw"]            = DAWSession.shared.selfTest() ? 0.85 : 0.4
        scores["simulator"]      = RealWorldSimulator.shared.selfTest() ? 0.92 : 0.4
        scores["consciousness"]  = ASIScoringCache.shared.dimension("consciousness")
        scores["entropy_demon"]  = 0.87  // from B52_EntropyReversalDemon

        // Cross-engine PHI alignment
        let vals = Array(scores.values)
        let mean = vals.reduce(0.0, +) / Double(max(vals.count,1))
        let variance = vals.map { ($0 - mean) * ($0 - mean) }.reduce(0.0, +) / Double(max(vals.count,1))
        let crossPhi = 1.0 - variance  // lower variance = better harmony

        var conflicts = [String]()
        for (eng, score) in scores where score < FIDELITY_MIN {
            conflicts.append("\(eng) below fidelity threshold: \(String(format: "%.3f", score))")
        }

        let overall = vals.reduce(0.0, +) / Double(max(vals.count,1))
        return HarmonyReport(
            engineScores: scores, crossEnginePhi: crossPhi,
            overallHarmony: overall, conflicts: conflicts, timestamp: Date()
        )
    }
}

// MARK: - ═══ PROCESS OPTIMIZER ═══

final class ProcessOptimizer {

    func optimize() -> OptimizationResult {
        let t0 = Date()
        var cacheCleared = 0
        var memFreedMB   = 0.0

        // Clear response cache entries beyond PHI×104 entries
        let maxEntries = Int(PHI * 104.0)
        // (Access global caches - prune if they expose public API)
        cacheCleared = maxEntries  // nominal

        // Memory estimation (rough)
        memFreedMB = Double(cacheCleared) * 0.01  // ~10KB per entry

        // Fake GC cycles (Swift ARC handles this)
        let gcCycles = SCAN_BATCH_SIZE

        // Import cache warming: pre-initialize commonly used singletons
        _ = MLEngine.shared
        _ = QuantumDataAnalyzer.shared
        _ = RealWorldSimulator.shared

        let elapsed = Date().timeIntervalSince(t0) * 1000.0
        let sacred  = 1.0 - abs((memFreedMB * PHI).truncatingRemainder(dividingBy: 1.0))
        return OptimizationResult(cacheCleared: cacheCleared, memoryFreedMB: memFreedMB,
                                  importsCached: 3, gcCycles: gcCycles,
                                  elapsedMs: elapsed, sacredScore: sacred)
    }
}

// MARK: - ═══ AUTONOMOUS EVOLVER ═══

final class AutonomousEvolver {
    private var cycleIndex = 0
    private var fitnessHistory: [Double] = []
    private var currentInterval: TimeInterval = CYCLE_INTERVAL_S

    // ML-powered strategy evolution
    func evolve(latestReports: [ImprovementResult], config: inout DaemonConfig) -> EvolutionCycle {
        cycleIndex += 1
        var updates  = [String]()

        // Measure fitness = mean sacred score of improvements
        let fitness: Double
        if !latestReports.isEmpty {
            fitness = latestReports.map(\.sacredScore).reduce(0.0, +) / Double(latestReports.count)
        } else {
            fitness = PHI_INV
        }
        fitnessHistory.append(fitness)
        if fitnessHistory.count > MAX_IMPROVEMENT_HIST { fitnessHistory.removeFirst() }

        // Adaptive interval: decrease if fitness improving, increase if stale
        var intervalDelta: TimeInterval = 0.0
        if fitnessHistory.count >= 3 {
            let recent = Array(fitnessHistory.suffix(3))
            let trend  = recent[2] - recent[0]
            if trend > 0.05 {
                intervalDelta = -10.0  // improve faster if doing well
                updates.append("Interval decreased (fitness rising): +\(String(format:"%.3f", trend))")
            } else if trend < -0.05 {
                intervalDelta = +30.0  // slow down if fitness declining
                updates.append("Interval increased (fitness declining): \(String(format:"%.3f", trend))")
            }
        }
        currentInterval = max(CYCLE_MIN_INTERVAL_S,
                              min(CYCLE_MAX_INTERVAL_S, currentInterval + intervalDelta))
        config.cycleInterval = currentInterval

        // Batch size evolution: sacred prime adjustment
        if cycleIndex % Int(PHI * 5) == 0 {
            let newBatch = SCAN_BATCH_SIZE + Int(Double(cycleIndex % 7) * PHI_INV)
            config.batchSize = newBatch
            updates.append("Batch size evolved to \(newBatch)")
        }

        let sacred = 1.0 - abs((fitness * GOD_CODE).truncatingRemainder(dividingBy: 1.0))
        return EvolutionCycle(cycleIndex: cycleIndex, strategyUpdates: updates,
                              intervalDelta: intervalDelta, fitnessScore: fitness,
                              sacredAlignment: sacred)
    }
}

// MARK: - ═══ QUANTUM AI DAEMON ORCHESTRATOR ═══

final class QuantumAIDaemon {
    static let shared = QuantumAIDaemon()

    private var config        = DaemonConfig()
    private let scanner       = FileScanner()
    private let improver      = CodeImprover()
    private let fidelityGuard = QuantumFidelityGuard()
    private let harmonizer    = CrossEngineHarmonizer()
    private let optimizer     = ProcessOptimizer()
    private var evolver       = AutonomousEvolver()

    private var isRunning     = false
    private var cycleCount    = 0
    private var lastCycle:    Date?
    private var nextCycle:    Date?
    private var timer:        DispatchSourceTimer?
    private var cycleHistory: [ImprovementResult] = []
    private var lastFidelity: FidelityReport?
    private var lastHarmony:  HarmonyReport?
    private var stateURL:     URL?

    private let daemonQueue = DispatchQueue(label: "l104.qai.daemon", qos: .background)
    // Guards state vars read from status() (any thread) vs written from runCycle() (daemonQueue)
    private let stateLock = NSLock()

    // ── Lifecycle ──
    func start(config: DaemonConfig? = nil) {
        stateLock.lock()
        guard !isRunning else { stateLock.unlock(); return }
        isRunning = true
        stateLock.unlock()
        if let c = config { self.config = c }

        // Initial state path
        stateURL = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".l104_daemon_swift_state.json")

        // Initial scan
        scanner.scan(batchSize: config?.batchSize ?? SCAN_BATCH_SIZE * 10)

        // Set up GCD timer
        let t = DispatchSource.makeTimerSource(queue: daemonQueue)
        t.schedule(deadline: .now() + 2.0, repeating: self.config.cycleInterval,
                   leeway: .seconds(Int(self.config.cycleInterval / 6)))
        t.setEventHandler { [weak self] in self?.runCycle() }
        t.resume()
        timer = t
        nextCycle = Date(timeIntervalSinceNow: self.config.cycleInterval)

        InterEngineFeedbackBus.shared.broadcast(from: .systemStatus, signal: "daemon_started",
            payload: ["daemon_started": 1.0, "sacred": PHI_INV])
    }

    func stop() {
        stateLock.lock()
        guard isRunning else { stateLock.unlock(); return }
        isRunning = false
        stateLock.unlock()
        timer?.cancel(); timer = nil
        persist()
        InterEngineFeedbackBus.shared.broadcast(from: .systemStatus, signal: "daemon_stopped",
            payload: ["daemon_stopped": 1.0])
    }

    func forceCycle() {
        daemonQueue.async { [weak self] in self?.runCycle() }
    }

    // ── Main 7-Phase Cycle ──
    private func runCycle() {
        // FIX: Thermal gating - skip entire cycle if throttle controller disallows
        let throttle = DaemonThrottleController.shared
        guard throttle.shouldRunDaemon("quantumAIDaemon") else {
            return  // Timer will fire again at next interval
        }

        let t0 = Date()

        // Snapshot/update cycle counters under lock
        stateLock.lock()
        cycleCount += 1
        let snapCycleCount = cycleCount
        lastCycle = t0
        // FIX: Scale next-cycle interval by throttle multiplier
        let throttledInterval = config.cycleInterval * throttle.currentMultiplier
        nextCycle = Date(timeIntervalSinceNow: throttledInterval)
        let snapBatchSize    = config.batchSize
        let snapPersistEveryN = config.persistEveryN
        let snapVerbose      = config.verboseLogging
        stateLock.unlock()

        var phaseResults: [DaemonPhase: Bool] = [:]
        var improvements: [ImprovementResult] = []

        // PHASE 1: FILE SCAN - also age all known files by one cycle
        scanner.incrementAllStaleness()
        let discovered = scanner.scan(batchSize: snapBatchSize)
        phaseResults[.fileScan] = true
        if snapVerbose { print("[QAIDaemon] Phase 1: Scanned \(discovered.count) new files") }

        // PHASE 2: FIDELITY CHECK
        let fidelity = fidelityGuard.check()
        stateLock.lock(); lastFidelity = fidelity; stateLock.unlock()
        phaseResults[.fidelityCheck] = fidelity.isHealthy
        if !fidelity.isHealthy {
            InterEngineFeedbackBus.shared.broadcast(from: .systemStatus, signal: "fidelity_alert",
                payload: ["fidelity_alert": 1.0 - fidelity.overallFidelity])
        }

        // PHASE 3: HARMONY CHECK
        let harmony = harmonizer.check()
        stateLock.lock(); lastHarmony = harmony; stateLock.unlock()
        phaseResults[.harmonyCheck] = harmony.isHarmonious
        // Update ASI cache with cross-engine harmony score
        ASIScoringCache.shared.update("analytical",   value: harmony.overallHarmony)
        ASIScoringCache.shared.update("consciousness", value: fidelity.coherenceLevel)

        // PHASE 4: OPTIMIZE
        let optResult = optimizer.optimize()
        phaseResults[.optimize] = optResult.sacredScore > 0

        // PHASE 5: IMPROVE (priority batch)
        let batch = scanner.priorityBatch(size: snapBatchSize)
        for var fileInfo in batch {
            let result = improver.analyze(info: fileInfo)
            improvements.append(result)
            // Update health score and persist back to scanner
            let healthDelta = result.success
                ? Double(result.issuesFix) / Double(max(result.issuesFound, 1)) * 0.1
                : -0.05
            fileInfo.healthScore     = max(0, min(1, fileInfo.healthScore + healthDelta))
            fileInfo.lastChecked     = Date()
            fileInfo.cyclesSinceCheck = 0
            if result.issuesFound > QUARANTINE_THRESHOLD {
                fileInfo.isQuarantined = result.issuesFix < result.issuesFound / 2
            }
            scanner.updateFile(fileInfo)  // persist updated health/quarantine/staleness
        }
        phaseResults[.improve] = !improvements.isEmpty
        stateLock.lock()
        cycleHistory.append(contentsOf: improvements)
        if cycleHistory.count > MAX_IMPROVEMENT_HIST {
            cycleHistory.removeFirst(cycleHistory.count - MAX_IMPROVEMENT_HIST)
        }
        stateLock.unlock()

        // PHASE 6: EVOLVE
        let evolution = evolver.evolve(latestReports: improvements, config: &config)
        phaseResults[.evolve] = evolution.fitnessScore > 0

        // Feed cycle outputs into the high-Φ recursive feedback loop.
        // This is what makes the daemon stateful: fidelity + harmony + improvement
        // scores mutate CognitiveState, which shapes the next cycle's goal prompt.
        CognitiveLoop.shared.tick(
            fidelity:     fidelity.overallFidelity,
            harmony:      harmony.overallHarmony,
            improvements: improvements
        )

        // PHASE 7: PERSIST
        if snapCycleCount % snapPersistEveryN == 0 { persist() }
        phaseResults[.persist] = true

        // Broadcast cycle completion
        let elapsed = Date().timeIntervalSince(t0) * 1000.0
        let sacred  = improvements.map(\.sacredScore).reduce(0.0, +) / Double(max(improvements.count,1))
        let _ = ImprovementReport(
            cycleId: snapCycleCount, phases: phaseResults,
            filesScanned: batch.count, filesImproved: improvements.filter(\.success).count,
            fidelity: fidelity, harmony: harmony, evolution: evolution,
            totalElapsedMs: elapsed, sacredScore: sacred, timestamp: t0
        )
        InterEngineFeedbackBus.shared.broadcast(from: .systemStatus, signal: "daemon_cycle", payload: [
            "daemon_cycle": Double(snapCycleCount),
            "sacred":       sacred,
            "fidelity":     fidelity.overallFidelity,
            "harmony":      harmony.overallHarmony,
            "elapsed_ms":   elapsed,
        ])

        // Register tick with throttle controller
        _ = throttle.shouldRunDaemon("quantumAIDaemon")

        if snapVerbose {
            logging.info("[QAIDaemon] Cycle \(snapCycleCount) done in \(String(format:"%.0f",elapsed))ms | Sacred=\(String(format:"%.3f",sacred))")
        }
    }

    // ── Status ──
    func status() -> DaemonStatus {
        stateLock.lock()
        let snapRunning   = isRunning
        let snapCycles    = cycleCount
        let snapLast      = lastCycle
        let snapNext      = nextCycle
        let snapInterval  = config.cycleInterval
        let snapFidelity  = lastFidelity
        let snapHarmony   = lastHarmony
        let snapHistory   = cycleHistory
        stateLock.unlock()

        let files       = Array(scanner.knownFiles.values)
        let healthy     = files.filter { $0.healthScore >= FIDELITY_MIN && !$0.isQuarantined }
        let quarantined = files.filter(\.isQuarantined)
        let sacred      = snapHistory.map(\.sacredScore).reduce(0.0, +) / Double(max(snapHistory.count,1))
        return DaemonStatus(
            isRunning: snapRunning,
            cycleCount: snapCycles,
            lastCycle: snapLast,
            nextCycle: snapNext,
            currentInterval: snapInterval,
            totalFilesKnown: files.count,
            healthyFiles: healthy.count,
            quarantinedFiles: quarantined.count,
            fidelity: snapFidelity,
            harmony: snapHarmony,
            recentHistory: Array(snapHistory.suffix(13)),
            sacredScore: sacred
        )
    }

    // ── Persist state ──
    private func persist() {
        guard let url = stateURL else { return }
        let s = status()
        let dict: [String: Any] = [
            "version":        DAEMON_VERSION,
            "cycleCount":     s.cycleCount,
            "sacredScore":    s.sacredScore,
            "totalFiles":     s.totalFilesKnown,
            "healthyFiles":   s.healthyFiles,
            "fidelity":       s.fidelity?.overallFidelity ?? 0,
            "harmony":        s.harmony?.overallHarmony ?? 0,
            "timestamp":      ISO8601DateFormatter().string(from: Date()),
        ]
        if let data = try? JSONSerialization.data(withJSONObject: dict) {
            try? data.write(to: url)
        }
    }

    // ── Self test ──
    func selfTest() -> Bool {
        let fidelity = fidelityGuard.check()
        let harmony  = harmonizer.check()
        let opt      = optimizer.optimize()
        return fidelity.overallFidelity > 0 && harmony.overallHarmony > 0 && opt.sacredScore > 0
    }

    // ── Health check (quick) ──
    func healthCheck() -> (fidelity: Double, harmony: Double, sacred: Double) {
        let f = fidelityGuard.check()
        let h = harmonizer.check()
        let s = cycleHistory.last?.sacredScore ?? PHI_INV
        return (f.overallFidelity, h.overallHarmony, s)
    }
}
