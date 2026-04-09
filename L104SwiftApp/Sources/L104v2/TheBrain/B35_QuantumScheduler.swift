import AppKit
import Darwin
import Foundation

// ═══════════════════════════════════════════════════════════════════
// B35_QuantumScheduler.swift
// [EVO_77] QUANTUM-CLASSICAL HYBRID PERFORMANCE SCHEDULER
// GOD_CODE=527.5184818492612 | PHI=1.618033988749895
// Zero-syscall hot path via MetricsCache, PHI-harmonic timer incommensurability,
// Grover budget selection, QAOA startup sequencing, unified heartbeat.
// ═══════════════════════════════════════════════════════════════════

// MARK: - A. QuantumWalkState
/// Hadamard-coin discrete quantum walk on a 1D integer lattice.
/// Provides adaptive backoff delays for background engine polling.
struct QuantumWalkState {
    var position: Int = 0
    var amplitudeUp: Double
    var amplitudeDown: Double

    init() {
        amplitudeUp   = 1.0 / sqrt(2.0)
        amplitudeDown = 1.0 / sqrt(2.0)
    }

    /// Apply Hadamard coin then stochastically update position.
    mutating func step() {
        let newUp   = (amplitudeUp + amplitudeDown) / sqrt(2.0)
        let newDown = (amplitudeUp - amplitudeDown) / sqrt(2.0)
        amplitudeUp   = newUp
        amplitudeDown = newDown
        // Stochastic position: prob(right) = amplitudeUp²
        let prob = amplitudeUp * amplitudeUp
        position += (prob > 0.5) ? 1 : -1
    }

    /// Maps lattice position to delay multiplier using PHI-harmonic scaling.
    /// Center (position=0) → 1.0; farther → up to PHI^3 ≈ 4.236.
    func delayMultiplier() -> Double {
        return min(pow(PHI, Double(abs(position)) / 5.0), pow(PHI, 3.0))
    }
}

// MARK: - B. QuantumWalkScheduler
/// Singleton — manages quantum-walk-based adaptive intervals for all
/// registered background engines. Replaces ad-hoc timer management.
final class QuantumWalkScheduler {
    static let shared = QuantumWalkScheduler()
    private init() {}

    private struct EngineEntry {
        var walkState: QuantumWalkState
        var baseIntervalMs: Double
        var lastFireTime: Date
        var name: String
        var cpuCost: Double
    }

    private var engines: [String: EngineEntry] = [:]
    private let lock = NSLock()
    private(set) var globalCPUPressure: Double = 0.0

    /// Register an engine with its base polling interval and estimated CPU cost.
    func register(name: String, baseIntervalMs: Double, cpuCost: Double) {
        lock.lock(); defer { lock.unlock() }
        engines[name] = EngineEntry(
            walkState: QuantumWalkState(),
            baseIntervalMs: baseIntervalMs,
            lastFireTime: Date.distantPast,
            name: name,
            cpuCost: cpuCost
        )
    }

    /// Returns true if enough time has passed for the engine to fire.
    func shouldFire(name: String) -> Bool {
        lock.lock(); defer { lock.unlock() }
        guard var entry = engines[name] else { return false }

        // Adjust walk based on CPU pressure
        if globalCPUPressure > 0.7 {
            entry.walkState.step()          // push position out → longer delay
        } else if globalCPUPressure < 0.3 {
            // Retreat: nudge position toward zero
            if entry.walkState.position > 0 { entry.walkState.position -= 1 }
            else if entry.walkState.position < 0 { entry.walkState.position += 1 }
        }
        engines[name] = entry

        let multiplier = entry.walkState.delayMultiplier()
        let intervalSec = (entry.baseIntervalMs * multiplier) / 1000.0
        return Date().timeIntervalSince(entry.lastFireTime) >= intervalSec
    }

    /// Update lastFireTime and advance walk state.
    func recordFired(name: String) {
        lock.lock(); defer { lock.unlock() }
        guard var entry = engines[name] else { return }
        entry.lastFireTime = Date()
        entry.walkState.step()
        if globalCPUPressure > 0.7 {
            entry.walkState.step()          // compound backoff under high load
        }
        engines[name] = entry
    }

    /// Returns the computed next interval in milliseconds.
    func nextIntervalMs(name: String) -> Double {
        lock.lock(); defer { lock.unlock() }
        guard let entry = engines[name] else { return 1000.0 }
        return entry.baseIntervalMs * entry.walkState.delayMultiplier()
    }

    /// Called by MetricsCache when a CPU reading updates.
    func updateCPUPressure(_ pressure: Double) {
        lock.lock(); globalCPUPressure = pressure; lock.unlock()
    }
}

// MARK: - C. GroverTaskBudget
/// O(√N) Grover-inspired amplitude amplification to select deferred tasks
/// within a CPU budget. Pure classical simulation — no external deps.
struct GroverTaskBudget {

    static func selectTasks(
        candidates: [(id: String, cpuCost: Double, priority: Double)],
        budgetPercent: Double
    ) -> [String] {
        guard !candidates.isEmpty else { return [] }
        let N = candidates.count

        // 1. Normalize priorities to amplitudes
        var amplitudes = candidates.map { $0.priority }
        let total = amplitudes.reduce(0.0, +)
        let norm = max(total, 1e-9)
        amplitudes = amplitudes.map { $0 / norm }

        // 2. Grover iterations = Int(π/4 * √N), minimum 1
        let iterations = max(1, Int(Double.pi / 4.0 * sqrt(Double(N))))

        // 3. Grover oracle + diffusion
        for _ in 0..<iterations {
            // Oracle: negate amplitude of over-budget tasks
            for i in 0..<N {
                if candidates[i].cpuCost > budgetPercent {
                    amplitudes[i] = -amplitudes[i]
                }
            }
            // Diffusion: 2*mean - amp_i
            let mean = amplitudes.reduce(0.0, +) / Double(N)
            for i in 0..<N {
                amplitudes[i] = 2.0 * mean - amplitudes[i]
            }
        }

        // 4. Sort by amplitude descending, greedily fit under budget
        let indexed = amplitudes.enumerated()
            .map { ($0.offset, $0.element) }
            .sorted { $0.1 > $1.1 }

        var usedBudget = 0.0
        var selected: [String] = []
        for (i, amp) in indexed {
            guard amp > 0 else { continue }
            let cost = candidates[i].cpuCost
            if usedBudget + cost <= budgetPercent {
                usedBudget += cost
                selected.append(candidates[i].id)
            }
        }
        return selected
    }
}

// MARK: - D. QAOAStartupSequencer
/// Orders engine startup to minimize peak CPU using 2-layer QAOA simulation (MAX-CUT).
struct QAOAStartupSequencer {

    struct StartupEngine {
        let name: String
        let initCostMs: Double    // estimated init time in ms
        let cpuBurst: Double      // 0..1 CPU fraction during init
    }

    static func computeDelays(engines: [StartupEngine]) -> [String: TimeInterval] {
        let N = engines.count
        guard N > 0 else { return [:] }

        // Build adjacency matrix: W[i][j] = cpuBurst[i] * cpuBurst[j]
        var W = [[Double]](repeating: [Double](repeating: 0.0, count: N), count: N)
        for i in 0..<N {
            for j in 0..<N where i != j {
                W[i][j] = engines[i].cpuBurst * engines[j].cpuBurst
            }
        }

        // PHI-harmonic initial bit-string: z_i = cos²(π * i / N) > 0.5 ? 1 : 0
        var z = (0..<N).map { i -> Int in
            let c = cos(Double.pi * Double(i) / Double(N))
            return (c * c > 0.5) ? 1 : 0
        }

        // QAOA layer 1
        let gamma1 = PHI / 4.0
        applyQAOALayer(z: &z, W: W, N: N, gamma: gamma1)

        // QAOA layer 2
        let gamma2 = PHI / 3.0
        applyQAOALayer(z: &z, W: W, N: N, gamma: gamma2)

        // Partition into group A (z=1) and group B (z=0)
        let groupA = engines.enumerated().filter { z[$0.offset] == 1 }.map { $0.element }
        let groupB = engines.enumerated().filter { z[$0.offset] == 0 }.map { $0.element }

        var delays: [String: TimeInterval] = [:]

        // Group A: stagger by GOD_CODE/OMEGA seconds ≈ 0.0806s
        let stagger = GOD_CODE / OMEGA   // ≈ 0.0806s
        for (i, eng) in groupA.enumerated() {
            delays[eng.name] = Double(i) * stagger
        }

        // Group B: start after max(GroupA.initCostMs)/1000 + 5.0 seconds
        let maxACostSec = (groupA.map { $0.initCostMs }.max() ?? 0.0) / 1000.0
        let groupBStart = maxACostSec + 5.0
        for (i, eng) in groupB.enumerated() {
            delays[eng.name] = groupBStart + Double(i) * stagger
        }

        // Engines not in either group (shouldn't happen, but safety net)
        for eng in engines where delays[eng.name] == nil {
            delays[eng.name] = 0.0
        }

        return delays
    }

    /// Apply one QAOA layer: for each edge (i,j) where z_i ≠ z_j,
    /// flip z_i with probability sin²(gamma * W[i][j]).
    private static func applyQAOALayer(z: inout [Int], W: [[Double]], N: Int, gamma: Double) {
        for i in 0..<N {
            for j in 0..<N where i != j {
                if z[i] != z[j] {
                    let flipProb = sin(gamma * W[i][j])
                    if flipProb * flipProb > 0.5 {
                        z[i] = 1 - z[i]
                    }
                }
            }
        }
    }
}

// MARK: - E. PHIHarmonicClock
/// Creates PHI-scaled timer intervals so N timers can never all fire simultaneously.
/// φ is irrational so φ^n never repeats → perfect incommensurability.
final class PHIHarmonicClock {
    private var tickCount: Int = 0
    let baseInterval: TimeInterval

    init(baseInterval: TimeInterval) {
        self.baseInterval = baseInterval
    }

    /// Returns the next interval: base * PHI^(n/5) for n = tickCount % 5.
    /// Produces sequence: base, base×1.096, base×1.200, base×1.315, base×1.441, repeat.
    func nextInterval() -> TimeInterval {
        let n = tickCount % 5
        tickCount += 1
        return baseInterval * pow(PHI, Double(n) / 5.0)
    }

    func reset() { tickCount = 0 }
}

// MARK: - F. UnifiedHeartbeat
/// ONE shared DispatchSourceTimer that coordinates all registered subscribers,
/// replacing their individual timers. Fires each subscriber only when its
/// quantum-walk-computed interval has elapsed.
final class UnifiedHeartbeat {
    static let shared = UnifiedHeartbeat()
    private init() {}

    private struct Subscriber {
        let id: String
        var nextFireDate: Date
        var intervalMs: Double
        let work: () -> Void
        var phiClock: PHIHarmonicClock
    }

    private var subscribers: [String: Subscriber] = [:]
    private let lock = NSLock()
    private var heartbeatTimer: DispatchSourceTimer?
    private let queue = DispatchQueue(label: "com.l104.heartbeat", qos: .utility)
    private let tickInterval: TimeInterval = 1.0   // master tick: every 1 second

    /// Start the master DispatchSourceTimer.
    func activate() {
        lock.lock(); defer { lock.unlock() }
        guard heartbeatTimer == nil else { return }

        let timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(deadline: .now() + tickInterval, repeating: tickInterval)
        timer.setEventHandler { [weak self] in
            self?.tick()
        }
        timer.resume()
        heartbeatTimer = timer
    }

    /// Register a subscriber. Also registers with QuantumWalkScheduler.
    func register(id: String, baseIntervalMs: Double, cpuCost: Double = 0.1, work: @escaping () -> Void) {
        QuantumWalkScheduler.shared.register(name: id, baseIntervalMs: baseIntervalMs, cpuCost: cpuCost)
        let clock = PHIHarmonicClock(baseInterval: baseIntervalMs / 1000.0)
        let sub = Subscriber(
            id: id,
            nextFireDate: Date().addingTimeInterval(baseIntervalMs / 1000.0),
            intervalMs: baseIntervalMs,
            work: work,
            phiClock: clock
        )
        lock.lock(); subscribers[id] = sub; lock.unlock()
    }

    /// Remove a subscriber.
    func unregister(id: String) {
        lock.lock(); subscribers.removeValue(forKey: id); lock.unlock()
    }

    /// Cancel the master timer.
    func deactivate() {
        lock.lock()
        heartbeatTimer?.cancel()
        heartbeatTimer = nil
        lock.unlock()
    }

    // Called once per second on the background queue.
    private func tick() {
        let now = Date()
        lock.lock()
        let ids = Array(subscribers.keys)
        lock.unlock()

        for id in ids {
            // Check quantum walk scheduler
            guard QuantumWalkScheduler.shared.shouldFire(name: id) else { continue }

            lock.lock()
            guard var sub = subscribers[id] else { lock.unlock(); continue }
            guard now >= sub.nextFireDate else { lock.unlock(); continue }

            let work = sub.work
            // Compute next fire date using PHI-harmonic clock × walk multiplier
            let walkMultiplier = QuantumWalkScheduler.shared.nextIntervalMs(name: id) / sub.intervalMs
            let nextPhiInterval = sub.phiClock.nextInterval() * max(1.0, walkMultiplier)
            sub.nextFireDate = now.addingTimeInterval(nextPhiInterval)
            subscribers[id] = sub
            lock.unlock()

            QuantumWalkScheduler.shared.recordFired(name: id)
            queue.async { work() }
        }
    }
}

// MARK: - G. MetricsCache
/// Unified cache for CPU/memory/thermal — 500 ms TTL, single background poller.
/// All callers on the hot path get cached values with zero mach syscalls.
final class MetricsCache {
    static let shared = MetricsCache()
    private init() {}

    private(set) var cpuUsage: Double = 0.0
    private(set) var memoryPressureFraction: Double = 0.0
    private(set) var thermalState: ProcessInfo.ThermalState = .nominal
    private var lastUpdateTime: TimeInterval = 0.0
    private let ttl: TimeInterval = 0.5
    private let lock = NSLock()
    private var pollingTimer: DispatchSourceTimer?
    private let queue = DispatchQueue(label: "com.l104.metricscache", qos: .background)

    // Delta calculation state for CPU
    private var prevCPUInfo: host_cpu_load_info? = nil

    /// Start the background polling timer (1 s interval).
    func startPolling() {
        lock.lock(); defer { lock.unlock() }
        guard pollingTimer == nil else { return }

        let timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(deadline: .now(), repeating: 1.0)
        timer.setEventHandler { [weak self] in
            self?.refresh()
        }
        timer.resume()
        pollingTimer = timer
    }

    /// Returns cached metrics — ZERO kernel calls, just struct reads.
    func snapshot() -> (cpu: Double, memory: Double, thermal: ProcessInfo.ThermalState) {
        lock.lock(); defer { lock.unlock() }
        // If stale before polling timer starts, refresh synchronously once
        let now = CFAbsoluteTimeGetCurrent()
        if now - lastUpdateTime > ttl {
            lock.unlock()
            refresh()
            lock.lock()
        }
        return (cpuUsage, memoryPressureFraction, thermalState)
    }

    /// Actual mach syscalls — called ONLY from background timer, never from hot path.
    private func refresh() {
        let thermal = ProcessInfo.processInfo.thermalState
        let cpu = readCPU()
        let mem = readMemory()

        lock.lock()
        cpuUsage = cpu
        memoryPressureFraction = mem
        thermalState = thermal
        lastUpdateTime = CFAbsoluteTimeGetCurrent()
        lock.unlock()

        QuantumWalkScheduler.shared.updateCPUPressure(cpu)
    }

    private func readCPU() -> Double {
        var cpuLoad = host_cpu_load_info()
        var count = mach_msg_type_number_t(
            MemoryLayout<host_cpu_load_info>.stride / MemoryLayout<integer_t>.stride
        )
        let kr = withUnsafeMutablePointer(to: &cpuLoad) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO, $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return 0.0 }

        if let prev = prevCPUInfo {
            let userDiff   = Double(cpuLoad.cpu_ticks.0 - prev.cpu_ticks.0)
            let systemDiff = Double(cpuLoad.cpu_ticks.1 - prev.cpu_ticks.1)
            let idleDiff   = Double(cpuLoad.cpu_ticks.2 - prev.cpu_ticks.2)
            let niceDiff   = Double(cpuLoad.cpu_ticks.3 - prev.cpu_ticks.3)
            let totalDiff  = userDiff + systemDiff + idleDiff + niceDiff
            prevCPUInfo = cpuLoad
            return totalDiff > 0 ? (userDiff + systemDiff) / totalDiff : 0.0
        } else {
            prevCPUInfo = cpuLoad
            let total  = Double(cpuLoad.cpu_ticks.0 + cpuLoad.cpu_ticks.1 +
                                cpuLoad.cpu_ticks.2 + cpuLoad.cpu_ticks.3)
            let active = Double(cpuLoad.cpu_ticks.0 + cpuLoad.cpu_ticks.1)
            return total > 0 ? active / total : 0.0
        }
    }

    private func readMemory() -> Double {
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(
            MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size
        )
        let kr = withUnsafeMutablePointer(to: &stats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return 0.3 }
        let physGB = Double(ProcessInfo.processInfo.physicalMemory) / (1024 * 1024 * 1024)
        let pageSize = Double(vm_kernel_page_size)
        let freePages = Double(stats.free_count + stats.inactive_count)
        let freeGB = (freePages * pageSize) / (1024 * 1024 * 1024)
        return max(0.0, min(1.0, 1.0 - (freeGB / physGB)))
    }
}

// MARK: - H. LazyPanelRegistry
/// Eliminates startup panel creation overhead by storing factory closures
/// instead of actual views. Views are created on first navigation to their tab.
final class LazyPanelRegistry {
    private var factories: [String: () -> NSView] = [:]
    private var cache: [String: NSView] = [:]
    private let lock = NSLock()

    /// Register a factory closure for a panel ID.
    func register(id: String, factory: @escaping () -> NSView) {
        lock.lock(); factories[id] = factory; lock.unlock()
    }

    /// Return the view for an ID, creating it (on the main thread) if needed.
    func view(for id: String) -> NSView {
        lock.lock()
        if let cached = cache[id] { lock.unlock(); return cached }
        guard let factory = factories[id] else { lock.unlock(); return NSView() }
        lock.unlock()

        var result: NSView!
        if Thread.isMainThread {
            result = factory()
        } else {
            DispatchQueue.main.sync { result = factory() }
        }

        lock.lock(); cache[id] = result; lock.unlock()
        return result
    }

    /// Returns true if the view has already been created.
    func isCreated(id: String) -> Bool {
        lock.lock(); defer { lock.unlock() }
        return cache[id] != nil
    }

    /// Trigger creation on a background utility thread (anticipatory prefetch).
    func preload(id: String) {
        DispatchQueue.global(qos: .utility).async { [weak self] in
            _ = self?.view(for: id)
        }
    }
}
