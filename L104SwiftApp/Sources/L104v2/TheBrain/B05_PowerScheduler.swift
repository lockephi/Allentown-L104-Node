import Accelerate
import AppKit
import Foundation
import NaturalLanguage
import simd

// ═══════════════════════════════════════════════════════════════════
// ⚡ POWER-AWARE COMPUTE SCHEDULER
// ═══════════════════════════════════════════════════════════════════

/// Schedules compute tasks based on power and thermal state
class PowerAwareScheduler {
    static let shared = PowerAwareScheduler()

    private let performanceQueue = DispatchQueue(label: "asi.performance", qos: .userInteractive, attributes: .concurrent)
    private let balancedQueue = DispatchQueue(label: "asi.balanced", qos: .userInitiated, attributes: .concurrent)
    private let efficiencyQueue = DispatchQueue(label: "asi.efficiency", qos: .utility, attributes: .concurrent)

    // EVO_58: Thread-safe counters - protected by statsLock
    private let statsLock = NSLock()
    private var _tasksScheduled: Int = 0
    private var _tasksCompleted: Int = 0
    private var _tasksFailed: Int = 0

    var tasksScheduled: Int { statsLock.lock(); defer { statsLock.unlock() }; return _tasksScheduled }
    var tasksCompleted: Int { statsLock.lock(); defer { statsLock.unlock() }; return _tasksCompleted }
    var tasksFailed: Int { statsLock.lock(); defer { statsLock.unlock() }; return _tasksFailed }

    private func incrementScheduled() { statsLock.lock(); _tasksScheduled += 1; statsLock.unlock() }
    private func incrementCompleted() { statsLock.lock(); _tasksCompleted += 1; statsLock.unlock() }
    private func incrementFailed() { statsLock.lock(); _tasksFailed += 1; statsLock.unlock() }
    private func incrementScheduled(by n: Int) { statsLock.lock(); _tasksScheduled += n; statsLock.unlock() }
    private func incrementCompleted(by n: Int) { statsLock.lock(); _tasksCompleted += n; statsLock.unlock() }

    /// Get optimal queue for current power mode — zero mach syscalls via MetricsCache TTL
    var optimalQueue: DispatchQueue {
        let (cpu, _, thermal) = MetricsCache.shared.snapshot()
        // Derive power mode from cached metrics — no mach syscall on the hot path
        switch thermal {
        case .nominal:
            return cpu < 0.5 ? performanceQueue : balancedQueue
        case .fair:
            return balancedQueue
        case .serious, .critical:
            return efficiencyQueue
        @unknown default:
            return balancedQueue
        }
    }

    /// Schedule compute task with power awareness
    func schedule(_ work: @escaping () -> Void) {
        incrementScheduled()
        optimalQueue.async { [weak self] in
            work()
            self?.incrementCompleted()
        }
    }

    /// Schedule with completion handler
    func schedule<T>(_ work: @escaping () -> T, completion: @escaping (T) -> Void) {
        incrementScheduled()
        optimalQueue.async { [weak self] in
            let result = work()
            self?.incrementCompleted()
            DispatchQueue.main.async {
                completion(result)
            }
        }
    }

    /// Parallel map with power-aware chunking
    func parallelMap<T, R>(_ array: [T], transform: @escaping (T) -> R) -> [R] {
        guard !array.isEmpty else { return [] }
        let count = array.count
        var results = [R?](repeating: nil, count: count)
        let group = DispatchGroup()
        let resultsLock = NSLock()

        incrementScheduled(by: count)

        for (index, element) in array.enumerated() {
            group.enter()
            optimalQueue.async { [weak self] in
                let value = transform(element)
                resultsLock.lock()
                results[index] = value
                resultsLock.unlock()
                self?.incrementCompleted()
                group.leave()
            }
        }

        group.wait()
        return results.compactMap { $0 }
    }

    /// EVO_58: Scheduler statistics
    func schedulerStats() -> [String: Any] {
        statsLock.lock()
        let s = _tasksScheduled
        let c = _tasksCompleted
        let f = _tasksFailed
        statsLock.unlock()
        return [
            "tasks_scheduled": s,
            "tasks_completed": c,
            "tasks_failed": f,
            "tasks_pending": s - c - f,
            "completion_rate": s > 0 ? Double(c) / Double(s) : 0.0
        ]
    }
}
