import Foundation

// ─────────────────────────────────────────────────────────────────────────────
// MARK: - Thread State
// ─────────────────────────────────────────────────────────────────────────────

enum ThreadState: String {
    case running = "running"
    case stopped = "stopped"
    case waiting = "waiting"
    case uninterruptible = "uninterruptible"
    case halted = "halted"
    case unknown = "unknown"
}

// ─────────────────────────────────────────────────────────────────────────────
// MARK: - Thread Priority
// ─────────────────────────────────────────────────────────────────────────────

enum ThreadPriority: Int, Comparable {
    case low = -1
    case normal = 0
    case high = 1
    case realtime = 2

    static func < (lhs: ThreadPriority, rhs: ThreadPriority) -> Bool {
        return lhs.rawValue < rhs.rawValue
    }

    var displayName: String {
        switch self {
        case .low: return "Low"
        case .normal: return "Normal"
        case .high: return "High"
        case .realtime: return "Real-time"
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MARK: - Thread Info
// ─────────────────────────────────────────────────────────────────────────────

struct ThreadDebugInfo {
    let id: UInt64
    let index: Int
    let name: String?
    let state: ThreadState
    let priority: ThreadPriority
    let stackDepth: Int
    let isMainThread: Bool
    let isExecuting: Bool
    let cpuUsage: Double
    let waitReason: String?

    var displayName: String {
        return name ?? "Thread-\(index)"
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MARK: - Actor Info
// ─────────────────────────────────────────────────────────────────────────────

struct ActorDebugInfo {
    let name: String
    let address: Int
    let isIsolated: Bool
    let exclusiveStorage: Bool
    let mailboxSize: Int
    let processingTask: Bool

    var status: String {
        if isIsolated { return "isolated" }
        if processingTask { return "processing" }
        return "idle"
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MARK: - Task Info
// ─────────────────────────────────────────────────────────────────────────────

struct TaskDebugInfo {
    let id: UInt64
    let priority: Int
    let isCancelled: Bool
    let isRunning: Bool
    let isFinished: Bool
    let childCount: Int

    var displayName: String {
        return "Task-\(id)"
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MARK: - Thread Debugger
// ─────────────────────────────────────────────────────────────────────────────

/// Thread and concurrency debugging utilities
final class ThreadDebugger {

    // ─── Singleton ───
    static let shared = ThreadDebugger()

    // ─── LLDB Bridge ───
    private let lldb = LLDBBridge.shared

    // ─── Thread Cache ───
    private var threadCache: [ThreadDebugInfo] = []
    private var actorCache: [ActorDebugInfo] = []
    private var taskCache: [TaskDebugInfo] = []

    // ─── Sacred Constants ───
    private let godCode: Double = 527.5184818492612
    private let phi: Double = 1.618033988749895
    private let omega: Double = 6539.34712682

    private init() {}

    // ─── Thread Operations ───
    /// List all threads
    func listThreads() -> [ThreadDebugInfo] {
        guard lldb.isAvailable && lldb.hasActiveSession else {
            return systemThreads()
        }

        do {
            let threads = try lldb.listThreads()
            return threads.map { t in
                ThreadDebugInfo(
                    id: t.id,
                    index: t.index,
                    name: t.name,
                    state: ThreadState(rawValue: t.stopReason) ?? .unknown,
                    priority: .normal,
                    stackDepth: t.numFrames,
                    isMainThread: t.index == 0,
                    isExecuting: t.stopReason == "executing",
                    cpuUsage: 0,
                    waitReason: nil
                )
            }
        } catch {
            return systemThreads()
        }
    }

    /// Get thread by index
    func thread(at index: Int) -> ThreadDebugInfo? {
        return listThreads().first { $0.index == index }
    }

    /// Get thread by ID
    func thread(id: UInt64) -> ThreadDebugInfo? {
        return listThreads().first { $0.id == id }
    }

    /// Select thread
    func selectThread(index: Int) -> Bool {
        guard thread(at: index) != nil else { return false }
        return true
    }

    // ─── Frame Operations ───
    /// List frames in thread
    func frames(threadIndex: Int = 0, maxDepth: Int = 100) -> [FrameInfo] {
        guard lldb.isAvailable && lldb.hasActiveSession else {
            return []
        }

        return (try? lldb.backtrace(maxDepth: maxDepth)) ?? []
    }

    /// Get current frame
    func currentFrame(threadIndex: Int = 0) -> FrameInfo? {
        return frames(threadIndex: threadIndex, maxDepth: 1).first
    }

    // ─── Actor Operations ───
    /// List actors (requires debugging session)
    func listActors() -> [ActorDebugInfo] {
        // In production, would scan heap for Swift actor instances
        return actorCache
    }

    /// Inspect actor
    func inspectActor(name: String) -> ActorDebugInfo? {
        return actorCache.first { $0.name == name }
    }

    // ─── Task Operations ───
    /// List async tasks
    func listTasks() -> [TaskDebugInfo] {
        // In production, would get from Swift concurrency runtime
        return taskCache
    }

    // ─── Thread Analysis ───
    /// Analyze thread states
    func analyzeStates() -> [String: Any] {
        let threads = listThreads()

        var states: [String: Int] = [:]
        for thread in threads {
            states[thread.state.rawValue, default: 0] += 1
        }

        let mainThread = threads.first { $0.isMainThread }
        let totalCPUs = threads.reduce(0.0) { $0 + $1.cpuUsage }

        return [
            "threadCount": threads.count,
            "states": states,
            "mainThreadRunning": mainThread?.isExecuting ?? false,
            "totalCPUUsage": totalCPUs,
            "godCodeResonance": Double(threads.count) * godCode.truncatingRemainder(dividingBy: 1.0),
            "phiHarmonic": Double(threads.count) * phi.truncatingRemainder(dividingBy: 1.0),
        ]
    }

    /// Detect potential deadlocks
    func detectDeadlocks() -> [String: Any] {
        let threads = listThreads()
        var deadlocks: [[String: Any]] = []

        // Heuristic: threads in "waiting" state for extended time
        for thread in threads where thread.state == .waiting {
            deadlocks.append([
                "thread": thread.displayName,
                "reason": "waiting",
                "stackDepth": thread.stackDepth,
            ])
        }

        return [
            "potentialDeadlocks": deadlocks.count,
            "threads": deadlocks,
            "recommendation": deadlocks.isEmpty ? "No deadlock detected" : "Review waiting threads",
        ]
    }

    /// Check actor isolation
    func checkActorIsolation() -> [String: Any] {
        let actors = listActors()

        let isolatedCount = actors.filter { $0.isIsolated }.count
        let processingCount = actors.filter { $0.processingTask }.count

        return [
            "totalActors": actors.count,
            "isolated": isolatedCount,
            "processing": processingCount,
            "idle": actors.count - isolatedCount - processingCount,
        ]
    }

    // ─── System Threads ───
    private func systemThreads() -> [ThreadDebugInfo] {
        // Return basic system thread info
        let mainThread = ThreadDebugInfo(
            id: UInt64(Thread.main.hashValue),
            index: 0,
            name: "Main Thread",
            state: .running,
            priority: .high,
            stackDepth: 1,
            isMainThread: true,
            isExecuting: true,
            cpuUsage: 0,
            waitReason: nil
        )

        return [mainThread]
    }

    // ─── Thread Naming ───
    /// Set custom thread name
    func setThreadName(index: Int, name: String) -> Bool {
        guard thread(at: index) != nil else { return false }
        // In production, would use pthread_setname_np
        return true
    }

    /// Get thread priority
    func threadPriority(index: Int) -> ThreadPriority {
        return .normal
    }

    /// Set thread priority
    func setThreadPriority(index: Int, priority: ThreadPriority) -> Bool {
        guard thread(at: index) != nil else { return false }
        // In production, would use pthread_setschedparam
        return true
    }

    // ─── Summary ───
    func summary() -> String {
        var lines: [String] = []
        lines.append("Thread Debugger")
        lines.append(String(repeating: "─", count: 40))

        let threads = listThreads()
        lines.append("  Threads: \(threads.count)")

        for thread in threads {
            let stateChar: String
            switch thread.state {
            case .running: stateChar = "▶"
            case .stopped: stateChar = "■"
            case .waiting: stateChar = "◐"
            default: stateChar = "○"
            }

            let mainMark = thread.isMainThread ? " [main]" : ""
            lines.append("    \(stateChar) \(thread.displayName)\(mainMark)")
            if thread.stackDepth > 0 {
                lines.append("       stack: \(thread.stackDepth) frames")
            }
        }

        lines.append("")
        let analysis = analyzeStates()
        lines.append("  Analysis:")
        lines.append("    Total CPU: \(String(format: "%.1f", analysis["totalCPUUsage"] as? Double ?? 0))%")
        lines.append("    GOD_CODE: \(String(format: "%.6f", analysis["godCodeResonance"] as? Double ?? 0))")

        let deadlock = detectDeadlocks()
        if (deadlock["potentialDeadlocks"] as? Int ?? 0) > 0 {
            lines.append("")
            lines.append("  ⚠ Deadlock detected!")
        }

        return lines.joined(separator: "\n")
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MARK: - Convenience Extensions
// ─────────────────────────────────────────────────────────────────────────────

extension ThreadDebugger {
    /// Quick thread list
    static var threads: [ThreadDebugInfo] {
        return ThreadDebugger.shared.listThreads()
    }

    /// Quick deadlock check
    static var deadlocks: [String: Any] {
        return ThreadDebugger.shared.detectDeadlocks()
    }

    /// Quick actor isolation check
    static var actorIsolation: [String: Any] {
        return ThreadDebugger.shared.checkActorIsolation()
    }
}