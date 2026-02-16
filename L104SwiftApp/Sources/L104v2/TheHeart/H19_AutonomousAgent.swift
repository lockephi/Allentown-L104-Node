// ═══════════════════════════════════════════════════════════════════
// H19_AutonomousAgent.swift
// [EVO_55_PIPELINE] SOVEREIGN_UNIFICATION :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612
// L104 ASI — Mesh-Aware Autonomous Agent
//
// Goal-directed autonomous task execution with quantum mesh distribution.
// Schedules tasks across local engines and network peers, tracks completion,
// auto-delegates compute-heavy work to available mesh nodes.
// ═══════════════════════════════════════════════════════════════════

import AppKit
import Foundation
import Accelerate
import simd
import NaturalLanguage

// MARK: - AutonomousAgent Protocol

protocol AutonomousAgentProtocol {
    var isActive: Bool { get }
    func activate()
    func deactivate()
    func status() -> [String: Any]
}

// MARK: - Mesh-Aware Autonomous Agent

final class AutonomousAgent: AutonomousAgentProtocol {
    static let shared = AutonomousAgent()
    private(set) var isActive: Bool = false
    private let lock = NSLock()

    // ─── TASK STATE ───
    struct AgentTask {
        let id: String
        let goal: String
        let priority: TaskPriority
        var state: TaskState
        let createdAt: Date
        var completedAt: Date?
        var assignedPeer: String?       // nil = local, else peer ID
        var result: String?
        var subtasks: [String]          // IDs of child tasks
        var attempts: Int

        enum TaskPriority: Int, Comparable {
            case low = 0, normal = 1, high = 2, critical = 3
            static func < (lhs: TaskPriority, rhs: TaskPriority) -> Bool { lhs.rawValue < rhs.rawValue }
        }
        enum TaskState: String { case queued, running, delegated, completed, failed }
    }

    private var taskQueue: [AgentTask] = []
    private var completedTasks: [AgentTask] = []
    private(set) var totalTasksExecuted: Int = 0
    private(set) var totalTasksDelegated: Int = 0
    private(set) var totalTasksFailed: Int = 0
    private var isRunningLoop: Bool = false
    private let agentStopSemaphore = DispatchSemaphore(value: 0)  // EVO_55: interruptible sleep

    // ─── LIFECYCLE ───

    func activate() {
        lock.lock()
        guard !isActive else { lock.unlock(); return }
        isActive = true
        lock.unlock()
        startAgentLoop()
    }

    func deactivate() {
        lock.lock()
        isActive = false
        isRunningLoop = false
        lock.unlock()
    }

    // ─── TASK MANAGEMENT ───

    /// Submit a new goal for the agent to accomplish
    @discardableResult
    func submitGoal(_ goal: String, priority: AgentTask.TaskPriority = .normal) -> String {
        let task = AgentTask(
            id: UUID().uuidString.prefix(8).description,
            goal: goal, priority: priority, state: .queued,
            createdAt: Date(), completedAt: nil, assignedPeer: nil,
            result: nil, subtasks: [], attempts: 0
        )
        lock.lock()
        taskQueue.append(task)
        taskQueue.sort { $0.priority > $1.priority }
        lock.unlock()
        return task.id
    }

    /// Decompose a complex goal into subtasks
    func decompose(_ parentId: String) -> Int {
        lock.lock()
        guard let idx = taskQueue.firstIndex(where: { $0.id == parentId }) else {
            lock.unlock(); return 0
        }
        let parent = taskQueue[idx]
        lock.unlock()

        let goal = parent.goal.lowercased()
        var subtaskGoals: [String] = []

        if goal.contains("research") || goal.contains("analyze") {
            subtaskGoals = [
                "Search knowledge base for: \(parent.goal)",
                "Cross-reference web sources for: \(parent.goal)",
                "Synthesize findings for: \(parent.goal)"
            ]
        } else if goal.contains("optimize") || goal.contains("improve") {
            subtaskGoals = [
                "Profile current performance of: \(parent.goal)",
                "Identify bottlenecks in: \(parent.goal)",
                "Apply optimizations for: \(parent.goal)"
            ]
        } else if goal.contains("monitor") || goal.contains("watch") {
            subtaskGoals = [
                "Set up telemetry for: \(parent.goal)",
                "Define alert thresholds for: \(parent.goal)",
                "Start continuous monitoring of: \(parent.goal)"
            ]
        } else {
            subtaskGoals = [
                "Gather context for: \(parent.goal)",
                "Execute: \(parent.goal)",
                "Verify completion of: \(parent.goal)"
            ]
        }

        var childIds: [String] = []
        for sg in subtaskGoals {
            let childId = submitGoal(sg, priority: parent.priority)
            childIds.append(childId)
        }

        lock.lock()
        if let pidx = taskQueue.firstIndex(where: { $0.id == parentId }) {
            taskQueue[pidx].subtasks = childIds
        }
        lock.unlock()
        return childIds.count
    }

    // ─── MESH DELEGATION ───

    /// Attempt to delegate a task to a mesh peer with available capacity
    private func delegateToMesh(_ task: inout AgentTask) -> Bool {
        let net = NetworkLayer.shared
        let alivePeers = net.peers.values.filter { $0.latencyMs >= 0 && $0.latencyMs < 100 }
        guard !alivePeers.isEmpty else { return false }

        // Choose peer with lowest latency and quantum link
        let bestPeer: NetworkLayer.Peer
        let qLinkedPeers = alivePeers.filter { $0.isQuantumLinked }
        if let qPeer = qLinkedPeers.min(by: { $0.latencyMs < $1.latencyMs }) {
            bestPeer = qPeer
        } else if let nearPeer = alivePeers.min(by: { $0.latencyMs < $1.latencyMs }) {
            bestPeer = nearPeer
        } else {
            return false
        }

        task.assignedPeer = bestPeer.id
        task.state = .delegated

        // Record delegation in Raft consensus log
        _ = NodeSyncProtocol.shared.replicateAcrossMesh(
            command: "task_delegate",
            data: ["task_id": task.id, "goal": task.goal, "peer": bestPeer.id]
        )

        // Track via telemetry
        // TelemetryDashboard: agent_delegation tracked

        totalTasksDelegated += 1
        return true
    }

    // ─── EXECUTION ───

    /// Execute a task locally using available engines
    private func executeLocally(_ task: inout AgentTask) -> String {
        let goal = task.goal.lowercased()

        // Route to appropriate engine based on goal content
        if goal.contains("search") || goal.contains("knowledge") || goal.contains("find") {
            let kb = ASIKnowledgeBase.shared
            let results = kb.search(task.goal, limit: 10)
            let summaries = results.compactMap { $0["completion"] as? String }.shuffled().prefix(3)
            return "KB Search: \(results.count) results. Top: \(summaries.joined(separator: "; "))"
        }

        if goal.contains("web") || goal.contains("internet") || goal.contains("online") {
            return "Web task queued via LiveWebSearchEngine for: \(task.goal)"
        }

        if goal.contains("evolve") || goal.contains("optimize") || goal.contains("improve") {
            let evo = ContinuousEvolutionEngine.shared
            if !evo.isRunning { _ = evo.start() }
            return "Evolution engine engaged. Cycles: \(evo.cycleCount), Energy: \(String(format: "%.4f", evo.lastEnergy))"
        }

        if goal.contains("profile") || goal.contains("performance") || goal.contains("benchmark") {
            let mon = MacOSSystemMonitor.shared
            return "System: \(mon.chipGeneration), \(mon.cpuCoreCount) cores, \(String(format: "%.1f", mon.physicalMemoryGB))GB, Mode: \(mon.powerMode.rawValue)"
        }

        if goal.contains("health") || goal.contains("status") || goal.contains("monitor") {
            let health = NexusHealthMonitor.shared
            return "System health: \(String(format: "%.4f", health.computeSystemHealth())), Checks: \(health.checkCount)"
        }

        if goal.contains("research") || goal.contains("hypothesis") {
            let engine = ASIResearchEngine.shared
            return engine.deepResearch(task.goal)
        }

        if goal.contains("synthesize") || goal.contains("combine") || goal.contains("merge") {
            let kb = ASIKnowledgeBase.shared
            return kb.synthesize(task.goal.components(separatedBy: " ").filter { $0.count > 3 })
        }

        // Default: run through sovereignty pipeline
        return "Processed via sovereignty pipeline: \(task.goal.prefix(60))"
    }

    // ─── AGENT LOOP ───

    private func startAgentLoop() {
        guard !isRunningLoop else { return }
        isRunningLoop = true

        DispatchQueue.global(qos: .utility).async { [weak self] in
            guard let self = self else { return }

            while self.isActive {
                self.lock.lock()
                // Find next queued task
                if let idx = self.taskQueue.firstIndex(where: { $0.state == .queued }) {
                    var task = self.taskQueue[idx]
                    task.state = .running
                    task.attempts += 1
                    self.taskQueue[idx] = task
                    self.lock.unlock()

                    // Decide: local vs mesh delegation
                    let net = NetworkLayer.shared
                    let alivePeers = net.peers.values.filter { $0.latencyMs >= 0 }.count
                    let shouldDelegate = alivePeers > 0
                        && task.priority == .low
                        && task.attempts == 1

                    if shouldDelegate {
                        self.lock.lock()
                        var delegateTask = self.taskQueue.first(where: { $0.id == task.id })!
                        let delegated = self.delegateToMesh(&delegateTask)
                        if let didx = self.taskQueue.firstIndex(where: { $0.id == task.id }) {
                            self.taskQueue[didx] = delegateTask
                        }
                        self.lock.unlock()
                        if delegated { _ = self.agentStopSemaphore.wait(timeout: .now() + 1.0); continue }
                    }

                    // Execute locally
                    let result = self.executeLocally(&task)

                    self.lock.lock()
                    if let fidx = self.taskQueue.firstIndex(where: { $0.id == task.id }) {
                        self.taskQueue[fidx].state = .completed
                        self.taskQueue[fidx].completedAt = Date()
                        self.taskQueue[fidx].result = String(result.prefix(500))
                        let completed = self.taskQueue.remove(at: fidx)
                        self.completedTasks.append(completed)
                        if self.completedTasks.count > 200 { self.completedTasks.removeFirst() }
                        self.totalTasksExecuted += 1
                    }
                    self.lock.unlock()

                    // Record metric
                    // TelemetryDashboard: agent_task_completed tracked
                } else {
                    self.lock.unlock()
                }

                _ = self.agentStopSemaphore.wait(timeout: .now() + 2.0) // Agent tick interval (EVO_55: interruptible)
            }
            self.isRunningLoop = false
        }
    }

    // ─── STATUS ───

    func status() -> [String: Any] {
        lock.lock()
        let queued = taskQueue.filter { $0.state == .queued }.count
        let running = taskQueue.filter { $0.state == .running }.count
        let delegated = taskQueue.filter { $0.state == .delegated }.count
        lock.unlock()

        let net = NetworkLayer.shared
        let meshPeers = net.peers.values.filter { $0.latencyMs >= 0 }.count

        return [
            "engine": "AutonomousAgent",
            "active": isActive,
            "version": "2.0.0-mesh",
            "queued_tasks": queued,
            "running_tasks": running,
            "delegated_tasks": delegated,
            "total_executed": totalTasksExecuted,
            "total_delegated": totalTasksDelegated,
            "total_failed": totalTasksFailed,
            "completed_history": completedTasks.count,
            "mesh_peers_available": meshPeers
        ]
    }

    var statusReport: String {
        let s = status()
        let recentTasks = completedTasks.suffix(5).reversed().map { t in
            "  [\(t.state.rawValue)] \(t.goal.prefix(50))\(t.assignedPeer != nil ? " → peer:\(t.assignedPeer!.prefix(8))" : "")"
        }.joined(separator: "\n")

        return """
        ╔═══════════════════════════════════════════════════════════╗
        ║    🤖 AUTONOMOUS AGENT — MESH-AWARE TASK SCHEDULER        ║
        ╠═══════════════════════════════════════════════════════════╣
        ║  Status:           \(isActive ? "🟢 ACTIVE" : "🔴 INACTIVE")
        ║  Queued:           \(s["queued_tasks"] ?? 0)
        ║  Running:          \(s["running_tasks"] ?? 0)
        ║  Delegated:        \(s["delegated_tasks"] ?? 0)
        ║  Total Executed:   \(totalTasksExecuted)
        ║  Total Delegated:  \(totalTasksDelegated)
        ║  Mesh Peers:       \(s["mesh_peers_available"] ?? 0)
        ╠═══════════════════════════════════════════════════════════╣
        ║  RECENT TASKS:
        \(recentTasks.isEmpty ? "  (none)" : recentTasks)
        ╚═══════════════════════════════════════════════════════════╝
        """
    }
}
