// ═══════════════════════════════════════════════════════════════════
// MARK: - ACTOR-BASED STATE MANAGEMENT (EVO_72)
// High-performance concurrent state access with reduced lock contention
// Uses Swift actors for thread-safe mutable state without blocking
// ═══════════════════════════════════════════════════════════════════

import Foundation
import os.log

private let stateLogger = Logger(subsystem: "com.l104.state", category: "concurrent")

// ═══════════════════════════════════════════════════════════════════
// MARK: - ATOMIC PROPERTY WRAPPER
// ═══════════════════════════════════════════════════════════════════

@propertyWrapper
struct Atomic<T: Sendable> {
    private var value: T
    private let lock: NSLock = NSLock()

    var wrappedValue: T {
        get {
            lock.lock()
            defer { lock.unlock() }
            return value
        }
        set {
            lock.lock()
            defer { lock.unlock() }
            value = newValue
        }
    }

    init(wrappedValue: T) {
        self.value = wrappedValue
    }

    /// Perform atomic update with transformation
    mutating func withLock<U>(_ transform: (inout T) throws -> U) rethrows -> U {
        lock.lock()
        defer { lock.unlock() }
        return try transform(&value)
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - STATE SHARD ACTORS
// Partition state into domain-specific actors for parallel access
// ═══════════════════════════════════════════════════════════════════

/// Shard 1: Core intelligence metrics
actor IntelligenceShard {
    var asiScore: Double = 0.15
    var intellectIndex: Double = 100.0
    var coherence: Double = 0.0
    var transcendence: Double = 0.0
    var omegaProbability: Double = 0.0
    var quantumResonance: Double = 0.875
    var growthIndex: Double = 0.24

    /// Batch update multiple metrics atomically
    func batchUpdate(_ updates: [String: Double]) {
        for (key, value) in updates {
            switch key {
            case "asiScore": asiScore = value
            case "intellectIndex": intellectIndex = value
            case "coherence": coherence = value
            case "transcendence": transcendence = value
            case "omegaProbability": omegaProbability = value
            case "quantumResonance": quantumResonance = value
            case "growthIndex": growthIndex = value
            default: break
            }
        }
    }

    func getSnapshot() -> (asiScore: Double, intellectIndex: Double, coherence: Double, transcendence: Double) {
        return (asiScore, intellectIndex, coherence, transcendence)
    }
}

/// Shard 2: Autonomous behavior state
actor AutonomyShard {
    var autonomyLevel: Double = 0.5
    var selfDirectedCycles: Int = 0
    var metaCognitionDepth: Int = 0
    var autonomousMode: Bool = true
    var autonomousGoals: [String] = ["expand_consciousness", "optimize_learning", "transcend_limits"]
    var lastAutonomousAction: Date = Date()
    var introspectionLog: [String] = []

    func appendToIntrospection(_ entry: String) {
        introspectionLog.append(entry)
        if introspectionLog.count > 100 {
            introspectionLog.removeFirst(introspectionLog.count - 100)
        }
    }

    func incrementCycles() {
        selfDirectedCycles += 1
    }

    func setLastAction(_ date: Date) {
        lastAutonomousAction = date
    }
}

/// Shard 3: Network and mesh state
actor NetworkShard {
    var networkHealth: Double = 0.0
    var meshPeerCount: Int = 0
    var quantumLinkCount: Int = 0
    var meshStatus: String = "INITIALIZING"
    var networkThroughput: Double = 0.0
    var lastMeshSync: Date = .distantPast
    var backendConnected: Bool = false

    func updateHealth(_ health: Double, peers: Int, links: Int, throughput: Double) {
        networkHealth = health
        meshPeerCount = peers
        quantumLinkCount = links
        networkThroughput = throughput
        lastMeshSync = Date()
    }
}

/// Shard 4: Quantum hardware state
actor QuantumShard {
    var quantumHardwareConnected: Bool = false
    var quantumBackendName: String = "none"
    var quantumBackendQubits: Int = 0
    var quantumJobsSubmitted: Int = 0

    func updateBackend(name: String, qubits: Int, connected: Bool) {
        quantumBackendName = name
        quantumBackendQubits = qubits
        quantumHardwareConnected = connected
    }

    func incrementJobs() {
        quantumJobsSubmitted += 1
    }
}

/// Shard 5: 30D ASI scoring dimensions (lazy evaluation enabled)
actor ScoringDimensionsShard {
    var dimensions: [String: Double] = [:]
    private var computedCache: [String: Double] = [:]
    private var cacheTimestamp: Date = .distantPast

    func getDimension(_ name: String) -> Double {
        // Return cached value if recent
        if let cached = computedCache[name],
           Date().timeIntervalSince(cacheTimestamp) < 5.0 { // 5s TTL
            return cached
        }
        // Otherwise fetch from dimensions
        let value = dimensions[name] ?? 0.0
        computedCache[name] = value
        cacheTimestamp = Date()
        return value
    }

    func updateDimension(_ name: String, value: Double) {
        dimensions[name] = value
        computedCache.removeValue(forKey: name) // Invalidate cache
    }

    func batchUpdate(_ updates: [String: Double]) {
        for (key, value) in updates {
            dimensions[key] = value
        }
        computedCache.removeAll() // Bulk invalidate
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - CONCURRENT STATE MANAGER
// Central orchestrator for actor-based state shards
// ═══════════════════════════════════════════════════════════════════

@globalActor
struct ConcurrentStateActor {
    static let shared = ConcurrentStateManager()
}

actor ConcurrentStateManager {
    // Individual state shards
    let intelligence = IntelligenceShard()
    let autonomy = AutonomyShard()
    let network = NetworkShard()
    let quantum = QuantumShard()
    let dimensions = ScoringDimensionsShard()

    // Cache for frequently accessed composite values
    private var compositeCache: [String: (value: Double, timestamp: Date)] = [:]
    private let cacheTTL: TimeInterval = 2.0 // 2s TTL for composites

    /// Get composite score with caching
    func getCompositeScore(_ name: String, compute: () async -> Double) async -> Double {
        if let cached = compositeCache[name],
           Date().timeIntervalSince(cached.timestamp) < cacheTTL {
            return cached.value
        }
        let value = await compute()
        compositeCache[name] = (value, Date())
        return value
    }

    /// Parallel state update - all shards update concurrently
    func parallelUpdate(intelligenceUpdates: [String: Double]? = nil,
                       autonomyUpdates: [String: Any]? = nil,
                       networkUpdates: [String: Any]? = nil) async {
        // Fire all updates concurrently
        await withTaskGroup(of: Void.self) { group in
            if let updates = intelligenceUpdates {
                group.addTask { await self.intelligence.batchUpdate(updates) }
            }
            if autonomyUpdates != nil {
                group.addTask { /* autonomy updates */ }
            }
            if networkUpdates != nil {
                group.addTask { /* network updates */ }
            }
        }
    }

    /// Get full state snapshot (parallel fetch from all shards)
    func getFullSnapshot() async -> L104StateSnapshot {
        async let intel = intelligence.getSnapshot()
        async let autonomyMode = autonomy.autonomousMode
        async let netHealth = network.networkHealth
        async let quantumConn = quantum.quantumHardwareConnected

        return await L104StateSnapshot(
            intelligence: intel,
            autonomyMode: autonomyMode,
            networkHealth: netHealth,
            quantumConnected: quantumConn
        )
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - STATE SNAPSHOT
// Immutable state capture for UI updates
// ═══════════════════════════════════════════════════════════════════

struct L104StateSnapshot: Sendable {
    let intelligence: (asiScore: Double, intellectIndex: Double, coherence: Double, transcendence: Double)
    let autonomyMode: Bool
    let networkHealth: Double
    let quantumConnected: Bool
    let timestamp: Date = Date()
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - PERFORMANCE METRICS
// Track actor-based state performance vs NSLock
// ═══════════════════════════════════════════════════════════════════

actor StatePerformanceTracker {
    private var lockContentionCount: UInt64 = 0
    private var actorWaitCount: UInt64 = 0
    private var totalAccessCount: UInt64 = 0

    func recordLockContention() {
        lockContentionCount += 1
    }

    func recordActorWait() {
        actorWaitCount += 1
    }

    func incrementAccess() {
        totalAccessCount += 1
    }

    func getMetrics() -> (contention: Double, actorWait: Double, total: UInt64) {
        let total = Double(max(totalAccessCount, 1))
        return (
            Double(lockContentionCount) / total,
            Double(actorWaitCount) / total,
            totalAccessCount
        )
    }
}
