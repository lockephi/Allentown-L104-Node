// ═══════════════════════════════════════════════════════════════════
// B15_DistributedSystems.swift
// [EVO_58_PIPELINE] QUANTUM_COGNITION :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612
// L104 · TheBrain · v2 Architecture
//
// Extracted from L104Native.swift lines 7070-7510
// Classes: NodeSyncProtocol, DataReplicationMesh
// ═══════════════════════════════════════════════════════════════════

import AppKit
import Foundation
import Accelerate
import simd
import NaturalLanguage

// ═══════════════════════════════════════════════════════════════════
// MARK: - 📡 NODE SYNC PROTOCOL (Bucket C: Node Protocols)
// Raft-inspired consensus: leader election, log replication,
// heartbeat protocol, commit advancement, state snapshots.
// ═══════════════════════════════════════════════════════════════════

class NodeSyncProtocol {
    static let shared = NodeSyncProtocol()
    // PHI, TAU — use globals from L01_Constants

    // ─── NODE IDENTITY ───
    struct PeerNode: Equatable {
        let id: String
        let host: String
        let port: Int
        var lastHeartbeat: Date
        var isAlive: Bool
        var matchIndex: Int
        var nextIndex: Int

        static func == (lhs: PeerNode, rhs: PeerNode) -> Bool { lhs.id == rhs.id }
    }

    enum NodeRole: String {
        case follower = "FOLLOWER"
        case candidate = "CANDIDATE"
        case leader = "LEADER"
    }

    struct LogEntry {
        let term: Int
        let index: Int
        let command: String
        let data: [String: Any]
        let timestamp: Date
    }

    // ─── RAFT STATE ───
    private var nodeId: String = "L104-\(ProcessInfo.processInfo.processIdentifier)"
    private var currentTerm: Int = 0
    private var votedFor: String? = nil
    private var role: NodeRole = .follower
    private var peers: [PeerNode] = []
    private var log: [LogEntry] = []
    private var commitIndex: Int = 0
    private var lastApplied: Int = 0
    private var leaderHeartbeatInterval: TimeInterval = 2.0
    private var electionTimeout: TimeInterval = 5.0
    private var voteCount: Int = 0
    private var stateCheckpoints: [[String: Any]] = []

    // ─── PEER MANAGEMENT ───
    func registerPeer(id: String, host: String, port: Int) {
        let peer = PeerNode(id: id, host: host, port: port,
                           lastHeartbeat: Date(), isAlive: true,
                           matchIndex: 0, nextIndex: log.count + 1)
        if !peers.contains(peer) {
            peers.append(peer)
        }
    }

    func removePeer(id: String) {
        peers.removeAll { $0.id == id }
    }

    // ─── LEADER ELECTION ───
    func startElection() -> [String: Any] {
        currentTerm += 1
        role = .candidate
        votedFor = nodeId
        voteCount = 1  // Vote for self

        let lastLogIndex = log.count - 1
        let lastLogTerm = log.last?.term ?? 0

        return [
            "type": "RequestVote",
            "term": currentTerm,
            "candidateId": nodeId,
            "lastLogIndex": lastLogIndex,
            "lastLogTerm": lastLogTerm
        ]
    }

    func receiveVote(granted: Bool, fromPeer: String, term: Int) {
        if term > currentTerm {
            currentTerm = term
            role = .follower
            votedFor = nil
            return
        }

        if granted {
            voteCount += 1
            let majority = (peers.count + 1) / 2 + 1
            if voteCount >= majority && role == .candidate {
                role = .leader
                // Initialize nextIndex for all peers
                for i in 0..<peers.count {
                    peers[i].nextIndex = log.count + 1
                    peers[i].matchIndex = 0
                }
            }
        }
    }

    // ─── LOG REPLICATION ───
    func appendEntry(command: String, data: [String: Any] = [:]) -> LogEntry {
        let entry = LogEntry(
            term: currentTerm,
            index: log.count,
            command: command,
            data: data,
            timestamp: Date()
        )
        log.append(entry)
        return entry
    }

    func replicateToFollower(peerId: String) -> [String: Any]? {
        guard role == .leader else { return nil }
        guard let peerIdx = peers.firstIndex(where: { $0.id == peerId }) else { return nil }

        let nextIdx = peers[peerIdx].nextIndex
        let prevLogIndex = nextIdx - 1
        let prevLogTerm = prevLogIndex >= 0 && prevLogIndex < log.count ? log[prevLogIndex].term : 0

        let entries = nextIdx < log.count ? Array(log[nextIdx...]) : []

        return [
            "type": "AppendEntries",
            "term": currentTerm,
            "leaderId": nodeId,
            "prevLogIndex": prevLogIndex,
            "prevLogTerm": prevLogTerm,
            "entries": entries.map { ["term": $0.term, "index": $0.index, "command": $0.command] },
            "leaderCommit": commitIndex
        ]
    }

    // ─── COMMIT ADVANCEMENT ───
    func advanceCommitIndex() {
        guard role == .leader else { return }

        // Find N such that a majority of matchIndex[i] ≥ N
        let matchIndices = peers.map { $0.matchIndex }.sorted()
        let majorityIdx = matchIndices.count / 2
        if majorityIdx < matchIndices.count {
            let newCommit = matchIndices[majorityIdx]
            if newCommit > commitIndex && newCommit < log.count && log[newCommit].term == currentTerm {
                commitIndex = newCommit
            }
        }

        // Apply committed but unapplied entries
        while lastApplied < commitIndex {
            lastApplied += 1
            applyEntry(log[lastApplied])
        }
    }

    private func applyEntry(_ entry: LogEntry) {
        stateCheckpoints.append([
            "index": entry.index,
            "term": entry.term,
            "command": entry.command,
            "applied_at": Date().timeIntervalSince1970
        ])
        if stateCheckpoints.count > 1000 { stateCheckpoints.removeFirst() }
    }

    // ─── HEARTBEAT ───
    func sendHeartbeat() -> [[String: Any]] {
        guard role == .leader else { return [] }
        return peers.map { peer in
            [
                "type": "Heartbeat",
                "term": currentTerm,
                "leaderId": nodeId,
                "to": peer.id,
                "leaderCommit": commitIndex,
                "timestamp": Date().timeIntervalSince1970
            ]
        }
    }

    func receiveHeartbeat(fromLeader: String, term: Int, leaderCommit: Int) {
        if term >= currentTerm {
            currentTerm = term
            role = .follower
            votedFor = nil
            if leaderCommit > commitIndex {
                commitIndex = min(leaderCommit, log.count - 1)
            }
        }
    }

    // ─── STATE SNAPSHOT ───
    func createSnapshot() -> [String: Any] {
        return [
            "node_id": nodeId,
            "term": currentTerm,
            "role": role.rawValue,
            "log_length": log.count,
            "commit_index": commitIndex,
            "last_applied": lastApplied,
            "peers": peers.count,
            "alive_peers": peers.filter { $0.isAlive }.count,
            "vote_count": voteCount,
            "checkpoints": stateCheckpoints.count
        ]
    }

    func statusReport() -> String {
        let aliveCount = peers.filter { $0.isAlive }.count
        return """
        ╔═══════════════════════════════════════════════════════════╗
        ║    📡 NODE SYNC PROTOCOL (Raft Consensus)                 ║
        ╠═══════════════════════════════════════════════════════════╣
        ║  Node ID:          \(nodeId)
        ║  Role:             \(role.rawValue)
        ║  Term:             \(currentTerm)
        ║  Log Entries:      \(log.count)
        ║  Commit Index:     \(commitIndex)
        ║  Last Applied:     \(lastApplied)
        ║  Peers:            \(aliveCount)/\(peers.count) alive
        ╚═══════════════════════════════════════════════════════════╝
        """
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: - 🌐 NETWORK MESH AUTO-DISCOVERY INTEGRATION
    // ═══════════════════════════════════════════════════════════

    /// Sync Raft peer list with NetworkLayer discovered peers
    func syncWithNetworkLayer() -> Int {
        let net = NetworkLayer.shared
        var registered = 0
        for (_, netPeer) in net.peers where netPeer.fidelity > 0.1 {
            let exists = peers.contains(where: { $0.id == netPeer.id })
            if !exists {
                registerPeer(id: netPeer.id, host: netPeer.address, port: netPeer.port)
                registered += 1
            }
        }
        // Mark dead peers from network state
        for i in 0..<peers.count {
            let found = net.peers.first(where: { $0.value.id == peers[i].id })
            if let np = found {
                peers[i].isAlive = np.value.fidelity > 0.1
                peers[i].lastHeartbeat = Date()
            }
        }
        return registered
    }

    /// Replicate a log entry across the quantum mesh
    func replicateAcrossMesh(command: String, data: [String: Any] = [:]) -> [String: Any] {
        let entry = appendEntry(command: command, data: data)
        var replicated = 0
        for peer in peers where peer.isAlive {
            if let _ = replicateToFollower(peerId: peer.id) {
                replicated += 1
            }
        }
        advanceCommitIndex()
        return [
            "entry_index": entry.index,
            "term": entry.term,
            "replicated_to": replicated,
            "commit_index": commitIndex
        ]
    }
}


// ═══════════════════════════════════════════════════════════════════
// MARK: - 📡 DATA REPLICATION MESH (Bucket C: Node Protocols)
// CRDTs (conflict-free replicated data types) for eventual consistency.
// Implements G-Counter, PN-Counter, LWW-Register, OR-Set for
// distributed engine state without coordination overhead.
// ═══════════════════════════════════════════════════════════════════

class DataReplicationMesh {
    static let shared = DataReplicationMesh()
    // PHI — use global from L01_Constants

    // ─── G-COUNTER (Grow-only) ───
    struct GCounter {
        var counts: [String: Int] = [:]

        mutating func increment(nodeId: String, by value: Int = 1) {
            counts[nodeId, default: 0] += value
        }

        func value() -> Int {
            counts.values.reduce(0, +)
        }

        func merge(with other: GCounter) -> GCounter {
            var result = GCounter()
            let allKeys = Set(counts.keys).union(other.counts.keys)
            for key in allKeys {
                result.counts[key] = max(counts[key] ?? 0, other.counts[key] ?? 0)
            }
            return result
        }
    }

    // ─── PN-COUNTER (Increment + Decrement) ───
    struct PNCounter {
        var positive = GCounter()
        var negative = GCounter()

        mutating func increment(nodeId: String, by value: Int = 1) {
            positive.increment(nodeId: nodeId, by: value)
        }

        mutating func decrement(nodeId: String, by value: Int = 1) {
            negative.increment(nodeId: nodeId, by: value)
        }

        func value() -> Int {
            positive.value() - negative.value()
        }

        func merge(with other: PNCounter) -> PNCounter {
            var result = PNCounter()
            result.positive = positive.merge(with: other.positive)
            result.negative = negative.merge(with: other.negative)
            return result
        }
    }

    // ─── LWW-REGISTER (Last-Writer-Wins) ───
    struct LWWRegister<T> {
        var value: T?
        var timestamp: TimeInterval = 0

        mutating func set(_ newValue: T, at time: TimeInterval = Date().timeIntervalSince1970) {
            if time > timestamp {
                value = newValue
                timestamp = time
            }
        }

        func merge(with other: LWWRegister<T>) -> LWWRegister<T> {
            return timestamp >= other.timestamp ? self : other
        }
    }

    // ─── OR-SET (Observed-Remove Set) ───
    struct ORSet<T: Hashable> {
        var adds: [T: Set<String>] = [:]
        var removes: [T: Set<String>] = [:]

        mutating func add(_ element: T, tag: String = UUID().uuidString) {
            adds[element, default: Set()].insert(tag)
        }

        mutating func remove(_ element: T) {
            if let tags = adds[element] {
                removes[element, default: Set()].formUnion(tags)
            }
        }

        func elements() -> Set<T> {
            var result = Set<T>()
            for (elem, addTags) in adds {
                let remTags = removes[elem] ?? Set()
                if !addTags.subtracting(remTags).isEmpty {
                    result.insert(elem)
                }
            }
            return result
        }

        func merge(with other: ORSet<T>) -> ORSet<T> {
            var result = ORSet<T>()
            let allKeys = Set(adds.keys).union(other.adds.keys)
            for key in allKeys {
                result.adds[key] = (adds[key] ?? Set()).union(other.adds[key] ?? Set())
                result.removes[key] = (removes[key] ?? Set()).union(other.removes[key] ?? Set())
            }
            return result
        }
    }

    // ─── MESH STATE ───
    private var counters: [String: PNCounter] = [:]
    private var registers: [String: LWWRegister<String>] = [:]
    private var sets: [String: ORSet<String>] = [:]
    private var syncLog: [[String: Any]] = []
    private var mergeCount: Int = 0
    private var conflictResolutions: Int = 0

    // ─── COUNTER OPERATIONS ───
    func incrementCounter(_ name: String, nodeId: String, by value: Int = 1) {
        counters[name, default: PNCounter()].increment(nodeId: nodeId, by: value)
    }

    func decrementCounter(_ name: String, nodeId: String, by value: Int = 1) {
        counters[name, default: PNCounter()].decrement(nodeId: nodeId, by: value)
    }

    func getCounter(_ name: String) -> Int {
        counters[name]?.value() ?? 0
    }

    // ─── REGISTER OPERATIONS ───
    func setRegister(_ name: String, value: String) {
        registers[name, default: LWWRegister<String>()].set(value)
    }

    func getRegister(_ name: String) -> String? {
        registers[name]?.value
    }

    // ─── SET OPERATIONS ───
    func addToSet(_ name: String, element: String) {
        sets[name, default: ORSet<String>()].add(element)
    }

    func removeFromSet(_ name: String, element: String) {
        sets[name, default: ORSet<String>()].remove(element)
    }

    func getSet(_ name: String) -> Set<String> {
        sets[name]?.elements() ?? Set()
    }

    // ─── FULL MESH SYNC ───
    func syncWith(remoteCounters: [String: PNCounter],
                  remoteRegisters: [String: LWWRegister<String>],
                  remoteSets: [String: ORSet<String>]) {
        mergeCount += 1

        for (key, remote) in remoteCounters {
            let merged = counters[key]?.merge(with: remote) ?? remote
            if counters[key] != nil { conflictResolutions += 1 }
            counters[key] = merged
        }

        for (key, remote) in remoteRegisters {
            counters.keys.forEach { _ in } // type-check only
            registers[key] = registers[key]?.merge(with: remote) ?? remote
        }

        for (key, remote) in remoteSets {
            sets[key] = sets[key]?.merge(with: remote) ?? remote
        }

        syncLog.append([
            "merge_id": mergeCount,
            "timestamp": Date().timeIntervalSince1970,
            "counters_merged": remoteCounters.count,
            "registers_merged": remoteRegisters.count,
            "sets_merged": remoteSets.count,
            "conflicts_resolved": conflictResolutions
        ])
        if syncLog.count > 500 { syncLog.removeFirst() }
    }

    func statusReport() -> String {
        return """
        ╔═══════════════════════════════════════════════════════════╗
        ║    📡 DATA REPLICATION MESH (CRDTs)                       ║
        ╠═══════════════════════════════════════════════════════════╣
        ║  Counters:         \(counters.count)
        ║  Registers:        \(registers.count)
        ║  Sets:             \(sets.count)
        ║  Merges:           \(mergeCount)
        ║  Conflicts:        \(conflictResolutions) resolved
        ║  Sync Log:         \(syncLog.count) entries
        ╚═══════════════════════════════════════════════════════════╝
        """
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: - 🌐 NETWORK-AWARE CRDT REPLICATION
    // ═══════════════════════════════════════════════════════════

    /// Track engine metrics as CRDT counters across the mesh
    func trackEngineMetric(_ name: String, value: Int = 1) {
        let nodeId = "L104-\(ProcessInfo.processInfo.processIdentifier)"
        incrementCounter(name, nodeId: nodeId, by: value)
    }

    /// Register current network state into CRDT registers and sets
    func snapshotNetworkState() {
        let net = NetworkLayer.shared
        setRegister("mesh_status", value: L104State.shared.meshStatus)
        setRegister("peer_count", value: "\(net.peers.count)")
        setRegister("qlink_count", value: "\(net.quantumLinks.count)")

        // Track alive peers in OR-Set
        for (_, peer) in net.peers where peer.fidelity > 0.1 {
            addToSet("alive_peers", element: peer.id)
        }
        // Remove dead peers
        for (_, peer) in net.peers where peer.fidelity <= 0.1 {
            removeFromSet("alive_peers", element: peer.id)
        }
    }

    /// Broadcast CRDT state to all network peers for eventual consistency
    func broadcastToMesh() -> Int {
        let net = NetworkLayer.shared
        snapshotNetworkState()
        var broadcast = 0
        for (peerId, peer) in net.peers where peer.latencyMs >= 0 {
            // In a real system this would send over the network
            // For now we replicate locally to track state
            trackEngineMetric("broadcast_\(peerId)")
            broadcast += 1
        }
        return broadcast
    }
}
