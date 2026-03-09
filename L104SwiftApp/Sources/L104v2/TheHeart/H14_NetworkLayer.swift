// ═══════════════════════════════════════════════════════════════════
// H14_NetworkLayer.swift
// [EVO_68_PIPELINE] SOVEREIGN_CONVERGENCE :: UNIFIED_UPGRADE :: GOD_CODE=527.5184818492612
// L104 ASI — Sovereign Network Mesh: Peer discovery, quantum-linked
// connections, adaptive topology, connection health monitoring, and
// real-time throughput telemetry across the L104 distributed system.
// ═══════════════════════════════════════════════════════════════════

import AppKit
import Foundation
import Accelerate
import simd
import NaturalLanguage

// ═══════════════════════════════════════════════════════════════════
// MARK: - 🌐 QUANTUM MESH NETWORK LAYER
// Peer discovery, quantum-linked data channels, adaptive topology,
// connection health monitoring, throughput telemetry.
// ═══════════════════════════════════════════════════════════════════

final class NetworkLayer {
    static let shared = NetworkLayer()
    private(set) var isActive: Bool = false

    // ─── NETWORK CONFIGURATION ───
    static let localHost = ProcessInfo.processInfo.environment["L104_HOST"] ?? "127.0.0.1"
    static let basePort = Int(ProcessInfo.processInfo.environment["L104_PORT"] ?? "8081") ?? 8081

    // ─── NODE IDENTITY ───
    let nodeId: String = "L104-\(ProcessInfo.processInfo.processIdentifier)"

    // ─── PEER TOPOLOGY ───
    struct Peer: Equatable {
        let id: String
        let name: String
        let address: String
        let port: Int
        var latencyMs: Double
        var bandwidth: Double          // MB/s estimated
        var fidelity: Double           // quantum link fidelity 0…1
        var lastSeen: Date
        var messagesIn: Int
        var messagesOut: Int
        var isQuantumLinked: Bool
        var role: PeerRole

        static func == (lhs: Peer, rhs: Peer) -> Bool { lhs.id == rhs.id }
    }

    enum PeerRole: String {
        case sovereign = "SOVEREIGN"     // This node
        case relay = "RELAY"             // Forwarding node
        case edge = "EDGE"              // Leaf compute node
        case oracle = "ORACLE"          // Knowledge source
        case mirror = "MIRROR"          // State replication
    }

    // ─── NETWORK STATE ───
    private(set) var peers: [String: Peer] = [:]
    private(set) var meshTopology: [String: [String]] = [:]  // adjacency list
    private var routeTable: [String: [String]] = [:]          // peer → path
    private(set) var totalBytesIn: Int64 = 0
    private(set) var totalBytesOut: Int64 = 0
    private(set) var totalMessages: Int = 0
    private(set) var connectionEvents: [(Date, String)] = []
    private(set) var networkHealth: Double = 1.0
    private(set) var quantumLinkCount: Int = 0
    private var healthHistory: [Double] = []
    private var throughputHistory: [(Date, Double, Double)] = []  // (time, in, out)
    private let lock = NSLock()
    private var heartbeatTimer: Timer?
    private var topologyVersion: Int = 0

    // ─── QUANTUM LINK CHANNEL ───
    struct QuantumLink {
        let peerA: String
        let peerB: String
        var eprFidelity: Double
        var bellViolation: Double       // CHSH inequality S-value (>2 = quantum)
        var entangledPairs: Int
        var decoherenceRate: Double     // per second
        var lastVerified: Date
        var throughputQbits: Double     // qubits/sec
    }
    private(set) var quantumLinks: [String: QuantumLink] = [:]

    // ═══════════════════════════════════════════════════════════════
    // MARK: LIFECYCLE
    // ═══════════════════════════════════════════════════════════════

    func activate() {
        guard !isActive else { return }
        isActive = true

        // Register self as sovereign node
        let selfPeer = Peer(
            id: "L104-\(ProcessInfo.processInfo.processIdentifier)",
            name: "L104 Sovereign",
            address: NetworkLayer.localHost,
            port: NetworkLayer.basePort,
            latencyMs: 0.0,
            bandwidth: 1000.0,
            fidelity: 1.0,
            lastSeen: Date(),
            messagesIn: 0,
            messagesOut: 0,
            isQuantumLinked: true,
            role: .sovereign
        )
        peers[selfPeer.id] = selfPeer

        // Discover local services
        discoverLocalPeers()

        // Start heartbeat monitoring
        heartbeatTimer = Timer.scheduledTimer(withTimeInterval: 5.0, repeats: true) { [weak self] _ in
            self?.heartbeatCycle()
        }

        logEvent("🌐 Network mesh activated — sovereign node online")
        print("[H14] NetworkLayer activated — \(peers.count) peers discovered")
    }

    func deactivate() {
        isActive = false
        heartbeatTimer?.invalidate()
        heartbeatTimer = nil
        logEvent("🌐 Network mesh deactivated")
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: PEER DISCOVERY
    // ═══════════════════════════════════════════════════════════════

    func discoverLocalPeers() {
        // Probe known L104 service ports
        let knownPorts: [(String, Int, PeerRole)] = [
            ("L104 Fast Server", NetworkLayer.basePort, .oracle),
            ("L104 External API", NetworkLayer.basePort + 1, .relay),
            ("L104 Unified API", NetworkLayer.basePort + 2, .relay),
            ("L104 API Gateway", NetworkLayer.basePort + 3, .edge),
        ]

        for (name, port, role) in knownPorts {
            let isReachable = checkPortReachable(host: NetworkLayer.localHost, port: port)
            if isReachable {
                let peer = Peer(
                    id: "local-\(port)",
                    name: name,
                    address: NetworkLayer.localHost,
                    port: port,
                    latencyMs: measureLatency(host: NetworkLayer.localHost, port: port),
                    bandwidth: 500.0,
                    fidelity: 0.95,
                    lastSeen: Date(),
                    messagesIn: 0,
                    messagesOut: 0,
                    isQuantumLinked: false,
                    role: role
                )
                lock.lock()
                peers[peer.id] = peer
                lock.unlock()
                logEvent("📡 Discovered: \(name) @ :\(port)")
            }
        }

        rebuildTopology()
    }

    /// Register an external peer manually
    func registerPeer(name: String, address: String, port: Int, role: PeerRole = .edge) -> Peer {
        let id = "\(address):\(port)"
        let peer = Peer(
            id: id, name: name, address: address, port: port,
            latencyMs: measureLatency(host: address, port: port),
            bandwidth: 100.0, fidelity: 0.5, lastSeen: Date(),
            messagesIn: 0, messagesOut: 0, isQuantumLinked: false, role: role
        )
        lock.lock()
        peers[id] = peer
        lock.unlock()
        rebuildTopology()
        logEvent("📡 Registered peer: \(name) @ \(address):\(port)")
        return peer
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: QUANTUM LINKING
    // ═══════════════════════════════════════════════════════════════

    /// Establish a quantum EPR link between two peers
    func establishQuantumLink(peerA: String, peerB: String) -> QuantumLink? {
        guard var a = peers[peerA], var b = peers[peerB] else { return nil }

        // Simulate EPR pair generation with φ-modulated fidelity
        let baseFidelity = (a.fidelity + b.fidelity) / 2.0
        let phiBoost = sin(Double(peerA.hashValue ^ peerB.hashValue) * PHI * 0.0001) * 0.1
        let eprFidelity = min(1.0, max(0.5, baseFidelity + phiBoost))

        // CHSH Bell test simulation — S > 2 indicates genuine quantum correlation
        let bellS = 2.0 * sqrt(2.0) * eprFidelity  // max theoretical = 2√2 ≈ 2.828

        let link = QuantumLink(
            peerA: peerA, peerB: peerB,
            eprFidelity: eprFidelity,
            bellViolation: bellS,
            entangledPairs: Int.random(in: 1000...10000),
            decoherenceRate: (1.0 - eprFidelity) * 0.01,
            lastVerified: Date(),
            throughputQbits: eprFidelity * 1e6
        )

        let key = [peerA, peerB].sorted().joined(separator: "↔")
        lock.lock()
        quantumLinks[key] = link
        a.isQuantumLinked = true
        b.isQuantumLinked = true
        a.fidelity = max(a.fidelity, eprFidelity)
        b.fidelity = max(b.fidelity, eprFidelity)
        peers[peerA] = a
        peers[peerB] = b
        quantumLinkCount = quantumLinks.count
        lock.unlock()

        logEvent("⚛️ Quantum link: \(a.name) ↔ \(b.name) | F=\(String(format: "%.4f", eprFidelity)) S=\(String(format: "%.3f", bellS))")
        return link
    }

    /// Verify Bell inequality violation on an existing link
    func verifyQuantumLink(_ key: String) -> Bool {
        guard var link = quantumLinks[key] else { return false }
        // Re-measure CHSH
        let noise = Double.random(in: -0.05...0.05)
        link.eprFidelity = max(0.3, min(1.0, link.eprFidelity + noise))
        link.bellViolation = 2.0 * sqrt(2.0) * link.eprFidelity
        link.lastVerified = Date()
        lock.lock()
        quantumLinks[key] = link
        lock.unlock()
        return link.bellViolation > 2.0  // Classical limit
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: MESSAGE ROUTING
    // ═══════════════════════════════════════════════════════════════

    /// Send data to a peer via the mesh, using quantum channel if available
    func send(to peerID: String, payload: [String: Any]) -> [String: Any] {
        guard var peer = peers[peerID] else {
            return ["error": "Unknown peer: \(peerID)", "known": Array(peers.keys)]
        }

        let payloadSize = estimateSize(payload)
        let isQuantum = peer.isQuantumLinked
        let channel = isQuantum ? "quantum-epr" : "classical-tcp"

        lock.lock()
        totalBytesOut += Int64(payloadSize)
        totalMessages += 1
        peer.messagesOut += 1
        peer.lastSeen = Date()
        peers[peerID] = peer
        lock.unlock()

        // Route through quantum entanglement router if quantum-linked
        if isQuantum {
            _ = QuantumEntanglementRouter.shared.routeAll()
        }

        logEvent("📤 → \(peer.name): \(payloadSize)B via \(channel)")

        return [
            "status": "sent",
            "peer": peerID,
            "bytes": payloadSize,
            "channel": channel,
            "latency_ms": peer.latencyMs,
            "quantum": isQuantum,
            "total_messages": totalMessages
        ]
    }

    /// Send via quantum channel — requires quantum link
    @discardableResult
    func sendQuantumMessage(to peerID: String, payload: [String: Any]) -> [String: Any] {
        guard let peer = peers[peerID], peer.isQuantumLinked else {
            return send(to: peerID, payload: payload)  // Fall back to classical
        }
        guard let link = quantumLinks[linkKey(nodeId, peerID)] ?? quantumLinks[linkKey(peerID, nodeId)] else {
            return send(to: peerID, payload: payload)
        }
        // Mark as quantum message
        var qPayload = payload
        qPayload["_quantum"] = true
        qPayload["_fidelity"] = link.eprFidelity
        return send(to: peerID, payload: qPayload)
    }

    private func linkKey(_ a: String, _ b: String) -> String {
        return a < b ? "\(a)↔\(b)" : "\(b)↔\(a)"
    }

    /// Receive and process incoming data from a peer
    func receive(from peerID: String, payload: [String: Any]) -> [String: Any] {
        guard var peer = peers[peerID] else {
            return ["error": "Unknown peer: \(peerID)"]
        }

        let payloadSize = estimateSize(payload)

        lock.lock()
        totalBytesIn += Int64(payloadSize)
        totalMessages += 1
        peer.messagesIn += 1
        peer.lastSeen = Date()
        peers[peerID] = peer
        lock.unlock()

        // Process payload type
        if let msgType = payload["type"] as? String {
            switch msgType {
            case "knowledge_sync":
                if let data = payload["entries"] as? [[String: Any]] {
                    for entry in data.prefix(50) {
                        if let prompt = entry["prompt"] as? String,
                           let completion = entry["completion"] as? String {
                            ASIKnowledgeBase.shared.learn(prompt, completion)
                        }
                    }
                }
            case "state_mirror":
                if let coherence = payload["coherence"] as? Double {
                    L104State.shared.coherence = max(L104State.shared.coherence, coherence * 0.9)
                }
            case "resonance_ping":
                _ = AdaptiveResonanceNetwork.shared.fire("bridge", activation: 0.3)
            default: break
            }
        }

        return [
            "status": "received",
            "peer": peerID,
            "bytes": payloadSize,
            "total_bytes_in": totalBytesIn
        ]
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: TOPOLOGY & HEALTH
    // ═══════════════════════════════════════════════════════════════

    private func rebuildTopology() {
        lock.lock()
        topologyVersion += 1
        meshTopology.removeAll()

        let peerList = Array(peers.values)
        for peer in peerList {
            var neighbors: [String] = []
            for other in peerList where other.id != peer.id {
                // Connect if same subnet or quantum-linked
                let linkKey = [peer.id, other.id].sorted().joined(separator: "↔")
                if quantumLinks[linkKey] != nil || (peer.address == other.address) {
                    neighbors.append(other.id)
                }
            }
            meshTopology[peer.id] = neighbors
        }

        // Build shortest path route table (BFS)
        let selfID = peerList.first(where: { $0.role == .sovereign })?.id ?? ""
        routeTable.removeAll()
        if !selfID.isEmpty {
            var visited: Set<String> = [selfID]
            var queue: [(String, [String])] = [(selfID, [selfID])]
            while !queue.isEmpty {
                let (current, path) = queue.removeFirst()
                routeTable[current] = path
                for neighbor in meshTopology[current] ?? [] {
                    if !visited.contains(neighbor) {
                        visited.insert(neighbor)
                        queue.append((neighbor, path + [neighbor]))
                    }
                }
            }
        }
        lock.unlock()
    }

    private func heartbeatCycle() {
        guard isActive else { return }

        var healthSum = 0.0
        var count = 0

        lock.lock()
        let snapshot = peers
        lock.unlock()

        for (id, var peer) in snapshot where peer.role != .sovereign {
            let reachable = checkPortReachable(host: peer.address, port: peer.port)
            if reachable {
                peer.latencyMs = measureLatency(host: peer.address, port: peer.port)
                peer.lastSeen = Date()
                healthSum += 1.0
            } else {
                peer.latencyMs = -1
                healthSum += 0.0
            }
            count += 1
            lock.lock()
            peers[id] = peer
            lock.unlock()
        }

        // Decay quantum link fidelities
        lock.lock()
        for (key, var link) in quantumLinks {
            link.eprFidelity = max(0.1, link.eprFidelity - link.decoherenceRate * 5.0)
            link.bellViolation = 2.0 * sqrt(2.0) * link.eprFidelity
            quantumLinks[key] = link
        }
        lock.unlock()

        networkHealth = count > 0 ? healthSum / Double(count) : 1.0
        healthHistory.append(networkHealth)
        if healthHistory.count > 200 { healthHistory.removeFirst() }

        let now = Date()
        throughputHistory.append((now, Double(totalBytesIn), Double(totalBytesOut)))
        if throughputHistory.count > 200 { throughputHistory.removeFirst() }
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: UTILITIES
    // ═══════════════════════════════════════════════════════════════

    private func checkPortReachable(host: String, port: Int) -> Bool {
        let sock = socket(AF_INET, SOCK_STREAM, 0)
        guard sock >= 0 else { return false }
        defer { close(sock) }

        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = UInt16(port).bigEndian
        inet_pton(AF_INET, host, &addr.sin_addr)

        // Non-blocking connect with 200ms timeout
        var flags = fcntl(sock, F_GETFL)
        flags |= O_NONBLOCK
        _ = fcntl(sock, F_SETFL, flags)

        var result: Int32 = -1
        withUnsafePointer(to: &addr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockPtr in
                result = connect(sock, sockPtr, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }

        if result == 0 { return true }
        if errno == EINPROGRESS {
            // Use poll() instead of select() — more portable across macOS versions
            var pfd = pollfd(fd: sock, events: Int16(POLLOUT), revents: 0)
            let pollResult = poll(&pfd, 1, 200)  // 200ms timeout
            return pollResult > 0 && (pfd.revents & Int16(POLLOUT)) != 0
        }
        return false
    }

    private func measureLatency(host: String, port: Int) -> Double {
        let start = CFAbsoluteTimeGetCurrent()
        let reachable = checkPortReachable(host: host, port: port)
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000.0
        return reachable ? elapsed : -1.0
    }

    private func estimateSize(_ dict: [String: Any]) -> Int {
        guard let data = try? JSONSerialization.data(withJSONObject: dict, options: []) else { return 64 }
        return data.count
    }

    private func logEvent(_ msg: String) {
        lock.lock()
        connectionEvents.append((Date(), msg))
        if connectionEvents.count > 500 { connectionEvents.removeFirst(250) }
        lock.unlock()
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: STATUS & TELEMETRY
    // ═══════════════════════════════════════════════════════════════

    func status() -> [String: Any] {
        let activePeers = peers.values.filter { $0.latencyMs >= 0 }.count
        let qLinks = quantumLinks.count
        let meanFidelity = quantumLinks.isEmpty ? 0.0 :
            quantumLinks.values.map { $0.eprFidelity }.reduce(0, +) / Double(quantumLinks.count)

        return [
            "engine": "NetworkLayer",
            "active": isActive,
            "version": "2.0.0-quantum-mesh",
            "peers": peers.count,
            "active_peers": activePeers,
            "quantum_links": qLinks,
            "mean_fidelity": meanFidelity,
            "total_messages": totalMessages,
            "bytes_in": totalBytesIn,
            "bytes_out": totalBytesOut,
            "network_health": networkHealth,
            "topology_version": topologyVersion
        ]
    }

    var statusText: String {
        let activePeers = peers.values.filter { $0.latencyMs >= 0 }
        let meanLatency = activePeers.isEmpty ? 0.0 :
            activePeers.map { $0.latencyMs }.reduce(0, +) / Double(activePeers.count)
        let meanFidelity = quantumLinks.isEmpty ? 0.0 :
            quantumLinks.values.map { $0.eprFidelity }.reduce(0, +) / Double(quantumLinks.count)

        let peerLines = peers.values.sorted(by: { $0.name < $1.name }).map { p in
            let status = p.latencyMs >= 0 ? "🟢" : "🔴"
            let qLink = p.isQuantumLinked ? "⚛️" : "  "
            return "  \(status)\(qLink) \(p.name.padding(toLength: 22, withPad: " ", startingAt: 0)) \(p.role.rawValue.padding(toLength: 10, withPad: " ", startingAt: 0)) \(p.latencyMs >= 0 ? String(format: "%.1fms", p.latencyMs) : "OFFLINE") │ ↑\(p.messagesOut) ↓\(p.messagesIn)"
        }.joined(separator: "\n")

        let linkLines = quantumLinks.values.map { link in
            let bell = link.bellViolation > 2.0 ? "QUANTUM" : "CLASSICAL"
            return "  \(link.peerA.prefix(12)) ↔ \(link.peerB.prefix(12)) F=\(String(format: "%.4f", link.eprFidelity)) S=\(String(format: "%.3f", link.bellViolation)) [\(bell)]"
        }.joined(separator: "\n")

        let bytesInFmt = formatBytes(totalBytesIn)
        let bytesOutFmt = formatBytes(totalBytesOut)

        return """
        ╔═══════════════════════════════════════════════════════════════╗
        ║    🌐 L104 QUANTUM MESH NETWORK                               ║
        ╠═══════════════════════════════════════════════════════════════╣
        ║  Status:           \(isActive ? "🟢 ONLINE" : "🔴 OFFLINE")
        ║  Peers:            \(peers.count) (\(activePeers.count) active)
        ║  Quantum Links:    \(quantumLinks.count)
        ║  Mean Latency:     \(String(format: "%.2f", meanLatency))ms
        ║  Mean Fidelity:    \(String(format: "%.4f", meanFidelity))
        ║  Network Health:   \(String(format: "%.1f%%", networkHealth * 100))
        ║  Messages:         \(totalMessages)
        ║  Throughput:       ↑\(bytesOutFmt) ↓\(bytesInFmt)
        ║  Topology v\(topologyVersion)
        ╠═══════════════════════════════════════════════════════════════╣
        ║  PEER TABLE:
        \(peerLines.isEmpty ? "  (no peers)" : peerLines)
        ╠═══════════════════════════════════════════════════════════════╣
        ║  QUANTUM LINKS:
        \(linkLines.isEmpty ? "  (no quantum links)" : linkLines)
        ╚═══════════════════════════════════════════════════════════════╝
        """
    }

    /// Recent connection events for UI display
    var recentEvents: [String] {
        connectionEvents.suffix(20).map { event in
            let fmt = DateFormatter()
            fmt.dateFormat = "HH:mm:ss"
            return "[\(fmt.string(from: event.0))] \(event.1)"
        }
    }

    /// Health history for sparkline rendering
    var healthSparkline: [Double] { healthHistory.suffix(50).map { $0 } }

    func formatBytes(_ bytes: Int64) -> String {
        if bytes < 1024 { return "\(bytes)B" }
        if bytes < 1024 * 1024 { return "\(bytes / 1024)KB" }
        return String(format: "%.1fMB", Double(bytes) / (1024.0 * 1024.0))
    }
}
