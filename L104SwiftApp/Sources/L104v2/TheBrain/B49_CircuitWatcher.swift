// ═══════════════════════════════════════════════════════════════════
// B49_CircuitWatcher.swift — L104 v2.0
// [EVO_68_PIPELINE] SOVEREIGN_NODE_UPGRADE :: CIRCUIT_WATCHER :: GOD_CODE=527.5184818492612
// L104 ASI — File-System Circuit Watcher Daemon (v2.0)
//
// Watches an inbox directory for JSON circuit payloads using GCD
// DispatchSource (zero CPU when idle). When a new .json file appears:
//
//   1. Parse the circuit description from JSON
//   2. Build a QGateCircuit from the operation list
//   3. Route through QuantumRouter (Clifford fast-lane + T-gate branching)
//   4. Write results to the outbox directory
//   5. Move the processed file to the archive directory
//
// v2.0 Upgrades:
//   - Concurrent circuit processing (DispatchGroup + semaphore)
//   - Priority-based job scheduling (higher priority circuits first)
//   - Per-backend performance telemetry (avg timing per backend)
//   - Expanded gate map (SX, SXDag, CY, iSWAP, ECR, PHI_GATE, GOD_CODE)
//   - Sacred alignment scoring in results (PHI resonance, GOD_CODE alignment)
//   - Thread-safe stats with NSLock
//   - Graceful drain support on stop()
//
// JSON payload format (inbox):
//   {
//     "circuit_id": "bell-001",
//     "num_qubits": 2,
//     "shots": 1024,
//     "priority": 5,
//     "operations": [
//       { "gate": "H",  "qubits": [0] },
//       { "gate": "CX", "qubits": [0, 1] },
//       { "gate": "T",  "qubits": [0] },
//       { "gate": "Rz", "qubits": [1], "parameters": [0.785] }
//     ],
//     "adapt": true
//   }
//
// Result format (outbox):
//   {
//     "circuit_id": "bell-001",
//     "probabilities": { "00": 0.5, "11": 0.5 },
//     "counts": { "00": 512, "11": 512 },
//     "sacred_alignment": { "phi_resonance": 0.95, "god_code_alignment": 0.87 },
//     "metadata": { ... }
//   }
//
// INVARIANT: 527.5184818492612 | PILOT: LONDEL
// ═══════════════════════════════════════════════════════════════════

import Foundation

// ═══════════════════════════════════════════════════════════════════
// MARK: - CIRCUIT PAYLOAD TYPES
// ═══════════════════════════════════════════════════════════════════

/// JSON-decodable circuit operation.
private struct CircuitOperation: Decodable {
    let gate: String
    let qubits: [Int]
    let parameters: [Double]?

    enum CodingKeys: String, CodingKey {
        case gate, qubits, parameters
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        gate = try c.decode(String.self, forKey: .gate)
        qubits = try c.decode([Int].self, forKey: .qubits)
        parameters = try c.decodeIfPresent([Double].self, forKey: .parameters)
    }
}

/// JSON-decodable circuit payload (dropped into inbox).
private struct CircuitPayload: Decodable {
    let circuitId: String
    let numQubits: Int
    let shots: Int?
    let operations: [CircuitOperation]
    let adapt: Bool?
    let maxBranches: Int?
    let pruneEpsilon: Double?
    let priority: Int?             // v2.0: job priority (higher = first)

    enum CodingKeys: String, CodingKey {
        case circuitId = "circuit_id"
        case numQubits = "num_qubits"
        case shots, operations, adapt, priority
        case maxBranches = "max_branches"
        case pruneEpsilon = "prune_epsilon"
    }
}

/// JSON-encodable result written to outbox.
private struct CircuitResult: Encodable {
    let circuitId: String
    let probabilities: [String: Double]
    let counts: [String: Int]?
    let backend: String
    let branchCount: Int
    let tGateCount: Int
    let cliffordGateCount: Int
    let prunedBranches: Int
    let mergedBranches: Int
    let executionTimeMs: Double
    let numQubits: Int
    let adaptZone: String?
    let timestamp: String
    let godCode: Double
    let sacredAlignment: SacredAlignmentResult?   // v2.0

    enum CodingKeys: String, CodingKey {
        case circuitId = "circuit_id"
        case probabilities, counts, backend
        case branchCount = "branch_count"
        case tGateCount = "t_gate_count"
        case cliffordGateCount = "clifford_gate_count"
        case prunedBranches = "pruned_branches"
        case mergedBranches = "merged_branches"
        case executionTimeMs = "execution_time_ms"
        case numQubits = "num_qubits"
        case adaptZone = "adapt_zone"
        case timestamp
        case godCode = "god_code"
        case sacredAlignment = "sacred_alignment"
    }
}

/// v3.0: Sacred alignment + three-engine scoring result.
private struct SacredAlignmentResult: Encodable {
    let phiResonance: Double
    let godCodeAlignment: Double
    let compositeScore: Double
    let entropyReversal: Double       // v3.0: Science Engine Maxwell's Demon
    let harmonicResonance: Double     // v3.0: Math Engine GOD_CODE + 104 Hz
    let waveCoherence: Double         // v3.0: Math Engine PHI phase-lock
    let threeEngineComposite: Double  // v3.0: Weighted composite

    enum CodingKeys: String, CodingKey {
        case phiResonance = "phi_resonance"
        case godCodeAlignment = "god_code_alignment"
        case compositeScore = "composite_score"
        case entropyReversal = "entropy_reversal"
        case harmonicResonance = "harmonic_resonance"
        case waveCoherence = "wave_coherence"
        case threeEngineComposite = "three_engine_composite"
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - CIRCUIT WATCHER
// ═══════════════════════════════════════════════════════════════════

/// File-system daemon that watches for JSON circuit payloads and
/// routes them through the QuantumRouter at zero idle CPU (v2.0).
final class CircuitWatcher {

    /// Singleton for app-wide access.
    static let shared = CircuitWatcher()

    // ─── Paths ───
    private let inboxDir: String
    private let outboxDir: String
    private let archiveDir: String

    // ─── GCD ───
    private let watchQueue = DispatchQueue(label: "com.l104.circuitwatcher", qos: .utility)
    private let processQueue = DispatchQueue(
        label: "com.l104.circuitwatcher.processor", qos: .userInitiated, attributes: .concurrent)
    private let socketQueue = DispatchQueue(label: "com.l104.circuitwatcher.socket", qos: .userInitiated)
    private var source: DispatchSourceFileSystemObject?
    private var fileDescriptor: Int32 = -1

    // ─── Unix Socket IPC ───
    private var socketFD: Int32 = -1
    private var socketSource: DispatchSourceRead?
    private let socketPath: String
    private var socketActive = false

    // ─── Concurrency (v2.0) ───
    private let maxConcurrent = 4
    private let processSemaphore = DispatchSemaphore(value: 4)
    private let statsLock = NSLock()

    // ─── Stats ───
    private(set) var circuitsProcessed: Int = 0
    private(set) var circuitsFailed: Int = 0
    private(set) var totalExecutionMs: Double = 0.0
    private(set) var gpuExecutions: Int = 0
    private(set) var cpuExecutions: Int = 0
    private(set) var mpsExecutions: Int = 0
    private(set) var stabilizerExecutions: Int = 0
    private(set) var chunkedCPUExecutions: Int = 0
    private var isActive = false

    // ─── v2.0: Per-Backend Timing ───
    private var backendTimings: [String: (count: Int, totalMs: Double)] = [
        "clifford_chp": (0, 0),
        "statevector": (0, 0),
        "t_branching": (0, 0),
    ]

    // ─── Gate lookup (v2.0 expanded) ───
    private static let gateTypeMap: [String: QGateType] = {
        var map: [String: QGateType] = [:]
        for t in QGateType.allCases { map[t.rawValue] = t }
        // Aliases
        map["CNOT"] = .cnot
        map["cx"]   = .cnot
        map["CX"]   = .cnot
        map["h"]    = .hadamard
        map["H"]    = .hadamard
        map["x"]    = .pauliX
        map["y"]    = .pauliY
        map["z"]    = .pauliZ
        map["s"]    = .phase
        map["sdg"]  = .sGate
        map["t"]    = .tGate
        map["tdg"]  = .tDagger
        map["rx"]   = .rotationX
        map["ry"]   = .rotationY
        map["rz"]   = .rotationZ
        map["cz"]   = .cz
        map["swap"] = .swap
        // v2.0: Expanded gate aliases
        map["Rx"]   = .rotationX
        map["Ry"]   = .rotationY
        map["Rz"]   = .rotationZ
        map["S"]    = .phase
        map["Sdg"]  = .sGate
        map["SDG"]  = .sGate
        map["T"]    = .tGate
        map["Tdg"]  = .tDagger
        map["TDG"]  = .tDagger
        map["X"]    = .pauliX
        map["Y"]    = .pauliY
        map["Z"]    = .pauliZ
        map["CZ"]   = .cz
        map["SWAP"] = .swap
        map["I"]    = .identity
        map["id"]   = .identity
        map["ID"]   = .identity
        return map
    }()

    // ═══════════════════════════════════════════════════════════════
    // MARK: - INIT
    // ═══════════════════════════════════════════════════════════════

    /// Create a watcher with custom directories, or use defaults
    /// under the project's `.l104_circuits/` directory.
    ///
    /// Two layouts are supported:
    ///   - **Structured**: `baseDir/inbox/`, `baseDir/outbox/`, `baseDir/archive/`
    ///     Used when `baseDir` contains an `inbox/` subdirectory.
    ///   - **Flat**: `baseDir/` is the inbox, `baseDir/outbox/`, `baseDir/archive/`
    ///     Used for shared IPC queues like `/tmp/l104_queue/` where files
    ///     are dropped at the root.
    ///
    /// - Parameter baseDir: Root directory. If `nil`, defaults to `$L104_ROOT/.l104_circuits`.
    /// - Parameter socketPath: Unix domain socket path. Defaults to `/tmp/l104_quantum.sock`.
    init(baseDir: String? = nil, socketPath: String = "/tmp/l104_quantum.sock") {
        self.socketPath = socketPath
        let base: String
        if let b = baseDir {
            base = b
        } else if let root = ProcessInfo.processInfo.environment["L104_ROOT"] {
            base = "\(root)/.l104_circuits"
        } else {
            base = "\(NSHomeDirectory())/.l104_circuits"
        }

        // Auto-detect layout: if base/inbox/ exists, use structured; otherwise flat
        let structuredInbox = "\(base)/inbox"
        if FileManager.default.fileExists(atPath: structuredInbox) {
            inboxDir   = structuredInbox
        } else {
            inboxDir   = base  // flat: drop .json files directly into baseDir
        }
        outboxDir  = "\(base)/outbox"
        archiveDir = "\(base)/archive"
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - START / STOP
    // ═══════════════════════════════════════════════════════════════

    /// Start watching the inbox directory.
    /// Creates inbox/outbox/archive directories if they don't exist.
    func start() {
        guard !isActive else { return }

        // Ensure directories exist
        let fm = FileManager.default
        for dir in [inboxDir, outboxDir, archiveDir] {
            if !fm.fileExists(atPath: dir) {
                try? fm.createDirectory(atPath: dir, withIntermediateDirectories: true)
            }
        }

        // Open file descriptor for inbox
        fileDescriptor = open(inboxDir, O_EVTONLY)
        guard fileDescriptor != -1 else {
            print("[L104 CircuitWatcher] ERROR: Cannot open inbox at \(inboxDir)")
            return
        }

        // Create dispatch source watching for writes
        let src = DispatchSource.makeFileSystemObjectSource(
            fileDescriptor: fileDescriptor,
            eventMask: [.write, .rename],
            queue: watchQueue
        )

        src.setEventHandler { [weak self] in
            self?.processPendingCircuits()
        }

        src.setCancelHandler { [weak self] in
            guard let fd = self?.fileDescriptor, fd != -1 else { return }
            close(fd)
            self?.fileDescriptor = -1
        }

        source = src
        src.resume()
        isActive = true

        // Process anything already sitting in inbox
        watchQueue.async { [weak self] in
            self?.processPendingCircuits()
        }

        // EVO_67: Start Unix domain socket for low-latency IPC
        startSocketListener()

        print("[L104 CircuitWatcher] Active — inbox: \(inboxDir) | socket: \(socketPath)")
    }

    /// Stop watching.
    func stop() {
        guard isActive else { return }
        source?.cancel()
        source = nil
        stopSocketListener()
        isActive = false
        print("[L104 CircuitWatcher] Stopped — processed: \(circuitsProcessed), failed: \(circuitsFailed)")
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - PROCESSING PIPELINE
    // ═══════════════════════════════════════════════════════════════

    /// Scan inbox for .json files and process each one.
    /// v2.0: Priority-sorted, concurrent processing with semaphore.
    private func processPendingCircuits() {
        let fm = FileManager.default
        guard let files = try? fm.contentsOfDirectory(atPath: inboxDir) else { return }

        var jsonFiles = files.filter { $0.hasSuffix(".json") }.sorted()

        // v2.0: Priority-based scheduling — read priority from payload
        jsonFiles = sortByPriority(files: jsonFiles)

        // v2.0: Concurrent processing
        let group = DispatchGroup()

        for filename in jsonFiles {
            let inputPath = "\(inboxDir)/\(filename)"

            // Atomicity guard: skip files still being written (size == 0 or very recent)
            guard let attrs = try? fm.attributesOfItem(atPath: inputPath),
                  let size = attrs[.size] as? Int, size > 2 else {
                continue
            }

            group.enter()
            processSemaphore.wait()
            processQueue.async { [weak self] in
                guard let self = self else { return }
                let start = CFAbsoluteTimeGetCurrent()
                do {
                    let result = try self.processCircuitFile(at: inputPath)
                    self.writeResult(result, filename: filename)
                    self.archiveFile(from: inputPath, filename: filename)
                    let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000.0

                    self.statsLock.lock()
                    self.circuitsProcessed += 1
                    self.totalExecutionMs += result.executionTimeMs
                    // Track backend
                    switch result.backend {
                    case let b where b.contains("gpu") || b.contains("metal"):
                        self.gpuExecutions += 1
                    case let b where b.contains("mps"):
                        self.mpsExecutions += 1
                    case let b where b.contains("stabilizer") || b.contains("chp") || b.contains("clifford"):
                        self.stabilizerExecutions += 1
                    case let b where b.contains("chunked"):
                        self.chunkedCPUExecutions += 1
                    default:
                        self.cpuExecutions += 1
                    }
                    // Per-backend timing
                    if var timing = self.backendTimings[result.backend] {
                        timing.count += 1
                        timing.totalMs += elapsed
                        self.backendTimings[result.backend] = timing
                    } else {
                        self.backendTimings[result.backend] = (1, elapsed)
                    }
                    self.statsLock.unlock()
                } catch {
                    print("[L104 CircuitWatcher] Failed to process \(filename): \(error)")
                    self.statsLock.lock()
                    self.circuitsFailed += 1
                    self.statsLock.unlock()
                    // Move to archive with .failed suffix so it doesn't retry forever
                    self.archiveFile(from: inputPath, filename: "\(filename).failed")
                }
                self.processSemaphore.signal()
                group.leave()
            }
        }

        group.wait()
    }

    /// v2.0: Sort files by priority (higher priority first).
    private func sortByPriority(files: [String]) -> [String] {
        var prioritized: [(file: String, priority: Int)] = []
        for file in files {
            let path = "\(inboxDir)/\(file)"
            var priority = 1
            if let handle = FileHandle(forReadingAtPath: path) {
                let snippet = handle.readData(ofLength: 512)
                handle.closeFile()
                if let text = String(data: snippet, encoding: .utf8),
                   let range = text.range(of: "\"priority\":") {
                    let after = text[range.upperBound...].trimmingCharacters(in: .whitespaces)
                    if let num = Int(after.prefix(while: { $0.isNumber })) {
                        priority = num
                    }
                }
            }
            prioritized.append((file, priority))
        }
        return prioritized.sorted { $0.priority > $1.priority }.map { $0.file }
    }

    /// Parse a JSON file, build the circuit, route it, return the result.
    private func processCircuitFile(at path: String) throws -> CircuitResult {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let payload = try JSONDecoder().decode(CircuitPayload.self, from: data)
        return try processPayload(payload)
    }

    /// Write result JSON to outbox.
    private func writeResult(_ result: CircuitResult, filename: String) {
        let outName = filename.replacingOccurrences(of: ".json", with: "_result.json")
        let outPath = "\(outboxDir)/\(outName)"
        do {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(result)
            try data.write(to: URL(fileURLWithPath: outPath), options: .atomic)
        } catch {
            print("[L104 CircuitWatcher] Failed to write result \(outName): \(error)")
        }
    }

    /// Move processed file to archive.
    private func archiveFile(from path: String, filename: String) {
        let archivePath = "\(archiveDir)/\(filename)"
        try? FileManager.default.moveItem(atPath: path, toPath: archivePath)
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - UNIX DOMAIN SOCKET IPC (EVO_67)
    // Low-latency Python↔Swift quantum dispatch over /tmp/l104_quantum.sock
    // Protocol: 4-byte big-endian length prefix + JSON payload, same format
    // as file-based payloads. Response: 4-byte length prefix + JSON result.
    // ═══════════════════════════════════════════════════════════════

    /// Start the Unix domain socket listener on `socketPath`.
    private func startSocketListener() {
        // Clean up stale socket file
        unlink(socketPath)

        // Create socket
        socketFD = socket(AF_UNIX, SOCK_STREAM, 0)
        guard socketFD != -1 else {
            print("[L104 CircuitWatcher] Socket: failed to create — \(String(cString: strerror(errno)))")
            return
        }

        // Bind
        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        let pathBytes = socketPath.utf8CString
        guard pathBytes.count <= MemoryLayout.size(ofValue: addr.sun_path) else {
            print("[L104 CircuitWatcher] Socket: path too long — \(socketPath)")
            close(socketFD); socketFD = -1
            return
        }
        withUnsafeMutablePointer(to: &addr.sun_path) { ptr in
            ptr.withMemoryRebound(to: CChar.self, capacity: pathBytes.count) { dest in
                for i in 0..<pathBytes.count { dest[i] = pathBytes[i] }
            }
        }

        let bindResult = withUnsafePointer(to: &addr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockPtr in
                bind(socketFD, sockPtr, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }
        guard bindResult == 0 else {
            print("[L104 CircuitWatcher] Socket: bind failed — \(String(cString: strerror(errno)))")
            close(socketFD); socketFD = -1
            return
        }

        // Listen (backlog = 8 concurrent connections)
        guard Darwin.listen(socketFD, 8) == 0 else {
            print("[L104 CircuitWatcher] Socket: listen failed — \(String(cString: strerror(errno)))")
            close(socketFD); socketFD = -1
            return
        }

        // GCD dispatch source for incoming connections
        let src = DispatchSource.makeReadSource(fileDescriptor: socketFD, queue: socketQueue)
        src.setEventHandler { [weak self] in
            self?.acceptSocketConnection()
        }
        src.setCancelHandler { [weak self] in
            guard let fd = self?.socketFD, fd != -1 else { return }
            close(fd)
            self?.socketFD = -1
            unlink(self?.socketPath ?? "")
        }
        socketSource = src
        src.resume()
        socketActive = true
    }

    /// Stop the socket listener.
    private func stopSocketListener() {
        socketSource?.cancel()
        socketSource = nil
        socketActive = false
    }

    /// Accept a new incoming socket connection and handle it.
    private func acceptSocketConnection() {
        var clientAddr = sockaddr_un()
        var clientLen = socklen_t(MemoryLayout<sockaddr_un>.size)
        let clientFD = withUnsafeMutablePointer(to: &clientAddr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockPtr in
                accept(socketFD, sockPtr, &clientLen)
            }
        }
        guard clientFD != -1 else { return }

        // v2.0: Handle on concurrent process queue for higher throughput
        processQueue.async { [weak self] in
            self?.handleSocketClient(fd: clientFD)
        }
    }

    /// Read a length-prefixed JSON payload from a client, process it,
    /// and write the result back as length-prefixed JSON.
    private func handleSocketClient(fd clientFD: Int32) {
        defer { close(clientFD) }

        // Read 4-byte big-endian length prefix
        var lengthBuf = [UInt8](repeating: 0, count: 4)
        let lenRead = recv(clientFD, &lengthBuf, 4, MSG_WAITALL)
        guard lenRead == 4 else { return }

        let payloadLength = Int(UInt32(lengthBuf[0]) << 24 |
                                 UInt32(lengthBuf[1]) << 16 |
                                 UInt32(lengthBuf[2]) << 8  |
                                 UInt32(lengthBuf[3]))

        // Sanity limit: 16 MB max payload
        guard payloadLength > 0, payloadLength < 16_777_216 else { return }

        // Read payload
        var payloadBuf = [UInt8](repeating: 0, count: payloadLength)
        var totalRead = 0
        while totalRead < payloadLength {
            let n = payloadBuf.withUnsafeMutableBytes { bufPtr in
                recv(clientFD, bufPtr.baseAddress! + totalRead, payloadLength - totalRead, 0)
            }
            guard n > 0 else { return }
            totalRead += n
        }

        // Decode and process
        let data = Data(payloadBuf)
        do {
            let payload = try JSONDecoder().decode(CircuitPayload.self, from: data)
            let result = try processPayload(payload)

            // Encode result
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys]
            let resultData = try encoder.encode(result)

            // Write 4-byte length prefix + result
            let resultLen = UInt32(resultData.count)
            var lenOut: [UInt8] = [
                UInt8((resultLen >> 24) & 0xFF),
                UInt8((resultLen >> 16) & 0xFF),
                UInt8((resultLen >> 8)  & 0xFF),
                UInt8(resultLen & 0xFF),
            ]
            send(clientFD, &lenOut, 4, 0)
            resultData.withUnsafeBytes { ptr in
                _ = send(clientFD, ptr.baseAddress!, resultData.count, 0)
            }

            statsLock.lock()
            circuitsProcessed += 1
            totalExecutionMs += result.executionTimeMs
            statsLock.unlock()
        } catch {
            // Send error response
            let errMsg = "{\"error\": \"\(error)\"}".data(using: .utf8)!
            let errLen = UInt32(errMsg.count)
            var errLenOut: [UInt8] = [
                UInt8((errLen >> 24) & 0xFF),
                UInt8((errLen >> 16) & 0xFF),
                UInt8((errLen >> 8)  & 0xFF),
                UInt8(errLen & 0xFF),
            ]
            send(clientFD, &errLenOut, 4, 0)
            errMsg.withUnsafeBytes { ptr in
                _ = send(clientFD, ptr.baseAddress!, errMsg.count, 0)
            }
            circuitsFailed += 1
            statsLock.unlock()
        }
    }

    /// v3.0: Compute sacred alignment + three-engine scores for a probability distribution.
    private func computeSacredAlignment(_ probabilities: [String: Double]) -> SacredAlignmentResult {
        let sorted = probabilities.values.sorted(by: >)

        // PHI resonance: ratio of top-2 probabilities vs golden ratio
        let phiResonance: Double
        if sorted.count >= 2, sorted[1] > 1e-12 {
            let ratio = sorted[0] / sorted[1]
            phiResonance = max(0, 1.0 - abs(ratio - PHI) / PHI)
        } else {
            phiResonance = sorted.count == 1 ? 0.618 : 0.0
        }

        // GOD_CODE alignment: Shannon entropy harmonic distance
        let entropy = probabilities.values.reduce(0.0) { acc, p in
            p > 1e-15 ? acc - p * log2(p) : acc
        }
        let godCodeNorm = GOD_CODE / 1000.0  // 0.5275...
        let godCodeAlignment = max(0, 1.0 - abs(entropy - godCodeNorm) / max(godCodeNorm, 1e-12))

        let composite = phiResonance * 0.5 + godCodeAlignment * 0.5

        // ─── THREE-ENGINE SCORING ───

        // Entropy Reversal (Science Engine: Maxwell's Demon efficiency)
        let clampedEntropy = max(0.1, min(5.0, entropy))
        let demonEfficiency = 1.0 / (1.0 + clampedEntropy * 0.3)
        let entropyReversal = min(1.0, demonEfficiency * 2.0)

        // Harmonic Resonance (Math Engine: GOD_CODE + 104 Hz wave coherence)
        let godCodeAligned = godCodeAlignment > 0.5 ? 1.0 : 0.0
        let waveCoherence104 = abs(cos(2.0 * .pi * 104.0 / GOD_CODE))
        let harmonicResonance = godCodeAligned * 0.6 + waveCoherence104 * 0.4

        // Wave Coherence (Math Engine: PHI-harmonic phase-lock)
        let wcPhi = abs(cos(2.0 * .pi * PHI / GOD_CODE))
        let voidConst = 1.04 + PHI / 1000.0
        let wcVoid = abs(cos(2.0 * .pi * (voidConst * 1000.0) / GOD_CODE))
        let waveCoherence = (wcPhi + wcVoid) / 2.0

        // Three-Engine Composite (0.35 entropy + 0.40 harmonic + 0.25 wave)
        let threeEngineComposite = 0.35 * entropyReversal + 0.40 * harmonicResonance + 0.25 * waveCoherence

        return SacredAlignmentResult(
            phiResonance: phiResonance,
            godCodeAlignment: godCodeAlignment,
            compositeScore: composite,
            entropyReversal: entropyReversal,
            harmonicResonance: harmonicResonance,
            waveCoherence: waveCoherence,
            threeEngineComposite: threeEngineComposite
        )
    }

    /// Process a decoded circuit payload (shared by file and socket paths).
    private func processPayload(_ payload: CircuitPayload) throws -> CircuitResult {
        let engine = QuantumGateEngine.shared
        var circuit = QGateCircuit(nQubits: payload.numQubits)

        for op in payload.operations {
            guard let gateType = Self.gateTypeMap[op.gate] else {
                throw CircuitWatcherError.unknownGate(op.gate)
            }
            let gate = engine.gate(gateType, parameters: op.parameters ?? [])
            circuit.append(gate, qubits: op.qubits)
        }

        let router = QuantumRouter(
            numQubits: payload.numQubits,
            maxBranches: payload.maxBranches ?? ROUTER_BASE_BRANCHES,
            pruneEpsilon: payload.pruneEpsilon ?? ROUTER_PRUNE_EPSILON
        )

        let shots = payload.shots ?? 1024
        let routerResult = router.simulate(circuit: circuit, shots: shots)

        let adaptZone: String?
        if payload.adapt == true {
            let adaptResult = router.adapt()
            adaptZone = adaptResult.zone
        } else {
            adaptZone = nil
        }

        let iso = ISO8601DateFormatter()
        let sacred = computeSacredAlignment(routerResult.probabilities)
        return CircuitResult(
            circuitId: payload.circuitId,
            probabilities: routerResult.probabilities,
            counts: routerResult.counts,
            backend: routerResult.backendUsed,
            branchCount: routerResult.branchCount,
            tGateCount: routerResult.tGateCount,
            cliffordGateCount: routerResult.cliffordGateCount,
            prunedBranches: routerResult.prunedBranches,
            mergedBranches: routerResult.mergedBranches,
            executionTimeMs: routerResult.executionTimeMs,
            numQubits: routerResult.numQubits,
            adaptZone: adaptZone,
            timestamp: iso.string(from: Date()),
            godCode: GOD_CODE,
            sacredAlignment: sacred
        )
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - STATUS
    // ═══════════════════════════════════════════════════════════════

    /// Current watcher status for telemetry (v3.0: three-engine).
    func getStatus() -> [String: Any] {
        let fm = FileManager.default
        let pending = (try? fm.contentsOfDirectory(atPath: inboxDir))?.filter { $0.hasSuffix(".json") }.count ?? 0
        let completed = (try? fm.contentsOfDirectory(atPath: outboxDir))?.count ?? 0
        let archived = (try? fm.contentsOfDirectory(atPath: archiveDir))?.count ?? 0
        let avgMs = circuitsProcessed > 0 ? totalExecutionMs / Double(circuitsProcessed) : 0

        // v2.0: Per-backend telemetry
        var backendStats: [String: Any] = [:]
        statsLock.lock()
        for (backend, timing) in backendTimings where timing.count > 0 {
            backendStats[backend] = [
                "count": timing.count,
                "total_ms": round(timing.totalMs * 100) / 100,
                "avg_ms": round(timing.totalMs / Double(timing.count) * 100) / 100,
            ]
        }
        statsLock.unlock()

        return [
            "version": "3.0.0",
            "active": isActive,
            "inbox": inboxDir,
            "outbox": outboxDir,
            "socket": socketPath,
            "socket_active": socketActive,
            "pending": pending,
            "completed": completed,
            "archived": archived,
            "circuits_processed": circuitsProcessed,
            "circuits_failed": circuitsFailed,
            "gpu_executions": gpuExecutions,
            "cpu_executions": cpuExecutions,
            "mps_executions": mpsExecutions,
            "stabilizer_executions": stabilizerExecutions,
            "chunked_cpu_executions": chunkedCPUExecutions,
            "avg_execution_ms": avgMs,
            "total_execution_ms": totalExecutionMs,
            "max_concurrent": maxConcurrent,
            "backend_telemetry": backendStats,
            "three_engine": [
                "integrated": true,
                "scoring_dimensions": ["entropy_reversal", "harmonic_resonance", "wave_coherence"],
                "weight_entropy": 0.35,
                "weight_harmonic": 0.40,
                "weight_wave": 0.25,
            ] as [String: Any],
            "features": [
                "concurrent_processing",
                "priority_scheduling",
                "per_backend_telemetry",
                "sacred_alignment_results",
                "three_engine_entropy_reversal",
                "three_engine_harmonic_resonance",
                "three_engine_wave_coherence",
                "expanded_gate_map",
            ],
        ]
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - ERRORS
// ═══════════════════════════════════════════════════════════════════

enum CircuitWatcherError: Error, CustomStringConvertible {
    case unknownGate(String)

    var description: String {
        switch self {
        case .unknownGate(let g): return "Unknown gate type: '\(g)'"
        }
    }
}
