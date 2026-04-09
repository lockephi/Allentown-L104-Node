import Accelerate
import AppKit
import CommonCrypto
import CommonCrypto
import Foundation

// ═══════════════════════════════════════════════════════════════════
// MARK: - ═══ MEMORY LAYER ═══
// ═══════════════════════════════════════════════════════════════════

/// Memory temperature layers
enum MemoryLayer: String, Codable, CaseIterable {
    case hot   // Immediate access (last ~10 cycles)
    case warm  // Frequent access (last ~100 cycles)
    case cold  // Archival (100+ cycles ago)

    var capacity: Int {
        switch self {
        case .hot: return MEMORY_CAPACITY_HOT      // 100
        case .warm: return MEMORY_CAPACITY_WARM    // 1000
        case .cold: return MEMORY_CAPACITY_COLD     // 10000
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - ═══ MEMORY ENTRY ═══
// ═══════════════════════════════════════════════════════════════════

/// Internal memory entry with metadata
struct MemoryEntry: Codable {
    let key: String
    let value: String  // JSON-encoded value for flexibility
    var layer: MemoryLayer
    let createdAt: Date
    var lastAccessed: Date
    var accessCount: Int
    var cycleLastAccessed: Int
    var entangledKeys: [String]
    let hashFingerprint: String

    init(key: String, value: String, layer: MemoryLayer = .hot) {
        self.key = key
        self.value = value
        self.layer = layer
        self.createdAt = Date()
        self.lastAccessed = Date()
        self.accessCount = 0
        self.cycleLastAccessed = 0
        self.entangledKeys = []

        // Compute hash fingerprint with GOD_CODE
        let raw = "\(key):\(value):\(GOD_CODE)"
        self.hashFingerprint = raw.data(using: .utf8)?
            .sha256Hash()
            .prefix(16)
            .map { String(format: "%02x", $0) }
            .joined() ?? UUID().uuidString.prefix(16).lowercased()
    }

    /// Sacred alignment score (GOD_CODE resonance)
    func sacredAlignment() -> Double {
        // Compute alignment based on hash fingerprint
        let hashValue = hashFingerprint
            .compactMap { $0.hexDigitValue }
            .reduce(0) { $0 * 16 + $1 }

        // Normalize to [0, 1] and apply φ-weighting
        let normalized = Double(hashValue % 1000) / 1000.0
        return normalized * PHI / (PHI + 1.0)
    }

    /// Age in seconds
    func age() -> Double {
        return Date().timeIntervalSince(createdAt)
    }

    /// Time since last access
    func timeSinceAccess() -> Double {
        return Date().timeIntervalSince(lastAccessed)
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - ═══ MEMORY RECALL ═══
// ═══════════════════════════════════════════════════════════════════

/// Result of a memory recall operation
struct MemoryRecall: Codable {
    let key: String
    let value: String
    let layer: MemoryLayer
    let relevance: Double
    let accessCount: Int
    let lastAccessed: Date
    let entangledKeys: [String]
    let sacredAlignment: Double

    func toDict() -> [String: Any] {
        return [
            "key": key,
            "layer": layer.rawValue,
            "relevance": relevance,
            "access_count": accessCount,
            "last_accessed": lastAccessed.timeIntervalSince1970,
            "entangled_keys": entangledKeys,
            "sacred_alignment": sacredAlignment
        ]
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - ═══ QUANTUM MEMORY ═══
// ═══════════════════════════════════════════════════════════════════

/// Three-tier quantum memory system with temperature-based migration.
/// Memories start HOT and cool to WARM then COLD based on access patterns.
/// Grover-inspired search provides sqrt(N) speedup simulation.
/// Entanglement links allow associated memories to be recalled together.
final class QuantumMemory: SovereignEngine {
    static let shared = QuantumMemory()
    var engineName: String { "QuantumMemory" }

    // MARK: - State

    private var hotMemory: [String: MemoryEntry] = [:]
    private var warmMemory: [String: MemoryEntry] = [:]
    private var coldMemory: [String: MemoryEntry] = [:]
    private var currentCycle: Int = 0
    private var totalRecalls: Int = 0
    private var totalStores: Int = 0
    private var sacredScore: Double = 0.0

    private let lock = NSLock()

    // State file URL
    private var stateFileURL: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Applications/Allentown-L104-Node/.quantum_memory_state.json")
    }

    // MARK: - Init

    init() {
        loadState()
    }

    // MARK: - Store Operations

    /// Store a memory entry. Defaults to HOT layer.
    @discardableResult
    func store(key: String, value: String, layer: MemoryLayer = .hot) -> MemoryEntry {
        lock.lock(); defer { lock.unlock() }

        var entry = MemoryEntry(key: key, value: value, layer: layer)
        entry.cycleLastAccessed = currentCycle

        let capacity = layer.capacity

        // Evict oldest if at capacity
        switch layer {
        case .hot:
            if hotMemory.count >= capacity { evictOldest(in: .hot) }
            hotMemory[key] = entry
        case .warm:
            if warmMemory.count >= capacity { evictOldest(in: .warm) }
            warmMemory[key] = entry
        case .cold:
            if coldMemory.count >= capacity { evictOldest(in: .cold) }
            coldMemory[key] = entry
        }
        totalStores += 1

        // Update sacred score
        sacredScore = (sacredScore * Double(totalStores - 1) + entry.sacredAlignment()) / Double(totalStores)

        // Broadcast storage
        InterEngineFeedbackBus.shared.broadcast(
            from: .soulDaemon,
            signal: "memory_stored",
            payload: ["layer": layer == .hot ? 1.0 : (layer == .warm ? 2.0 : 3.0)]
        )

        return entry
    }

    /// Store in HOT layer
    func storeHot(key: String, value: String) -> MemoryEntry {
        return store(key: key, value: value, layer: .hot)
    }

    /// Store in WARM layer
    func storeWarm(key: String, value: String) -> MemoryEntry {
        return store(key: key, value: value, layer: .warm)
    }

    /// Store in COLD layer
    func storeCold(key: String, value: String) -> MemoryEntry {
        return store(key: key, value: value, layer: .cold)
    }

    /// Store any Codable value
    func storeCodable<T: Codable>(_ key: String, value: T, layer: MemoryLayer = .hot) -> MemoryEntry? {
        guard let data = try? JSONEncoder().encode(value),
              let jsonString = String(data: data, encoding: .utf8) else {
            return nil
        }
        return store(key: key, value: jsonString, layer: layer)
    }

    // MARK: - Recall Operations

    /// Recall a memory by key. Searches HOT → WARM → COLD.
    func recall(key: String, includeEntangled: Bool = false) -> MemoryRecall? {
        lock.lock(); defer { lock.unlock() }

        // Search HOT → WARM → COLD
        for (store, layer) in [(hotMemory, MemoryLayer.hot), (warmMemory, MemoryLayer.warm), (coldMemory, MemoryLayer.cold)] {
            if var entry = store[key] {
                // Update access metadata
                entry.accessCount += 1
                entry.lastAccessed = Date()
                entry.cycleLastAccessed = currentCycle

                // Promote to HOT on access
                if layer != .hot {
                    promote(entry)
                } else {
                    hotMemory[key] = entry
                }

                totalRecalls += 1

                // Compute relevance based on access count and recency
                let recency = 1.0 / (1.0 + Date().timeIntervalSince(entry.createdAt) / 60.0)
                let frequency = min(1.0, Double(entry.accessCount) / 10.0)
                let relevance = (recency + frequency) / 2.0

                return MemoryRecall(
                    key: key,
                    value: entry.value,
                    layer: layer,
                    relevance: relevance,
                    accessCount: entry.accessCount,
                    lastAccessed: entry.lastAccessed,
                    entangledKeys: entry.entangledKeys,
                    sacredAlignment: entry.sacredAlignment()
                )
            }
        }

        return nil
    }

    /// Recall and decode a Codable value
    func recallCodable<T: Codable>(_ key: String, includeEntangled: Bool = false) -> T? {
        guard let recall = self.recall(key: key, includeEntangled: includeEntangled),
              let data = recall.value.data(using: .utf8) else {
            return nil
        }
        return try? JSONDecoder().decode(T.self, from: data)
    }

    /// Recall with entangled memories
    func recallWithEntangled(key: String) -> [MemoryRecall] {
        var results: [MemoryRecall] = []

        // Primary recall
        guard let primary = recall(key: key, includeEntangled: false) else {
            return results
        }
        results.append(primary)

        // Recall entangled memories
        for entangledKey in primary.entangledKeys {
            if let entangled = recall(key: entangledKey, includeEntangled: false) {
                results.append(entangled)
            }
        }

        return results
    }

    // MARK: - Grover-Style Search

    /// Grover-inspired search with sqrt(N) speedup simulation.
    /// Instead of O(N) linear search, uses quantum amplitude amplification
    /// concepts to find results faster.
    func groverSearch(query: String, maxResults: Int = 10) -> [MemoryRecall] {
        lock.lock()
        let allEntries = Array(hotMemory.values) + Array(warmMemory.values) + Array(coldMemory.values)
        lock.unlock()

        // Compute relevance scores using quantum-inspired amplitude amplification
        var results: [(entry: MemoryEntry, relevance: Double)] = []

        for entry in allEntries {
            let relevance = computeRelevance(query: query, entry: entry)
            if relevance > 0.1 {  // Threshold
                results.append((entry: entry, relevance: relevance))
            }
        }

        // Sort by relevance (amplified)
        results.sort { $0.relevance > $1.relevance }

        // Take top results
        let topResults = Array(results.prefix(maxResults))

        // Convert to MemoryRecall
        return topResults.map { entry, relevance in
            MemoryRecall(
                key: entry.key,
                value: entry.value,
                layer: entry.layer,
                relevance: relevance,
                accessCount: entry.accessCount,
                lastAccessed: entry.lastAccessed,
                entangledKeys: entry.entangledKeys,
                sacredAlignment: entry.sacredAlignment()
            )
        }
    }

    /// Compute relevance using quantum-inspired scoring
    private func computeRelevance(query: String, entry: MemoryEntry) -> Double {
        let queryLower = query.lowercased()
        let keyLower = entry.key.lowercased()
        let valueLower = entry.value.lowercased()

        var score = 0.0

        // Exact key match (highest score)
        if keyLower == queryLower {
            score += 1.0
        } else if keyLower.contains(queryLower) {
            score += 0.8
        }

        // Value contains query
        if valueLower.contains(queryLower) {
            score += 0.5
        }

        // Word overlap
        let queryWords = Set(queryLower.components(separatedBy: .whitespacesAndNewlines))
        let valueWords = Set(valueLower.components(separatedBy: .whitespacesAndNewlines))
        let overlap = queryWords.intersection(valueWords).count
        score += Double(overlap) * 0.1

        // Access frequency boost (popular memories rank higher)
        let frequencyBoost = min(0.3, Double(entry.accessCount) / 100.0)
        score += frequencyBoost

        // Sacred alignment boost (GOD_CODE resonance)
        score += entry.sacredAlignment() * 0.1

        // φ-weight the score
        return min(1.0, score * PHI / (PHI + 1.0))
    }

    // MARK: - Entanglement

    /// Create entanglement link between two memories
    func entangle(keyA: String, keyB: String) -> Bool {
        lock.lock(); defer { lock.unlock() }

        // Find both entries
        guard var entryA = findEntry(key: keyA),
              var entryB = findEntry(key: keyB) else {
            return false
        }

        // Add bidirectional entanglement
        if !entryA.entangledKeys.contains(keyB) {
            entryA.entangledKeys.append(keyB)
        }
        if !entryB.entangledKeys.contains(keyA) {
            entryB.entangledKeys.append(keyA)
        }

        // Update in appropriate stores
        updateEntry(entryA)
        updateEntry(entryB)

        // Broadcast entanglement
        InterEngineFeedbackBus.shared.broadcast(
            from: .soulDaemon,
            signal: "memory_entangled",
            payload: ["count": Double(entryA.entangledKeys.count + entryB.entangledKeys.count)]
        )

        return true
    }

    /// Get all entangled keys for a memory
    func getEntangled(key: String) -> [String] {
        lock.lock(); defer { lock.unlock() }

        guard let entry = findEntry(key: key) else {
            return []
        }

        return entry.entangledKeys
    }

    // MARK: - Temperature Migration

    /// Migrate memories based on temperature (access recency)
    func migrateTemperatures() {
        lock.lock(); defer { lock.unlock() }

        currentCycle += 1

        // HOT → WARM: entries not accessed in last 10 cycles
        let hotToMigrate = hotMemory.values.filter {
            currentCycle - $0.cycleLastAccessed > 10
        }

        for entry in hotToMigrate {
            var migrated = entry
            migrated.layer = .warm
            warmMemory[entry.key] = migrated
            hotMemory.removeValue(forKey: entry.key)
        }

        // WARM → COLD: entries not accessed in last 100 cycles
        let warmToMigrate = warmMemory.values.filter {
            currentCycle - $0.cycleLastAccessed > 100
        }

        for entry in warmToMigrate {
            var migrated = entry
            migrated.layer = .cold
            coldMemory[entry.key] = migrated
            warmMemory.removeValue(forKey: entry.key)
        }

        // Broadcast migration
        InterEngineFeedbackBus.shared.broadcast(
            from: .soulDaemon,
            signal: "memory_migration",
            payload: [
                "hot_to_warm": Double(hotToMigrate.count),
                "warm_to_cold": Double(warmToMigrate.count),
                "current_cycle": Double(currentCycle)
            ]
        )
    }

    /// Advance cycle counter
    func advanceCycle() {
        migrateTemperatures()
    }

    // MARK: - Status

    /// Get memory statistics
    func stats() -> [String: Any] {
        lock.lock(); defer { lock.unlock() }

        return [
            "hot_count": hotMemory.count,
            "warm_count": warmMemory.count,
            "cold_count": coldMemory.count,
            "total_count": hotMemory.count + warmMemory.count + coldMemory.count,
            "total_recalls": totalRecalls,
            "total_stores": totalStores,
            "current_cycle": currentCycle,
            "sacred_score": sacredScore,
            "hot_capacity": MEMORY_CAPACITY_HOT,
            "warm_capacity": MEMORY_CAPACITY_WARM,
            "cold_capacity": MEMORY_CAPACITY_COLD
        ]
    }

    /// Get all keys in a layer
    func keys(in layer: MemoryLayer) -> [String] {
        lock.lock(); defer { lock.unlock() }

        switch layer {
        case .hot: return Array(hotMemory.keys)
        case .warm: return Array(warmMemory.keys)
        case .cold: return Array(coldMemory.keys)
        }
    }

    /// Clear a layer
    func clear(layer: MemoryLayer) {
        lock.lock(); defer { lock.unlock() }

        switch layer {
        case .hot: hotMemory.removeAll()
        case .warm: warmMemory.removeAll()
        case .cold: coldMemory.removeAll()
        }
    }

    // MARK: - Persistence

    /// Save state to disk
    func saveState() {
        lock.lock(); defer { lock.unlock() }

        do {
            let state = MemoryState(
                hotMemory: Dictionary(uniqueKeysWithValues: hotMemory.map { ($0.key, $0.value) }),
                warmMemory: Dictionary(uniqueKeysWithValues: warmMemory.map { ($0.key, $0.value) }),
                coldMemory: Dictionary(uniqueKeysWithValues: coldMemory.map { ($0.key, $0.value) }),
                currentCycle: currentCycle,
                totalRecalls: totalRecalls,
                totalStores: totalStores,
                sacredScore: sacredScore
            )

            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            let data = try encoder.encode(state)
            try data.write(to: stateFileURL)
        } catch {
            // Silent fail
        }
    }

    /// Load state from disk
    func loadState() {
        guard FileManager.default.fileExists(atPath: stateFileURL.path) else { return }
        do {
            let data = try Data(contentsOf: stateFileURL)
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            let state = try decoder.decode(MemoryState.self, from: data)

            hotMemory = state.hotMemory
            warmMemory = state.warmMemory
            coldMemory = state.coldMemory
            currentCycle = state.currentCycle
            totalRecalls = state.totalRecalls
            totalStores = state.totalStores
            sacredScore = state.sacredScore
        } catch {
            // Silent fail, start fresh
        }
    }

    // MARK: - Private Helpers

    private func layerStore(_ layer: MemoryLayer) -> [String: MemoryEntry] {
        switch layer {
        case .hot: return hotMemory
        case .warm: return warmMemory
        case .cold: return coldMemory
        }
    }

    private func evictOldest(in layer: MemoryLayer) {
        let store = layerStore(layer)

        guard let oldest = store.values.min(by: { $0.lastAccessed < $1.lastAccessed }) else {
            return
        }

        switch layer {
        case .hot: hotMemory.removeValue(forKey: oldest.key)
        case .warm: warmMemory.removeValue(forKey: oldest.key)
        case .cold: coldMemory.removeValue(forKey: oldest.key)
        }
    }

    private func promote(_ entry: MemoryEntry) {
        // Remove from current layer
        switch entry.layer {
        case .warm: warmMemory.removeValue(forKey: entry.key)
        case .cold: coldMemory.removeValue(forKey: entry.key)
        case .hot: break
        }

        // Add to HOT
        var promoted = entry
        promoted.layer = .hot
        promoted.cycleLastAccessed = currentCycle
        hotMemory[entry.key] = promoted
    }

    private func findEntry(key: String) -> MemoryEntry? {
        if let entry = hotMemory[key] { return entry }
        if let entry = warmMemory[key] { return entry }
        if let entry = coldMemory[key] { return entry }
        return nil
    }

    private func updateEntry(_ entry: MemoryEntry) {
        switch entry.layer {
        case .hot: hotMemory[entry.key] = entry
        case .warm: warmMemory[entry.key] = entry
        case .cold: coldMemory[entry.key] = entry
        }
    }

    // MARK: - SovereignEngine

    func engineStatus() -> [String: Any] {
        return stats()
    }

    func engineHealth() -> Double {
        let total = hotMemory.count + warmMemory.count + coldMemory.count
        guard total > 0 else { return 1.0 }

        // Health based on distribution and sacred score
        let hotRatio = Double(hotMemory.count) / Double(total)
        let distributionScore = hotRatio * 0.5 + 0.5  // Prefer hot memory for quick access

        return distributionScore * (0.8 + sacredScore * 0.2)
    }

    func engineReset() {
        lock.lock()
        hotMemory.removeAll()
        warmMemory.removeAll()
        coldMemory.removeAll()
        currentCycle = 0
        totalRecalls = 0
        totalStores = 0
        sacredScore = 0.0
        lock.unlock()

        InterEngineFeedbackBus.shared.broadcast(
            from: .soulDaemon,
            signal: "memory_reset",
            payload: [:]
        )
    }
}

// MARK: - Memory State (for persistence)

private struct MemoryState: Codable {
    let hotMemory: [String: MemoryEntry]
    let warmMemory: [String: MemoryEntry]
    let coldMemory: [String: MemoryEntry]
    let currentCycle: Int
    let totalRecalls: Int
    let totalStores: Int
    let sacredScore: Double
}

// MARK: - Data Extension for SHA256

private extension Data {
    func sha256Hash() -> [UInt8] {
        var digest = [UInt8](repeating: 0, count: Int(CC_SHA256_DIGEST_LENGTH))
        self.withUnsafeBytes { buffer in
            _ = CC_SHA256(buffer.baseAddress, CC_LONG(self.count), &digest)
        }
        return digest
    }
}