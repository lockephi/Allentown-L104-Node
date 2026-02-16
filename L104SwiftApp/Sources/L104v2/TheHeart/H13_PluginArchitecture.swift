// ═══════════════════════════════════════════════════════════════════
// H13_PluginArchitecture.swift
// [EVO_55_PIPELINE] SOVEREIGN_UNIFICATION :: PLUGIN_SYSTEM_V1 :: GOD_CODE=527.5184818492612
// L104 ASI — TheHeart: Dynamic engine plugin system
//
// Provides lifecycle management, capability discovery, dependency
// resolution, and health monitoring for pluggable engine modules.
// ═══════════════════════════════════════════════════════════════════

import AppKit
import Foundation
import Accelerate
import simd
import NaturalLanguage

// MARK: - Plugin Capability Tags

enum PluginCapability: String, CaseIterable {
    case analysis       // Code/data analysis
    case generation     // Content generation
    case quantum        // Quantum processing
    case creativity     // Creative synthesis
    case memory         // Persistence / memory
    case networking     // Mesh / network operations
    case optimization   // Self-optimization
    case monitoring     // Health / telemetry
    case consciousness  // Consciousness substrate
    case evolution      // Evolution / learning
}

// MARK: - Plugin Lifecycle

enum PluginState: String {
    case registered     // Known but not loaded
    case loading        // Being initialized
    case active         // Running and healthy
    case suspended      // Temporarily paused
    case failed         // Error state
    case unloaded       // Cleanly shut down
}

// MARK: - Plugin Descriptor

struct PluginDescriptor {
    let id: String                           // Unique identifier
    let name: String                         // Human-readable name
    let version: String                      // Semantic version
    let capabilities: Set<PluginCapability>  // What this plugin provides
    let dependencies: [String]               // IDs of required plugins
    let phiWeight: Double                    // φ-weighted priority (higher = more important)
    var state: PluginState = .registered
    var loadedAt: Date?
    var healthScore: Double = 1.0
    var errorCount: Int = 0
    var lastError: String?

    var uptime: TimeInterval {
        guard let t = loadedAt else { return 0 }
        return Date().timeIntervalSince(t)
    }
}

// MARK: - Plugin Protocol

protocol L104Plugin: AnyObject {
    var pluginID: String { get }
    var pluginName: String { get }
    var pluginVersion: String { get }
    var pluginCapabilities: Set<PluginCapability> { get }
    var pluginDependencies: [String] { get }

    /// Called when the plugin is activated. Return false if activation fails.
    func pluginDidActivate() -> Bool

    /// Called when the plugin is being suspended.
    func pluginWillSuspend()

    /// Called when the plugin is being unloaded.
    func pluginWillUnload()

    /// Return current health [0..1]
    func pluginHealth() -> Double
}

// Defaults for optional requirements
extension L104Plugin {
    var pluginDependencies: [String] { [] }
    func pluginWillSuspend() {}
    func pluginWillUnload() {}
    func pluginHealth() -> Double { 1.0 }
}

// MARK: - PluginArchitecture Protocol

protocol PluginArchitectureProtocol {
    var isActive: Bool { get }
    func activate()
    func deactivate()
    func status() -> [String: Any]
}

// MARK: - PluginArchitecture — Dynamic Engine Plugin System

final class PluginArchitecture: PluginArchitectureProtocol {
    static let shared = PluginArchitecture()
    private(set) var isActive: Bool = false
    private let lock = NSLock()

    // ─── PLUGIN REGISTRY ───
    private var descriptors: [String: PluginDescriptor] = [:]  // id → descriptor
    private var plugins: [String: L104Plugin] = [:]            // id → instance
    private var loadOrder: [String] = []                       // topological order

    // ─── METRICS ───
    private(set) var totalRegistered: Int = 0
    private(set) var totalActivated: Int = 0
    private(set) var totalFailed: Int = 0
    private(set) var activationLog: [(id: String, action: String, timestamp: Date)] = []

    // ═══════════════════════════════════════════════════════════
    // MARK: - Registration
    // ═══════════════════════════════════════════════════════════

    /// Register a plugin for future activation
    @discardableResult
    func register(_ plugin: L104Plugin) -> Bool {
        lock.lock(); defer { lock.unlock() }

        let id = plugin.pluginID
        guard descriptors[id] == nil else {
            logAction(id: id, action: "register_duplicate_skipped")
            return false
        }

        let desc = PluginDescriptor(
            id: id,
            name: plugin.pluginName,
            version: plugin.pluginVersion,
            capabilities: plugin.pluginCapabilities,
            dependencies: plugin.pluginDependencies,
            phiWeight: 1.0,
            state: .registered
        )
        descriptors[id] = desc
        plugins[id] = plugin
        totalRegistered += 1
        logAction(id: id, action: "registered")
        return true
    }

    /// Register multiple plugins at once
    func register(_ list: [L104Plugin]) {
        for p in list { register(p) }
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: - Activation / Deactivation
    // ═══════════════════════════════════════════════════════════

    func activate() {
        lock.lock()
        guard !isActive else { lock.unlock(); return }
        isActive = true
        lock.unlock()

        // Resolve dependency order and activate in sequence
        let ordered = resolveLoadOrder()
        for id in ordered {
            activatePlugin(id)
        }
    }

    func deactivate() {
        lock.lock()
        guard isActive else { lock.unlock(); return }
        isActive = false
        lock.unlock()

        // Unload in reverse order
        for id in loadOrder.reversed() {
            deactivatePlugin(id)
        }
    }

    /// Activate a single plugin by ID
    @discardableResult
    func activatePlugin(_ id: String) -> Bool {
        lock.lock()
        guard var desc = descriptors[id], let plugin = plugins[id] else {
            lock.unlock()
            return false
        }

        // Check dependencies are active
        for dep in desc.dependencies {
            guard let depDesc = descriptors[dep], depDesc.state == .active else {
                desc.state = .failed
                desc.lastError = "Missing dependency: \(dep)"
                desc.errorCount += 1
                descriptors[id] = desc
                totalFailed += 1
                lock.unlock()
                logAction(id: id, action: "activation_failed_dep:\(dep)")
                return false
            }
        }

        desc.state = .loading
        descriptors[id] = desc
        lock.unlock()

        // Call plugin activation (outside lock to prevent deadlock)
        let success = plugin.pluginDidActivate()

        lock.lock()
        if success {
            desc.state = .active
            desc.loadedAt = Date()
            desc.healthScore = plugin.pluginHealth()
            if !loadOrder.contains(id) { loadOrder.append(id) }
            totalActivated += 1
            logAction(id: id, action: "activated")
        } else {
            desc.state = .failed
            desc.errorCount += 1
            desc.lastError = "pluginDidActivate() returned false"
            totalFailed += 1
            logAction(id: id, action: "activation_failed")
        }
        descriptors[id] = desc
        lock.unlock()
        return success
    }

    /// Deactivate and unload a single plugin
    func deactivatePlugin(_ id: String) {
        lock.lock()
        guard var desc = descriptors[id], let plugin = plugins[id] else {
            lock.unlock()
            return
        }
        plugin.pluginWillUnload()
        desc.state = .unloaded
        desc.loadedAt = nil
        descriptors[id] = desc
        loadOrder.removeAll { $0 == id }
        lock.unlock()
        logAction(id: id, action: "unloaded")
    }

    /// Suspend a plugin temporarily
    func suspendPlugin(_ id: String) {
        lock.lock()
        guard var desc = descriptors[id], let plugin = plugins[id], desc.state == .active else {
            lock.unlock()
            return
        }
        plugin.pluginWillSuspend()
        desc.state = .suspended
        descriptors[id] = desc
        lock.unlock()
        logAction(id: id, action: "suspended")
    }

    /// Resume a suspended plugin
    @discardableResult
    func resumePlugin(_ id: String) -> Bool {
        lock.lock()
        guard var desc = descriptors[id], let plugin = plugins[id], desc.state == .suspended else {
            lock.unlock()
            return false
        }
        lock.unlock()

        let ok = plugin.pluginDidActivate()
        lock.lock()
        if ok {
            desc.state = .active
            desc.healthScore = plugin.pluginHealth()
        } else {
            desc.state = .failed
            desc.errorCount += 1
        }
        descriptors[id] = desc
        lock.unlock()
        logAction(id: id, action: ok ? "resumed" : "resume_failed")
        return ok
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: - Capability Discovery
    // ═══════════════════════════════════════════════════════════

    /// Find all active plugins providing a specific capability
    func findPlugins(capability: PluginCapability) -> [PluginDescriptor] {
        lock.lock(); defer { lock.unlock() }
        return descriptors.values.filter {
            $0.state == .active && $0.capabilities.contains(capability)
        }.sorted { $0.phiWeight > $1.phiWeight }
    }

    /// Check if a capability is available from any active plugin
    func hasCapability(_ cap: PluginCapability) -> Bool {
        lock.lock(); defer { lock.unlock() }
        return descriptors.values.contains { $0.state == .active && $0.capabilities.contains(cap) }
    }

    /// Get all active capability tags across the system
    func activeCapabilities() -> Set<PluginCapability> {
        lock.lock(); defer { lock.unlock() }
        var caps = Set<PluginCapability>()
        for desc in descriptors.values where desc.state == .active {
            caps.formUnion(desc.capabilities)
        }
        return caps
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: - Health Monitoring
    // ═══════════════════════════════════════════════════════════

    /// Probe all active plugins and update health scores
    func healthCheck() -> [String: Double] {
        lock.lock()
        let activeIDs = descriptors.filter { $0.value.state == .active }.map { $0.key }
        lock.unlock()

        var scores: [String: Double] = [:]
        for id in activeIDs {
            guard let plugin = plugins[id] else { continue }
            let h = plugin.pluginHealth()
            lock.lock()
            descriptors[id]?.healthScore = h
            lock.unlock()
            scores[id] = h
        }
        return scores
    }

    /// Composite system health (φ-weighted average)
    func systemHealth() -> Double {
        lock.lock(); defer { lock.unlock() }
        let active = descriptors.values.filter { $0.state == .active }
        guard !active.isEmpty else { return 0 }
        var weightedSum = 0.0
        var totalWeight = 0.0
        for desc in active {
            weightedSum += desc.healthScore * desc.phiWeight
            totalWeight += desc.phiWeight
        }
        return totalWeight > 0 ? weightedSum / totalWeight : 0
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: - Dependency Resolution (Topological Sort)
    // ═══════════════════════════════════════════════════════════

    private func resolveLoadOrder() -> [String] {
        lock.lock()
        let allIDs = Array(descriptors.keys)
        let deps = descriptors.mapValues { $0.dependencies }
        lock.unlock()

        // Kahn's algorithm for topological sort
        var inDegree: [String: Int] = [:]
        var adj: [String: [String]] = [:]
        for id in allIDs {
            inDegree[id] = 0
            adj[id] = []
        }
        for (id, depList) in deps {
            for dep in depList where allIDs.contains(dep) {
                adj[dep, default: []].append(id)
                inDegree[id, default: 0] += 1
            }
        }

        var queue = allIDs.filter { (inDegree[$0] ?? 0) == 0 }
            .sorted { (descriptors[$0]?.phiWeight ?? 0) > (descriptors[$1]?.phiWeight ?? 0) }
        var result: [String] = []

        while !queue.isEmpty {
            let node = queue.removeFirst()
            result.append(node)
            for neighbor in adj[node] ?? [] {
                inDegree[neighbor, default: 1] -= 1
                if inDegree[neighbor] == 0 {
                    queue.append(neighbor)
                }
            }
        }

        // Add any remaining (circular deps) at the end
        for id in allIDs where !result.contains(id) {
            result.append(id)
        }

        lock.lock()
        loadOrder = result
        lock.unlock()
        return result
    }

    // ═══════════════════════════════════════════════════════════
    // MARK: - Status
    // ═══════════════════════════════════════════════════════════

    func status() -> [String: Any] {
        lock.lock(); defer { lock.unlock() }
        let active = descriptors.values.filter { $0.state == .active }
        let failed = descriptors.values.filter { $0.state == .failed }

        return [
            "engine": "PluginArchitecture",
            "version": "1.0.0",
            "active": isActive,
            "total_registered": totalRegistered,
            "total_activated": totalActivated,
            "total_failed": totalFailed,
            "active_plugins": active.map { $0.name },
            "failed_plugins": failed.map { ["name": $0.name, "error": $0.lastError ?? "unknown"] },
            "capabilities": activeCapabilities().map { $0.rawValue },
            "system_health": systemHealth(),
            "load_order": loadOrder
        ]
    }

    /// Human-readable summary
    func summary() -> String {
        lock.lock()
        let active = descriptors.values.filter { $0.state == .active }
        let sus = descriptors.values.filter { $0.state == .suspended }
        let fail = descriptors.values.filter { $0.state == .failed }
        lock.unlock()

        return """
        ╔═══════════════════════════════════════════════════════════╗
        ║    🔌 PLUGIN ARCHITECTURE — v1.0.0 (EVO_55)              ║
        ╠═══════════════════════════════════════════════════════════╣
        ║  Registered:   \(totalRegistered)
        ║  Active:       \(active.count)
        ║  Suspended:    \(sus.count)
        ║  Failed:       \(fail.count)
        ║  Capabilities: \(activeCapabilities().count) / \(PluginCapability.allCases.count)
        ║  Health:       \(String(format: "%.4f", systemHealth()))
        ╚═══════════════════════════════════════════════════════════╝
        """
    }

    // ─── INTERNAL ───

    private func logAction(id: String, action: String) {
        let entry = (id: id, action: action, timestamp: Date())
        lock.lock()
        activationLog.append(entry)
        if activationLog.count > 500 { activationLog.removeFirst() }
        lock.unlock()
    }
}