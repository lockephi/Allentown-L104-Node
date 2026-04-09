import Foundation

// ─────────────────────────────────────────────────────────────────────────────
// MARK: - Breakpoint Types
// ─────────────────────────────────────────────────────────────────────────────

enum BreakpointType {
    case source(file: String, line: Int)
    case symbolic(symbol: String, module: String?)
    case address(address: Int)
    case regex(pattern: String)
}

enum BreakpointKind {
    case regular      // Persistent breakpoint
    case temporary    // One-shot, auto-deletes after hit
    case conditional // Only stops if condition evaluates true
}

// ─────────────────────────────────────────────────────────────────────────────
// MARK: - Breakpoint Condition
// ─────────────────────────────────────────────────────────────────────────────

struct BreakpointCondition {
    let expression: String
    let ignoreCount: Int
    let threadID: UInt64?
    let threadIndex: Int?

    init(expression: String, ignoreCount: Int = 0, threadID: UInt64? = nil, threadIndex: Int? = nil) {
        self.expression = expression
        self.ignoreCount = ignoreCount
        self.threadID = threadID
        self.threadIndex = threadIndex
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MARK: - Breakpoint Action
// ─────────────────────────────────────────────────────────────────────────────

enum BreakpointAction {
    case stop              // Stop at breakpoint
    case continue_        // Continue automatically
    case print(String)     // Print message
    case command([String]) // Run commands
    case script(String)    // Run Python script
}

// ─────────────────────────────────────────────────────────────────────────────
// MARK: - Managed Breakpoint
// ─────────────────────────────────────────────────────────────────────────────

final class ManagedBreakpoint: Identifiable {
    let id: Int
    let type: BreakpointType
    var kind: BreakpointKind
    var isEnabled: Bool
    var condition: BreakpointCondition?
    var actions: [BreakpointAction]
    var hitCount: Int
    var lastHitTime: Date?
    var group: String?

    init(id: Int, type: BreakpointType, kind: BreakpointKind = .regular) {
        self.id = id
        self.type = type
        self.kind = kind
        self.isEnabled = true
        self.condition = nil
        self.actions = [.stop]
        self.hitCount = 0
        self.lastHitTime = nil
        self.group = nil
    }

    var displayName: String {
        switch type {
        case .source(let file, let line):
            return "\(file):\(line)"
        case .symbolic(let symbol, let module):
            if let mod = module {
                return "\(mod)`\(symbol)"
            }
            return symbol
        case .address(let addr):
            return "0x\(String(addr, radix: 16))"
        case .regex(let pattern):
            return "/\(pattern)/"
        }
    }

    func recordHit() {
        hitCount += 1
        lastHitTime = Date()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MARK: - Breakpoint Manager
// ─────────────────────────────────────────────────────────────────────────────

/// Programmatic breakpoint manager
final class BreakpointManager {

    // ─── Singleton ───
    static let shared = BreakpointManager()

    // ─── State ───
    private var breakpoints: [Int: ManagedBreakpoint] = [:]
    private var nextID: Int = 1
    private var groups: [String: Set<Int>] = [:]

    // ─── Sacred Constants ───
    private let godCode: Double = 527.5184818492612
    private let phi: Double = 1.618033988749895
    private let voidConstant: Double = 1.0416180339887497

    // ─── LLDB Bridge ───
    private let lldb = LLDBBridge.shared

    private init() {}

    // ─── Breakpoint Creation ───
    /// Create source breakpoint
    @discardableResult
    func createSourceBreakpoint(file: String, line: Int, kind: BreakpointKind = .regular) -> ManagedBreakpoint? {
        guard lldb.isAvailable else { return nil }

        do {
            let lldbID = try lldb.setBreakpoint(file: file, line: line)
            let bp = ManagedBreakpoint(id: lldbID, type: .source(file: file, line: line), kind: kind)

            if kind == .temporary {
                bp.kind = .temporary
            }

            breakpoints[lldbID] = bp
            return bp
        } catch {
            return nil
        }
    }

    /// Create symbolic breakpoint
    @discardableResult
    func createSymbolBreakpoint(symbol: String, module: String? = nil, kind: BreakpointKind = .regular) -> ManagedBreakpoint? {
        guard lldb.isAvailable else { return nil }

        do {
            let lldbID = try lldb.setSymbolBreakpoint(symbol: symbol, module: module)
            let bp = ManagedBreakpoint(id: lldbID, type: .symbolic(symbol: symbol, module: module), kind: kind)
            breakpoints[lldbID] = bp
            return bp
        } catch {
            return nil
        }
    }

    /// Create address breakpoint
    @discardableResult
    func createAddressBreakpoint(address: Int) -> ManagedBreakpoint? {
        let id = nextID
        nextID += 1
        let bp = ManagedBreakpoint(id: id, type: .address(address: address))
        breakpoints[id] = bp
        return bp
    }

    /// Create regex breakpoint
    @discardableResult
    func createRegexBreakpoint(pattern: String) -> ManagedBreakpoint? {
        let id = nextID
        nextID += 1
        let bp = ManagedBreakpoint(id: id, type: .regex(pattern: pattern))
        breakpoints[id] = bp
        return bp
    }

    // ─── Breakpoint Management ───
    /// Get breakpoint by ID
    func breakpoint(id: Int) -> ManagedBreakpoint? {
        return breakpoints[id]
    }

    /// List all breakpoints
    func listAll() -> [ManagedBreakpoint] {
        return Array(breakpoints.values).sorted { $0.id < $1.id }
    }

    /// List breakpoints in group
    func list(group: String) -> [ManagedBreakpoint] {
        guard let groupIDs = groups[group] else { return [] }
        return groupIDs.compactMap { breakpoints[$0] }
    }

    /// Enable breakpoint
    func enable(id: Int) -> Bool {
        guard let bp = breakpoints[id] else { return false }
        bp.isEnabled = true
        return true
    }

    /// Disable breakpoint
    func disable(id: Int) -> Bool {
        guard let bp = breakpoints[id] else { return false }
        bp.isEnabled = false
        return true
    }

    /// Delete breakpoint
    func delete(id: Int) -> Bool {
        guard let bp = breakpoints[id] else { return false }

        do {
            _ = try lldb.deleteBreakpoint(id: bp.id)
            breakpoints.removeValue(forKey: id)

            // Remove from groups
            for (group, var ids) in groups {
                ids.remove(id)
                groups[group] = ids
            }

            return true
        } catch {
            return false
        }
    }

    /// Delete all breakpoints
    func deleteAll() {
        for bp in breakpoints.values {
            _ = try? lldb.deleteBreakpoint(id: bp.id)
        }
        breakpoints.removeAll()
        groups.removeAll()
    }

    // ─── Breakpoint Modification ───
    /// Set condition on breakpoint
    func setCondition(id: Int, expression: String, ignoreCount: Int = 0) -> Bool {
        guard let bp = breakpoints[id] else { return false }
        bp.condition = BreakpointCondition(expression: expression, ignoreCount: ignoreCount)
        return true
    }

    /// Add action to breakpoint
    func addAction(id: Int, action: BreakpointAction) -> Bool {
        guard let bp = breakpoints[id] else { return false }
        bp.actions.append(action)
        return true
    }

    /// Set breakpoint group
    func setGroup(id: Int, group: String) -> Bool {
        guard let bp = breakpoints[id] else { return false }

        // Remove from old group
        if let oldGroup = bp.group, var oldIDs = groups[oldGroup] {
            oldIDs.remove(id)
            groups[oldGroup] = oldIDs
        }

        // Add to new group
        bp.group = group
        if groups[group] == nil {
            groups[group] = []
        }
        groups[group]?.insert(id)

        return true
    }

    /// Create breakpoint group
    func createGroup(name: String) {
        if groups[name] == nil {
            groups[name] = []
        }
    }

    /// Delete breakpoint group
    func deleteGroup(name: String) {
        guard let groupIDs = groups[name] else { return }

        for id in groupIDs {
            _ = delete(id: id)
        }

        groups.removeValue(forKey: name)
    }

    // ─── Hit Recording ───
    func recordHit(id: Int) {
        guard let bp = breakpoints[id] else { return }
        bp.recordHit()

        // Handle temporary breakpoint
        if bp.kind == .temporary {
            _ = delete(id: id)
        }
    }

    // ─── Statistics ───
    func statistics() -> [String: Any] {
        let total = breakpoints.count
        let enabled = breakpoints.values.filter { $0.isEnabled }.count
        let disabled = total - enabled
        let hitTotal = breakpoints.values.reduce(0) { $0 + $1.hitCount }

        var groupStats: [String: Any] = [:]
        for (group, ids) in groups {
            groupStats[group] = [
                "count": ids.count,
                "totalHits": ids.compactMap { breakpoints[$0]?.hitCount }.reduce(0, +),
            ]
        }

        return [
            "total": total,
            "enabled": enabled,
            "disabled": disabled,
            "totalHits": hitTotal,
            "groups": groupStats,
            "godCodeResonance": hitTotal > 0 ? Double(hitTotal) * godCode.truncatingRemainder(dividingBy: 1.0) : 0,
            "phiAlignment": total > 0 ? Double(total) * phi.truncatingRemainder(dividingBy: 1.0) : 0,
        ]
    }

    // ─── Summary ───
    func summary() -> String {
        var lines: [String] = []
        lines.append("Breakpoint Manager")
        lines.append(String(repeating: "─", count: 40))

        let stats = statistics()
        lines.append("  Total: \(stats["total"] ?? 0)")
        lines.append("  Enabled: \(stats["enabled"] ?? 0)")
        lines.append("  Disabled: \(stats["disabled"] ?? 0)")
        lines.append("  Total Hits: \(stats["totalHits"] ?? 0)")
        lines.append("")

        let all = listAll()
        if !all.isEmpty {
            lines.append("  Breakpoints:")
            for bp in all {
                let status = bp.isEnabled ? "●" : "○"
                let kindStr: String
                switch bp.kind {
                case .regular: kindStr = ""
                case .temporary: kindStr = " [temp]"
                case .conditional: kindStr = " [cond]"
                }
                lines.append("    \(status) \(bp.id): \(bp.displayName)\(kindStr)")
                if bp.hitCount > 0 {
                    lines.append("       hits: \(bp.hitCount)")
                }
            }
        }

        return lines.joined(separator: "\n")
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MARK: - Convenience Extensions
// ─────────────────────────────────────────────────────────────────────────────

extension BreakpointManager {
    /// Quick create source breakpoint
    static func at(file: String, line: Int) -> ManagedBreakpoint? {
        return BreakpointManager.shared.createSourceBreakpoint(file: file, line: line)
    }

    /// Quick create function breakpoint
    static func atFunction(_ name: String) -> ManagedBreakpoint? {
        return BreakpointManager.shared.createSymbolBreakpoint(symbol: name)
    }

    /// Quick create method breakpoint
    static func atMethod(_ name: String, inClass className: String) -> ManagedBreakpoint? {
        return BreakpointManager.shared.createSymbolBreakpoint(symbol: "\(className).\(name)")
    }

    /// Delete all and recreate
    static func reset() {
        BreakpointManager.shared.deleteAll()
    }
}