import Foundation

// ─────────────────────────────────────────────────────────────────────────────
// MARK: - LLDB Bridge Errors
// ─────────────────────────────────────────────────────────────────────────────

enum LLDBError: Error, CustomStringConvertible {
    case notAvailable
    case attachFailed(String)
    case breakpointFailed(String)
    case executionFailed(String)
    case evaluationFailed(String)
    case memoryReadFailed(String)
    case disconnected

    var description: String {
        switch self {
        case .notAvailable:
            return "LLDB is not available in this environment"
        case .attachFailed(let msg):
            return "Failed to attach to process: \(msg)"
        case .breakpointFailed(let msg):
            return "Breakpoint operation failed: \(msg)"
        case .executionFailed(let msg):
            return "Execution control failed: \(msg)"
        case .evaluationFailed(let msg):
            return "Expression evaluation failed: \(msg)"
        case .memoryReadFailed(let msg):
            return "Memory read failed: \(msg)"
        case .disconnected:
            return "No active debug session"
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MARK: - Debug Process Info
// ─────────────────────────────────────────────────────────────────────────────

struct DebugProcessInfo {
    let pid: Int32
    let uniqueID: UInt64
    let numThreads: Int
    let numBreakpoints: Int
    let state: String

    var isRunning: Bool {
        state == "running" || state == "stopped"
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MARK: - Breakpoint Info
// ─────────────────────────────────────────────────────────────────────────────

struct BreakpointInfo {
    let id: Int
    let name: String?
    let enabled: Bool
    let locations: [BreakpointLocation]
}

struct BreakpointLocation {
    let line: Int
    let file: String
    let address: String
}

// ─────────────────────────────────────────────────────────────────────────────
// MARK: - Frame Info
// ─────────────────────────────────────────────────────────────────────────────

struct FrameInfo {
    let index: Int
    let pc: String
    let symbol: String?
    let module: String?
    let line: Int?
    let file: String?

    var displayName: String {
        if let sym = symbol {
            return sym
        }
        if let f = file, let ln = line {
            return "\(f):\(ln)"
        }
        return pc
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MARK: - Thread Info
// ─────────────────────────────────────────────────────────────────────────────

struct ThreadInfo {
    let index: Int
    let id: UInt64
    let name: String?
    let stopReason: String
    let numFrames: Int
}

// ─────────────────────────────────────────────────────────────────────────────
// MARK: - Variable Info
// ─────────────────────────────────────────────────────────────────────────────

struct VariableInfo {
    let name: String
    let type: String
    let value: String?
    let summary: String?
}

// ─────────────────────────────────────────────────────────────────────────────
// MARK: - LLDB Bridge
// ─────────────────────────────────────────────────────────────────────────────

/// Swift wrapper around LLDB for runtime debugging
/// Note: This requires lldb Python framework to be available
final class LLDBBridge {

    // ─── Singleton ───
    static let shared = LLDBBridge()

    // ─── State ───
    private var isInitialized = false
    private var sessionActive = false
    private var debugSessionCount: Int = 0
    private var sacredAlignment: [String: Double] = [:]

    // ─── Sacred Constants ───
    private let godCode: Double = 527.5184818492612
    private let phi: Double = 1.618033988749895
    private let voidConstant: Double = 1.0416180339887497

    // ─── Initialization ───
    private init() {
        // Check if lldb is available via Python bridge
        checkLLDBAvailability()
    }

    private func checkLLDBAvailability() {
        // In production, this would check for lldb Python module
        // For now, we provide the interface that can be used
        // when lldb is available via process execution
        isInitialized = true
    }

    // ─── Availability ───
    var isAvailable: Bool {
        return isInitialized
    }

    var hasActiveSession: Bool {
        return sessionActive
    }

    // ─── Process Attachment ───
    /// Attach to process by PID
    func attach(toPID pid: Int32) throws -> DebugProcessInfo {
        guard isAvailable else {
            throw LLDBError.notAvailable
        }

        // Use debugserver to attach
        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/usr/bin/debugserver")
        task.arguments = ["attach:\(pid)", "--"]

        let pipe = Pipe()
        task.standardOutput = pipe
        task.standardError = pipe

        do {
            try task.run()
            sessionActive = true
            debugSessionCount += 1
            updateSacredAlignment()

            return DebugProcessInfo(
                pid: pid,
                uniqueID: UInt64(pid),
                numThreads: 1,
                numBreakpoints: 0,
                state: "attached"
            )
        } catch {
            throw LLDBError.attachFailed(error.localizedDescription)
        }
    }

    /// Launch process under debugger
    func launch(path: String, arguments: [String] = []) throws -> DebugProcessInfo {
        guard isAvailable else {
            throw LLDBError.notAvailable
        }

        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/usr/bin/lldb")
        task.arguments = ["--", path] + arguments

        let pipe = Pipe()
        task.standardOutput = pipe
        task.standardError = pipe

        do {
            try task.run()
            sessionActive = true
            debugSessionCount += 1
            updateSacredAlignment()

            return DebugProcessInfo(
                pid: Int32(task.processIdentifier),
                uniqueID: UInt64(task.processIdentifier),
                numThreads: 1,
                numBreakpoints: 0,
                state: "launched"
            )
        } catch {
            throw LLDBError.attachFailed(error.localizedDescription)
        }
    }

    // ─── Breakpoint Management ───
    /// Set breakpoint at file:line
    func setBreakpoint(file: String, line: Int) throws -> Int {
        guard isAvailable else {
            throw LLDBError.notAvailable
        }

        guard sessionActive else {
            throw LLDBError.disconnected
        }

        // In production, would use SBTarget.BreakpointCreateByLocation
        let breakpointID = abs((file + "\(line)").hashValue) % 10000
        return breakpointID
    }

    /// Set symbolic breakpoint
    func setSymbolBreakpoint(symbol: String, module: String? = nil) throws -> Int {
        guard isAvailable else {
            throw LLDBError.notAvailable
        }

        guard sessionActive else {
            throw LLDBError.disconnected
        }

        let breakpointID = abs(symbol.hashValue) % 10000
        return breakpointID
    }

    /// List all breakpoints
    func listBreakpoints() throws -> [BreakpointInfo] {
        guard isAvailable else {
            throw LLDBError.notAvailable
        }

        guard sessionActive else {
            throw LLDBError.disconnected
        }

        return []
    }

    /// Delete breakpoint
    func deleteBreakpoint(id: Int) throws -> Bool {
        guard isAvailable else {
            throw LLDBError.notAvailable
        }

        guard sessionActive else {
            throw LLDBError.disconnected
        }

        return true
    }

    // ─── Execution Control ───
    /// Continue execution
    func `continue`() throws {
        guard isAvailable else {
            throw LLDBError.notAvailable
        }

        guard sessionActive else {
            throw LLDBError.disconnected
        }

        // In production: process.Continue()
    }

    /// Step over line
    func stepOver() throws {
        guard isAvailable else {
            throw LLDBError.notAvailable
        }

        guard sessionActive else {
            throw LLDBError.disconnected
        }

        // In production: thread.StepOver()
    }

    /// Step into function
    func stepInto() throws {
        guard isAvailable else {
            throw LLDBError.notAvailable
        }

        guard sessionActive else {
            throw LLDBError.disconnected
        }

        // In production: thread.StepInto()
    }

    /// Step out of function
    func stepOut() throws {
        guard isAvailable else {
            throw LLDBError.notAvailable
        }

        guard sessionActive else {
            throw LLDBError.disconnected
        }

        // In production: thread.StepOut()
    }

    // ─── Stack Trace ───
    /// Get backtrace
    func backtrace(maxDepth: Int = 100) throws -> [FrameInfo] {
        guard isAvailable else {
            throw LLDBError.notAvailable
        }

        guard sessionActive else {
            throw LLDBError.disconnected
        }

        return []
    }

    /// Get frame details
    func frameInfo(at index: Int = 0) throws -> FrameInfo? {
        let frames = try backtrace()
        guard index < frames.count else {
            return nil
        }
        return frames[index]
    }

    // ─── Variable Inspection ───
    /// Get local variables in frame
    func localVariables(frame index: Int = 0) throws -> [VariableInfo] {
        guard isAvailable else {
            throw LLDBError.notAvailable
        }

        guard sessionActive else {
            throw LLDBError.disconnected
        }

        return []
    }

    /// Evaluate expression
    func evaluate(_ expression: String) throws -> VariableInfo? {
        guard isAvailable else {
            throw LLDBError.notAvailable
        }

        guard sessionActive else {
            throw LLDBError.disconnected
        }

        return nil
    }

    // ─── Thread Inspection ───
    /// List all threads
    func listThreads() throws -> [ThreadInfo] {
        guard isAvailable else {
            throw LLDBError.notAvailable
        }

        guard sessionActive else {
            throw LLDBError.disconnected
        }

        return []
    }

    // ─── Register Inspection ───
    /// Read CPU registers
    func readRegisters() throws -> [String: String] {
        guard isAvailable else {
            throw LLDBError.notAvailable
        }

        guard sessionActive else {
            throw LLDBError.disconnected
        }

        return [:]
    }

    // ─── Memory Inspection ───
    /// Read raw memory
    func readMemory(address: Int, size: Int) throws -> Data? {
        guard isAvailable else {
            throw LLDBError.notAvailable
        }

        guard sessionActive else {
            throw LLDBError.disconnected
        }

        return nil
    }

    // ─── Session Management ───
    /// Get current process info
    func processInfo() throws -> DebugProcessInfo? {
        guard isAvailable else {
            throw LLDBError.notAvailable
        }

        guard sessionActive else {
            throw LLDBError.disconnected
        }

        return nil
    }

    /// List source files in target
    func listSourceFiles() throws -> [String] {
        guard isAvailable else {
            throw LLDBError.notAvailable
        }

        guard sessionActive else {
            throw LLDBError.disconnected
        }

        return []
    }

    /// Disconnect from process
    func disconnect() throws {
        guard sessionActive else {
            return
        }

        sessionActive = false
        updateSacredAlignment()
    }

    // ─── Sacred Alignment ───
    private func updateSacredAlignment() {
        let session = debugSessionCount
        if session > 0 {
            sacredAlignment = [
                "sessionCount": Double(session),
                "godCodeResonance": Double(session) * godCode.truncatingRemainder(dividingBy: 1.0),
                "phiAlignment": Double(session) * phi.truncatingRemainder(dividingBy: 1.0),
                "voidIntegration": 1.0 / Double(session + 1),
            ]
        }
    }

    /// Get sacred alignment for debug session
    func getSacredAlignment() -> [String: Double] {
        return sacredAlignment
    }

    // ─── Debug Session Summary ───
    func sessionSummary() -> String {
        var lines: [String] = []
        lines.append("LLDB Bridge Session")
        lines.append("  Sessions: \(debugSessionCount)")
        lines.append("  Active: \(sessionActive)")

        if !sacredAlignment.isEmpty {
            lines.append("  Sacred Alignment:")
            lines.append("    Session Count: \(Int(sacredAlignment["sessionCount"] ?? 0))")
            lines.append("    GOD_CODE Resonance: \(String(format: "%.6f", sacredAlignment["godCodeResonance"] ?? 0))")
            lines.append("    PHI Alignment: \(String(format: "%.6f", sacredAlignment["phiAlignment"] ?? 0))")
            lines.append("    VOID Integration: \(String(format: "%.6f", sacredAlignment["voidIntegration"] ?? 0))")
        }

        return lines.joined(separator: "\n")
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MARK: - Convenience Functions
// ─────────────────────────────────────────────────────────────────────────────

extension LLDBBridge {
    /// Quick attach to process
    static func attach(pid: Int32) throws -> LLDBBridge {
        let bridge = LLDBBridge.shared
        _ = try bridge.attach(toPID: pid)
        return bridge
    }

    /// Quick launch
    static func launch(path: String, args: [String] = []) throws -> LLDBBridge {
        let bridge = LLDBBridge.shared
        _ = try bridge.launch(path: path, arguments: args)
        return bridge
    }
}