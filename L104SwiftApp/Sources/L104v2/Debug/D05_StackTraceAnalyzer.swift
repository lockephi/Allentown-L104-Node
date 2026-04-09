import Foundation

// ─────────────────────────────────────────────────────────────────────────────
// MARK: - Stack Frame Types
// ─────────────────────────────────────────────────────────────────────────────

enum FrameType: String {
    case swift = "swift"
    case objc = "objc"
    case c = "c"
    case kernel = "kernel"
    case async = "async"
    case unknown = "unknown"
}

// ─────────────────────────────────────────────────────────────────────────────
// MARK: - Analyzed Frame
// ─────────────────────────────────────────────────────────────────────────────

struct AnalyzedFrame: Identifiable {
    let id: Int
    let index: Int
    let address: String
    let symbol: String?
    let module: String?
    let file: String?
    let line: Int?
    let column: Int?
    let frameType: FrameType

    var displayName: String {
        if let sym = symbol {
            return sym
        }
        if let f = file, let ln = line {
            return "\(f):\(ln)"
        }
        return address
    }

    var isInProjectCode: Bool {
        guard let module = module else { return false }
        return module.contains("L104")
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MARK: - Call Graph Node
// ─────────────────────────────────────────────────────────────────────────────

final class CallGraphNode: Identifiable {
    let id = UUID()
    let function: String
    let callCount: Int
    let totalTime: Double
    var children: [CallGraphNode]

    init(function: String, callCount: Int = 1, totalTime: Double = 0) {
        self.function = function
        self.callCount = callCount
        self.totalTime = totalTime
        self.children = []
    }

    func addChild(_ node: CallGraphNode) {
        children.append(node)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MARK: - Stack Trace Analyzer
// ─────────────────────────────────────────────────────────────────────────────

/// Stack trace analysis and symbolication
final class StackTraceAnalyzer {

    // ─── Singleton ───
    static let shared = StackTraceAnalyzer()

    // ─── LLDB Bridge ───
    private let lldb = LLDBBridge.shared

    // ─── Sacred Constants ───
    private let godCode: Double = 527.5184818492612
    private let phi: Double = 1.618033988749895

    private init() {}

    // ─── Parsing ───
    /// Parse raw stack trace string
    func parse(rawTrace: String) -> [AnalyzedFrame] {
        var frames: [AnalyzedFrame] = []

        let lines = rawTrace.components(separatedBy: "\n")
        for (index, line) in lines.enumerated() {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            guard !trimmed.isEmpty else { continue }

            if let frame = parseFrame(line: trimmed, index: index) {
                frames.append(frame)
            }
        }

        return frames
    }

    /// Parse single frame line
    private func parseFrame(line: String, index: Int) -> AnalyzedFrame? {
        // Common formats:
        // 0   L104SwiftApp                      0x0000000100001234 _main + 4
        // 0   L104SwiftApp                      0x0000000100001234 ViewController.viewDidLoad() + 20
        // 0   libswiftCore.dylib                0x00007fff12345678 _assertionFailure() + 100

        // Try address pattern
        let addressPattern = "(0x[0-9a-fA-F]+)"
        guard let addressMatch = line.range(of: addressPattern, options: .regularExpression) else {
            return nil
        }

        let address = String(line[addressMatch])

        // Try to extract symbol
        var symbol: String?
        let module: String? = nil
        let file: String? = nil
        let lineNum: Int? = nil

        // Extract function after address
        let parts = line.components(separatedBy: " ")
        var foundAddress = false

        for i in 0..<parts.count {
            if parts[i].contains("0x") {
                foundAddress = true
                continue
            }

            if foundAddress && !parts[i].isEmpty {
                symbol = parts[i]
                break
            }
        }

        // Determine frame type
        let frameType: FrameType
        if let sym = symbol {
            if sym.contains("async") || sym.contains("await") {
                frameType = .async
            } else if sym.hasPrefix("_") || sym.hasPrefix("swift_") {
                frameType = .swift
            } else if sym.hasPrefix("objc") || sym.contains("objc_msg") {
                frameType = .objc
            } else {
                frameType = .swift
            }
        } else {
            frameType = .unknown
        }

        return AnalyzedFrame(
            id: index,
            index: index,
            address: address,
            symbol: symbol,
            module: module,
            file: file,
            line: lineNum,
            column: nil,
            frameType: frameType
        )
    }

    /// Get backtrace from debugger
    func getBacktrace(maxDepth: Int = 100) -> [AnalyzedFrame] {
        guard lldb.isAvailable && lldb.hasActiveSession else {
            return []
        }

        do {
            let frames = try lldb.backtrace(maxDepth: maxDepth)
            return frames.enumerated().map { i, f in
                AnalyzedFrame(
                    id: i,
                    index: i,
                    address: f.pc,
                    symbol: f.symbol,
                    module: f.module,
                    file: f.file,
                    line: f.line,
                    column: nil,
                    frameType: .unknown
                )
            }
        } catch {
            return []
        }
    }

    // ─── Filtering ───
    /// Filter frames by type
    func filter(frames: [AnalyzedFrame], type: FrameType) -> [AnalyzedFrame] {
        return frames.filter { $0.frameType == type }
    }

    /// Filter frames by module
    func filter(frames: [AnalyzedFrame], module: String) -> [AnalyzedFrame] {
        return frames.filter { $0.module?.contains(module) ?? false }
    }

    /// Get project frames only
    func projectFrames(frames: [AnalyzedFrame]) -> [AnalyzedFrame] {
        return frames.filter { $0.isInProjectCode }
    }

    /// Get system frames only
    func systemFrames(frames: [AnalyzedFrame]) -> [AnalyzedFrame] {
        return frames.filter { !$0.isInProjectCode }
    }

    /// Search for function
    func search(frames: [AnalyzedFrame], pattern: String) -> [AnalyzedFrame] {
        return frames.filter { frame in
            frame.symbol?.contains(pattern) ?? false ||
            frame.module?.contains(pattern) ?? false ||
            frame.file?.contains(pattern) ?? false
        }
    }

    // ─── Analysis ───
    /// Detect recursion
    func detectRecursion(frames: [AnalyzedFrame]) -> [String: Any] {
        var functionCounts: [String: Int] = [:]

        for frame in frames {
            if let symbol = frame.symbol {
                functionCounts[symbol, default: 0] += 1
            }
        }

        let recursive = functionCounts.filter { $0.value > 1 }

        return [
            "recursiveFunctions": recursive.map { ["function": $0.key, "depth": $0.value] },
            "isRecursive": !recursive.isEmpty,
        ]
    }

    /// Analyze call graph
    func analyzeCallGraph(frames: [AnalyzedFrame]) -> CallGraphNode {
        let root = CallGraphNode(function: "root", callCount: 0, totalTime: 0)

        var lastNode = root
        for frame in frames {
            let node = CallGraphNode(
                function: frame.symbol ?? frame.address,
                callCount: 1,
                totalTime: 0
            )
            lastNode.addChild(node)
            lastNode = node
        }

        return root
    }

    /// Find hotspots (frequently called functions)
    func findHotspots(frames: [AnalyzedFrame], minCalls: Int = 2) -> [String: Int] {
        var functionCounts: [String: Int] = [:]

        for frame in frames {
            if let symbol = frame.symbol {
                functionCounts[symbol, default: 0] += 1
            }
        }

        return functionCounts.filter { $0.value >= minCalls }
            .sorted { $0.value > $1.value }
            .reduce(into: [:]) { result, pair in
                result[pair.key] = pair.value
            }
    }

    // ─── Symbolication ───
    /// Symbolicate addresses (requires debug info)
    func symbolicate(frames: [AnalyzedFrame]) -> [AnalyzedFrame] {
        // In production, would use atos or debug info
        return frames
    }

    /// Get source location for frame
    func sourceLocation(frame: AnalyzedFrame) -> (file: String, line: Int)? {
        guard let file = frame.file, let line = frame.line else {
            return nil
        }
        return (file, line)
    }

    // ─── Formatting ───
    /// Format stack trace for display
    func format(frames: [AnalyzedFrame], showProjectOnly: Bool = false) -> String {
        var displayFrames = frames

        if showProjectOnly {
            displayFrames = projectFrames(frames: frames)
        }

        var lines: [String] = []
        lines.append("Stack Trace (\(displayFrames.count) frames)")
        lines.append(String(repeating: "═", count: 60))

        for frame in displayFrames {
            let indent = String(repeating: " ", count: min(frame.index, 10))
            let typeChar: String
            switch frame.frameType {
            case .swift: typeChar = "S"
            case .objc: typeChar = "O"
            case .c: typeChar = "C"
            case .async: typeChar = "A"
            case .kernel: typeChar = "K"
            case .unknown: typeChar = "?"
            }

            lines.append("\(frame.index)[\(typeChar)]\(indent) \(frame.displayName)")

            if let file = frame.file, let line = frame.line {
                lines.append("      at \(file):\(line)")
            }
        }

        return lines.joined(separator: "\n")
    }

    /// Compact format
    func formatCompact(frames: [AnalyzedFrame]) -> String {
        return frames.map { $0.displayName }.joined(separator: " <- ")
    }

    // ─── Summary ───
    func summary(frames: [AnalyzedFrame]) -> String {
        var lines: [String] = []
        lines.append("Stack Trace Analysis")
        lines.append(String(repeating: "═", count: 40))

        lines.append("  Total frames: \(frames.count)")

        let projectFrames = projectFrames(frames: frames)
        let sysFrames = systemFrames(frames: frames)

        lines.append("  Project code: \(projectFrames.count)")
        lines.append("  System code: \(sysFrames.count)")

        let recursion = detectRecursion(frames: frames)
        if let isRecursive = recursion["isRecursive"] as? Bool, isRecursive {
            lines.append("  ⚠ Recursion detected")
        }

        let hotspots = findHotspots(frames: frames)
        if !hotspots.isEmpty {
            lines.append("  Hotspots:")
            for (fn, count) in hotspots.prefix(5) {
                lines.append("    \(fn): \(count)")
            }
        }

        // Sacred alignment
        let sacredScore = Double(frames.count) * godCode.truncatingRemainder(dividingBy: 1.0)
        lines.append("  Sacred alignment: \(String(format: "%.6f", sacredScore))")

        return lines.joined(separator: "\n")
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MARK: - Convenience Extensions
// ─────────────────────────────────────────────────────────────────────────────

extension StackTraceAnalyzer {
    /// Quick parse
    static func parse(_ trace: String) -> [AnalyzedFrame] {
        return StackTraceAnalyzer.shared.parse(rawTrace: trace)
    }

    /// Quick format
    static func format(_ frames: [AnalyzedFrame]) -> String {
        return StackTraceAnalyzer.shared.format(frames: frames)
    }

    /// Quick summary
    static func summarize(_ frames: [AnalyzedFrame]) -> String {
        return StackTraceAnalyzer.shared.summary(frames: frames)
    }
}