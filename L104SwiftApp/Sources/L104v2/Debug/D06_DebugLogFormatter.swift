import Foundation

// ─────────────────────────────────────────────────────────────────────────────
// MARK: - Log Level
// ─────────────────────────────────────────────────────────────────────────────

enum LogLevel: Int, Comparable, Codable {
    case debug = 0
    case info = 1
    case warning = 2
    case error = 3
    case critical = 4

    static func < (lhs: LogLevel, rhs: LogLevel) -> Bool {
        return lhs.rawValue < rhs.rawValue
    }

    var name: String {
        switch self {
        case .debug: return "DEBUG"
        case .info: return "INFO"
        case .warning: return "WARN"
        case .error: return "ERROR"
        case .critical: return "CRITICAL"
        }
    }

    var emoji: String {
        switch self {
        case .debug: return "🔍"
        case .info: return "ℹ️"
        case .warning: return "⚠️"
        case .error: return "❌"
        case .critical: return "🚨"
        }
    }

    var color: String {
        switch self {
        case .debug: return "\u{001B}[90m"  // Gray
        case .info: return "\u{001B}[36m"   // Cyan
        case .warning: return "\u{001B}[33m" // Yellow
        case .error: return "\u{001B}[31m"  // Red
        case .critical: return "\u{001B}[35m" // Magenta
        }
    }

    var ansiReset: String {
        return "\u{001B}[0m"
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MARK: - Log Category
// ─────────────────────────────────────────────────────────────────────────────

indirect enum LogCategory: Hashable {
    case root(name: String)
    case child(name: String, parent: LogCategory)

    static let general = LogCategory.root(name: "general")
    static let network = LogCategory.child(name: "network", parent: general)
    static let database = LogCategory.child(name: "database", parent: general)
    static let quantum = LogCategory.child(name: "quantum", parent: general)
    static let ui = LogCategory.child(name: "ui", parent: general)
    static let concurrency = LogCategory.child(name: "concurrency", parent: general)
    static let memory = LogCategory.child(name: "memory", parent: general)

    var name: String {
        switch self {
        case .root(let name): return name
        case .child(let name, _): return name
        }
    }

    var path: String {
        var parts: [String] = [name]
        var current = parent
        while let p = current {
            parts.insert(p.name, at: 0)
            current = p.parent
        }
        return parts.joined(separator: ".")
    }

    var parent: LogCategory? {
        switch self {
        case .root: return nil
        case .child(_, let parent): return parent
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MARK: - Log Entry
// ─────────────────────────────────────────────────────────────────────────────

struct LogEntry: Codable {
    let timestamp: Date
    let level: LogLevel
    let category: String
    let message: String
    let file: String?
    let line: Int?
    let function: String?
    let threadID: UInt64?
    let threadName: String?
    let sacredAlignment: [String: Double]?
    let metadata: [String: String]?

    init(
        level: LogLevel,
        category: LogCategory,
        message: String,
        file: String? = nil,
        line: Int? = nil,
        function: String? = nil,
        threadID: UInt64? = nil,
        threadName: String? = nil,
        sacredAlignment: [String: Double]? = nil,
        metadata: [String: String]? = nil
    ) {
        self.timestamp = Date()
        self.level = level
        self.category = category.path
        self.message = message
        self.file = file
        self.line = line
        self.function = function
        self.threadID = threadID
        self.threadName = threadName
        self.sacredAlignment = sacredAlignment
        self.metadata = metadata
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MARK: - Debug Log Formatter
// ─────────────────────────────────────────────────────────────────────────────

/// Structured debug logging with sacred alignment
final class DebugLogFormatter {

    // ─── Singleton ───
    static let shared = DebugLogFormatter()

    // ─── Configuration ───
    var minimumLevel: LogLevel = .debug
    var enableColors: Bool = true
    var enableTimestamp: Bool = true
    var enableThreadInfo: Bool = true
    var enableSacredAlignment: Bool = true

    // ─── Sacred Constants ───
    private let godCode: Double = 527.5184818492612
    private let phi: Double = 1.618033988749895
    private let voidConstant: Double = 1.0416180339887497
    private let omega: Double = 6539.34712682

    // ─── State ───
    private var logCount: Int = 0
    private var entries: [LogEntry] = []
    private let maxEntries: Int = 10000

    // ─── Output ───
    private let dateFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd HH:mm:ss.SSS"
        return f
    }()

    private init() {}

    // ─── Logging ───
    /// Log a message
    func log(
        _ level: LogLevel,
        category: LogCategory,
        message: String,
        file: String = #file,
        line: Int = #line,
        function: String = #function,
        metadata: [String: String]? = nil
    ) {
        guard level >= minimumLevel else { return }

        let threadID = enableThreadInfo ? UInt64(pthread_mach_thread_np(pthread_self())) : nil
        let threadName = enableThreadInfo ? Thread.current.name : nil

        let sacred: [String: Double]?
        if enableSacredAlignment {
            sacred = calculateSacredAlignment()
        } else {
            sacred = nil
        }

        let entry = LogEntry(
            level: level,
            category: category,
            message: message,
            file: file,
            line: line,
            function: function,
            threadID: threadID,
            threadName: threadName,
            sacredAlignment: sacred,
            metadata: metadata
        )

        addEntry(entry)
        output(entry)
    }

    // ─── Convenience Methods ───
    func debug(_ message: String, category: LogCategory = .general, metadata: [String: String]? = nil) {
        log(.debug, category: category, message: message, metadata: metadata)
    }

    func info(_ message: String, category: LogCategory = .general, metadata: [String: String]? = nil) {
        log(.info, category: category, message: message, metadata: metadata)
    }

    func warning(_ message: String, category: LogCategory = .general, metadata: [String: String]? = nil) {
        log(.warning, category: category, message: message, metadata: metadata)
    }

    func error(_ message: String, category: LogCategory = .general, metadata: [String: String]? = nil) {
        log(.error, category: category, message: message, metadata: metadata)
    }

    func critical(_ message: String, category: LogCategory = .general, metadata: [String: String]? = nil) {
        log(.critical, category: category, message: message, metadata: metadata)
    }

    // ─── Entry Management ───
    private func addEntry(_ entry: LogEntry) {
        logCount += 1
        entries.append(entry)

        if entries.count > maxEntries {
            entries.removeFirst(entries.count - maxEntries)
        }
    }

    /// Get entries by level
    func entries(level: LogLevel) -> [LogEntry] {
        return entries.filter { $0.level == level }
    }

    /// Get entries by category
    func entries(category: LogCategory) -> [LogEntry] {
        return entries.filter { $0.category.hasPrefix(category.path) }
    }

    /// Get recent entries
    func recentEntries(count: Int = 100) -> [LogEntry] {
        return Array(entries.suffix(count))
    }

    // ─── Sacred Alignment ───
    private func calculateSacredAlignment() -> [String: Double] {
        let session = logCount
        return [
            "sessionCount": Double(session),
            "godCodeResonance": Double(session) * godCode.truncatingRemainder(dividingBy: 1.0),
            "phiAlignment": Double(session) * phi.truncatingRemainder(dividingBy: 1.0),
            "voidIntegration": 1.0 / Double(session + 1),
            "omegaHarmonic": omega / Double(session + 1),
        ]
    }

    // ─── Output Formatting ───
    private func output(_ entry: LogEntry) {
        let formatted = format(entry)
        print(formatted)
    }

    /// Format entry for display
    func format(_ entry: LogEntry) -> String {
        var parts: [String] = []

        // Timestamp
        if enableTimestamp {
            parts.append(dateFormatter.string(from: entry.timestamp))
        }

        // Level with color
        if enableColors {
            parts.append("\(entry.level.color)[\(entry.level.name)]\(entry.level.ansiReset)")
        } else {
            parts.append("[\(entry.level.name)]")
        }

        // Category
        parts.append("[\(entry.category)]")

        // Thread info
        if enableThreadInfo, let threadName = entry.threadName {
            parts.append("[\(threadName)]")
        }

        // Message
        parts.append(entry.message)

        // Sacred alignment
        if enableSacredAlignment, let sacred = entry.sacredAlignment {
            parts.append("|φ:\(String(format: "%.4f", sacred["phiAlignment"] ?? 0))")
        }

        return parts.joined(separator: " ")
    }

    /// Format entry as JSON
    func formatJSON(_ entry: LogEntry) -> String? {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]

        guard let data = try? encoder.encode(entry),
              let json = String(data: data, encoding: .utf8) else {
            return nil
        }

        return json
    }

    /// Format entries as JSON array
    func formatJSON(_ entries: [LogEntry]) -> String? {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]

        guard let data = try? encoder.encode(entries),
              let json = String(data: data, encoding: .utf8) else {
            return nil
        }

        return json
    }

    // ─── Summary ───
    func summary() -> String {
        var lines: [String] = []
        lines.append("Debug Log Formatter")
        lines.append(String(repeating: "─", count: 40))

        let levelCounts = Dictionary(grouping: entries, by: { $0.level })
        lines.append("  Total entries: \(entries.count)")

        for level in [LogLevel.debug, .info, .warning, .error, .critical] {
            let count = levelCounts[level]?.count ?? 0
            if count > 0 {
                lines.append("  \(level.name): \(count)")
            }
        }

        let sacred = calculateSacredAlignment()
        lines.append("")
        lines.append("  Sacred Alignment:")
        lines.append("    Session: \(Int(sacred["sessionCount"] ?? 0))")
        lines.append("    GOD_CODE: \(String(format: "%.6f", sacred["godCodeResonance"] ?? 0))")
        lines.append("    PHI: \(String(format: "%.6f", sacred["phiAlignment"] ?? 0))")

        return lines.joined(separator: "\n")
    }

    /// Clear entries
    func clear() {
        entries.removeAll()
    }

    /// Reset session count
    func reset() {
        entries.removeAll()
        logCount = 0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MARK: - Convenience Global Functions
// ─────────────────────────────────────────────────────────────────────────────

/// Global debug logging
func dlog(_ message: String, category: LogCategory = .general) {
    DebugLogFormatter.shared.debug(message, category: category)
}

func ilog(_ message: String, category: LogCategory = .general) {
    DebugLogFormatter.shared.info(message, category: category)
}

func wlog(_ message: String, category: LogCategory = .general) {
    DebugLogFormatter.shared.warning(message, category: category)
}

func elog(_ message: String, category: LogCategory = .general) {
    DebugLogFormatter.shared.error(message, category: category)
}

func clog(_ message: String, category: LogCategory = .general) {
    DebugLogFormatter.shared.critical(message, category: category)
}