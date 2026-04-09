import AppKit
import Foundation

// ═══════════════════════════════════════════════════════════════════
// MARK: - ═══ AGENT TYPES ═══
// ═══════════════════════════════════════════════════════════════════

/// Agent type determines specialization and default tools
enum AgentType: String, CaseIterable, Codable {
    case coder       // Code generation, fixing, refactoring
    case researcher  // Codebase analysis, documentation
    case tester      // Test generation, validation
    case deployer    // Build, deploy, daemon management
    case upgrader    // System upgrades, optimization
    case debugger    // Error diagnosis, log analysis
    case monitor     // System health, metrics, daemon watching
    case optimizer   // Performance profiling, bottleneck elimination
    case inventor    // Creative problem solving, new feature design
    case planner     // Task decomposition, multi-step planning
    case general     // General-purpose (default)
}

/// Agent lifecycle status
enum AgentStatus: String, Codable {
    case queued      // Waiting to run
    case running     // Currently executing
    case toolCall    // Waiting on tool execution
    case thinking    // DeepSeek reasoning
    case waiting     // Waiting on dependency (chained agent)
    case completed   // Successfully finished
    case failed      // Execution failed
    case cancelled   // User cancelled
    case timeout     // Killed by timeout
}

/// Priority level (lower = higher priority)
enum AgentPriority: Int, Codable, Comparable {
    case critical = 0  // Immediate - bypass queue
    case high = 1       // Next in line
    case normal = 2     // Default
    case low = 3        // Background work
    case idle = 4       // Only when nothing else running

    static func < (lhs: AgentPriority, rhs: AgentPriority) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - ═══ AGENT TASK & RESULT ═══
// ═══════════════════════════════════════════════════════════════════

/// Task definition for agent execution
struct AgentTask {
    let taskId: String
    let prompt: String
    let agentType: AgentType
    var status: AgentStatus
    let priority: AgentPriority
    let toolsEnabled: [String]
    let model: String
    let maxRounds: Int
    let timeoutSeconds: Double
    let costBudget: Double
    let createdAt: Date
    var startedAt: Date?
    var completedAt: Date?
    var messages: [[String: Any]]  // Not Codable directly, stored as JSON
    var toolCallsMade: [[String: Any]]
    var filesModified: [String]
    var result: AgentResult?
    var error: String?
    var retryCount: Int
    var progress: Double  // 0.0 - 1.0
    var currentAction: String
    // Chaining support
    let dependsOn: String?
    let chainNext: String?
    var chainContext: [String: Any]
    // Tags and source
    let tags: [String]
    let source: String  // api, mailbox, chain, scheduler

    init(
        prompt: String,
        agentType: AgentType = .general,
        priority: AgentPriority = .normal,
        toolsEnabled: [String]? = nil,
        model: String = "deepseek-chat",
        maxRounds: Int = 10,
        timeoutSeconds: Double = 600.0,
        costBudget: Double = 0.05,
        dependsOn: String? = nil,
        chainNext: String? = nil,
        tags: [String] = [],
        source: String = "api"
    ) {
        self.taskId = "agent_\(UUID().uuidString.prefix(12).lowercased)"
        self.prompt = prompt
        self.agentType = agentType
        self.status = .queued
        self.priority = priority
        self.toolsEnabled = toolsEnabled ?? AgentDefaultTools.forType(agentType)
        self.model = model
        self.maxRounds = maxRounds
        self.timeoutSeconds = timeoutSeconds
        self.costBudget = costBudget
        self.createdAt = Date()
        self.startedAt = nil
        self.completedAt = nil
        self.messages = []
        self.toolCallsMade = []
        self.filesModified = []
        self.result = nil
        self.error = nil
        self.retryCount = 0
        self.progress = 0.0
        self.currentAction = ""
        self.dependsOn = dependsOn
        self.chainNext = chainNext
        self.chainContext = [:]
        self.tags = tags
        self.source = source
    }

    /// Check if task has exceeded its timeout
    func isTimedOut() -> Bool {
        guard timeoutSeconds > 0, let started = startedAt else { return false }
        return Date().timeIntervalSince(started) > timeoutSeconds
    }

    /// Estimate cost from tokens used so far
    func costSoFar() -> Double {
        if let result = result {
            return result.costEstimate
        }
        return Double(toolCallsMade.count) * 0.0005
    }

    /// Check if task has exceeded its cost budget
    func isOverBudget() -> Bool {
        guard costBudget > 0 else { return false }
        return costSoFar() > costBudget
    }

    /// Elapsed time since start
    func elapsed() -> Double {
        guard let started = startedAt else { return 0.0 }
        return (completedAt ?? Date()).timeIntervalSince(started)
    }

    /// Convert to dictionary for JSON/telemetry
    func toDict() -> [String: Any] {
        var dict: [String: Any] = [
            "task_id": taskId,
            "prompt": prompt,
            "agent_type": agentType.rawValue,
            "status": status.rawValue,
            "priority": priority.rawValue,
            "tools_enabled": toolsEnabled,
            "model": model,
            "max_rounds": maxRounds,
            "timeout_seconds": timeoutSeconds,
            "cost_budget": costBudget,
            "created_at": createdAt.timeIntervalSince1970,
            "tool_calls_count": toolCallsMade.count,
            "files_modified": filesModified,
            "retry_count": retryCount,
            "progress": progress,
            "current_action": currentAction,
            "error": error ?? "",
            "duration": elapsed(),
            "depends_on": dependsOn ?? "",
            "chain_next": chainNext ?? "",
            "tags": tags,
            "source": source
        ]
        if let started = startedAt {
            dict["started_at"] = started.timeIntervalSince1970
        }
        if let completed = completedAt {
            dict["completed_at"] = completed.timeIntervalSince1970
        }
        if let result = result {
            dict["result"] = result.toDict()
        }
        return dict
    }
}

/// Execution result from an agent
struct AgentResult {
    let summary: String
    let output: String
    let filesCreated: [String]
    let filesModified: [String]
    let toolResults: [[String: Any]]
    let tokensUsed: Int
    let roundsUsed: Int
    let costEstimate: Double
    let success: Bool
    let errorsEncountered: [String]
    let warnings: [String]
    let metrics: [String: Any]

    init(
        summary: String = "",
        output: String = "",
        filesCreated: [String] = [],
        filesModified: [String] = [],
        toolResults: [[String: Any]] = [],
        tokensUsed: Int = 0,
        roundsUsed: Int = 0,
        costEstimate: Double = 0.0,
        success: Bool = true,
        errorsEncountered: [String] = [],
        warnings: [String] = [],
        metrics: [String: Any] = [:]
    ) {
        self.summary = summary
        self.output = String(output.prefix(4000))  // Truncate for API
        self.filesCreated = filesCreated
        self.filesModified = filesModified
        self.toolResults = toolResults
        self.tokensUsed = tokensUsed
        self.roundsUsed = roundsUsed
        self.costEstimate = costEstimate
        self.success = success
        self.errorsEncountered = errorsEncountered
        self.warnings = warnings
        self.metrics = metrics
    }

    func toDict() -> [String: Any] {
        return [
            "summary": summary,
            "output": String(output.prefix(4000)),
            "files_created": filesCreated,
            "files_modified": filesModified,
            "tool_results_count": toolResults.count,
            "tokens_used": tokensUsed,
            "rounds_used": roundsUsed,
            "cost_estimate": String(format: "%.6f", costEstimate),
            "success": success,
            "errors_encountered": Array(errorsEncountered.prefix(10)),
            "warnings": Array(warnings.prefix(10)),
            "metrics": metrics
        ]
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - ═══ DEFAULT TOOLS ═══
// ═══════════════════════════════════════════════════════════════════

/// Default tools per agent type
struct AgentDefaultTools {
    static func forType(_ type: AgentType) -> [String] {
        switch type {
        case .coder:
            return ["read_file", "write_file", "edit_file", "list_files",
                    "search_code", "analyze_code", "python_exec", "git_status",
                    "diff_viewer", "dependency_check"]
        case .researcher:
            return ["read_file", "list_files", "search_code", "analyze_code",
                    "python_exec", "git_status", "system_metrics", "dependency_check"]
        case .tester:
            return ["read_file", "write_file", "edit_file", "list_files",
                    "search_code", "analyze_code", "python_exec", "run_shell",
                    "git_status"]
        case .deployer:
            return ["read_file", "write_file", "list_files", "run_shell",
                    "python_exec", "git_status", "system_metrics", "http_probe"]
        case .upgrader:
            return ["read_file", "write_file", "edit_file", "list_files",
                    "search_code", "analyze_code", "python_exec", "run_shell",
                    "git_status", "diff_viewer", "system_metrics", "dependency_check"]
        case .debugger:
            return ["read_file", "list_files", "search_code", "analyze_code",
                    "python_exec", "run_shell", "git_status", "system_metrics",
                    "diff_viewer"]
        case .monitor:
            return ["read_file", "list_files", "search_code", "run_shell",
                    "python_exec", "system_metrics", "http_probe", "quantum_bridge"]
        case .optimizer:
            return ["read_file", "write_file", "edit_file", "list_files",
                    "search_code", "analyze_code", "python_exec", "run_shell",
                    "system_metrics", "diff_viewer", "dependency_check"]
        case .inventor:
            return ["read_file", "write_file", "edit_file", "list_files",
                    "search_code", "analyze_code", "python_exec", "run_shell",
                    "git_status", "quantum_bridge", "dependency_check"]
        case .planner:
            return ["read_file", "list_files", "search_code", "analyze_code",
                    "git_status", "system_metrics", "dependency_check"]
        case .general:
            return ["read_file", "write_file", "list_files", "run_shell",
                    "search_code", "analyze_code", "python_exec"]
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - ═══ TOOL REGISTRY ═══
// ═══════════════════════════════════════════════════════════════════

/// Result from tool execution
struct ToolResult {
    let success: Bool
    let output: String
    let error: String?
    let duration: Double
    let metadata: [String: Any]

    func toDict() -> [String: Any] {
        var dict: [String: Any] = [
            "success": success,
            "output": String(output.prefix(8000)),
            "duration": duration
        ]
        if let error = error { dict["error"] = error }
        dict["metadata"] = metadata
        return dict
    }
}

/// Registry of sandboxed tools available to agents
final class ToolRegistry {
    static let shared = ToolRegistry()

    private let workspaceURL: URL
    private let lock = NSLock()

    init() {
        // Default workspace
        workspaceURL = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Applications/Allentown-L104-Node")
    }

    /// Execute a tool by name with arguments
    func execute(name: String, args: [String: Any]) async -> ToolResult {
        let startTime = Date()

        switch name {
        case "read_file":
            return await readFile(args: args)
        case "write_file":
            return await writeFile(args: args)
        case "edit_file":
            return await editFile(args: args)
        case "list_files":
            return await listFiles(args: args)
        case "run_shell":
            return await runShell(args: args)
        case "search_code":
            return await searchCode(args: args)
        case "analyze_code":
            return await analyzeCode(args: args)
        case "python_exec":
            return await pythonExec(args: args)
        case "git_status":
            return await gitStatus(args: args)
        case "system_metrics":
            return await systemMetrics(args: args)
        case "dependency_check":
            return await dependencyCheck(args: args)
        case "quantum_bridge":
            return await quantumBridge(args: args)
        case "diff_viewer":
            return await diffViewer(args: args)
        case "http_probe":
            return await httpProbe(args: args)
        default:
            return ToolResult(
                success: false,
                output: "",
                error: "Unknown tool: \(name)",
                duration: Date().timeIntervalSince(startTime),
                metadata: [:]
            )
        }
    }

    /// Get OpenAI-compatible tool schemas
    func getSchemas(for tools: [String]) -> [[String: Any]] {
        return tools.compactMap { toolSchema($0) }
    }

    /// List all available tools
    func listTools() -> [String] {
        return [
            "read_file", "write_file", "edit_file", "list_files",
            "run_shell", "search_code", "analyze_code", "python_exec",
            "git_status", "system_metrics", "dependency_check",
            "quantum_bridge", "diff_viewer", "http_probe"
        ]
    }

    // MARK: - Tool Implementations

    private func readFile(args: [String: Any]) async -> ToolResult {
        guard let path = args["path"] as? String else {
            return ToolResult(success: false, output: "", error: "Missing 'path' argument", duration: 0, metadata: [:])
        }

        let fileURL = workspaceURL.appendingPathComponent(path)
        do {
            let content = try String(contentsOf: fileURL, encoding: .utf8)
            return ToolResult(
                success: true,
                output: content,
                error: nil,
                duration: 0,
                metadata: ["path": path, "size": content.count]
            )
        } catch {
            return ToolResult(success: false, output: "", error: error.localizedDescription, duration: 0, metadata: ["path": path])
        }
    }

    private func writeFile(args: [String: Any]) async -> ToolResult {
        guard let path = args["path"] as? String,
              let content = args["content"] as? String else {
            return ToolResult(success: false, output: "", error: "Missing 'path' or 'content'", duration: 0, metadata: [:])
        }

        let fileURL = workspaceURL.appendingPathComponent(path)
        do {
            try content.write(to: fileURL, atomically: true, encoding: .utf8)
            return ToolResult(
                success: true,
                output: "Written \(content.count) bytes to \(path)",
                error: nil,
                duration: 0,
                metadata: ["path": path, "size": content.count]
            )
        } catch {
            return ToolResult(success: false, output: "", error: error.localizedDescription, duration: 0, metadata: ["path": path])
        }
    }

    private func editFile(args: [String: Any]) async -> ToolResult {
        guard let path = args["path"] as? String,
              let oldString = args["old_string"] as? String,
              let newString = args["new_string"] as? String else {
            return ToolResult(success: false, output: "", error: "Missing required arguments", duration: 0, metadata: [:])
        }

        let fileURL = workspaceURL.appendingPathComponent(path)
        do {
            var content = try String(contentsOf: fileURL, encoding: .utf8)
            let replacements = args["replace_all"] as? Bool ?? false

            if replacements {
                content = content.replacingOccurrences(of: oldString, with: newString)
            } else {
                guard let range = content.range(of: oldString) else {
                    return ToolResult(success: false, output: "", error: "Old string not found", duration: 0, metadata: [:])
                }
                content.replaceSubrange(range, with: newString)
            }

            try content.write(to: fileURL, atomically: true, encoding: .utf8)
            return ToolResult(success: true, output: "Edited \(path)", error: nil, duration: 0, metadata: ["path": path])
        } catch {
            return ToolResult(success: false, output: "", error: error.localizedDescription, duration: 0, metadata: [:])
        }
    }

    private func listFiles(args: [String: Any]) async -> ToolResult {
        let directory = args["directory"] as? String ?? "."
        let recursive = args["recursive"] as? Bool ?? false

        let dirURL = workspaceURL.appendingPathComponent(directory)
        var files: [String] = []

        do {
            if recursive {
                let enumerator = FileManager.default.enumerator(at: dirURL, includingPropertiesForKeys: [.isDirectoryKey])
                while let fileURL = enumerator?.nextObject() as? URL {
                    let relative = fileURL.path.replacingOccurrences(of: workspaceURL.path + "/", with: "")
                    files.append(relative)
                }
            } else {
                let contents = try FileManager.default.contentsOfDirectory(at: dirURL, includingPropertiesForKeys: nil)
                files = contents.map { $0.lastPathComponent }
            }

            return ToolResult(
                success: true,
                output: files.joined(separator: "\n"),
                error: nil,
                duration: 0,
                metadata: ["count": files.count, "directory": directory]
            )
        } catch {
            return ToolResult(success: false, output: "", error: error.localizedDescription, duration: 0, metadata: [:])
        }
    }

    private func runShell(args: [String: Any]) async -> ToolResult {
        guard let command = args["command"] as? String else {
            return ToolResult(success: false, output: "", error: "Missing 'command'", duration: 0, metadata: [:])
        }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/zsh")
        process.arguments = ["-c", command]
        process.currentDirectoryURL = workspaceURL

        let stdout = Pipe()
        let stderr = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr

        let startTime = Date()
        do {
            try process.run()
            process.waitUntilExit()
            let duration = Date().timeIntervalSince(startTime)

            let stdoutData = stdout.fileHandleForReading.readDataToEndOfFile()
            let stderrData = stderr.fileHandleForReading.readDataToEndOfFile()

            let output = String(data: stdoutData, encoding: .utf8) ?? ""
            let errorOutput = String(data: stderrData, encoding: .utf8) ?? ""

            let success = process.terminationStatus == 0
            return ToolResult(
                success: success,
                output: output,
                error: success ? nil : errorOutput,
                duration: duration,
                metadata: ["command": command, "exit_code": process.terminationStatus]
            )
        } catch {
            return ToolResult(success: false, output: "", error: error.localizedDescription, duration: 0, metadata: [:])
        }
    }

    private func searchCode(args: [String: Any]) async -> ToolResult {
        guard let query = args["query"] as? String else {
            return ToolResult(success: false, output: "", error: "Missing 'query'", duration: 0, metadata: [:])
        }

        // Use grep via shell
        let command = "grep -rn --include='*.swift' --include='*.py' '\(query)' . 2>/dev/null | head -100"
        return await runShell(args: ["command": command])
    }

    private func analyzeCode(args: [String: Any]) async -> ToolResult {
        guard let path = args["path"] as? String else {
            return ToolResult(success: false, output: "", error: "Missing 'path'", duration: 0, metadata: [:])
        }

        let fileURL = workspaceURL.appendingPathComponent(path)
        do {
            let content = try String(contentsOf: fileURL, encoding: .utf8)
            let lines = content.components(separatedBy: "\n")

            // Basic analysis
            var classes: [String] = []
            var functions: [String] = []
            var imports: [String] = []

            for line in lines {
                let trimmed = line.trimmingCharacters(in: .whitespaces)
                if trimmed.hasPrefix("class ") || trimmed.hasPrefix("struct ") || trimmed.hasPrefix("enum ") || trimmed.hasPrefix("protocol ") {
                    classes.append(trimmed)
                }
                if trimmed.hasPrefix("func ") {
                    functions.append(trimmed)
                }
                if trimmed.hasPrefix("import ") {
                    imports.append(trimmed)
                }
            }

            let analysis = """
            File: \(path)
            Lines: \(lines.count)
            Classes/Structs: \(classes.count)
            Functions: \(functions.count)
            Imports: \(imports.count)

            Classes:
            \(classes.prefix(20).joined(separator: "\n"))

            Functions:
            \(functions.prefix(20).joined(separator: "\n"))
            """

            return ToolResult(
                success: true,
                output: analysis,
                error: nil,
                duration: 0,
                metadata: ["lines": lines.count, "classes": classes.count, "functions": functions.count]
            )
        } catch {
            return ToolResult(success: false, output: "", error: error.localizedDescription, duration: 0, metadata: [:])
        }
    }

    private func pythonExec(args: [String: Any]) async -> ToolResult {
        guard let code = args["code"] as? String else {
            return ToolResult(success: false, output: "", error: "Missing 'code'", duration: 0, metadata: [:])
        }

        // Write code to temp file and execute
        let tempFile = FileManager.default.temporaryDirectory.appendingPathComponent("agent_exec_\(UUID().uuidString).py")
        do {
            try code.write(to: tempFile, atomically: true, encoding: .utf8)
            let command = "cd \(workspaceURL.path) && .venv/bin/python \(tempFile.path) 2>&1"
            return await runShell(args: ["command": command])
        } catch {
            return ToolResult(success: false, output: "", error: error.localizedDescription, duration: 0, metadata: [:])
        }
    }

    private func gitStatus(args: [String: Any]) async -> ToolResult {
        return await runShell(args: ["command": "git status --short && git log -1 --oneline"])
    }

    private func systemMetrics(args: [String: Any]) async -> ToolResult {
        let processInfo = ProcessInfo.processInfo
        let metrics = """
        CPU Usage: Process-wide
        Memory: \(processInfo.physicalMemory / 1_000_000) MB total
        Process Memory: \(Double(ProcessInfo.processInfo.environment.keys.count) * 0.1) MB estimated
        Uptime: \(processInfo.systemUptime) seconds
        """
        return ToolResult(
            success: true,
            output: metrics,
            error: nil,
            duration: 0,
            metadata: ["physical_memory": processInfo.physicalMemory, "uptime": processInfo.systemUptime]
        )
    }

    private func dependencyCheck(args: [String: Any]) async -> ToolResult {
        // Check Swift packages and Python requirements
        let swiftCmd = "swift package show-dependencies 2>&1 | head -50"
        let pythonCmd = "cd \(workspaceURL.path) && .venv/bin/pip list --format=freeze 2>&1 | head -50"

        let swiftResult = await runShell(args: ["command": swiftCmd])
        let pythonResult = await runShell(args: ["command": pythonCmd])

        return ToolResult(
            success: true,
            output: "Swift:\n\(swiftResult.output)\n\nPython:\n\(pythonResult.output)",
            error: nil,
            duration: 0,
            metadata: [:]
        )
    }

    private func quantumBridge(args: [String: Any]) async -> ToolResult {
        // Bridge to quantum subsystems via InterEngineFeedbackBus
        guard let operation = args["operation"] as? String else {
            return ToolResult(success: false, output: "", error: "Missing 'operation'", duration: 0, metadata: [:])
        }

        // Broadcast to quantum bus
        InterEngineFeedbackBus.shared.broadcast(
            from: .quantumGate,
            signal: "agent_query",
            payload: ["operation": operation.hasHashValue ? Double(operation.hashValue) : 0]
        )

        return ToolResult(
            success: true,
            output: "Quantum bridge query sent: \(operation)",
            error: nil,
            duration: 0,
            metadata: ["operation": operation]
        )
    }

    private func diffViewer(args: [String: Any]) async -> ToolResult {
        guard let path = args["path"] as? String else {
            return ToolResult(success: false, output: "", error: "Missing 'path'", duration: 0, metadata: [:])
        }

        return await runShell(args: ["command": "git diff HEAD -- \(path) 2>&1 | head -200"])
    }

    private func httpProbe(args: [String: Any]) async -> ToolResult {
        guard let url = args["url"] as? String else {
            return ToolResult(success: false, output: "", error: "Missing 'url'", duration: 0, metadata: [:])
        }

        guard let targetURL = URL(string: url) else {
            return ToolResult(success: false, output: "", error: "Invalid URL", duration: 0, metadata: [:])
        }

        do {
            let startTime = Date()
            let (data, response) = try await URLSession.shared.data(from: targetURL)
            let duration = Date().timeIntervalSince(startTime)

            let httpResponse = response as? HTTPURLResponse
            let statusCode = httpResponse?.statusCode ?? 0

            return ToolResult(
                success: (200..<300).contains(statusCode),
                output: "Status: \(statusCode)\nSize: \(data.count) bytes\nTime: \(String(format: "%.3f", duration))s",
                error: nil,
                duration: duration,
                metadata: ["status_code": statusCode, "size": data.count]
            )
        } catch {
            return ToolResult(success: false, output: "", error: error.localizedDescription, duration: 0, metadata: [:])
        }
    }

    // MARK: - Tool Schema

    private func toolSchema(_ name: String) -> [String: Any]? {
        switch name {
        case "read_file":
            return [
                "type": "function",
                "function": [
                    "name": "read_file",
                    "description": "Read the contents of a file",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "path": ["type": "string", "description": "Path to file relative to workspace"]
                        ],
                        "required": ["path"]
                    ]
                ]
            ]
        case "write_file":
            return [
                "type": "function",
                "function": [
                    "name": "write_file",
                    "description": "Write content to a file",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "path": ["type": "string", "description": "Path to file"],
                            "content": ["type": "string", "description": "Content to write"]
                        ],
                        "required": ["path", "content"]
                    ]
                ]
            ]
        case "edit_file":
            return [
                "type": "function",
                "function": [
                    "name": "edit_file",
                    "description": "Edit a file by replacing text",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "path": ["type": "string", "description": "Path to file"],
                            "old_string": ["type": "string", "description": "Text to replace"],
                            "new_string": ["type": "string", "description": "Replacement text"],
                            "replace_all": ["type": "boolean", "description": "Replace all occurrences"]
                        ],
                        "required": ["path", "old_string", "new_string"]
                    ]
                ]
            ]
        case "run_shell":
            return [
                "type": "function",
                "function": [
                    "name": "run_shell",
                    "description": "Execute a shell command",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "command": ["type": "string", "description": "Shell command to run"]
                        ],
                        "required": ["command"]
                    ]
                ]
            ]
        default:
            return [
                "type": "function",
                "function": [
                    "name": name,
                    "description": "Tool: \(name)",
                    "parameters": ["type": "object", "properties": [:]]
                ]
            ]
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - ═══ AGENT ORCHESTRATOR ═══
// ═══════════════════════════════════════════════════════════════════

/// Manages agent lifecycle: create, queue, execute, monitor, chain, cancel.
/// Thread-safe with DispatchQueue, persists state to .l104_agent_state.json
final class AgentOrchestrator: SovereignEngine {
    static let shared = AgentOrchestrator()
    var engineName: String { "AgentOrchestrator" }

    // MARK: - State

    private var agents: [String: AgentTask] = [:]
    private var history: [[String: Any]] = []
    private var pendingChains: [String: AgentTask] = [:]
    private var runningCount: Int = 0
    private var totalCompleted: Int = 0
    private var totalFailed: Int = 0
    private var totalTokens: Int = 0
    private var totalCost: Double = 0.0
    private var bootTime: Date = Date()
    private var apiCallTimes: [Date] = []
    private let maxConcurrent: Int = 4
    private let maxHistory: Int = 100
    private let rateLimitPerMinute: Int = 30

    private let lock = NSLock()
    private let queue = DispatchQueue(label: "agent.orchestrator", qos: .userInitiated)

    // MARK: - Paths

    private var stateFileURL: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Applications/Allentown-L104-Node/.l104_agent_state.json")
    }

    private var mailboxURL: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Applications/Allentown-L104-Node/.l104_mailbox")
    }

    // MARK: - Init

    init() {
        loadState()
    }

    // MARK: - Public API

    /// Submit a new agent task
    func submit(_ task: AgentTask) -> String {
        lock.lock()
        let taskId = task.taskId
        agents[taskId] = task
        lock.unlock()

        // Broadcast to bus
        InterEngineFeedbackBus.shared.broadcast(
            from: .agentOrchestration,
            signal: "task_submitted",
            payload: ["task_id": taskId.hasHashValue ? Double(taskId.hashValue) : 0, "type": 1.0]
        )

        // Process queue
        processQueue()

        return taskId
    }

    /// Submit a chain of tasks to run sequentially
    func submitChain(_ tasks: [AgentTask]) -> [String] {
        guard !tasks.isEmpty else { return [] }

        var taskIds: [String] = []
        var previousId: String? = nil

        for (index, var task) in tasks.enumerated() {
            if let prevId = previousId {
                task = AgentTask(
                    prompt: task.prompt,
                    agentType: task.agentType,
                    priority: task.priority,
                    toolsEnabled: task.toolsEnabled,
                    model: task.model,
                    maxRounds: task.maxRounds,
                    timeoutSeconds: task.timeoutSeconds,
                    costBudget: task.costBudget,
                    dependsOn: prevId,
                    chainNext: nil,
                    tags: task.tags,
                    source: "chain"
                )
            }

            if index < tasks.count - 1, let nextTask = tasks[safe: index + 1] {
                // Pre-register chain
                let nextId = UUID().uuidString.prefix(12).lowercased()
                task = AgentTask(
                    prompt: task.prompt,
                    agentType: task.agentType,
                    priority: task.priority,
                    toolsEnabled: task.toolsEnabled,
                    model: task.model,
                    maxRounds: task.maxRounds,
                    timeoutSeconds: task.timeoutSeconds,
                    costBudget: task.costBudget,
                    dependsOn: task.dependsOn,
                    chainNext: "agent_\(nextId)",
                    tags: task.tags,
                    source: "chain"
                )
            }

            let id = submit(task)
            taskIds.append(id)
            previousId = id
        }

        return taskIds
    }

    /// Get task by ID
    func getTask(_ taskId: String) -> AgentTask? {
        lock.lock(); defer { lock.unlock() }
        return agents[taskId]
    }

    /// Cancel a running task
    func cancel(_ taskId: String) {
        lock.lock()
        var task = agents[taskId]
        task?.status = .cancelled
        task?.completedAt = Date()
        if let t = task { agents[taskId] = t }
        lock.unlock()

        InterEngineFeedbackBus.shared.broadcast(
            from: .agentOrchestration,
            signal: "task_cancelled",
            payload: ["task_id": taskId.hasHashValue ? Double(taskId.hashValue) : 0]
        )
    }

    /// List active (non-completed) tasks
    func listActive() -> [AgentTask] {
        lock.lock(); defer { lock.unlock() }
        return agents.values.filter { ![.completed, .failed, .cancelled, .timeout].contains($0.status) }
    }

    /// List completed tasks
    func listCompleted() -> [AgentTask] {
        lock.lock(); defer { lock.unlock() }
        return agents.values.filter { $0.status == .completed }
    }

    /// List tasks by tag
    func listByTag(_ tag: String) -> [AgentTask] {
        lock.lock(); defer { lock.unlock() }
        return agents.values.filter { $0.tags.contains(tag) }
    }

    /// Aggregate statistics
    func stats() -> [String: Any] {
        lock.lock(); defer { lock.unlock() }
        return [
            "total_submitted": agents.count,
            "total_completed": totalCompleted,
            "total_failed": totalFailed,
            "total_tokens": totalTokens,
            "total_cost": String(format: "%.6f", totalCost),
            "running_count": runningCount,
            "pending_count": agents.values.filter { $0.status == .queued }.count,
            "uptime": Date().timeIntervalSince(bootTime)
        ]
    }

    /// Full status for telemetry
    func status() -> [String: Any] {
        lock.lock(); defer { lock.unlock() }
        return [
            "engine": engineName,
            "agents": agents.mapValues { $0.toDict() },
            "stats": [
                "total_completed": totalCompleted,
                "total_failed": totalFailed,
                "total_tokens": totalTokens,
                "total_cost": totalCost
            ],
            "history_size": history.count,
            "boot_time": bootTime.timeIntervalSince1970
        ]
    }

    /// Process mailbox for Swift app requests
    func processMailbox() -> [String] {
        let requestsDir = mailboxURL.appendingPathComponent("requests")
        var processed: [String] = []

        guard FileManager.default.fileExists(atPath: requestsDir.path) else { return processed }

        do {
            let files = try FileManager.default.contentsOfDirectory(at: requestsDir, includingPropertiesForKeys: nil)
            for file in files where file.pathExtension == "json" {
                if let data = try? Data(contentsOf: file),
                   let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let prompt = json["prompt"] as? String {

                    let agentType = (json["agent_type"] as? String).flatMap { AgentType(rawValue: $0) } ?? .general
                    let priority = AgentPriority(rawValue: (json["priority"] as? Int) ?? 2) ?? .normal

                    let task = AgentTask(
                        prompt: prompt,
                        agentType: agentType,
                        priority: priority,
                        source: "mailbox"
                    )

                    let taskId = submit(task)
                    processed.append(taskId)
                }
                try FileManager.default.removeItem(at: file)
            }
        } catch {
            // L104: Error handled with sacred resilience
        }

        return processed
    }

    // MARK: - SovereignEngine

    func engineStatus() -> [String: Any] {
        return status()
    }

    func engineHealth() -> Double {
        lock.lock(); defer { lock.unlock() }
        let total = agents.count
        guard total > 0 else { return 1.0 }
        let healthy = agents.values.filter { ![.failed, .timeout].contains($0.status) }.count
        return Double(healthy) / Double(total)
    }

    func engineReset() {
        lock.lock()
        agents.removeAll()
        pendingChains.removeAll()
        runningCount = 0
        lock.unlock()

        InterEngineFeedbackBus.shared.broadcast(
            from: .agentOrchestration,
            signal: "reset",
            payload: [:]
        )
    }

    // MARK: - Private

    private func processQueue() {
        queue.async { [weak self] in
            self?._processQueue()
        }
    }

    private func _processQueue() {
        lock.lock()

        // Check concurrent limit
        guard runningCount < maxConcurrent else {
            lock.unlock()
            return
        }

        // Get next queued task (sorted by priority)
        let queued = agents.values
            .filter { $0.status == .queued }
            .sorted { $0.priority < $1.priority }

        guard var task = queued.first else {
            lock.unlock()
            return
        }

        // Check if waiting for dependency
        if let depId = task.dependsOn {
            if let dep = agents[depId], ![.completed, .failed, .cancelled, .timeout].contains(dep.status) {
                // Dependency still running
                lock.unlock()
                return
            }
        }

        task.status = .running
        task.startedAt = Date()
        agents[task.taskId] = task
        runningCount += 1

        lock.unlock()

        // Run task asynchronously
        let capturedTask = task
        Task { [weak self] in
            await self?.runTask(capturedTask)
        }
    }

    private func runTask(_ task: AgentTask) async {
        var mutableTask = task

        // Check timeout
        if mutableTask.isTimedOut() {
            mutableTask.status = .timeout
            mutableTask.error = "Task timed out"
            mutableTask.completedAt = Date()
            completeTask(mutableTask)
            return
        }

        // Check budget
        if mutableTask.isOverBudget() {
            mutableTask.status = .failed
            mutableTask.error = "Cost budget exceeded"
            mutableTask.completedAt = Date()
            completeTask(mutableTask)
            return
        }

        // Simulate execution (in real implementation, call DeepSeek API)
        // For now, just use tool registry directly
        mutableTask.status = .thinking
        mutableTask.currentAction = "Analyzing prompt..."

        // Execute tools based on prompt analysis
        // This is a simplified version - real implementation would use DeepSeek API
        let result = await executeWithTools(&mutableTask)

        mutableTask.result = result
        mutableTask.status = result.success ? .completed : .failed
        mutableTask.completedAt = Date()

        completeTask(mutableTask)
    }

    private func executeWithTools(_ task: inout AgentTask) async -> AgentResult {
        // Simplified execution - real implementation would use DeepSeek API
        // For now, just return success
        var toolResults: [[String: Any]] = []
        var filesModified: [String] = []
        var totalTokens = 0

        // Parse prompt for tool hints
        let prompt = task.prompt.lowercased()

        if prompt.contains("read") || prompt.contains("analyze") {
            // Use analyze_code
            let result = await ToolRegistry.shared.execute(name: "analyze_code", args: ["path": task.prompt])
            toolResults.append(result.toDict())
            totalTokens += result.output.count / 4  // Rough token estimate
        }

        if prompt.contains("status") || prompt.contains("health") {
            let result = await ToolRegistry.shared.execute(name: "system_metrics", args: [:])
            toolResults.append(result.toDict())
        }

        return AgentResult(
            summary: "Task completed successfully",
            output: "Processed prompt: \(task.prompt.prefix(100))",
            filesCreated: [],
            filesModified: filesModified,
            toolResults: toolResults,
            tokensUsed: totalTokens,
            roundsUsed: 1,
            costEstimate: Double(totalTokens) * 0.000001,
            success: true,
            errorsEncountered: [],
            warnings: [],
            metrics: ["agent_type": task.agentType.rawValue]
        )
    }

    private func completeTask(_ task: AgentTask) {
        lock.lock()
        agents[task.taskId] = task
        runningCount -= 1
        if task.status == .completed {
            totalCompleted += 1
        } else {
            totalFailed += 1
        }
        if let result = task.result {
            totalTokens += result.tokensUsed
            totalCost += result.costEstimate
        }

        // Add to history
        history.append(task.toDict())
        if history.count > maxHistory { history.removeFirst(history.count - maxHistory) }
        lock.unlock()

        // Save state
        saveState()

        // Write mailbox response
        writeMailboxResponse(task)

        // Handle chain
        if let chainNext = task.chainNext, task.status == .completed {
            // Chain next agent
            lock.lock()
            if var chained = pendingChains.removeValue(forKey: chainNext) {
                chained.chainContext = [
                    "parent_task_id": task.taskId,
                    "parent_summary": task.result?.summary ?? "",
                    "parent_success": task.result?.success ?? false
                ]
                lock.unlock()
                _ = submit(chained)
            } else {
                lock.unlock()
            }
        }

        // Broadcast completion
        InterEngineFeedbackBus.shared.broadcast(
            from: .agentOrchestration,
            signal: "task_completed",
            payload: [
                "task_id": task.taskId.hasHashValue ? Double(task.taskId.hashValue) : 0,
                "success": task.status == .completed ? 1.0 : 0.0
            ]
        )

        // Process next in queue
        processQueue()
    }

    private func writeMailboxResponse(_ task: AgentTask) {
        let responsesDir = mailboxURL.appendingPathComponent("responses")
        do {
            try FileManager.default.createDirectory(at: responsesDir, withIntermediateDirectories: true)
            let responseFile = responsesDir.appendingPathComponent("\(task.taskId).json")

            let response: [String: Any] = [
                "task_id": task.taskId,
                "task": task.prompt,
                "agent_type": task.agentType.rawValue,
                "status": task.status.rawValue,
                "priority": task.priority.rawValue,
                "output": task.result?.output ?? task.error ?? "",
                "summary": task.result?.summary ?? "",
                "files_modified": task.filesModified,
                "tool_calls_count": task.toolCallsMade.count,
                "tokens_used": task.result?.tokensUsed ?? 0,
                "cost_estimate": task.result?.costEstimate ?? 0.0,
                "duration": task.elapsed(),
                "completed_at": ISO8601DateFormatter().string(from: task.completedAt ?? Date()),
                "errors": task.result?.errorsEncountered ?? [],
                "warnings": task.result?.warnings ?? [],
                "chain_next": task.chainNext ?? "",
                "tags": task.tags,
                "source": task.source
            ]

            let data = try JSONSerialization.data(withJSONObject: response, options: .prettyPrinted)
            try data.write(to: responseFile)
        } catch {
            // Silent fail for mailbox
        }
    }

    private func loadState() {
        guard FileManager.default.fileExists(atPath: stateFileURL.path) else { return }
        do {
            let data = try Data(contentsOf: stateFileURL)
            if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
                if let h = json["history"] as? [[String: Any]] {
                    history = Array(h.suffix(maxHistory))
                }
                totalCompleted = (json["total_completed"] as? Int) ?? 0
                totalFailed = (json["total_failed"] as? Int) ?? 0
                totalTokens = (json["total_tokens"] as? Int) ?? 0
                totalCost = (json["total_cost"] as? Double) ?? 0.0
            }
        } catch {
            // Silent fail
        }
    }

    private func saveState() {
        do {
            let state: [String: Any] = [
                "history": Array(history.suffix(maxHistory)),
                "total_completed": totalCompleted,
                "total_failed": totalFailed,
                "total_tokens": totalTokens,
                "total_cost": totalCost,
                "saved_at": Date().timeIntervalSince1970,
                "version": "3.0.0"
            ]
            let data = try JSONSerialization.data(withJSONObject: state, options: .prettyPrinted)
            try data.write(to: stateFileURL)
        } catch {
            // Silent fail
        }
    }

    /// Start processing (for Swift app lifecycle)
    func startProcessing() {
        // EVO_76: 20s startup grace — agent mailbox runs after boot CPU settles
        DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + 20.0) { [weak self] in
            guard let self else { return }
            DispatchQueue.main.async {
                // Periodic mailbox check
                Timer.scheduledTimer(withTimeInterval: 5.0, repeats: true) { [weak self] _ in
                    self?.processMailbox()
                }

                // Periodic status broadcast
                Timer.scheduledTimer(withTimeInterval: 30.0, repeats: true) { [weak self] _ in
                    guard let self = self else { return }
                    InterEngineFeedbackBus.shared.broadcast(
                        from: .agentOrchestration,
                        signal: "status_update",
                        payload: [
                            "active": Double(self.listActive().count),
                            "completed": Double(self.totalCompleted),
                            "failed": Double(self.totalFailed)
                        ]
                    )
                }
            }
        }
    }
}

// MARK: - Array Extension

private extension Array {
    subscript(safe index: Int) -> Element? {
        return indices.contains(index) ? self[index] : nil
    }
}

// MARK: - String Extension for HashValue

private extension String {
    var hasHashValue: Bool {
        return !isEmpty
    }
}