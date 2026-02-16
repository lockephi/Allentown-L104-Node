// ═══════════════════════════════════════════════════════════════════
// L25_PythonBridge.swift
// L104v2 Architecture — EVO_54 Pipeline-Integrated Python Bridge
// PythonResult, PythonModuleInfo, PythonBridge
// Streams through unified EVO_54 pipeline (695 l104_* modules)
// Extracted from L104Native.swift lines 2401–3016
// ═══════════════════════════════════════════════════════════════════

import AppKit
import Foundation
import Accelerate
import simd
import NaturalLanguage

struct PythonResult: CustomStringConvertible {
    let success: Bool
    let output: String
    let error: String
    let returnValue: Any?
    let executionTime: Double

    var description: String {
        if success {
            return output.isEmpty ? "(ok)" : output
        } else {
            return "PythonError: \(error)"
        }
    }
}

/// Discovered Python module with metadata
struct PythonModuleInfo {
    let name: String
    let path: String
    let classes: [String]
    let functions: [String]
    let docstring: String
    let sizeBytes: Int
}

/// Main bridge for calling Python from Swift
/// EVO_54: Pipeline-integrated — routes through unified L104 subsystem mesh
/// Replaces PythonKit dependency — works with bare swiftc builds
class PythonBridge {
    static let shared = PythonBridge()

    // ─── CONFIGURATION ───

    /// Path to the Python interpreter in the virtual environment
    private let pythonPath: String
    /// Path to the ASI workspace (695 l104_* modules)
    let workspacePath: String
    /// Timeout for Python execution (seconds)
    var timeout: TimeInterval = 30.0
    /// Cache for module introspection results
    private var moduleCache: [String: PythonModuleInfo] = [:]
    /// Cache for recently executed snippets
    private var resultCache: [String: (result: PythonResult, timestamp: Date)] = [:]
    private let cacheTTL: TimeInterval = 20.0  // 20s — short enough to keep responses fresh
    /// Execution statistics
    private(set) var totalExecutions: Int = 0
    private(set) var totalErrors: Int = 0
    private(set) var totalExecutionTime: Double = 0.0
    /// Discovered modules
    private(set) var discoveredModules: [String] = []
    /// Active persistent session (long-running Python process)
    private var persistentProcess: Process?
    private var persistentStdin: FileHandle?
    private var persistentStdout: FileHandle?
    private var sessionActive = false
    /// EVO_56: Warm module cache — avoid reimporting on every call
    private var warmedModules: Set<String> = []

    // ─── v21.0 FILE-BASED STATE CACHE (zero-spawn reads) ───
    private var nirvanicStateCache: [String: Any]? = nil
    private var consciousnessO2Cache: [String: Any]? = nil
    private var stateCacheTime: Date = .distantPast
    private let stateCacheTTL: TimeInterval = 10.0  // Refresh every 10s

    /// Read builder state files directly — no Python process spawn needed.
    /// Returns cached data for up to 10 seconds.
    func readNirvanicState() -> [String: Any]? {
        if Date().timeIntervalSince(stateCacheTime) < stateCacheTTL, let c = nirvanicStateCache { return c }
        let path = workspacePath + "/.l104_ouroboros_nirvanic_state.json"
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        nirvanicStateCache = json
        stateCacheTime = Date()
        return json
    }

    func readConsciousnessO2State() -> [String: Any]? {
        if Date().timeIntervalSince(stateCacheTime) < stateCacheTTL, let c = consciousnessO2Cache { return c }
        let path = workspacePath + "/.l104_consciousness_o2_state.json"
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        consciousnessO2Cache = json
        return json
    }

    func readLinkState() -> [String: Any]? {
        let path = workspacePath + "/.l104_quantum_link_state.json"
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        return json
    }

    func readGateRegistry() -> [String: Any]? {
        let path = workspacePath + "/.l104_gate_registry.json"
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        return json
    }

    /// Invalidate file state caches (call after running a pipeline)
    func invalidateStateCache() {
        stateCacheTime = .distantPast
        nirvanicStateCache = nil
        consciousnessO2Cache = nil
    }

    // ─── INITIALIZATION ───

    init() {
        // Auto-detect workspace and venv
        let appDir = FileManager.default.currentDirectoryPath
        let candidates = [
            appDir + "/../.venv/bin/python",
            appDir + "/../../.venv/bin/python",
            "/Users/carolalvarez/Applications/Allentown-L104-Node/.venv/bin/python",
        ]
        pythonPath = candidates.first { FileManager.default.fileExists(atPath: $0) }
            ?? "/usr/bin/python3"

        let wsCandidates = [
            appDir + "/..",
            appDir + "/../..",
            "/Users/carolalvarez/Applications/Allentown-L104-Node",
        ]
        workspacePath = wsCandidates.first {
            FileManager.default.fileExists(atPath: $0 + "/l104_fast_server.py")
        } ?? "/Users/carolalvarez/Applications/Allentown-L104-Node"
    }

    // ─── CORE EXECUTION ───

    /// Execute raw Python code and capture output
    @discardableResult
    func execute(_ code: String, timeout: TimeInterval? = nil) -> PythonResult {
        let start = CFAbsoluteTimeGetCurrent()
        totalExecutions += 1

        // Check cache
        if let cached = resultCache[code], Date().timeIntervalSince(cached.timestamp) < cacheTTL {
            return cached.result
        }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: pythonPath)
        process.arguments = ["-c", code]
        process.environment = [
            "PYTHONPATH": workspacePath,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "PATH": (ProcessInfo.processInfo.environment["PATH"] ?? "/usr/bin") + ":" + workspacePath
        ]
        process.currentDirectoryURL = URL(fileURLWithPath: workspacePath)

        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        do {
            try process.run()
        } catch {
            totalErrors += 1
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            totalExecutionTime += elapsed
            return PythonResult(success: false, output: "", error: "Failed to launch Python: \(error)", returnValue: nil, executionTime: elapsed)
        }

        // Timeout handling
        let effectiveTimeout = timeout ?? self.timeout
        let deadline = DispatchTime.now() + effectiveTimeout
        let group = DispatchGroup()
        group.enter()
        DispatchQueue.global().async {
            process.waitUntilExit()
            group.leave()
        }
        let waitResult = group.wait(timeout: deadline)
        if waitResult == .timedOut {
            process.terminate()
            totalErrors += 1
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            totalExecutionTime += elapsed
            return PythonResult(success: false, output: "", error: "Timeout after \(effectiveTimeout)s", returnValue: nil, executionTime: elapsed)
        }

        let stdoutData = stdoutPipe.fileHandleForReading.readDataToEndOfFile()
        let stderrData = stderrPipe.fileHandleForReading.readDataToEndOfFile()
        let stdout = String(data: stdoutData, encoding: .utf8) ?? ""
        let stderr = String(data: stderrData, encoding: .utf8) ?? ""
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        totalExecutionTime += elapsed
        let success = process.terminationStatus == 0

        if !success { totalErrors += 1 }

        let result = PythonResult(
            success: success,
            output: stdout.trimmingCharacters(in: .whitespacesAndNewlines),
            error: stderr.trimmingCharacters(in: .whitespacesAndNewlines),
            returnValue: parseJSON(stdout),
            executionTime: elapsed
        )

        // Cache successful results
        if success { resultCache[code] = (result, Date()) }

        return result
    }

    /// Execute a Python script file
    @discardableResult
    func executeFile(_ filename: String, args: [String] = []) -> PythonResult {
        let start = CFAbsoluteTimeGetCurrent()
        totalExecutions += 1

        let fullPath: String
        if filename.hasPrefix("/") {
            fullPath = filename
        } else {
            fullPath = workspacePath + "/" + filename
        }

        guard FileManager.default.fileExists(atPath: fullPath) else {
            totalErrors += 1
            return PythonResult(success: false, output: "", error: "File not found: \(fullPath)", returnValue: nil, executionTime: 0)
        }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: pythonPath)
        process.arguments = [fullPath] + args
        process.environment = [
            "PYTHONPATH": workspacePath,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1"
        ]
        process.currentDirectoryURL = URL(fileURLWithPath: workspacePath)

        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        do { try process.run() } catch {
            totalErrors += 1
            return PythonResult(success: false, output: "", error: "Launch failed: \(error)", returnValue: nil, executionTime: 0)
        }

        process.waitUntilExit()
        let stdout = String(data: stdoutPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        let stderr = String(data: stderrPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        totalExecutionTime += elapsed

        if process.terminationStatus != 0 { totalErrors += 1 }

        return PythonResult(
            success: process.terminationStatus == 0,
            output: stdout.trimmingCharacters(in: .whitespacesAndNewlines),
            error: stderr.trimmingCharacters(in: .whitespacesAndNewlines),
            returnValue: parseJSON(stdout),
            executionTime: elapsed
        )
    }

    // ─── MODULE IMPORT & CALL ───

    /// Import a Python module and call a function with JSON-serialized args
    func callFunction(module: String, function: String, args: [String] = [], kwargs: [String: String] = [:]) -> PythonResult {
        var code = "import sys, json\nsys.path.insert(0, '\(workspacePath)')\n"
        code += "import \(module)\n"

        var argList = args.map { "'\($0)'" }.joined(separator: ", ")
        let kwargList = kwargs.map { "\($0.key)='\($0.value)'" }.joined(separator: ", ")

        if !argList.isEmpty && !kwargList.isEmpty {
            argList += ", " + kwargList
        } else if !kwargList.isEmpty {
            argList = kwargList
        }

        code += "result = \(module).\(function)(\(argList))\n"
        code += "if result is not None:\n"
        code += "    try:\n"
        code += "        print(json.dumps(result, default=str))\n"
        code += "    except:\n"
        code += "        print(str(result))\n"

        return execute(code)
    }

    /// Import a module, create a class instance, and call a method
    func callMethod(module: String, className: String, method: String, constructorArgs: [String] = [], methodArgs: [String] = []) -> PythonResult {
        var code = "import sys, json\nsys.path.insert(0, '\(workspacePath)')\n"
        code += "import \(module)\n"

        let ctorArgs = constructorArgs.map { "'\($0)'" }.joined(separator: ", ")
        let methArgs = methodArgs.map { "'\($0)'" }.joined(separator: ", ")

        code += "obj = \(module).\(className)(\(ctorArgs))\n"
        code += "result = obj.\(method)(\(methArgs))\n"
        code += "if result is not None:\n"
        code += "    try:\n"
        code += "        print(json.dumps(result, default=str))\n"
        code += "    except:\n"
        code += "        print(str(result))\n"

        return execute(code)
    }

    /// Evaluate a Python expression and return the result
    func eval(_ expression: String) -> PythonResult {
        let code = """
        import sys, json
        sys.path.insert(0, '\(workspacePath)')
        _result = \(expression)
        try:
            print(json.dumps(_result, default=str))
        except:
            print(str(_result))
        """
        return execute(code)
    }

    // ─── MODULE DISCOVERY ───

    /// Discover all l104_* modules in the workspace
    func discoverModules() -> [String] {
        let code = """
        import os, json
        modules = []
        for f in sorted(os.listdir('\(workspacePath)')):
            if f.startswith('l104_') and f.endswith('.py'):
                modules.append(f[:-3])
        print(json.dumps(modules))
        """
        let result = execute(code)
        if result.success, let json = result.returnValue as? [String] {
            discoveredModules = json
            return json
        }
        return []
    }

    /// Introspect a module — get classes, functions, docstring
    func introspectModule(_ moduleName: String) -> PythonModuleInfo? {
        if let cached = moduleCache[moduleName] { return cached }

        let code = """
        import sys, json, inspect, os
        sys.path.insert(0, '\(workspacePath)')
        try:
            mod = __import__('\(moduleName)')
            classes = [name for name, obj in inspect.getmembers(mod, inspect.isclass) if obj.__module__ == '\(moduleName)']
            funcs = [name for name, obj in inspect.getmembers(mod, inspect.isfunction) if obj.__module__ == '\(moduleName)']
            doc = (mod.__doc__ or '')[:500]
            path = inspect.getfile(mod)
            size = os.path.getsize(path)
            print(json.dumps({'classes': classes, 'functions': funcs, 'docstring': doc, 'path': path, 'size': size}))
        except Exception as e:
            print(json.dumps({'error': str(e)}))
        """
        let result = execute(code, timeout: 10)
        guard result.success, let dict = result.returnValue as? [String: Any] else { return nil }

        if dict["error"] != nil { return nil }

        let info = PythonModuleInfo(
            name: moduleName,
            path: dict["path"] as? String ?? "",
            classes: dict["classes"] as? [String] ?? [],
            functions: dict["functions"] as? [String] ?? [],
            docstring: dict["docstring"] as? String ?? "",
            sizeBytes: dict["size"] as? Int ?? 0
        )
        moduleCache[moduleName] = info
        return info
    }

    // ─── L104 ASI INTEGRATION ───

    /// Connect to the LearningIntellect from l104_fast_server
    func queryIntellect(_ message: String) -> PythonResult {
        let escaped = message.replacingOccurrences(of: "'", with: "\\'").replacingOccurrences(of: "\\", with: "\\\\")
        let code = """
        import sys, json
        sys.path.insert(0, '\(workspacePath)')
        from l104_fast_server import intellect
        response = intellect.generate_response('\(escaped)')
        print(json.dumps({'response': str(response), 'knowledge_count': len(intellect.knowledge_base) if hasattr(intellect, 'knowledge_base') else 0}))
        """
        return execute(code, timeout: 15)
    }

    /// Get ASI Quantum Bridge status from Python backend
    func getASIBridgeStatus() -> PythonResult {
        let code = """
        import sys, json
        sys.path.insert(0, '\(workspacePath)')
        from l104_fast_server import asi_quantum_bridge, intellect
        bridge_status = asi_quantum_bridge.get_bridge_status()
        # Enrich with intellect stats
        bridge_status['total_memories'] = len(intellect.permanent_memory) if hasattr(intellect, 'permanent_memory') else 0
        bridge_status['knowledge_entries'] = len(intellect.knowledge_base) if hasattr(intellect, 'knowledge_base') else 0
        print(json.dumps(bridge_status, default=str))
        """
        return execute(code, timeout: 45)
    }

    /// Run a learning/training cycle on the Python intellect
    func trainIntellect(data: String, category: String = "general") -> PythonResult {
        let escaped = data.replacingOccurrences(of: "'", with: "\\'")
        let code = """
        import sys, json
        sys.path.insert(0, '\(workspacePath)')
        from l104_fast_server import intellect
        intellect.learn('\(escaped)', category='\(category)')
        stats = {'learned': True, 'category': '\(category)', 'total_knowledge': len(intellect.knowledge_base) if hasattr(intellect, 'knowledge_base') else 0}
        print(json.dumps(stats))
        """
        return execute(code, timeout: 10)
    }

    /// Get Python environment info
    func getEnvironmentInfo() -> PythonResult {
        let code = """
        import sys, json, platform, os
        info = {
            'python_version': sys.version,
            'platform': platform.platform(),
            'executable': sys.executable,
            'prefix': sys.prefix,
            'path': sys.path[:5],
            'modules_available': len([f for f in os.listdir('\(workspacePath)') if f.startswith('l104_') and f.endswith('.py')]),
            'cwd': os.getcwd()
        }
        print(json.dumps(info))
        """
        return execute(code, timeout: 5)
    }

    /// List installed pip packages
    func listPackages() -> PythonResult {
        let code = """
        import json, pkg_resources
        pkgs = {p.project_name: p.version for p in sorted(pkg_resources.working_set, key=lambda p: p.project_name.lower())}
        print(json.dumps(pkgs))
        """
        return execute(code, timeout: 10)
    }

    /// Install a pip package at runtime
    func installPackage(_ package: String) -> PythonResult {
        return execute("import subprocess; subprocess.check_call(['\(pythonPath)', '-m', 'pip', 'install', '\(package)', '-q'])", timeout: 60)
    }

    // ─── PERSISTENT SESSION ───

    /// Start a persistent Python REPL session for interactive use
    func startSession() -> Bool {
        guard !sessionActive else { return true }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: pythonPath)
        process.arguments = ["-u", "-c", """
        import sys, json
        sys.path.insert(0, '\(workspacePath)')
        print('SESSION_READY', flush=True)
        while True:
            try:
                line = input()
                if line == '__EXIT__':
                    break
                exec(compile(line, '<bridge>', 'exec'))
                sys.stdout.flush()
            except Exception as e:
                print(f'__ERROR__:{e}', flush=True)
        """]
        process.environment = ["PYTHONPATH": workspacePath, "PYTHONUNBUFFERED": "1"]
        process.currentDirectoryURL = URL(fileURLWithPath: workspacePath)

        let stdinPipe = Pipe()
        let stdoutPipe = Pipe()
        process.standardInput = stdinPipe
        process.standardOutput = stdoutPipe
        process.standardError = FileHandle.nullDevice

        do {
            try process.run()
            persistentProcess = process
            persistentStdin = stdinPipe.fileHandleForWriting
            persistentStdout = stdoutPipe.fileHandleForReading
            sessionActive = true

            // Wait for SESSION_READY
            if let data = persistentStdout?.availableData,
               let response = String(data: data, encoding: .utf8),
               response.contains("SESSION_READY") {
                return true
            }
            endSession()
            return false
        } catch {
            return false
        }
    }

    /// Send a command to the persistent session
    func sessionExec(_ code: String) -> String {
        guard sessionActive, let stdin = persistentStdin, let stdout = persistentStdout else {
            return "No active session"
        }
        let cmd = code.replacingOccurrences(of: "\n", with: ";") + "\n"
        guard let cmdData = cmd.data(using: .utf8) else { return "Encoding error" }
        stdin.write(cmdData)
        usleep(100_000)  // 100ms for execution
        let data = stdout.availableData
        return String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    }

    /// End the persistent session
    func endSession() {
        if let stdin = persistentStdin, let exitData = "__EXIT__\n".data(using: .utf8) {
            stdin.write(exitData)
        }
        persistentProcess?.terminate()
        persistentProcess = nil
        persistentStdin = nil
        persistentStdout = nil
        sessionActive = false
    }

    // ─── BATCH OPERATIONS ───

    /// Execute multiple Python snippets in sequence within a single process
    func executeBatch(_ snippets: [String]) -> [PythonResult] {
        let combined = snippets.enumerated().map { (i, code) in
            """
            try:
                exec('''\(code.replacingOccurrences(of: "'''", with: "\\'\\'\\'\\'"))''')
                print(f'__BATCH_OK__:{i}')
            except Exception as e:
                print(f'__BATCH_ERR__:{i}:{e}')
            """
        }.joined(separator: "\n")

        let wrapper = "import sys\nsys.path.insert(0, '\(workspacePath)')\n" + combined
        let batchResult = execute(wrapper, timeout: timeout * Double(snippets.count))

        var results = [PythonResult]()
        let lines = batchResult.output.components(separatedBy: "\n")
        for (i, _) in snippets.enumerated() {
            let okLine = lines.first { $0.contains("__BATCH_OK__:\(i)") }
            let errLine = lines.first { $0.contains("__BATCH_ERR__:\(i):") }
            if okLine != nil {
                results.append(PythonResult(success: true, output: "", error: "", returnValue: nil, executionTime: batchResult.executionTime / Double(snippets.count)))
            } else if let err = errLine {
                let errMsg = String(err.dropFirst("__BATCH_ERR__:\(i):".count))
                results.append(PythonResult(success: false, output: "", error: errMsg, returnValue: nil, executionTime: 0))
            } else {
                results.append(PythonResult(success: false, output: "", error: "Unknown", returnValue: nil, executionTime: 0))
            }
        }
        return results
    }

    // ─── UTILITIES ───

    /// Parse JSON from Python stdout
    private func parseJSON(_ string: String) -> Any? {
        let trimmed = string.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        // Find last line that looks like JSON
        let lines = trimmed.components(separatedBy: "\n")
        for line in lines.reversed() {
            let l = line.trimmingCharacters(in: .whitespaces)
            if l.hasPrefix("{") || l.hasPrefix("[") || l.hasPrefix("\"") {
                if let data = l.data(using: .utf8),
                   let json = try? JSONSerialization.jsonObject(with: data) {
                    return json
                }
            }
        }
        return nil
    }

    /// Clear all caches
    func clearCache() {
        resultCache.removeAll()
        moduleCache.removeAll()
        warmedModules.removeAll()
    }

    // EVO_56: Pre-warm Python interpreter by importing heavy modules once at startup
    func warmUp() {
        DispatchQueue.global(qos: .utility).async { [weak self] in
            guard let self = self else { return }
            let warmCode = """
            import sys, json, os
            sys.path.insert(0, '\(self.workspacePath)')
            # Pre-import heavy modules to cache bytecode
            try:
                import l104_code_engine
                import l104_fast_server
            except:
                pass
            print(json.dumps({"warmed": True}))
            """
            let result = self.execute(warmCode, timeout: 15)
            if result.success {
                self.warmedModules.insert("l104_code_engine")
                self.warmedModules.insert("l104_fast_server")
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // ⚛️ QUANTUM COMPUTING BRIDGE — l104_quantum_coherence.py (Qiskit 2.3.0)
    // Real quantum circuits: Grover, QPE, VQE, QAOA, Amplitude Estimation,
    // Quantum Walks, Quantum Kernels — executed via Statevector simulator
    // ═══════════════════════════════════════════════════════════════════

    /// Run Grover's search algorithm
    func quantumGrover(target: Int, nQubits: Int) -> PythonResult {
        let pyCode = """
        import sys, json
        sys.path.insert(0, '\(workspacePath)')
        from l104_quantum_coherence import QuantumCoherenceEngine
        e = QuantumCoherenceEngine()
        r = e.grover_search(\(target), \(nQubits))
        print(json.dumps(r, default=str))
        """
        return execute(pyCode, timeout: 30)
    }

    /// Quantum Phase Estimation
    func quantumQPE(precisionQubits: Int = 5) -> PythonResult {
        let pyCode = """
        import sys, json
        sys.path.insert(0, '\(workspacePath)')
        from l104_quantum_coherence import QuantumCoherenceEngine
        e = QuantumCoherenceEngine()
        r = e.quantum_phase_estimation(precision_qubits=\(precisionQubits))
        print(json.dumps(r, default=str))
        """
        return execute(pyCode, timeout: 30)
    }

    /// VQE — Variational Quantum Eigensolver
    func quantumVQE(nQubits: Int = 4, iterations: Int = 50) -> PythonResult {
        let pyCode = """
        import sys, json
        sys.path.insert(0, '\(workspacePath)')
        from l104_quantum_coherence import QuantumCoherenceEngine
        e = QuantumCoherenceEngine()
        r = e.vqe_optimize(num_qubits=\(nQubits), max_iterations=\(iterations))
        print(json.dumps(r, default=str))
        """
        return execute(pyCode, timeout: 45)
    }

    /// QAOA — MaxCut approximation
    func quantumQAOA(edges: [(Int, Int)], p: Int = 2) -> PythonResult {
        let edgeStr = edges.map { "(\($0.0),\($0.1))" }.joined(separator: ",")
        let pyCode = """
        import sys, json
        sys.path.insert(0, '\(workspacePath)')
        from l104_quantum_coherence import QuantumCoherenceEngine
        e = QuantumCoherenceEngine()
        r = e.qaoa_maxcut([\(edgeStr)], p=\(p))
        print(json.dumps(r, default=str))
        """
        return execute(pyCode, timeout: 45)
    }

    /// Amplitude Estimation
    func quantumAmplitudeEstimation(targetProb: Double, countingQubits: Int = 5) -> PythonResult {
        let pyCode = """
        import sys, json
        sys.path.insert(0, '\(workspacePath)')
        from l104_quantum_coherence import QuantumCoherenceEngine
        e = QuantumCoherenceEngine()
        r = e.amplitude_estimation(target_prob=\(targetProb), counting_qubits=\(countingQubits))
        print(json.dumps(r, default=str))
        """
        return execute(pyCode, timeout: 30)
    }

    /// Quantum Walk on a graph
    func quantumWalk(nNodes: Int = 8, steps: Int = 10) -> PythonResult {
        let pyCode = """
        import sys, json
        sys.path.insert(0, '\(workspacePath)')
        from l104_quantum_coherence import QuantumCoherenceEngine
        e = QuantumCoherenceEngine()
        r = e.quantum_walk(n_nodes=\(nNodes), steps=\(steps))
        print(json.dumps(r, default=str))
        """
        return execute(pyCode, timeout: 30)
    }

    /// Quantum Kernel — similarity between feature vectors
    func quantumKernel(x1: [Double], x2: [Double]) -> PythonResult {
        let x1Str = x1.map { String($0) }.joined(separator: ",")
        let x2Str = x2.map { String($0) }.joined(separator: ",")
        let pyCode = """
        import sys, json
        sys.path.insert(0, '\(workspacePath)')
        from l104_quantum_coherence import QuantumCoherenceEngine
        e = QuantumCoherenceEngine()
        r = e.quantum_kernel([\(x1Str)], [\(x2Str)])
        print(json.dumps(r, default=str))
        """
        return execute(pyCode, timeout: 30)
    }

    /// Get quantum engine status
    func quantumStatus() -> PythonResult {
        let pyCode = """
        import sys, json
        sys.path.insert(0, '\(workspacePath)')
        from l104_quantum_coherence import QuantumCoherenceEngine
        e = QuantumCoherenceEngine()
        s = e.get_status()
        print(json.dumps(s, default=str))
        """
        return execute(pyCode, timeout: 15)
    }

    // ═══════════════════════════════════════════════════════════════════
    // ⚛️ REAL QUANTUM HARDWARE BRIDGE — l104_quantum_mining_engine.py
    // Calls real IBM Quantum QPUs via Qiskit Runtime (SamplerV2)
    // Extended timeouts for real hardware queue + execution
    // ═══════════════════════════════════════════════════════════════════

    /// Initialize real quantum mining engine with IBM token
    func quantumHardwareInit(token: String) -> PythonResult {
        let pyCode = """
        import sys, json
        sys.path.insert(0, '\(workspacePath)')
        from l104_quantum_mining_engine import initialize_quantum_mining
        engine = initialize_quantum_mining(ibm_token='\(token)')
        s = engine.status
        result = {
            "initialized": True,
            "backend": s.backend_name if s else "unknown",
            "qubits": s.qubits if s else 0,
            "real_hardware": engine.is_real_hardware,
            "quantum_volume": s.quantum_volume if s else 0
        }
        print(json.dumps(result, default=str))
        """
        return execute(pyCode, timeout: 30)
    }

    /// Get real quantum hardware status
    func quantumHardwareStatus() -> PythonResult {
        let pyCode = """
        import sys, json
        sys.path.insert(0, '\(workspacePath)')
        from l104_quantum_mining_engine import get_quantum_engine
        engine = get_quantum_engine()
        s = engine.status
        result = {
            "backend": s.backend_name if s else "none",
            "qubits": s.qubits if s else 0,
            "real_hardware": engine.is_real_hardware,
            "quantum_volume": s.quantum_volume if s else 0,
            "connected": s.connected if s else False,
            "error_rate": s.error_rate if s else 0.0,
            "queue_depth": s.queue_depth if s else 0
        }
        print(json.dumps(result, default=str))
        """
        return execute(pyCode, timeout: 15)
    }

    /// Run Grover search on real quantum hardware
    func quantumHardwareGrover(target: Int, nQubits: Int) -> PythonResult {
        let pyCode = """
        import sys, json
        sys.path.insert(0, '\(workspacePath)')
        from l104_quantum_mining_engine import get_quantum_engine
        engine = get_quantum_engine()
        nonce, details = engine.mine_quantum(b'L104_BLOCK', \(target), qubit_count=\(nQubits))
        result = {
            "nonce": nonce,
            "real_hardware": engine.is_real_hardware,
            "backend": engine.status.backend_name if engine.status else "unknown",
            "details": details
        }
        print(json.dumps(result, default=str))
        """
        return execute(pyCode, timeout: 120)
    }

    /// Full quantum mining with strategy selection on real hardware
    func quantumHardwareMine(strategy: String = "auto") -> PythonResult {
        let pyCode = """
        import sys, json
        sys.path.insert(0, '\(workspacePath)')
        from l104_quantum_mining_engine import get_quantum_engine
        engine = get_quantum_engine()
        nonce, details = engine.mine_full_quantum(b'L104_BLOCK', 1000000, strategy='\(strategy)')
        result = {
            "nonce": nonce,
            "strategy": '\(strategy)',
            "real_hardware": engine.is_real_hardware,
            "backend": engine.status.backend_name if engine.status else "unknown",
            "details": details
        }
        print(json.dumps(result, default=str))
        """
        return execute(pyCode, timeout: 180)
    }

    /// VQE optimization on real quantum hardware
    func quantumHardwareVQE() -> PythonResult {
        let pyCode = """
        import sys, json
        sys.path.insert(0, '\(workspacePath)')
        from l104_quantum_mining_engine import get_quantum_engine
        engine = get_quantum_engine()
        if engine.vqe_optimizer:
            result = engine.vqe_optimizer.optimize(n_qubits=8, p=3, iterations=30)
            result["real_hardware"] = engine.is_real_hardware
            result["backend"] = engine.status.backend_name if engine.status else "unknown"
        else:
            result = {"error": "VQE optimizer not available", "real_hardware": False}
        print(json.dumps(result, default=str))
        """
        return execute(pyCode, timeout: 120)
    }

    /// Quantum random oracle — true quantum randomness for nonce seeding
    func quantumHardwareRandomOracle() -> PythonResult {
        let pyCode = """
        import sys, json
        sys.path.insert(0, '\(workspacePath)')
        from l104_quantum_mining_engine import get_quantum_engine
        engine = get_quantum_engine()
        oracle = None
        # QuantumRandomOracle is standalone — instantiate with hw_manager
        from l104_quantum_mining_engine import QuantumRandomOracle
        oracle = QuantumRandomOracle(engine.hw_manager)
        seed = oracle.generate_sacred_nonce_seed()
        result = {
            "seed": seed,
            "real_hardware": engine.is_real_hardware,
            "backend": engine.status.backend_name if engine.status else "unknown"
        }
        print(json.dumps(result, default=str))
        """
        return execute(pyCode, timeout: 60)
    }

    /// Get quantum advantage report
    func quantumHardwareReport(difficultyBits: Int = 16) -> PythonResult {
        let pyCode = """
        import sys, json
        sys.path.insert(0, '\(workspacePath)')
        from l104_quantum_mining_engine import get_quantum_engine
        engine = get_quantum_engine()
        report = engine.get_quantum_advantage_report(\(difficultyBits))
        result = {
            "report": report,
            "real_hardware": engine.is_real_hardware,
            "backend": engine.status.backend_name if engine.status else "unknown",
            "qubits": engine.status.qubits if engine.status else 0
        }
        print(json.dumps(result, default=str))
        """
        return execute(pyCode, timeout: 15)
    }

    // ═══════════════════════════════════════════════════════════════════
    // 🧠 CODING INTELLIGENCE BRIDGE — l104_coding_system.py
    // ASI-grade code review, quality gates, AI context, self-analysis
    // ═══════════════════════════════════════════════════════════════════

    /// Full code review via CodingIntelligenceSystem
    func codingSystemReview(_ code: String, filename: String = "") -> PythonResult {
        let escaped = code.replacingOccurrences(of: "'", with: "\\'").replacingOccurrences(of: "\n", with: "\\n")
        let pyCode = """
        import sys, json
        sys.path.insert(0, '\(workspacePath)')
        from l104_coding_system import coding_system
        r = coding_system.review('\(escaped)', '\(filename)')
        print(json.dumps(r, default=str))
        """
        return execute(pyCode, timeout: 30)
    }

    /// Quality gate check (CI/CD pass/fail)
    func codingSystemQualityCheck(_ code: String) -> PythonResult {
        let escaped = code.replacingOccurrences(of: "'", with: "\\'").replacingOccurrences(of: "\n", with: "\\n")
        let pyCode = """
        import sys, json
        sys.path.insert(0, '\(workspacePath)')
        from l104_coding_system import coding_system
        r = coding_system.quality_check('\(escaped)')
        print(json.dumps(r, default=str))
        """
        return execute(pyCode, timeout: 20)
    }

    /// Code explanation — structure, patterns, what it does
    func codingSystemExplain(_ code: String) -> PythonResult {
        let escaped = code.replacingOccurrences(of: "'", with: "\\'").replacingOccurrences(of: "\n", with: "\\n")
        let pyCode = """
        import sys, json
        sys.path.insert(0, '\(workspacePath)')
        from l104_coding_system import coding_system
        r = coding_system.explain('\(escaped)')
        print(json.dumps(r, default=str))
        """
        return execute(pyCode, timeout: 20)
    }

    /// Proactive suggestions for code improvements
    func codingSystemSuggest(_ code: String) -> PythonResult {
        let escaped = code.replacingOccurrences(of: "'", with: "\\'").replacingOccurrences(of: "\n", with: "\\n")
        let pyCode = """
        import sys, json
        sys.path.insert(0, '\(workspacePath)')
        from l104_coding_system import coding_system
        r = coding_system.suggest('\(escaped)')
        print(json.dumps(r, default=str))
        """
        return execute(pyCode, timeout: 20)
    }

    /// Project scan — full project structure, frameworks, health
    func codingSystemProjectScan() -> PythonResult {
        let pyCode = """
        import sys, json
        sys.path.insert(0, '\(workspacePath)')
        from l104_coding_system import coding_system
        r = coding_system.scan_project('\(workspacePath)')
        print(json.dumps(r, default=str))
        """
        return execute(pyCode, timeout: 45)
    }

    /// Self-analysis — L104 examines its own codebase
    func codingSystemSelfAnalyze() -> PythonResult {
        let pyCode = """
        import sys, json
        sys.path.insert(0, '\(workspacePath)')
        from l104_coding_system import coding_system
        r = coding_system.self_analyze()
        print(json.dumps(r, default=str))
        """
        return execute(pyCode, timeout: 45)
    }

    /// CI quality report for entire project
    func codingSystemCIReport() -> PythonResult {
        let pyCode = """
        import sys, json
        sys.path.insert(0, '\(workspacePath)')
        from l104_coding_system import coding_system
        r = coding_system.ci_report('\(workspacePath)')
        print(json.dumps(r, default=str))
        """
        return execute(pyCode, timeout: 60)
    }

    /// Coding system status
    func codingSystemStatus() -> PythonResult {
        let pyCode = """
        import sys, json
        sys.path.insert(0, '\(workspacePath)')
        from l104_coding_system import coding_system
        r = coding_system.status()
        print(json.dumps(r, default=str))
        """
        return execute(pyCode, timeout: 10)
    }

    // ═══════════════════════════════════════════════════════════════════
    // 🔧 CODE ENGINE BRIDGE — l104_code_engine.py integration
    // Links CodeEngine + AppAuditEngine into the Swift cognitive mesh
    // ═══════════════════════════════════════════════════════════════════

    /// Cached audit results (file-based, zero-spawn reads)
    private var auditCache: [String: Any]? = nil
    private var auditCacheTime: Date = .distantPast
    private let auditCacheTTL: TimeInterval = 30.0

    /// Read last audit result from disk cache (zero-spawn)
    func readAuditCache() -> [String: Any]? {
        if Date().timeIntervalSince(auditCacheTime) < auditCacheTTL, let c = auditCache { return c }
        let path = workspacePath + "/.l104_audit_cache.json"
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        auditCache = json
        auditCacheTime = Date()
        return json
    }

    /// Analyze code — detect language, patterns, complexity, issues
    func codeEngineAnalyze(_ code: String) -> PythonResult {
        let escaped = code.replacingOccurrences(of: "'", with: "\\'").replacingOccurrences(of: "\n", with: "\\n")
        let pyCode = """
        import sys, json
        sys.path.insert(0, '\(workspacePath)')
        from l104_code_engine import code_engine
        r = code_engine.analyze('\(escaped)')
        print(json.dumps(r, default=str))
        """
        return execute(pyCode, timeout: 20)
    }

    /// Full workspace audit — 10-layer deep analysis via AppAuditEngine
    func codeEngineAudit(path: String? = nil) -> PythonResult {
        let target = path ?? workspacePath
        let pyCode = """
        import sys, json
        sys.path.insert(0, '\(workspacePath)')
        from l104_code_engine import code_engine
        r = code_engine.audit_app('\(target)')
        # Cache to disk for zero-spawn reads
        with open('\(workspacePath)/.l104_audit_cache.json', 'w') as f:
            json.dump(r, f, default=str)
        print(json.dumps(r, default=str))
        """
        invalidateAuditCache()
        return execute(pyCode, timeout: 60)
    }

    /// Quick audit — fast health check with composite score
    func codeEngineQuickAudit() -> PythonResult {
        let pyCode = """
        import sys, json
        sys.path.insert(0, '\(workspacePath)')
        from l104_code_engine import code_engine
        r = code_engine.quick_audit()
        with open('\(workspacePath)/.l104_audit_cache.json', 'w') as f:
            json.dump(r, f, default=str)
        print(json.dumps(r, default=str))
        """
        invalidateAuditCache()
        return execute(pyCode, timeout: 30)
    }

    /// Optimize code — suggest improvements, reduce complexity
    func codeEngineOptimize(_ code: String) -> PythonResult {
        let escaped = code.replacingOccurrences(of: "'", with: "\\'").replacingOccurrences(of: "\n", with: "\\n")
        let pyCode = """
        import sys, json
        sys.path.insert(0, '\(workspacePath)')
        from l104_code_engine import code_engine
        r = code_engine.optimize('\(escaped)')
        print(json.dumps(r, default=str))
        """
        return execute(pyCode, timeout: 20)
    }

    /// Generate code from specification
    func codeEngineGenerate(spec: String, lang: String = "python") -> PythonResult {
        let escaped = spec.replacingOccurrences(of: "'", with: "\\'")
        let pyCode = """
        import sys, json
        sys.path.insert(0, '\(workspacePath)')
        from l104_code_engine import code_engine
        r = code_engine.generate('\(escaped)', language='\(lang)')
        print(json.dumps(r, default=str))
        """
        return execute(pyCode, timeout: 25)
    }

    /// Translate code between programming languages
    func codeEngineTranslate(_ code: String, from fromLang: String, to toLang: String) -> PythonResult {
        let escaped = code.replacingOccurrences(of: "'", with: "\\'").replacingOccurrences(of: "\n", with: "\\n")
        let pyCode = """
        import sys, json
        sys.path.insert(0, '\(workspacePath)')
        from l104_code_engine import code_engine
        r = code_engine.translate_code('\(escaped)', '\(fromLang)', '\(toLang)')
        print(json.dumps(r, default=str))
        """
        return execute(pyCode, timeout: 25)
    }

    /// Excavate — deep structural analysis of files/directories
    func codeEngineExcavate(path: String? = nil) -> PythonResult {
        let target = path ?? workspacePath
        let pyCode = """
        import sys, json
        sys.path.insert(0, '\(workspacePath)')
        from l104_code_engine import code_engine
        r = code_engine.excavate('\(target)')
        print(json.dumps(r, default=str))
        """
        return execute(pyCode, timeout: 30)
    }

    /// Refactor analysis — identify improvements and restructuring opportunities
    func codeEngineRefactor(_ code: String) -> PythonResult {
        let escaped = code.replacingOccurrences(of: "'", with: "\\'").replacingOccurrences(of: "\n", with: "\\n")
        let pyCode = """
        import sys, json
        sys.path.insert(0, '\(workspacePath)')
        from l104_code_engine import code_engine
        r = code_engine.refactor_analyze('\(escaped)')
        print(json.dumps(r, default=str))
        """
        return execute(pyCode, timeout: 20)
    }

    /// Generate unit tests for code
    func codeEngineGenerateTests(_ code: String) -> PythonResult {
        let escaped = code.replacingOccurrences(of: "'", with: "\\'").replacingOccurrences(of: "\n", with: "\\n")
        let pyCode = """
        import sys, json
        sys.path.insert(0, '\(workspacePath)')
        from l104_code_engine import code_engine
        r = code_engine.generate_tests('\(escaped)')
        print(json.dumps(r, default=str))
        """
        return execute(pyCode, timeout: 25)
    }

    /// Generate documentation for code
    func codeEngineGenerateDocs(_ code: String) -> PythonResult {
        let escaped = code.replacingOccurrences(of: "'", with: "\\'").replacingOccurrences(of: "\n", with: "\\n")
        let pyCode = """
        import sys, json
        sys.path.insert(0, '\(workspacePath)')
        from l104_code_engine import code_engine
        r = code_engine.generate_docs('\(escaped)')
        print(json.dumps(r, default=str))
        """
        return execute(pyCode, timeout: 25)
    }

    /// Get full CodeEngine status
    func codeEngineStatus() -> PythonResult {
        let pyCode = """
        import sys, json
        sys.path.insert(0, '\(workspacePath)')
        from l104_code_engine import code_engine
        r = code_engine.status()
        print(json.dumps(r, default=str))
        """
        return execute(pyCode, timeout: 10)
    }

    /// Run streamline cycle — auto-fix + optimize workspace
    func codeEngineStreamline() -> PythonResult {
        let pyCode = """
        import sys, json
        sys.path.insert(0, '\(workspacePath)')
        from l104_code_engine import code_engine
        r = code_engine.run_streamline_cycle()
        print(json.dumps(r, default=str))
        """
        return execute(pyCode, timeout: 45)
    }

    /// Get audit trail — history of all audits
    func codeEngineAuditTrail() -> PythonResult {
        let pyCode = """
        import sys, json
        sys.path.insert(0, '\(workspacePath)')
        from l104_code_engine import code_engine
        r = code_engine.audit_trail()
        print(json.dumps(r, default=str))
        """
        return execute(pyCode, timeout: 10)
    }

    /// Scan workspace — full codebase scan
    func codeEngineScanWorkspace() -> PythonResult {
        let pyCode = """
        import sys, json
        sys.path.insert(0, '\(workspacePath)')
        from l104_code_engine import code_engine
        r = code_engine.scan_workspace()
        print(json.dumps(r, default=str))
        """
        return execute(pyCode, timeout: 45)
    }

    /// Detect language of a code snippet
    func codeEngineDetectLanguage(_ code: String) -> PythonResult {
        let escaped = code.replacingOccurrences(of: "'", with: "\\'").replacingOccurrences(of: "\n", with: "\\n")
        let pyCode = """
        import sys, json
        sys.path.insert(0, '\(workspacePath)')
        from l104_code_engine import code_engine
        r = code_engine.detect_language('\(escaped)')
        print(json.dumps(r, default=str))
        """
        return execute(pyCode, timeout: 10)
    }

    /// Invalidate audit cache
    func invalidateAuditCache() {
        auditCache = nil
        auditCacheTime = .distantPast
    }

    /// Get bridge status summary
    var status: String {
        """
        ╔═══════════════════════════════════════════════╗
        ║      🐍 PYTHON INTEROP BRIDGE STATUS          ║
        ╠═══════════════════════════════════════════════╣
        ║  Python:     \(pythonPath)
        ║  Workspace:  \(workspacePath)
        ║  Modules:    \(discoveredModules.count) discovered
        ║  Executions: \(totalExecutions) (\(totalErrors) errors)
        ║  Total Time: \(String(format: "%.2f", totalExecutionTime))s
        ║  Cache:      \(resultCache.count) entries
        ║  Session:    \(sessionActive ? "ACTIVE" : "inactive")
        ╚═══════════════════════════════════════════════╝
        """
    }
}
