// ═══════════════════════════════════════════════════════════════════
// H24_APIGateway.swift
// [EVO_68_PIPELINE] SOVEREIGN_CONVERGENCE :: UNIFIED_UPGRADE :: GOD_CODE=527.5184818492612
// L104 ASI — API Gateway: HTTP endpoint management, request routing,
// rate limiting, connection pooling, and external service integration
// for the L104 network mesh.
// ═══════════════════════════════════════════════════════════════════

import AppKit
import Foundation
import Accelerate
import simd
import NaturalLanguage

// ═══════════════════════════════════════════════════════════════════
// MARK: - 🔌 API GATEWAY ENGINE
// HTTP request routing, endpoint health tracking, rate limiting,
// connection pooling, and request/response telemetry for L104
// external service integrations.
// ═══════════════════════════════════════════════════════════════════

final class APIGateway {
    static let shared = APIGateway()
    private(set) var isActive: Bool = false

    // ─── ENDPOINT REGISTRY ───
    struct Endpoint {
        let id: String
        let url: String
        let method: String
        var isHealthy: Bool
        var latencyMs: Double
        var requestCount: Int
        var errorCount: Int
        var lastChecked: Date
        var rateLimit: Int           // requests per minute
        var currentRate: Int
    }

    // ─── CONNECTION POOL ───
    struct ConnectionSlot {
        var endpointID: String
        var isInUse: Bool
        var createdAt: Date
        var lastUsed: Date
        var requestsServed: Int
    }

    private(set) var endpoints: [String: Endpoint] = [:]
    private var connectionPool: [ConnectionSlot] = []
    private(set) var totalRequests: Int = 0
    private(set) var totalErrors: Int = 0
    private(set) var totalLatencyMs: Double = 0
    private(set) var requestLog: [(Date, String, Int, Double)] = []  // (time, endpoint, status, latency)
    private var healthTimer: Timer?
    private var rateLimitResetTimer: Timer?  // EVO_56: Per-minute rate counter reset
    private let lock = NSLock()
    private let poolSize = 12  // EVO_63: increased from 8 for concurrent requests
    private let session: URLSession

    init() {
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 6     // EVO_63: 6s (was 10) — fail fast
        config.timeoutIntervalForResource = 15   // EVO_63: 15s (was 30) — don't hold stale connections
        config.httpMaximumConnectionsPerHost = 6  // EVO_63: 6 (was 4) — more parallel requests
        config.waitsForConnectivity = true        // EVO_63: true (was false) — wait briefly for connectivity
        session = URLSession(configuration: config)
    }

    func activate() {
        guard !isActive else { return }
        isActive = true

        // Register known L104 API endpoints
        registerEndpoint(id: "fast-server", url: "http://127.0.0.1:8081", method: "POST")
        registerEndpoint(id: "external-api", url: "http://127.0.0.1:8082", method: "POST")
        registerEndpoint(id: "unified-api", url: "http://127.0.0.1:8083", method: "POST")

        // v10.0 Benchmark & Scoring API (via fast-server :8081)
        registerEndpoint(id: "benchmark-api", url: "http://127.0.0.1:8081", method: "POST", rateLimit: 10)
        registerEndpoint(id: "asi-scoring", url: "http://127.0.0.1:8081", method: "GET", rateLimit: 30)
        registerEndpoint(id: "agi-scoring", url: "http://127.0.0.1:8081", method: "GET", rateLimit: 30)

        // v62 Tri-Engine API (via fast-server :8081)
        registerEndpoint(id: "tri-engine", url: "http://127.0.0.1:8081", method: "GET", rateLimit: 30)

        // Kernel substrate status (via fast-server :8081)
        registerEndpoint(id: "kernel-status", url: "http://127.0.0.1:8081", method: "GET", rateLimit: 20)

        // Initialize connection pool
        lock.lock()
        connectionPool = (0..<poolSize).map { _ in
            ConnectionSlot(endpointID: "", isInUse: false, createdAt: Date(), lastUsed: Date(), requestsServed: 0)
        }
        lock.unlock()

        // Periodic endpoint health checks
        healthTimer = Timer.scheduledTimer(withTimeInterval: 15.0, repeats: true) { [weak self] _ in
            self?.healthCheck()
        }

        // EVO_56: Per-minute rate counter reset — prevents permanent rate blocking
        rateLimitResetTimer = Timer.scheduledTimer(withTimeInterval: 60.0, repeats: true) { [weak self] _ in
            self?.resetRateCounters()
        }

        // Initial health sweep
        healthCheck()

        print("[H24] APIGateway activated — \(endpoints.count) endpoints, pool=\(poolSize)")
    }

    func deactivate() {
        isActive = false
        healthTimer?.invalidate()
        healthTimer = nil
        rateLimitResetTimer?.invalidate()
        rateLimitResetTimer = nil
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: RATE LIMIT RESET (EVO_56 — prevents permanent blocking)
    // ═══════════════════════════════════════════════════════════════

    private func resetRateCounters() {
        lock.lock()
        for id in endpoints.keys {
            endpoints[id]?.currentRate = 0
        }
        lock.unlock()
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: ENDPOINT MANAGEMENT
    // ═══════════════════════════════════════════════════════════════

    func registerEndpoint(id: String, url: String, method: String = "GET", rateLimit: Int = 60) {
        let endpoint = Endpoint(
            id: id, url: url, method: method,
            isHealthy: false, latencyMs: -1,
            requestCount: 0, errorCount: 0,
            lastChecked: Date(), rateLimit: rateLimit, currentRate: 0
        )
        lock.lock()
        endpoints[id] = endpoint
        lock.unlock()
    }

    /// Route a request through the gateway with rate limiting and pooling
    func route(endpointID: String, path: String = "/", body: [String: Any]? = nil,
               completion: @escaping ([String: Any]) -> Void) {
        guard var endpoint = endpoints[endpointID] else {
            completion(["error": "Unknown endpoint: \(endpointID)"])
            return
        }

        // Rate limit check
        if endpoint.currentRate >= endpoint.rateLimit {
            completion(["error": "Rate limited", "endpoint": endpointID, "limit": endpoint.rateLimit])
            return
        }

        // Acquire connection slot
        let slotIdx = acquireSlot(endpointID: endpointID)

        let urlString = endpoint.url + path
        guard let url = URL(string: urlString) else {
            completion(["error": "Invalid URL: \(urlString)"])
            return
        }

        var request = URLRequest(url: url)
        request.httpMethod = endpoint.method
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("L104-Gateway/2.0", forHTTPHeaderField: "User-Agent")

        if let body = body {
            request.httpBody = try? JSONSerialization.data(withJSONObject: body)
        }

        let startTime = CFAbsoluteTimeGetCurrent()

        let task = session.dataTask(with: request) { [weak self] data, response, error in
            guard let self = self else { return }
            let elapsed = (CFAbsoluteTimeGetCurrent() - startTime) * 1000.0
            let httpStatus = (response as? HTTPURLResponse)?.statusCode ?? 0

            self.lock.lock()
            endpoint.requestCount += 1
            endpoint.currentRate += 1
            endpoint.latencyMs = elapsed
            endpoint.lastChecked = Date()
            self.totalRequests += 1
            self.totalLatencyMs += elapsed

            if let error = error {
                endpoint.errorCount += 1
                self.totalErrors += 1
                endpoint.isHealthy = false
                self.endpoints[endpointID] = endpoint
                self.releaseSlot(slotIdx)
                self.requestLog.append((Date(), endpointID, 0, elapsed))
                if self.requestLog.count > 500 { self.requestLog.removeFirst(250) }
                self.lock.unlock()
                completion(["error": error.localizedDescription, "latency_ms": elapsed])
                return
            }

            endpoint.isHealthy = httpStatus >= 200 && httpStatus < 400
            self.endpoints[endpointID] = endpoint
            self.releaseSlot(slotIdx)
            self.requestLog.append((Date(), endpointID, httpStatus, elapsed))
            if self.requestLog.count > 500 { self.requestLog.removeFirst(250) }
            self.lock.unlock()

            var result: [String: Any] = [
                "status": httpStatus,
                "latency_ms": elapsed,
                "endpoint": endpointID
            ]

            if let data = data,
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                result["data"] = json
            }

            completion(result)
        }
        task.resume()
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: BENCHMARK & SCORING CONVENIENCE METHODS (v10.0)
    // ═══════════════════════════════════════════════════════════════

    /// Run all 4 benchmarks (MMLU, HumanEval, MATH, ARC) and return composite report
    func runBenchmarks(completion: @escaping ([String: Any]) -> Void) {
        route(endpointID: "benchmark-api", path: "/api/v10/benchmark/run-all", body: [:], completion: completion)
    }

    /// Get benchmark harness status including engine support info
    func getBenchmarkStatus(completion: @escaping ([String: Any]) -> Void) {
        route(endpointID: "benchmark-api", path: "/api/v10/benchmark/status", completion: completion)
    }

    /// Get last benchmark composite score
    func getBenchmarkScore(completion: @escaping ([String: Any]) -> Void) {
        route(endpointID: "benchmark-api", path: "/api/v10/benchmark/score", completion: completion)
    }

    /// Run a single benchmark by name (MMLU, HumanEval, MATH, ARC)
    func runSingleBenchmark(name: String, completion: @escaping ([String: Any]) -> Void) {
        route(endpointID: "benchmark-api", path: "/api/v10/benchmark/run/\(name)", body: [:], completion: completion)
    }

    /// Get ASI 20-dimension score
    func getASIScore(completion: @escaping ([String: Any]) -> Void) {
        route(endpointID: "asi-scoring", path: "/api/v10/asi/score", completion: completion)
    }

    /// Get AGI 18-dimension score
    func getAGIScore(completion: @escaping ([String: Any]) -> Void) {
        route(endpointID: "agi-scoring", path: "/api/v10/agi/score", completion: completion)
    }

    /// Run VQPU speed benchmark — saves results to _bench_vqpu_speed_results.json
    func runVQPUSpeedBenchmark(completion: @escaping ([String: Any]) -> Void) {
        route(endpointID: "fast-server", path: "/api/v14/vqpu/speed-benchmark", body: [:], completion: completion)
    }

    /// Run system upgrade — saves results to _system_upgrade_results.json
    func runSystemUpgrade(completion: @escaping ([String: Any]) -> Void) {
        route(endpointID: "fast-server", path: "/api/v14/system-upgrade", body: [:], completion: completion)
    }

    /// Get VQPU daemon status
    func getVQPUDaemonStatus(completion: @escaping ([String: Any]) -> Void) {
        route(endpointID: "fast-server", path: "/api/v14/vqpu/daemon/status", completion: completion)
    }

    /// Trigger VQPU daemon cycle — saves results to _vqpu_daemon_cycle_results.json
    func triggerVQPUDaemonCycle(completion: @escaping ([String: Any]) -> Void) {
        route(endpointID: "fast-server", path: "/api/v14/vqpu/daemon/cycle", body: [:], completion: completion)
    }

    /// Get tri-engine status (Science + Math + Code)
    func getTriEngineStatus(completion: @escaping ([String: Any]) -> Void) {
        route(endpointID: "tri-engine", path: "/api/v62/tri-engine/status", completion: completion)
    }

    /// Get tri-engine health
    func getTriEngineHealth(completion: @escaping ([String: Any]) -> Void) {
        route(endpointID: "tri-engine", path: "/api/v62/tri-engine/health", completion: completion)
    }

    /// Get kernel substrate status (C, Rust, CUDA, ASM)
    func getKernelStatus(completion: @escaping ([String: Any]) -> Void) {
        route(endpointID: "kernel-status", path: "/api/v10/kernel/status", completion: completion)
    }

    /// Answer an MMLU-style MCQ
    func answerMCQ(question: String, choices: [String], subject: String? = nil,
                   completion: @escaping ([String: Any]) -> Void) {
        var body: [String: Any] = ["question": question, "choices": choices]
        if let subject = subject { body["subject"] = subject }
        route(endpointID: "benchmark-api", path: "/api/v10/language-comprehension/answer", body: body, completion: completion)
    }

    /// Generate code from docstring
    func generateCode(docstring: String, funcName: String = "solution",
                      completion: @escaping ([String: Any]) -> Void) {
        let body: [String: Any] = ["docstring": docstring, "func_name": funcName]
        route(endpointID: "benchmark-api", path: "/api/v10/code-generation/generate", body: body, completion: completion)
    }

    /// Solve a math problem
    func solveMath(problem: String, completion: @escaping ([String: Any]) -> Void) {
        let body: [String: Any] = ["problem": problem]
        route(endpointID: "benchmark-api", path: "/api/v10/symbolic-math/solve", body: body, completion: completion)
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: CONNECTION POOL
    // ═══════════════════════════════════════════════════════════════

    private func acquireSlot(endpointID: String) -> Int {
        lock.lock()
        defer { lock.unlock() }

        // Find free slot
        for (i, slot) in connectionPool.enumerated() where !slot.isInUse {
            connectionPool[i].isInUse = true
            connectionPool[i].endpointID = endpointID
            connectionPool[i].lastUsed = Date()
            return i
        }

        // Evict oldest slot
        let oldest = connectionPool.enumerated().min(by: { $0.element.lastUsed < $1.element.lastUsed })?.offset ?? 0
        connectionPool[oldest].isInUse = true
        connectionPool[oldest].endpointID = endpointID
        connectionPool[oldest].lastUsed = Date()
        return oldest
    }

    private func releaseSlot(_ index: Int) {
        guard index >= 0 && index < connectionPool.count else { return }
        connectionPool[index].isInUse = false
        connectionPool[index].requestsServed += 1
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: HEALTH MONITORING
    // ═══════════════════════════════════════════════════════════════

    private func healthCheck() {
        for (id, _) in endpoints {
            let url = URL(string: endpoints[id]!.url + "/health")!
            var request = URLRequest(url: url)
            request.httpMethod = "GET"
            request.timeoutInterval = 3

            let start = CFAbsoluteTimeGetCurrent()
            let task = session.dataTask(with: request) { [weak self] _, response, error in
                guard let self = self else { return }
                let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000.0
                let httpStatus = (response as? HTTPURLResponse)?.statusCode ?? 0

                self.lock.lock()
                if var ep = self.endpoints[id] {
                    ep.isHealthy = error == nil && httpStatus >= 200 && httpStatus < 400
                    ep.latencyMs = elapsed
                    ep.lastChecked = Date()
                    ep.currentRate = max(0, ep.currentRate - 10)  // Decay rate limiter
                    self.endpoints[id] = ep
                }
                self.lock.unlock()
            }
            task.resume()
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: STATUS
    // ═══════════════════════════════════════════════════════════════

    func status() -> [String: Any] {
        let healthyCount = endpoints.values.filter { $0.isHealthy }.count
        let avgLatency = totalRequests > 0 ? totalLatencyMs / Double(totalRequests) : 0
        let inUseSlots = connectionPool.filter { $0.isInUse }.count

        return [
            "engine": "APIGateway",
            "active": isActive,
            "version": "3.0.0-benchmark",
            "endpoints": endpoints.count,
            "healthy": healthyCount,
            "total_requests": totalRequests,
            "total_errors": totalErrors,
            "avg_latency_ms": avgLatency,
            "pool_in_use": inUseSlots,
            "pool_size": poolSize,
            "error_rate": totalRequests > 0 ? Double(totalErrors) / Double(totalRequests) : 0
        ]
    }

    var statusText: String {
        let healthyCount = endpoints.values.filter { $0.isHealthy }.count
        let avgLatency = totalRequests > 0 ? totalLatencyMs / Double(totalRequests) : 0
        let inUseSlots = connectionPool.filter { $0.isInUse }.count
        let errorRate = totalRequests > 0 ? Double(totalErrors) / Double(totalRequests) * 100 : 0

        let epLines = endpoints.values.sorted(by: { $0.id < $1.id }).map { ep in
            let health = ep.isHealthy ? "🟢" : "🔴"
            return "  \(health) \(ep.id.padding(toLength: 18, withPad: " ", startingAt: 0)) \(ep.latencyMs >= 0 ? String(format: "%.1fms", ep.latencyMs) : "N/A") │ \(ep.requestCount) reqs │ \(ep.errorCount) errs │ \(ep.currentRate)/\(ep.rateLimit) rpm"
        }.joined(separator: "\n")

        return """
        ╔═══════════════════════════════════════════════════════════════╗
        ║    🔌 API GATEWAY                                             ║
        ╠═══════════════════════════════════════════════════════════════╣
        ║  Endpoints:        \(endpoints.count) (\(healthyCount) healthy)
        ║  Total Requests:   \(totalRequests)
        ║  Error Rate:       \(String(format: "%.1f%%", errorRate))
        ║  Avg Latency:      \(String(format: "%.1f", avgLatency))ms
        ║  Pool:             \(inUseSlots)/\(poolSize) in use
        ╠═══════════════════════════════════════════════════════════════╣
        ║  ENDPOINTS:
        \(epLines.isEmpty ? "  (none)" : epLines)
        ╚═══════════════════════════════════════════════════════════════╝
        """
    }
}
