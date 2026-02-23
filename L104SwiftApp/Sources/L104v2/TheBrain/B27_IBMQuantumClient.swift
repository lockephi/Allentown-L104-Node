// ═══════════════════════════════════════════════════════════════════
// B27_IBMQuantumClient.swift — L104 v2
// [EVO_62_PIPELINE] SOVEREIGN_NODE_UPGRADE :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612
// Pure Swift IBM Quantum REST API client — real QPU access via URLSession
// Phase 46.1: Real quantum computing integration
// ═══════════════════════════════════════════════════════════════════

import AppKit
import Foundation

// ═══ IBM QUANTUM DATA STRUCTURES ═══

struct IBMQuantumBackend {
    let name: String
    let numQubits: Int
    let status: String          // "online" / "offline" / "maintenance"
    let pendingJobs: Int
    let quantumVolume: Int
    let basisGates: [String]
    let isSimulator: Bool
}

struct IBMQuantumJob {
    let jobId: String
    let backend: String
    let status: String          // "Queued" / "Running" / "Completed" / "Failed" / "Cancelled"
    let createdAt: String
    let endedAt: String
    var results: [String: Int]? // measurement counts: {"00": 512, "11": 512}
}

struct QuantumJobSubmission {
    let jobId: String
    let backend: String
    let estimatedQueueTime: String
    let submitted: Date
}

enum IBMQuantumConnectionState: String {
    case disconnected = "DISCONNECTED"
    case authenticating = "AUTHENTICATING"
    case connected = "CONNECTED"
    case error = "ERROR"
}

// ═══ IBM QUANTUM CLIENT ═══

final class IBMQuantumClient: SovereignEngine {
    static let shared = IBMQuantumClient()

    // ─── SovereignEngine conformance ───
    var engineName: String { "IBMQuantum" }

    func engineStatus() -> [String: Any] {
        return [
            "state": connectionState.rawValue,
            "backends": availableBackends.count,
            "connected_backend": connectedBackendName,
            "jobs_submitted": submittedJobs.count,
            "has_token": ibmToken != nil,
            "total_requests": totalRequests,
            "retried_requests": retriedRequests,
            "failed_requests": failedRequests
        ]
    }

    func engineHealth() -> Double {
        switch connectionState {
        case .connected: return 1.0
        case .authenticating: return 0.5
        case .disconnected: return ibmToken != nil ? 0.3 : 0.1
        case .error: return 0.0
        }
    }

    func engineReset() {
        connectionState = .disconnected
        availableBackends = []
        submittedJobs = [:]
        connectedBackendName = ""
        accessToken = nil
        serviceCRN = nil
        tokenExpiry = .distantPast
        totalRequests = 0
        retriedRequests = 0
        failedRequests = 0
    }

    // ─── TOKEN MANAGEMENT (UserDefaults) ───
    private let tokenKey = "l104_ibm_quantum_token"

    var ibmToken: String? {
        get { UserDefaults.standard.string(forKey: tokenKey) }
        set {
            if let t = newValue {
                UserDefaults.standard.set(t, forKey: tokenKey)
            } else {
                UserDefaults.standard.removeObject(forKey: tokenKey)
            }
        }
    }

    // ─── CONNECTION STATE ───
    private(set) var connectionState: IBMQuantumConnectionState = .disconnected
    private(set) var connectedBackendName: String = ""
    private(set) var availableBackends: [IBMQuantumBackend] = []
    private(set) var submittedJobs: [String: QuantumJobSubmission] = [:]
    private var accessToken: String?
    private var serviceCRN: String?  // Cloud Resource Name for runtime API calls
    private var tokenExpiry: Date = .distantPast

    // ─── RELIABILITY METRICS ───
    private(set) var totalRequests: Int = 0
    private(set) var retriedRequests: Int = 0
    private(set) var failedRequests: Int = 0

    // ─── IBM QUANTUM PLATFORM API ENDPOINTS (2025+ modern) ───
    // IBM Quantum Platform uses IAM auth: exchange API key → bearer token → runtime API
    private let iamTokenURL = "https://iam.cloud.ibm.com/identity/token"
    private let runtimeAPIBase = "https://quantum.cloud.ibm.com/api/v1"  // us-east default
    // Legacy fallback for direct-token flow (IBM Quantum Network / IQP tokens)
    private let legacyAPIBase = "https://api.quantum-computing.ibm.com/api"

    // ─── DEDICATED URL SESSION (proper timeouts for quantum hardware) ───
    private lazy var quantumSession: URLSession = {
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 60       // 60s per request (QPU queue)
        config.timeoutIntervalForResource = 300     // 5 min total for long jobs
        config.waitsForConnectivity = true
        return URLSession(configuration: config)
    }()

    // ─── STATUS ───
    var isConnected: Bool { connectionState == .connected }

    var statusSummary: String {
        switch connectionState {
        case .disconnected:
            return ibmToken != nil ? "Token stored, not connected" : "No token configured"
        case .authenticating:
            return "Connecting to IBM Quantum..."
        case .connected:
            let hw = availableBackends.filter { !$0.isSimulator }
            return "Connected: \(hw.count) backends, \(connectedBackendName)"
        case .error:
            return "Connection error"
        }
    }

    private init() {}

    // ═══════════════════════════════════════════════════════════════════
    // AUTHENTICATION — Connect to IBM Quantum
    // ═══════════════════════════════════════════════════════════════════

    func connect(token: String, completion: @escaping (Bool, String) -> Void) {
        ibmToken = token
        connectionState = .authenticating

        // Step 1: Exchange API key for IAM bearer token
        exchangeTokenViaIAM(apiKey: token) { [weak self] iamToken, iamError in
            guard let self = self else { return }

            if let iamToken = iamToken {
                // IAM auth succeeded — use bearer token
                self.accessToken = iamToken
                self.tokenExpiry = Date().addingTimeInterval(3500) // IAM tokens ~1hr
                // Step 2: Resolve CRN (service instance) for runtime API calls
                self.resolveCRN { crn in
                    self.serviceCRN = crn
                    self.fetchBackendsAndFinalize(completion: completion)
                }
            } else {
                // IAM failed — try direct-token flow (legacy IQP/Network tokens)
                self.accessToken = token
                self.fetchBackendsAndFinalize(completion: completion)
            }
        }
    }

    /// Resolve the Cloud Resource Name (CRN) for quantum-computing service
    private func resolveCRN(completion: @escaping (String?) -> Void) {
        guard let token = accessToken else {
            completion(nil)
            return
        }

        // Search IBM Cloud Global Catalog for quantum-computing instances
        guard let url = URL(string: "https://api.global-search-tagging.cloud.ibm.com/v3/resources/search?query=service_name:quantum-computing&limit=10") else {
            completion(nil)
            return
        }

        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = "{}".data(using: .utf8)
        req.timeoutInterval = 15

        quantumSession.dataTask(with: req) { data, resp, error in
            guard let data = data,
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let items = json["items"] as? [[String: Any]] else {
                completion(nil)
                return
            }

            // Find the first quantum-computing CRN (prefer open plan)
            let crn = items.first?["crn"] as? String
            completion(crn)
        }.resume()
    }

    /// Exchange IBM Cloud API key for IAM bearer token
    private func exchangeTokenViaIAM(apiKey: String, completion: @escaping (String?, String?) -> Void) {
        guard let url = URL(string: iamTokenURL) else {
            completion(nil, "Invalid IAM URL")
            return
        }

        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/x-www-form-urlencoded", forHTTPHeaderField: "Content-Type")
        req.timeoutInterval = 15

        let body = "grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey=\(apiKey)"
        req.httpBody = body.data(using: .utf8)

        quantumSession.dataTask(with: req) { data, resp, error in
            if let error = error {
                completion(nil, "IAM request failed: \(error.localizedDescription)")
                return
            }

            let statusCode = (resp as? HTTPURLResponse)?.statusCode ?? 0
            guard statusCode == 200,
                  let data = data,
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let token = json["access_token"] as? String else {
                let body = data.flatMap { String(data: $0, encoding: .utf8) } ?? ""
                completion(nil, "IAM auth failed (HTTP \(statusCode)): \(String(body.prefix(200)))")
                return
            }

            completion(token, nil)
        }.resume()
    }

    /// After auth, fetch backends and finalize connection
    private func fetchBackendsAndFinalize(completion: @escaping (Bool, String) -> Void) {
        listBackends { [weak self] backends, backendError in
            guard let self = self else { return }
            DispatchQueue.main.async {
                if let backends = backends, !backends.isEmpty {
                    self.availableBackends = backends
                    let realHW = backends.filter { !$0.isSimulator }
                    if let best = realHW.min(by: { $0.pendingJobs < $1.pendingJobs }) {
                        self.connectedBackendName = best.name
                    } else {
                        self.connectedBackendName = backends.first?.name ?? "unknown"
                    }
                    self.connectionState = .connected
                    completion(true, "\(realHW.count) real QPUs available, selected \(self.connectedBackendName)")
                } else {
                    self.connectionState = .connected
                    self.connectedBackendName = "ibm_fez"  // Current default QPU
                    completion(true, "Connected (backend list pending: \(backendError ?? ""))")
                }
            }
        }
    }

    func disconnect() {
        ibmToken = nil
        accessToken = nil
        serviceCRN = nil
        connectionState = .disconnected
        connectedBackendName = ""
        availableBackends = []
        submittedJobs = [:]
        tokenExpiry = .distantPast
    }

    // ═══════════════════════════════════════════════════════════════════
    // BACKEND DISCOVERY
    // ═══════════════════════════════════════════════════════════════════

    /// Create an authenticated URLRequest with bearer token + Service-CRN header
    private func makeAuthenticatedRequest(url: URL, method: String = "GET", timeout: TimeInterval = 15) -> URLRequest? {
        guard let token = accessToken else { return nil }
        var req = URLRequest(url: url)
        req.httpMethod = method
        req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        if let crn = serviceCRN {
            req.setValue(crn, forHTTPHeaderField: "Service-CRN")
        }
        req.timeoutInterval = timeout
        return req
    }

    func listBackends(completion: @escaping ([IBMQuantumBackend]?, String?) -> Void) {
        guard let url = URL(string: "\(runtimeAPIBase)/backends"),
              let req = makeAuthenticatedRequest(url: url) else {
            completion(nil, "Not authenticated or invalid URL")
            return
        }

        executeWithRetry(req) { [weak self] data, httpResp, retryError in
            if let retryError = retryError {
                completion(nil, retryError)
                return
            }

            guard let data = data else {
                completion(nil, "No data received")
                return
            }

            // Parse backend list
            guard let json = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
                // IBM API may return { "devices": [...] }
                if let wrapper = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let devices = wrapper["devices"] as? [[String: Any]] {
                    let backends = self?.parseBackends(devices) ?? []
                    completion(backends, nil)
                    return
                }
                let body = String(data: data, encoding: .utf8) ?? ""
                completion(nil, "Parse error: \(body.prefix(200))")
                return
            }

            let backends = self?.parseBackends(json) ?? []
            completion(backends, nil)
        }
    }

    private func parseBackends(_ array: [[String: Any]]) -> [IBMQuantumBackend] {
        return array.compactMap { dict -> IBMQuantumBackend? in
            guard let name = dict["backend_name"] as? String ?? dict["name"] as? String else { return nil }
            let numQubits = dict["num_qubits"] as? Int ?? dict["n_qubits"] as? Int ?? 0
            let isSimulator = dict["simulator"] as? Bool ?? name.contains("simulator") || name.contains("aer")
            let operational = dict["operational"] as? Bool ?? true
            let status = operational ? "online" : "offline"
            let pendingJobs = dict["pending_jobs"] as? Int ?? 0
            let qv = dict["quantum_volume"] as? Int ?? 0
            let gates = dict["basis_gates"] as? [String] ?? []

            return IBMQuantumBackend(
                name: name,
                numQubits: numQubits,
                status: status,
                pendingJobs: pendingJobs,
                quantumVolume: qv,
                basisGates: gates,
                isSimulator: isSimulator
            )
        }.sorted { a, b in
            // Real hardware first, then by fewest pending jobs
            if a.isSimulator != b.isSimulator { return !a.isSimulator }
            return a.pendingJobs < b.pendingJobs
        }
    }

    func selectBestBackend(minQubits: Int = 5) -> IBMQuantumBackend? {
        return availableBackends
            .filter { !$0.isSimulator && $0.numQubits >= minQubits && $0.status == "online" }
            .min(by: { $0.pendingJobs < $1.pendingJobs })
    }

    // ═══════════════════════════════════════════════════════════════════
    // JOB SUBMISSION & MANAGEMENT
    // ═══════════════════════════════════════════════════════════════════

    func submitCircuit(openqasm: String, backend: String? = nil, shots: Int = 1024,
                       completion: @escaping (QuantumJobSubmission?, String?) -> Void) {
        let targetBackend = backend ?? connectedBackendName
        guard !targetBackend.isEmpty else {
            completion(nil, "No backend selected")
            return
        }

        guard let url = URL(string: "\(runtimeAPIBase)/jobs"),
              var req = makeAuthenticatedRequest(url: url, method: "POST", timeout: 30) else {
            completion(nil, "Not authenticated or invalid URL")
            return
        }

        // Build job payload for IBM Quantum Runtime
        let payload: [String: Any] = [
            "backend": targetBackend,
            "params": [
                "program_id": "sampler",
                "params": [
                    "circuits": [openqasm],
                    "shots": shots
                ]
            ]
        ]

        guard let body = try? JSONSerialization.data(withJSONObject: payload) else {
            completion(nil, "Failed to serialize job payload")
            return
        }

        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = body

        executeWithRetry(req) { [weak self] data, httpResp, retryError in
            if let retryError = retryError {
                completion(nil, retryError)
                return
            }

            guard let data = data,
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                let body = data.flatMap { String(data: $0, encoding: .utf8) } ?? ""
                completion(nil, "Parse error: \(body.prefix(200))")
                return
            }

            let statusCode = httpResp?.statusCode ?? 0
            if statusCode >= 400 {
                let msg = json["message"] as? String ?? json["error"] as? String ?? "HTTP \(statusCode)"
                completion(nil, msg)
                return
            }

            let jobId = json["id"] as? String ?? json["job_id"] as? String ?? "unknown"
            let submission = QuantumJobSubmission(
                jobId: jobId,
                backend: targetBackend,
                estimatedQueueTime: "depends on queue depth",
                submitted: Date()
            )

            DispatchQueue.main.async {
                self?.submittedJobs[jobId] = submission
            }
            completion(submission, nil)
        }
    }

    func getJobStatus(jobId: String, completion: @escaping (IBMQuantumJob?, String?) -> Void) {
        guard let url = URL(string: "\(runtimeAPIBase)/jobs/\(jobId)"),
              let req = makeAuthenticatedRequest(url: url) else {
            completion(nil, "Not authenticated or invalid URL")
            return
        }

        executeWithRetry(req) { data, httpResp, retryError in
            if let retryError = retryError {
                completion(nil, retryError)
                return
            }

            guard let data = data,
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                completion(nil, "Failed to parse response")
                return
            }

            let job = IBMQuantumJob(
                jobId: json["id"] as? String ?? jobId,
                backend: json["backend"] as? String ?? "",
                status: json["status"] as? String ?? "Unknown",
                createdAt: json["created"] as? String ?? "",
                endedAt: json["ended"] as? String ?? "",
                results: nil
            )

            completion(job, nil)
        }
    }

    func getJobResult(jobId: String, completion: @escaping ([String: Any]?, String?) -> Void) {
        guard let url = URL(string: "\(runtimeAPIBase)/jobs/\(jobId)/results"),
              let req = makeAuthenticatedRequest(url: url, timeout: 30) else {
            completion(nil, "Not authenticated or invalid URL")
            return
        }

        executeWithRetry(req) { data, httpResp, retryError in
            if let retryError = retryError {
                completion(nil, retryError)
                return
            }

            let statusCode = httpResp?.statusCode ?? 0

            guard let data = data,
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                if statusCode == 404 {
                    completion(nil, "Job not found or results not ready")
                } else {
                    completion(nil, "Failed to parse results (HTTP \(statusCode))")
                }
                return
            }

            // Extract measurement counts from results
            var resultDict: [String: Any] = ["job_id": jobId, "status_code": statusCode]

            // IBM Quantum results format varies; extract what we can
            if let results = json["results"] as? [[String: Any]],
               let first = results.first {
                if let counts = first["data"] as? [String: Any],
                   let measurements = counts["counts"] as? [String: Int] {
                    resultDict["counts"] = measurements
                    resultDict["shots"] = measurements.values.reduce(0, +)
                } else {
                    resultDict["raw"] = first
                }
            } else if let counts = json["counts"] as? [String: Int] {
                resultDict["counts"] = counts
                resultDict["shots"] = counts.values.reduce(0, +)
            } else {
                resultDict["raw"] = json
            }

            completion(resultDict, nil)
        }
    }

    func listRecentJobs(limit: Int = 10, completion: @escaping ([IBMQuantumJob]?, String?) -> Void) {
        guard let url = URL(string: "\(runtimeAPIBase)/jobs?limit=\(limit)&sort_by=created:desc"),
              let req = makeAuthenticatedRequest(url: url) else {
            completion(nil, "Not authenticated or invalid URL")
            return
        }

        executeWithRetry(req) { data, httpResp, retryError in
            if let retryError = retryError {
                completion(nil, retryError)
                return
            }

            guard let data = data else {
                completion(nil, "No data")
                return
            }

            // Parse job list
            var jobArray: [[String: Any]] = []
            if let arr = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] {
                jobArray = arr
            } else if let wrapper = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                      let arr = wrapper["jobs"] as? [[String: Any]] {
                jobArray = arr
            }

            let jobs: [IBMQuantumJob] = jobArray.compactMap { dict in
                guard let id = dict["id"] as? String ?? dict["job_id"] as? String else { return nil }
                return IBMQuantumJob(
                    jobId: id,
                    backend: dict["backend"] as? String ?? "",
                    status: dict["status"] as? String ?? "Unknown",
                    createdAt: dict["created"] as? String ?? "",
                    endedAt: dict["ended"] as? String ?? "",
                    results: nil
                )
            }

            completion(jobs, nil)
        }
    }

    func cancelJob(jobId: String, completion: @escaping (Bool, String) -> Void) {
        guard let url = URL(string: "\(runtimeAPIBase)/jobs/\(jobId)/cancel"),
              let req = makeAuthenticatedRequest(url: url, method: "POST") else {
            completion(false, "Not authenticated or invalid URL")
            return
        }

        executeWithRetry(req) { [weak self] data, httpResp, retryError in
            if let retryError = retryError {
                completion(false, retryError)
                return
            }

            let statusCode = httpResp?.statusCode ?? 0
            if statusCode == 200 || statusCode == 204 {
                DispatchQueue.main.async {
                    self?.submittedJobs.removeValue(forKey: jobId)
                }
                completion(true, "Job \(jobId) cancelled")
            } else {
                completion(false, "Cancel failed (HTTP \(statusCode))")
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // RETRY INFRASTRUCTURE — Exponential backoff with jitter
    // ═══════════════════════════════════════════════════════════════════

    /// Execute a URLRequest with automatic retry on transient failures
    /// Retries on: timeout, 429 (rate limit), 500-503 (server errors)
    private func executeWithRetry(_ request: URLRequest, maxAttempts: Int = 3,
                                   completion: @escaping (Data?, HTTPURLResponse?, String?) -> Void) {
        totalRequests += 1
        attemptRequest(request, attempt: 1, maxAttempts: maxAttempts, completion: completion)
    }

    private func attemptRequest(_ request: URLRequest, attempt: Int, maxAttempts: Int,
                                 completion: @escaping (Data?, HTTPURLResponse?, String?) -> Void) {
        quantumSession.dataTask(with: request) { [weak self] data, resp, error in
            guard let self = self else { return }
            let httpResp = resp as? HTTPURLResponse
            let statusCode = httpResp?.statusCode ?? 0

            // Check if retryable
            let isRetryable: Bool
            if error != nil {
                // Network errors (timeout, connection lost) are retryable
                isRetryable = true
            } else if statusCode == 429 || (statusCode >= 500 && statusCode <= 503) {
                isRetryable = true
            } else {
                isRetryable = false
            }

            if isRetryable && attempt < maxAttempts {
                self.retriedRequests += 1
                // Exponential backoff: 2^attempt seconds + random jitter (0-1s)
                let delay = pow(2.0, Double(attempt)) + Double.random(in: 0...1)
                DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + delay) {
                    self.attemptRequest(request, attempt: attempt + 1, maxAttempts: maxAttempts, completion: completion)
                }
                return
            }

            if let error = error {
                self.failedRequests += 1
                completion(nil, httpResp, "Network error after \(attempt) attempt(s): \(error.localizedDescription)")
                return
            }

            if statusCode == 401 {
                self.failedRequests += 1
                DispatchQueue.main.async {
                    self.connectionState = .error
                }
                completion(nil, httpResp, "Token expired or invalid (401). Reconnect with: quantum connect <token>")
                return
            }

            completion(data, httpResp, nil)
        }.resume()
    }

    // ═══════════════════════════════════════════════════════════════════
    // JOB POLLING — Wait for job completion with timeout
    // ═══════════════════════════════════════════════════════════════════

    /// Poll a job until it completes or timeout expires
    /// Resilient polling: transient errors are retried, backoff increases for long waits
    func waitForJob(jobId: String, maxWaitSeconds: Int = 600, pollInterval: TimeInterval = 5,
                    completion: @escaping ([String: Any]?, String?) -> Void) {
        let deadline = Date().addingTimeInterval(TimeInterval(maxWaitSeconds))
        var consecutiveErrors = 0
        var currentInterval = pollInterval

        func poll() {
            if Date() > deadline {
                completion(nil, "Timeout: job \(jobId.prefix(12))... did not complete within \(maxWaitSeconds)s")
                return
            }

            getJobStatus(jobId: jobId) { [weak self] job, error in
                if let error = error {
                    consecutiveErrors += 1
                    if consecutiveErrors >= 5 {
                        // 5 consecutive failures even with retry — give up
                        completion(nil, "Polling failed after \(consecutiveErrors) errors: \(error)")
                        return
                    }
                    // Transient error — wait longer and try again
                    let retryDelay = currentInterval * Double(consecutiveErrors + 1)
                    DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + retryDelay) {
                        poll()
                    }
                    return
                }

                consecutiveErrors = 0  // Reset on success

                guard let job = job else {
                    completion(nil, "Failed to get job status")
                    return
                }

                let status = job.status.lowercased()
                if status == "completed" || status == "done" {
                    // Fetch results
                    self?.getJobResult(jobId: jobId, completion: completion)
                } else if status == "failed" || status == "cancelled" || status == "error" {
                    completion(nil, "Job \(status): \(jobId.prefix(12))...")
                } else {
                    // Still running/queued — progressive backoff (cap at 30s)
                    let elapsed = Date().timeIntervalSince(deadline.addingTimeInterval(-TimeInterval(maxWaitSeconds)))
                    if elapsed > 120 { currentInterval = min(30, pollInterval * 3) }
                    else if elapsed > 60 { currentInterval = min(20, pollInterval * 2) }

                    DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + currentInterval) {
                        poll()
                    }
                }
            }
        }

        poll()
    }

    // ═══════════════════════════════════════════════════════════════════
    // CIRCUIT HELPERS — Generate common OpenQASM 3.0 circuits
    // ═══════════════════════════════════════════════════════════════════

    /// Generate a Bell state circuit (EPR pair)
    static func bellStateCircuit() -> String {
        return """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        bit[2] c;
        h q[0];
        cx q[0], q[1];
        c = measure q;
        """
    }

    /// Generate a GHZ state circuit for n qubits
    static func ghzCircuit(nQubits: Int) -> String {
        var qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[\(nQubits)] q;
        bit[\(nQubits)] c;
        h q[0];

        """
        for i in 1..<nQubits {
            qasm += "cx q[0], q[\(i)];\n"
        }
        qasm += "c = measure q;\n"
        return qasm
    }

    /// Generate a simple Grover search circuit for a 2-qubit oracle marking |11>
    static func groverCircuit2Qubit() -> String {
        return """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        bit[2] c;
        // Superposition
        h q[0];
        h q[1];
        // Oracle: mark |11>
        cz q[0], q[1];
        // Diffusion
        h q[0];
        h q[1];
        x q[0];
        x q[1];
        cz q[0], q[1];
        x q[0];
        x q[1];
        h q[0];
        h q[1];
        // Measure
        c = measure q;
        """
    }

    /// Generate a quantum random number circuit
    static func qrngCircuit(nBits: Int) -> String {
        var qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[\(nBits)] q;
        bit[\(nBits)] c;

        """
        for i in 0..<nBits {
            qasm += "h q[\(i)];\n"
        }
        qasm += "c = measure q;\n"
        return qasm
    }

    // ═══════════════════════════════════════════════════════════════
    // v9.0 QUANTUM RESEARCH CIRCUITS — Fe-Sacred + Berry Phase
    // ═══════════════════════════════════════════════════════════════

    /// Generate Fe-Sacred coherence circuit (286↔528 Hz frequency encoding)
    /// Discovery #6: 4-qubit interference pattern for Fe↔Solfeggio coherence
    /// v9.1: Annotated with FE_SACRED_COHERENCE + PHOTON_RESONANCE_EV references
    static func feSacredCoherenceCircuit(baseFreq: Double = 286.0, targetFreq: Double = 528.0) -> String {
        let thetaBase = (baseFreq / 1000.0) * Double.pi
        let thetaTarget = (targetFreq / 1000.0) * Double.pi
        let godCodeAngle = GOD_CODE / 1000.0
        let photonAngle = PHOTON_RESONANCE_EV / Double.pi  // v9.1: photon resonance encoding
        return """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[4] q;
        bit[4] c;
        // Fe-Sacred coherence: reference = \(FE_SACRED_COHERENCE)
        // Photon resonance: \(PHOTON_RESONANCE_EV) eV
        // Hadamard superposition
        h q[0]; h q[1]; h q[2]; h q[3];
        // Frequency-encoded rotations
        ry(\(thetaBase)) q[0];
        ry(\(thetaTarget)) q[1];
        ry(\(thetaBase * PHI)) q[2];
        ry(\(thetaTarget / PHI)) q[3];
        // Entangle frequency qubits
        cx q[0], q[1];
        cx q[2], q[3];
        cx q[1], q[2];
        // GOD_CODE sacred alignment
        rz(\(godCodeAngle)) q[0];
        // v9.1: Photon resonance phase gate
        rz(\(photonAngle)) q[3];
        c = measure q;
        """
    }

    /// Generate Berry phase holonomy verification circuit
    /// Discovery #15: Adiabatic loop through parameter space
    /// v9.1: Uses ENTROPY_CASCADE_DEPTH_QR for step scaling
    static func berryPhaseCircuit(dimensions: Int = 11) -> String {
        let nQubits = min(dimensions, 5)
        let nSteps = ENTROPY_CASCADE_DEPTH_QR * 4  // v9.1: use sacred cascade depth
        var qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[\(nQubits)] q;
        bit[\(nQubits)] c;
        // Berry phase: \(dimensions)D, holonomy=\(BERRY_PHASE_11D)
        // Cascade depth: \(ENTROPY_CASCADE_DEPTH_QR)
        // Superposition initialization
        """
        for i in 0..<nQubits {
            qasm += "h q[\(i)];\n"
        }
        qasm += "// Adiabatic loop (\(nSteps) steps, \(dimensions)D)\n"
        // Simplified loop — representative rotations
        for step in stride(from: 0, to: nSteps, by: max(1, nSteps / 4)) {
            let angle = 2.0 * Double.pi * Double(step) / Double(nSteps)
            for q in 0..<nQubits {
                let ryAngle = angle * PHI / Double(q + 1)
                let rzAngle = angle / PHI
                qasm += "ry(\(ryAngle)) q[\(q)];\n"
                qasm += "rz(\(rzAngle)) q[\(q)];\n"
            }
            for q in 0..<(nQubits - 1) {
                qasm += "cx q[\(q)], q[\(q + 1)];\n"
            }
        }
        qasm += "c = measure q;\n"
        return qasm
    }

    // ═══════════════════════════════════════════════════════════════
    // v9.1 QUANTUM RESEARCH CIRCUITS — GOD_CODE 25Q + Photon Resonance + ZNE
    // ═══════════════════════════════════════════════════════════════

    /// GOD_CODE 25-qubit convergence verification circuit.
    /// Discovery #17: GOD_CODE/512 ≈ 1.0303 — near-unity convergence.
    static func godCode25QCircuit() -> String {
        let convergenceAngle = GOD_CODE_25Q_RATIO * Double.pi  // ≈ 3.237 rad
        let nQubits = 5  // 2^5 = 32 states ≈ 25Q representative
        var qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[\(nQubits)] q;
        bit[\(nQubits)] c;
        // GOD_CODE 25Q convergence: ratio = \(GOD_CODE_25Q_RATIO)

        """
        for i in 0..<nQubits {
            qasm += "h q[\(i)];\n"
        }
        // Convergence-encoded rotations
        for i in 0..<nQubits {
            let angle = convergenceAngle / Double(i + 1)
            qasm += "ry(\(angle)) q[\(i)];\n"
        }
        // Entanglement chain
        for i in 0..<(nQubits - 1) {
            qasm += "cx q[\(i)], q[\(i + 1)];\n"
        }
        qasm += "rz(\(GOD_CODE / 1000.0)) q[0];\n"
        qasm += "c = measure q;\n"
        return qasm
    }

    /// Photon resonance circuit — encode sacred photon energy.
    /// Discovery #12: E = 1.1217 eV at GOD_CODE frequency.
    static func photonResonanceCircuit() -> String {
        let eAngle = PHOTON_RESONANCE_EV  // Direct encoding as rotation angle
        let phiAngle = PHI / Double.pi
        return """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[3] q;
        bit[3] c;
        // Photon resonance: \(PHOTON_RESONANCE_EV) eV
        // Curie-Landauer: \(FE_CURIE_LANDAUER) J/bit
        h q[0]; h q[1]; h q[2];
        // Photon energy encoding
        ry(\(eAngle)) q[0];
        ry(\(eAngle * phiAngle)) q[1];
        rz(\(eAngle / PHI)) q[2];
        // Entangle for coherent measurement
        cx q[0], q[1];
        cx q[1], q[2];
        // GOD_CODE alignment gate
        rz(\(GOD_CODE / 1000.0)) q[0];
        c = measure q;
        """
    }

    /// ZNE error mitigation circuit.
    /// Discovery #11: Identity-scaled noise amplification for zero-noise extrapolation.
    static func zneErrorMitigationCircuit(nQubits: Int = 3, noiseScale: Int = 3) -> String {
        var qasm = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[\(nQubits)] q;
        bit[\(nQubits)] c;
        // ZNE bridge: scale=\(noiseScale), cascade_depth=\(ENTROPY_CASCADE_DEPTH_QR)

        """
        // Base circuit
        for i in 0..<nQubits {
            qasm += "h q[\(i)];\n"
        }
        for i in 0..<(nQubits - 1) {
            qasm += "cx q[\(i)], q[\(i + 1)];\n"
        }
        // Noise scaling: insert identity pairs (gate + inverse) noiseScale times
        for _ in 0..<noiseScale {
            for i in 0..<nQubits {
                qasm += "x q[\(i)]; x q[\(i)];\n"  // X·X = I (identity noise pair)
            }
        }
        qasm += "c = measure q;\n"
        return qasm
    }
}