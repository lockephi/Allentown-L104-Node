// ═══════════════════════════════════════════════════════════════════
// B27_IBMQuantumClient.swift — L104 v2
// [EVO_56_APEX_WIRED] SOVEREIGN_UNIFICATION :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612
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
            "has_token": ibmToken != nil
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
        tokenExpiry = .distantPast
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
    private var tokenExpiry: Date = .distantPast

    // ─── IBM QUANTUM API ENDPOINTS ───
    // IBM Quantum Platform (ibm_quantum channel) uses API token directly as bearer
    private let quantumAPIBase = "https://api.quantum-computing.ibm.com/api"

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
        accessToken = token  // IBM Quantum Platform: API token IS the bearer token

        // Validate by fetching user info
        guard let url = URL(string: "\(quantumAPIBase)/users/me") else {
            connectionState = .error
            completion(false, "Invalid API URL")
            return
        }

        var req = URLRequest(url: url)
        req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        req.timeoutInterval = 15

        URLSession.shared.dataTask(with: req) { [weak self] data, resp, error in
            guard let self = self else { return }

            if let error = error {
                DispatchQueue.main.async {
                    self.connectionState = .error
                    completion(false, "Connection failed: \(error.localizedDescription)")
                }
                return
            }

            let statusCode = (resp as? HTTPURLResponse)?.statusCode ?? 0

            if statusCode == 200 || statusCode == 201 {
                // Auth succeeded — fetch backends
                self.listBackends { backends, backendError in
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
                            self.connectedBackendName = "ibm_brisbane"  // Common default
                            completion(true, "Connected (backend list unavailable: \(backendError ?? ""))")
                        }
                    }
                }
            } else if statusCode == 401 {
                DispatchQueue.main.async {
                    self.connectionState = .error
                    completion(false, "Invalid API token (401). Get your token at https://quantum.ibm.com/account")
                }
            } else {
                DispatchQueue.main.async {
                    self.connectionState = .error
                    let body = data.flatMap { String(data: $0, encoding: .utf8) } ?? ""
                    completion(false, "HTTP \(statusCode): \(body.prefix(200))")
                }
            }
        }.resume()
    }

    func disconnect() {
        ibmToken = nil
        accessToken = nil
        connectionState = .disconnected
        connectedBackendName = ""
        availableBackends = []
        submittedJobs = [:]
        tokenExpiry = .distantPast
    }

    // ═══════════════════════════════════════════════════════════════════
    // BACKEND DISCOVERY
    // ═══════════════════════════════════════════════════════════════════

    func listBackends(completion: @escaping ([IBMQuantumBackend]?, String?) -> Void) {
        guard let token = accessToken else {
            completion(nil, "Not authenticated")
            return
        }

        guard let url = URL(string: "\(quantumAPIBase)/backends") else {
            completion(nil, "Invalid URL")
            return
        }

        var req = URLRequest(url: url)
        req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        req.timeoutInterval = 15

        URLSession.shared.dataTask(with: req) { data, resp, error in
            if let error = error {
                completion(nil, error.localizedDescription)
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
                    let backends = self.parseBackends(devices)
                    completion(backends, nil)
                    return
                }
                let body = String(data: data, encoding: .utf8) ?? ""
                completion(nil, "Parse error: \(body.prefix(200))")
                return
            }

            let backends = self.parseBackends(json)
            completion(backends, nil)
        }.resume()
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
        guard let token = accessToken else {
            completion(nil, "Not authenticated")
            return
        }

        let targetBackend = backend ?? connectedBackendName
        guard !targetBackend.isEmpty else {
            completion(nil, "No backend selected")
            return
        }

        guard let url = URL(string: "\(quantumAPIBase)/jobs") else {
            completion(nil, "Invalid URL")
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

        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = body
        req.timeoutInterval = 30

        URLSession.shared.dataTask(with: req) { [weak self] data, resp, error in
            if let error = error {
                completion(nil, error.localizedDescription)
                return
            }

            guard let data = data,
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                let body = data.flatMap { String(data: $0, encoding: .utf8) } ?? ""
                completion(nil, "Parse error: \(body.prefix(200))")
                return
            }

            let statusCode = (resp as? HTTPURLResponse)?.statusCode ?? 0
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
        }.resume()
    }

    func getJobStatus(jobId: String, completion: @escaping (IBMQuantumJob?, String?) -> Void) {
        guard let token = accessToken else {
            completion(nil, "Not authenticated")
            return
        }

        guard let url = URL(string: "\(quantumAPIBase)/jobs/\(jobId)") else {
            completion(nil, "Invalid URL")
            return
        }

        var req = URLRequest(url: url)
        req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        req.timeoutInterval = 15

        URLSession.shared.dataTask(with: req) { data, _, error in
            if let error = error {
                completion(nil, error.localizedDescription)
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
        }.resume()
    }

    func getJobResult(jobId: String, completion: @escaping ([String: Any]?, String?) -> Void) {
        guard let token = accessToken else {
            completion(nil, "Not authenticated")
            return
        }

        guard let url = URL(string: "\(quantumAPIBase)/jobs/\(jobId)/results") else {
            completion(nil, "Invalid URL")
            return
        }

        var req = URLRequest(url: url)
        req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        req.timeoutInterval = 30

        URLSession.shared.dataTask(with: req) { data, resp, error in
            if let error = error {
                completion(nil, error.localizedDescription)
                return
            }

            let statusCode = (resp as? HTTPURLResponse)?.statusCode ?? 0

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
        }.resume()
    }

    func listRecentJobs(limit: Int = 10, completion: @escaping ([IBMQuantumJob]?, String?) -> Void) {
        guard let token = accessToken else {
            completion(nil, "Not authenticated")
            return
        }

        guard let url = URL(string: "\(quantumAPIBase)/jobs?limit=\(limit)&sort_by=created:desc") else {
            completion(nil, "Invalid URL")
            return
        }

        var req = URLRequest(url: url)
        req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        req.timeoutInterval = 15

        URLSession.shared.dataTask(with: req) { data, _, error in
            if let error = error {
                completion(nil, error.localizedDescription)
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
        }.resume()
    }

    func cancelJob(jobId: String, completion: @escaping (Bool, String) -> Void) {
        guard let token = accessToken else {
            completion(false, "Not authenticated")
            return
        }

        guard let url = URL(string: "\(quantumAPIBase)/jobs/\(jobId)/cancel") else {
            completion(false, "Invalid URL")
            return
        }

        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        req.timeoutInterval = 15

        URLSession.shared.dataTask(with: req) { data, resp, error in
            if let error = error {
                completion(false, error.localizedDescription)
                return
            }

            let statusCode = (resp as? HTTPURLResponse)?.statusCode ?? 0
            if statusCode == 200 || statusCode == 204 {
                DispatchQueue.main.async { [weak self] in
                    self?.submittedJobs.removeValue(forKey: jobId)
                }
                completion(true, "Job \(jobId) cancelled")
            } else {
                completion(false, "Cancel failed (HTTP \(statusCode))")
            }
        }.resume()
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
}
