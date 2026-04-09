import Foundation

// ─────────────────────────────────────────────────────────────────────────────
// MARK: - Memory Region
// ─────────────────────────────────────────────────────────────────────────────

struct MemoryRegion {
    let address: Int
    let size: Int
    let permissions: String
    let name: String?
    let path: String?

    var endAddress: Int {
        return address + size
    }

    var isReadable: Bool {
        permissions.contains("r")
    }

    var isWritable: Bool {
        permissions.contains("w")
    }

    var isExecutable: Bool {
        permissions.contains("x")
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MARK: - Heap Object
// ─────────────────────────────────────────────────────────────────────────────

struct HeapObject {
    let address: Int
    let size: Int
    let typeName: String?
    let className: String?
    let referenceCount: Int
    let isARC: Bool

    var displayName: String {
        return className ?? typeName ?? "Unknown"
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MARK: - Memory Usage
// ─────────────────────────────────────────────────────────────────────────────

struct MemoryUsage {
    let total: Int64
    let used: Int64
    let free: Int64
    let active: Int64
    let inactive: Int64
    let wired: Int64
    let compressed: Int64

    var usedPercent: Double {
        guard total > 0 else { return 0 }
        return Double(used) / Double(total) * 100
    }

    func formatted() -> String {
        let fmt = ByteCountFormatter()
        fmt.allowedUnits = [.useAll]
        fmt.countStyle = .memory

        return """
          Total: \(fmt.string(fromByteCount: total))
          Used: \(fmt.string(fromByteCount: used)) (\(String(format: "%.1f", usedPercent))%)
          Free: \(fmt.string(fromByteCount: free))
          Active: \(fmt.string(fromByteCount: active))
          Inactive: \(fmt.string(fromByteCount: inactive))
          Wired: \(fmt.string(fromByteCount: wired))
          Compressed: \(fmt.string(fromByteCount: compressed))
        """
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MARK: - Memory Debugger
// ─────────────────────────────────────────────────────────────────────────────

/// Memory introspection utility
final class MemoryDebugger {

    // ─── Singleton ───
    static let shared = MemoryDebugger()

    // ─── LLDB Bridge ───
    private let lldb = LLDBBridge.shared

    // ─── Sacred Constants ───
    private let godCode: Double = 527.5184818492612
    private let phi: Double = 1.618033988749895

    private init() {}

    // ─── Memory Reading ───
    /// Read memory at address
    func read(address: Int, size: Int) -> Data? {
        guard lldb.isAvailable && lldb.hasActiveSession else {
            // Fallback: try direct read (limited)
            return nil
        }

        return try? lldb.readMemory(address: address, size: size)
    }

    /// Read memory as hex string
    func readHex(address: Int, size: Int, bytesPerLine: Int = 16) -> String? {
        guard let data = read(address: address, size: size) else {
            return nil
        }

        var lines: [String] = []
        for offset in stride(from: 0, to: data.count, by: bytesPerLine) {
            let end = min(offset + bytesPerLine, data.count)
            let chunk = data.subdata(in: offset..<end)

            let hexPart = chunk.map { String(format: "%02x", $0) }.joined(separator: " ")
            let addrStr = String(format: "0x%08llx", address + offset)

            // Add ASCII representation
            let asciiPart = chunk.map { byte -> Character in
                if byte >= 32 && byte < 127 {
                    return Character(UnicodeScalar(byte))
                }
                return "."
            }

            lines.append("\(addrStr)  \(hexPart.padding(toLength: bytesPerLine * 3 - 1, withPad: " ", startingAt: 0))  |\(String(asciiPart))|")
        }

        return lines.joined(separator: "\n")
    }

    /// Read null-terminated string
    func readCString(address: Int, maxLength: Int = 256) -> String? {
        guard let data = read(address: address, size: maxLength) else {
            return nil
        }

        var result = ""
        for byte in data {
            if byte == 0 { break }
            if byte >= 32 && byte < 127 {
                result.append(Character(UnicodeScalar(byte)))
            }
        }

        return result.isEmpty ? nil : result
    }

    // ─── Memory Regions ───
    /// List memory regions (requires root)
    func listRegions() -> [MemoryRegion] {
        var regions: [MemoryRegion] = []

        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/usr/bin/vmmap")
        task.arguments = ["-c", String(Foundation.ProcessInfo().processIdentifier)]

        let pipe = Pipe()
        task.standardOutput = pipe

        do {
            try task.run()
            task.waitUntilExit()

            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            if let output = String(data: data, encoding: .utf8) {
                regions = parseRegions(from: output)
            }
        } catch {
            // Fallback: basic regions
            regions = basicRegions()
        }

        return regions
    }

    private func parseRegions(from output: String) -> [MemoryRegion] {
        var regions: [MemoryRegion] = []

        let lines = output.components(separatedBy: "\n")
        for line in lines {
            guard line.contains("---") == false else { continue }

            let parts = line.split(separator: " ", omittingEmptySubsequences: true)
            guard parts.count >= 4 else { continue }

            // Parse vmmap format (simplified)
            if let start = Int(parts[0].description.dropFirst(), radix: 16),
               let size = Int(parts[1].description, radix: 16) {
                let perms = String(parts[2])
                let name = parts.count > 3 ? parts[3].description : nil

                regions.append(MemoryRegion(
                    address: start,
                    size: size,
                    permissions: perms,
                    name: name,
                    path: nil
                ))
            }
        }

        return regions
    }

    private func basicRegions() -> [MemoryRegion] {
        return [
            MemoryRegion(address: 0, size: 0, permissions: "", name: "null", path: nil)
        ]
    }

    // ─── Memory Usage ───
    /// Get system memory usage
    func memoryUsage() -> MemoryUsage {
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)

        let hostPort = mach_host_self()
        let result = withUnsafeMutablePointer(to: &stats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(hostPort, HOST_VM_INFO64, $0, &count)
            }
        }

        let pageSize = Int64(vm_kernel_page_size)

        if result == KERN_SUCCESS {
            return MemoryUsage(
                total: Int64(Foundation.ProcessInfo().physicalMemory),
                used: Int64(stats.free_count + stats.active_count + stats.inactive_count) * pageSize,
                free: Int64(stats.free_count) * pageSize,
                active: Int64(stats.active_count) * pageSize,
                inactive: Int64(stats.inactive_count) * pageSize,
                wired: Int64(stats.wire_count) * pageSize,
                compressed: Int64(stats.compressor_page_count) * pageSize
            )
        }

        return MemoryUsage(
            total: Int64(Foundation.ProcessInfo().physicalMemory),
            used: 0,
            free: 0,
            active: 0,
            inactive: 0,
            wired: 0,
            compressed: 0
        )
    }

    // ─── Heap Scanning ───
    /// Scan heap for objects (requires lldb session)
    func scanHeap(pattern: Data? = nil) -> [HeapObject] {
        guard lldb.isAvailable && lldb.hasActiveSession else {
            return []
        }

        // In production, would scan heap memory for known object patterns
        return []
    }

    /// Find pattern in memory
    func findPattern(_ pattern: Data, maxResults: Int = 100) -> [Int] {
        guard lldb.isAvailable && lldb.hasActiveSession else {
            return []
        }

        // memoryFindPattern not available in current LLDB bridge
        return []
    }

    // ─── Object Introspection ───
    /// Inspect object at address
    func inspect(address: Int) -> [String: Any]? {
        guard lldb.isAvailable && lldb.hasActiveSession else {
            return nil
        }

        guard let vars = try? lldb.localVariables() else {
            return nil
        }

        // Find variable at address
        for v in vars {
            if let value = v.value, let _ = Int(value.trimmingCharacters(in: .whitespaces)) {
                return [
                    "name": v.name,
                    "type": v.type,
                    "value": value,
                    "summary": v.summary ?? "",
                ]
            }
        }

        return nil
    }

    // ─── Leak Detection ───
    /// Heuristic leak detection
    func detectLeaks() -> [String: Any] {
        let usage = memoryUsage()

        return [
            "usedMemory": usage.used,
            "wiredMemory": usage.wired,
            "compressedMemory": usage.compressed,
            "leakProbability": usage.wired > Int64(Foundation.ProcessInfo().physicalMemory) / 4 ? "high" : "low",
            "recommendation": usage.usedPercent > 90 ? "High memory pressure - consider reducing allocations" : "Memory usage normal",
        ]
    }

    // ─── Summary ───
    func summary() -> String {
        var lines: [String] = []
        lines.append("Memory Debugger")
        lines.append(String(repeating: "─", count: 40))

        let usage = memoryUsage()
        lines.append("  \(usage.formatted())")
        lines.append("")

        if lldb.hasActiveSession {
            let regions = listRegions()
            lines.append("  Regions: \(regions.count)")

            if let layout = try? lldb.processInfo() {
                lines.append("  Process Memory:")
                lines.append("    PID: \(layout.pid)")
                lines.append("    Threads: \(layout.numThreads)")
                lines.append("    Breakpoints: \(layout.numBreakpoints)")
            }
        } else {
            lines.append("  No active debug session")
        }

        return lines.joined(separator: "\n")
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MARK: - Convenience Extensions
// ─────────────────────────────────────────────────────────────────────────────

extension MemoryDebugger {
    /// Quick hex dump
    static func hexDump(address: Int, size: Int) -> String? {
        return MemoryDebugger.shared.readHex(address: address, size: size)
    }

    /// Quick C string read
    static func string(at address: Int) -> String? {
        return MemoryDebugger.shared.readCString(address: address)
    }

    /// Quick memory usage
    static func usage() -> MemoryUsage {
        return MemoryDebugger.shared.memoryUsage()
    }
}