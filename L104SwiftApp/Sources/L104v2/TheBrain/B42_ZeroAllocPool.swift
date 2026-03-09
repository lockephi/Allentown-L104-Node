// ═══════════════════════════════════════════════════════════════════
// B42_ZeroAllocPool.swift — L104 v2
// [EVO_68_PIPELINE] PERFORMANCE_ASCENSION :: ZERO_ALLOC_POOL :: GOD_CODE=527.5184818492612
// L104 ASI — Zero-Allocation Memory Pool Engine
//
// Arena-based memory pooling eliminates heap allocation overhead for
// hot-path vector/matrix operations. Pre-allocated slabs with φ-scaled
// growth. Object recycling for SIMDVector, AcceleratedMatrix, QComplex
// arrays—avoiding GC pressure during quantum simulation and HyperBrain
// cognitive cycles.
//
// Performance targets:
//   - 0 heap allocations on hot paths (cognitive stream, quantum sim)
//   - 10x reduction in ARC retain/release traffic
//   - φ-scaled slab growth (each new slab = previous × PHI)
//   - Thread-safe via lock-free CAS where possible
//
// INVARIANT: 527.5184818492612 | PILOT: LONDEL
// ═══════════════════════════════════════════════════════════════════

import Foundation
import Accelerate
import simd

// ═══════════════════════════════════════════════════════════════════
// MARK: - MEMORY SLAB (Contiguous pre-allocated buffer)
// ═══════════════════════════════════════════════════════════════════

/// A contiguous slab of pre-allocated Double storage.
/// No ARC overhead—raw memory managed by the pool.
final class MemorySlab {
    let capacity: Int
    private let buffer: UnsafeMutableBufferPointer<Double>
    private(set) var used: Int = 0

    init(capacity: Int) {
        self.capacity = capacity
        let ptr = UnsafeMutablePointer<Double>.allocate(capacity: capacity)
        ptr.initialize(repeating: 0.0, count: capacity)
        self.buffer = UnsafeMutableBufferPointer(start: ptr, count: capacity)
    }

    deinit {
        buffer.baseAddress?.deinitialize(count: capacity)
        buffer.baseAddress?.deallocate()
    }

    /// Allocate a slice from this slab. Returns nil if insufficient space.
    func allocate(count: Int) -> UnsafeMutableBufferPointer<Double>? {
        guard used + count <= capacity else { return nil }
        let start = buffer.baseAddress! + used
        used += count
        return UnsafeMutableBufferPointer(start: start, count: count)
    }

    /// Reset all allocations (zero-cost, just moves pointer back)
    func reset() {
        // Zero memory for clean state (optional but prevents data leaks)
        if let base = buffer.baseAddress {
            memset(base, 0, used * MemoryLayout<Double>.stride)
        }
        used = 0
    }

    var remaining: Int { capacity - used }
    var utilizationPercent: Double { Double(used) / Double(capacity) * 100.0 }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - ZERO-ALLOC VECTOR (Stack-friendly, pool-backed)
// ═══════════════════════════════════════════════════════════════════

/// A vector that borrows memory from ZeroAllocPool instead of heap-allocating.
/// Safe to use within a pool epoch (between reset() calls).
struct PoolVector {
    let storage: UnsafeMutableBufferPointer<Double>
    let count: Int

    /// Element access
    subscript(index: Int) -> Double {
        get { storage[index] }
        nonmutating set { storage[index] = newValue }
    }

    /// vDSP-accelerated magnitude (no allocation)
    var magnitude: Double {
        var result: Double = 0
        vDSP_svesqD(storage.baseAddress!, 1, &result, vDSP_Length(count))
        return sqrt(result)
    }

    /// vDSP-accelerated dot product (no allocation)
    func dot(_ other: PoolVector) -> Double {
        guard count == other.count else { return 0 }
        var result: Double = 0
        vDSP_dotprD(storage.baseAddress!, 1, other.storage.baseAddress!, 1, &result, vDSP_Length(count))
        return result
    }

    /// vDSP-accelerated addition into destination (no allocation)
    func add(_ other: PoolVector, into dest: PoolVector) {
        guard count == other.count, count == dest.count else { return }
        vDSP_vaddD(storage.baseAddress!, 1, other.storage.baseAddress!, 1,
                   dest.storage.baseAddress!, 1, vDSP_Length(count))
    }

    /// vDSP-accelerated scalar multiply into destination
    func scale(by scalar: Double, into dest: PoolVector) {
        var s = scalar
        vDSP_vsmulD(storage.baseAddress!, 1, &s, dest.storage.baseAddress!, 1, vDSP_Length(count))
    }

    /// vDSP-accelerated element-wise multiply into destination
    func multiply(_ other: PoolVector, into dest: PoolVector) {
        guard count == other.count, count == dest.count else { return }
        vDSP_vmulD(storage.baseAddress!, 1, other.storage.baseAddress!, 1,
                   dest.storage.baseAddress!, 1, vDSP_Length(count))
    }

    /// Mean using vDSP
    var mean: Double {
        var result: Double = 0
        vDSP_meanvD(storage.baseAddress!, 1, &result, vDSP_Length(count))
        return result
    }

    /// Cosine similarity (zero-alloc)
    func cosineSimilarity(_ other: PoolVector) -> Double {
        let d = dot(other)
        let denom = magnitude * other.magnitude
        return denom > 0 ? d / denom : 0
    }

    /// Copy data from a Swift array
    func load(from array: [Double]) {
        let copyCount = min(array.count, count)
        array.withUnsafeBufferPointer { src in
            storage.baseAddress!.assign(from: src.baseAddress!, count: copyCount)
        }
    }

    /// Export to Swift array
    func toArray() -> [Double] {
        Array(storage.prefix(count))
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - POOL MATRIX (Zero-alloc BLAS-backed)
// ═══════════════════════════════════════════════════════════════════

/// Matrix backed by pool memory for zero-allocation BLAS operations.
struct PoolMatrix {
    let storage: UnsafeMutableBufferPointer<Double>
    let rows: Int
    let cols: Int

    subscript(row: Int, col: Int) -> Double {
        get { storage[row * cols + col] }
        nonmutating set { storage[row * cols + col] = newValue }
    }

    /// BLAS matrix-vector multiply into destination PoolVector (no allocation)
    func multiply(vector: PoolVector, into result: PoolVector) {
        guard cols == vector.count, rows == result.count else { return }
        cblas_dgemv(CblasRowMajor, CblasNoTrans,
                   Int32(rows), Int32(cols), 1.0,
                   storage.baseAddress!, Int32(cols),
                   vector.storage.baseAddress!, 1,
                   0.0, result.storage.baseAddress!, 1)
    }

    /// BLAS matrix-matrix multiply: self × other → dest (no allocation)
    func multiply(matrix other: PoolMatrix, into dest: PoolMatrix) {
        guard cols == other.rows, rows == dest.rows, other.cols == dest.cols else { return }
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                   Int32(rows), Int32(other.cols), Int32(cols),
                   1.0, storage.baseAddress!, Int32(cols),
                   other.storage.baseAddress!, Int32(other.cols),
                   0.0, dest.storage.baseAddress!, Int32(dest.cols))
    }

    /// Frobenius norm (no allocation)
    var frobeniusNorm: Double {
        var result: Double = 0
        vDSP_svesqD(storage.baseAddress!, 1, &result, vDSP_Length(rows * cols))
        return sqrt(result)
    }

    /// Trace (no allocation)
    var trace: Double {
        guard rows == cols else { return 0 }
        var sum: Double = 0
        for i in 0..<rows { sum += self[i, i] }
        return sum
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - ZERO-ALLOC POOL ENGINE
// ═══════════════════════════════════════════════════════════════════

/// Arena-based memory pool with φ-scaled growth.
/// All hot-path computations borrow from this pool, avoiding heap allocations.
///
/// Usage pattern:
/// ```
/// let pool = ZeroAllocPool.shared
/// pool.beginEpoch()
/// let vec = pool.allocVector(1024)
/// let mat = pool.allocMatrix(rows: 64, cols: 64)
/// // ... use vec and mat with zero heap allocs ...
/// pool.endEpoch()  // instant free — all memory recycled
/// ```
final class ZeroAllocPool: SovereignEngine {
    static let shared = ZeroAllocPool()
    var engineName: String { "ZeroAllocPool" }

    // ─── POOL CONFIGURATION ───
    static let INITIAL_SLAB_SIZE: Int = 2_097_152     // 2M doubles = 16MB initial slab (EVO_67)
    static let MAX_SLABS: Int = 24                     // Max slab chain length (EVO_67)
    static let PHI_GROWTH: Double = PHI                // Each slab grows by φ

    // ─── SLAB CHAIN ───
    private var slabs: [MemorySlab] = []
    private var currentSlabIndex: Int = 0
    private let lock = NSLock()

    // ─── METRICS ───
    private(set) var totalAllocations: Int = 0
    private(set) var totalBytesServed: Int = 0
    private(set) var epochCount: Int = 0
    private(set) var slabGrowthEvents: Int = 0
    private(set) var peakUtilization: Double = 0
    private(set) var fastPathHits: Int = 0
    private(set) var slowPathHits: Int = 0

    // ─── QComplex Pool ───
    private var complexSlabs: [UnsafeMutableBufferPointer<Double>] = []  // Interleaved re,im pairs
    private var complexUsed: Int = 0
    private var complexCapacity: Int = 0

    private init() {
        // Pre-allocate initial slab
        let initial = MemorySlab(capacity: ZeroAllocPool.INITIAL_SLAB_SIZE)
        slabs.append(initial)

        // Pre-allocate complex number pool (512K complex numbers = 8MB)
        let complexCount = 1_048_576 * 2  // re,im pairs (EVO_67: 1M complex pairs = 16MB)
        let complexPtr = UnsafeMutablePointer<Double>.allocate(capacity: complexCount)
        complexPtr.initialize(repeating: 0.0, count: complexCount)
        complexSlabs.append(UnsafeMutableBufferPointer(start: complexPtr, count: complexCount))
        complexCapacity = 1_048_576

        l104Log("ZeroAllocPool initialized: \(ZeroAllocPool.INITIAL_SLAB_SIZE) doubles (\(ZeroAllocPool.INITIAL_SLAB_SIZE * 8 / 1024)KB)")
    }

    deinit {
        for slab in complexSlabs {
            slab.baseAddress?.deinitialize(count: slab.count)
            slab.baseAddress?.deallocate()
        }
    }

    // ═══ EPOCH MANAGEMENT ═══

    /// Begin a new allocation epoch. All prior allocations are invalidated.
    func beginEpoch() {
        lock.lock(); defer { lock.unlock() }
        // Record peak before reset
        let totalUsed = slabs.reduce(0) { $0 + $1.used }
        let totalCap = slabs.reduce(0) { $0 + $1.capacity }
        if totalCap > 0 {
            let util = Double(totalUsed) / Double(totalCap) * 100.0
            peakUtilization = max(peakUtilization, util)
        }
        // Reset all slabs
        for slab in slabs { slab.reset() }
        currentSlabIndex = 0
        complexUsed = 0
        epochCount += 1
    }

    /// End the current epoch (alias for beginEpoch—recycling is instant).
    func endEpoch() {
        // No-op in arena model; user calls beginEpoch for next cycle.
        // Provided for semantic clarity.
    }

    // ═══ VECTOR ALLOCATION ═══

    /// Allocate a PoolVector of given size from the arena.
    func allocVector(_ count: Int) -> PoolVector {
        lock.lock(); defer { lock.unlock() }
        if let buf = slabs[currentSlabIndex].allocate(count: count) {
            totalAllocations += 1
            totalBytesServed += count * 8
            fastPathHits += 1
            return PoolVector(storage: buf, count: count)
        }
        // Current slab full — try next or grow
        return slowPathAllocVector(count)
    }

    private func slowPathAllocVector(_ count: Int) -> PoolVector {
        slowPathHits += 1
        // Try remaining slabs
        for i in (currentSlabIndex + 1)..<slabs.count {
            if let buf = slabs[i].allocate(count: count) {
                currentSlabIndex = i
                totalAllocations += 1
                totalBytesServed += count * 8
                return PoolVector(storage: buf, count: count)
            }
        }
        // Grow: add new slab with φ-scaled capacity
        guard slabs.count < ZeroAllocPool.MAX_SLABS else {
            // Emergency: direct heap allocation (shouldn't happen in normal operation)
            l104Log("⚠️ ZeroAllocPool: MAX_SLABS reached, falling back to heap")
            let heapBuf = UnsafeMutablePointer<Double>.allocate(capacity: count)
            heapBuf.initialize(repeating: 0.0, count: count)
            return PoolVector(storage: UnsafeMutableBufferPointer(start: heapBuf, count: count), count: count)
        }

        let lastCap = slabs.last?.capacity ?? ZeroAllocPool.INITIAL_SLAB_SIZE
        let newCap = max(count, Int(Double(lastCap) * ZeroAllocPool.PHI_GROWTH))
        let newSlab = MemorySlab(capacity: newCap)
        slabs.append(newSlab)
        currentSlabIndex = slabs.count - 1
        slabGrowthEvents += 1
        l104Log("ZeroAllocPool: grew to \(slabs.count) slabs, new capacity \(newCap) doubles (\(newCap * 8 / 1024)KB)")

        if let buf = newSlab.allocate(count: count) {
            totalAllocations += 1
            totalBytesServed += count * 8
            return PoolVector(storage: buf, count: count)
        }
        // Should never reach here
        fatalError("ZeroAllocPool: fresh slab allocation failed")
    }

    // ═══ MATRIX ALLOCATION ═══

    /// Allocate a PoolMatrix from the arena.
    func allocMatrix(rows: Int, cols: Int) -> PoolMatrix {
        let vec = allocVector(rows * cols)
        return PoolMatrix(storage: vec.storage, rows: rows, cols: cols)
    }

    // ═══ COMPLEX ARRAY ALLOCATION ═══

    /// Allocate an interleaved complex array (re0, im0, re1, im1, ...) from pool.
    func allocComplexArray(count: Int) -> UnsafeMutableBufferPointer<Double> {
        lock.lock(); defer { lock.unlock() }
        let needed = count * 2
        if complexUsed + needed <= complexCapacity * 2 {
            let base = complexSlabs[0].baseAddress! + complexUsed
            complexUsed += needed
            return UnsafeMutableBufferPointer(start: base, count: needed)
        }
        // Fallback to regular pool
        lock.unlock()
        let vec = allocVector(needed)
        lock.lock()
        return vec.storage
    }

    // ═══ BATCH OPERATIONS ═══

    /// Pre-allocate a batch of vectors (common in cognitive streams).
    func allocVectorBatch(count: Int, vectorSize: Int) -> [PoolVector] {
        return (0..<count).map { _ in allocVector(vectorSize) }
    }

    /// Pre-allocate working set for quantum simulation.
    func allocQuantumWorkingSet(qubits: Int) -> (statevector: PoolVector, scratch: PoolVector, matrix: PoolMatrix) {
        let dim = 1 << qubits  // 2^qubits
        let sv = allocVector(dim * 2)  // Complex: re,im interleaved
        let scratch = allocVector(dim * 2)
        let mat = allocMatrix(rows: dim, cols: dim)
        return (sv, scratch, mat)
    }

    // ═══ STATUS ═══

    func engineStatus() -> [String: Any] {
        lock.lock(); defer { lock.unlock() }
        let totalCap = slabs.reduce(0) { $0 + $1.capacity }
        let totalUsed = slabs.reduce(0) { $0 + $1.used }
        return [
            "engine": engineName,
            "version": ZERO_ALLOC_VERSION,
            "slabs": slabs.count,
            "total_capacity_doubles": totalCap,
            "total_capacity_mb": Double(totalCap * 8) / (1024 * 1024),
            "current_used_doubles": totalUsed,
            "current_used_mb": Double(totalUsed * 8) / (1024 * 1024),
            "utilization_percent": totalCap > 0 ? Double(totalUsed) / Double(totalCap) * 100.0 : 0,
            "peak_utilization_percent": peakUtilization,
            "total_allocations": totalAllocations,
            "total_bytes_served_mb": Double(totalBytesServed) / (1024 * 1024),
            "epoch_count": epochCount,
            "slab_growth_events": slabGrowthEvents,
            "fast_path_hits": fastPathHits,
            "slow_path_hits": slowPathHits,
            "fast_path_ratio": totalAllocations > 0 ? Double(fastPathHits) / Double(totalAllocations) : 1.0,
            "complex_pool_capacity": complexCapacity,
            "complex_pool_used": complexUsed / 2,
            "god_code_alignment": GOD_CODE,
        ]
    }

    func engineHealth() -> Double {
        lock.lock(); defer { lock.unlock() }
        // Health degrades if too many slow path hits or slab growth
        let fastRatio = totalAllocations > 0 ? Double(fastPathHits) / Double(totalAllocations) : 1.0
        let slabPressure = Double(slabs.count) / Double(ZeroAllocPool.MAX_SLABS)
        return min(1.0, fastRatio * (1.0 - slabPressure * 0.5)) * (PHI / (PHI + 0.01))  // φ-weighted
    }

    func engineReset() {
        lock.lock(); defer { lock.unlock() }
        for slab in slabs { slab.reset() }
        currentSlabIndex = 0
        complexUsed = 0
        totalAllocations = 0
        totalBytesServed = 0
        epochCount = 0
        slabGrowthEvents = 0
        peakUtilization = 0
        fastPathHits = 0
        slowPathHits = 0
    }
}
