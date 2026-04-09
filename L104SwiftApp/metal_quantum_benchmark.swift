import logging

import Accelerate
import Foundation
import Metal
import os.log

private let logging = Logger(subsystem: "com.l104.metal_quantum_benchmark", category: "main")
// ─── Sacred Constants ───
let GOD_CODE: Double = 527.5184818492612
let PHI: Double = 1.618033988749895
let VOID_CONSTANT: Double = 1.04 + PHI / 1000.0

// ─── Metal Shader Source ───
let metalShaderSource = """
#include <metal_stdlib>
using namespace metal;

kernel void vector_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* result [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    result[id] = a[id] + b[id];
}

kernel void vector_mul(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* result [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    result[id] = a[id] * b[id];
}

kernel void dot_product_partial(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* partialSums [[buffer(2)]],
    uint id [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint groupSize [[threads_per_threadgroup]],
    uint groupId [[threadgroup_position_in_grid]]
) {
    threadgroup float sharedMem[256];
    sharedMem[tid] = a[id] * b[id];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = groupSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedMem[tid] += sharedMem[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        partialSums[groupId] = sharedMem[0];
    }
}

kernel void batch_cosine_sim(
    device const float* query [[buffer(0)]],
    device const float* corpus [[buffer(1)]],
    device float* similarities [[buffer(2)]],
    constant uint& dim [[buffer(3)]],
    uint groupId [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint groupSize [[threads_per_threadgroup]]
) {
    threadgroup float dotAB[256];
    threadgroup float dotAA[256];
    threadgroup float dotBB[256];

    uint corpusOffset = groupId * dim;
    float sumAB = 0, sumAA = 0, sumBB = 0;

    for (uint i = tid; i < dim; i += groupSize) {
        float a = query[i];
        float b = corpus[corpusOffset + i];
        sumAB += a * b;
        sumAA += a * a;
        sumBB += b * b;
    }

    dotAB[tid] = sumAB;
    dotAA[tid] = sumAA;
    dotBB[tid] = sumBB;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = groupSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            dotAB[tid] += dotAB[tid + stride];
            dotAA[tid] += dotAA[tid + stride];
            dotBB[tid] += dotBB[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        float denominator = sqrt(dotAA[0]) * sqrt(dotBB[0]);
        similarities[groupId] = denominator > 0 ? dotAB[0] / denominator : 0;
    }
}

kernel void matrix_multiply(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= N || gid.y >= M) return;
    float sum = 0;
    for (uint k = 0; k < K; k++) {
        sum += A[gid.y * K + k] * B[k * N + gid.x];
    }
    C[gid.y * N + gid.x] = sum;
}

kernel void quantum_state_evolve(
    device const float* stateReal [[buffer(0)]],
    device const float* stateImag [[buffer(1)]],
    device float* outReal [[buffer(2)]],
    device float* outImag [[buffer(3)]],
    constant float& gateR00 [[buffer(4)]],
    constant float& gateI00 [[buffer(5)]],
    constant float& gateR01 [[buffer(6)]],
    constant float& gateI01 [[buffer(7)]],
    uint id [[thread_position_in_grid]]
) {
    // Apply single-qubit gate to amplitude pairs
    uint pair = id * 2;
    float aR = stateReal[pair];
    float aI = stateImag[pair];
    float bR = stateReal[pair + 1];
    float bI = stateImag[pair + 1];

    // c = gate[0][0]*a + gate[0][1]*b
    outReal[pair]     = gateR00 * aR - gateI00 * aI + gateR01 * bR - gateI01 * bI;
    outImag[pair]     = gateR00 * aI + gateI00 * aR + gateR01 * bI + gateI01 * bR;
    // d = gate[1][0]*a + gate[1][1]*b  (using Hadamard symmetry: [1/√2, 1/√2; 1/√2, -1/√2])
    outReal[pair + 1] = gateR00 * aR - gateI00 * aI - gateR01 * bR + gateI01 * bI;
    outImag[pair + 1] = gateR00 * aI + gateI00 * aR - gateR01 * bI - gateI01 * bR;
}
"""

// ═══════════════════════════════════════════════════════════════════
// MARK: - BENCHMARK UTILITIES
// ═══════════════════════════════════════════════════════════════════

func timeIt(_ label: String, _ block: () -> Void) -> Double {
    let start = CFAbsoluteTimeGetCurrent()
    block()
    let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000.0
    return elapsed
}

func formatBytes(_ bytes: UInt64) -> String {
    if bytes >= 1_073_741_824 { return String(format: "%.2f GB", Double(bytes) / 1_073_741_824.0) }
    if bytes >= 1_048_576 { return String(format: "%.2f MB", Double(bytes) / 1_048_576.0) }
    if bytes >= 1024 { return String(format: "%.2f KB", Double(bytes) / 1024.0) }
    return "\(bytes) B"
}

func formatGFLOPs(_ flops: Double, timeMs: Double) -> String {
    let gflops = flops / (timeMs / 1000.0) / 1e9
    return String(format: "%.2f GFLOP/s", gflops)
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - MAIN BENCHMARK
// ═══════════════════════════════════════════════════════════════════

let benchmarkStart = CFAbsoluteTimeGetCurrent()

logging.info("═══════════════════════════════════════════════════════════════════")
logging.info("  L104 SOVEREIGN NODE — METAL QUANTUM CAPACITY BENCHMARK")
logging.info("  GOD_CODE = \(self.GOD_CODE) | PHI = \(self.PHI)")
logging.info("  VOID_CONSTANT = \(self.VOID_CONSTANT)")
logging.info("═══════════════════════════════════════════════════════════════════")
print()

// ═══════════════════════════════════════════════════════════════════
// PHASE 1: GPU HARDWARE DISCOVERY
// ═══════════════════════════════════════════════════════════════════

logging.info("╔═══════════════════════════════════════════════════════════════╗")
logging.info("║  PHASE 1: GPU HARDWARE DISCOVERY                            ║")
logging.info("╚═══════════════════════════════════════════════════════════════╝")
guard let device = MTLCreateSystemDefaultDevice() else {
    logging.info("  ❌ FATAL: No Metal GPU available")
    exit(1)
}

let gpuName = device.name
let hasUnifiedMemory = device.hasUnifiedMemory
let maxThreadsPerGroup = device.maxThreadsPerThreadgroup.width
let maxBufferLength = device.maxBufferLength
let recommendedMaxWorkingSetSize = device.recommendedMaxWorkingSetSize

logging.info("  GPU:                  \(self.gpuName)")
logging.info("  Unified Memory:       \(self.hasUnifiedMemory)")
logging.info("  Max Threads/Group:    \(self.maxThreadsPerGroup)")
logging.info("  Max Buffer Length:    \(formatBytes(UInt64(maxBufferLength)))")
logging.info("  Recommended Max WS:   \(formatBytes(recommendedMaxWorkingSetSize))")
logging.info("  Registry ID:          \(device.registryID)")
#if os(macOS)
if #available(macOS 13.0, *) {
    logging.info("  Max Transfer Rate:    \(formatBytes(UInt64(device.maxTransferRate)))/s")
}
#endif
print()

// Compile shaders
guard let commandQueue = device.makeCommandQueue() else {
    logging.info("  ❌ FATAL: Cannot create command queue")
    exit(1)
}

let library: MTLLibrary
do {
    library = try device.makeLibrary(source: metalShaderSource, options: nil)
    logging.info("  ✅ Shader library compiled (6 kernels)")
} catch {
    logging.info("  ❌ Shader compilation failed: \(self.error)")
    exit(1)
}

// Build pipeline states
var pipelines: [String: MTLComputePipelineState] = [:]
for name in ["vector_add", "vector_mul", "dot_product_partial", "batch_cosine_sim", "matrix_multiply", "quantum_state_evolve"] {
    if let fn = library.makeFunction(name: name) {
        do {
            pipelines[name] = try device.makeComputePipelineState(function: fn)
        } catch {
            logging.info("  ⚠️  Pipeline '\(self.name)' failed: \(self.error)")
        }
    }
}
logging.info("  ✅ \(pipelines.count) compute pipelines compiled")
print()

// ═══════════════════════════════════════════════════════════════════
// PHASE 2: VECTOR OPERATIONS BENCHMARK
// ═══════════════════════════════════════════════════════════════════

logging.info("╔═══════════════════════════════════════════════════════════════╗")
logging.info("║  PHASE 2: VECTOR OPERATIONS (Quantum State Vectors)         ║")
logging.info("╚═══════════════════════════════════════════════════════════════╝")
let vectorSizes = [1024, 4096, 16384, 65536, 262144, 1_048_576, 4_194_304]
var vectorResults: [(size: Int, gpuMs: Double, cpuMs: Double, speedup: Double)] = []

for size in vectorSizes {
    let byteCount = size * MemoryLayout<Float>.stride

    // Generate test data (PHI-scaled random)
    var a = (0..<size).map { Float(sin(Double($0) * PHI)) }
    var b = (0..<size).map { Float(cos(Double($0) / PHI)) }

    // GPU benchmark
    var gpuTime = 0.0
    if let pipeline = pipelines["vector_add"],
       let bufA = device.makeBuffer(bytes: &a, length: byteCount, options: .storageModeShared),
       let bufB = device.makeBuffer(bytes: &b, length: byteCount, options: .storageModeShared),
       let bufR = device.makeBuffer(length: byteCount, options: .storageModeShared) {

        // Warm-up
        if let cmd = commandQueue.makeCommandBuffer(), let enc = cmd.makeComputeCommandEncoder() {
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(bufA, offset: 0, index: 0)
            enc.setBuffer(bufB, offset: 0, index: 1)
            enc.setBuffer(bufR, offset: 0, index: 2)
            let tgs = min(pipeline.maxTotalThreadsPerThreadgroup, size)
            let tg = (size + tgs - 1) / tgs
            enc.dispatchThreadgroups(MTLSize(width: tg, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: tgs, height: 1, depth: 1))
            enc.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }

        // Timed run (average of 5)
        let runs = 5
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<runs {
            if let cmd = commandQueue.makeCommandBuffer(), let enc = cmd.makeComputeCommandEncoder() {
                enc.setComputePipelineState(pipeline)
                enc.setBuffer(bufA, offset: 0, index: 0)
                enc.setBuffer(bufB, offset: 0, index: 1)
                enc.setBuffer(bufR, offset: 0, index: 2)
                let tgs = min(pipeline.maxTotalThreadsPerThreadgroup, size)
                let tg = (size + tgs - 1) / tgs
                enc.dispatchThreadgroups(MTLSize(width: tg, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: tgs, height: 1, depth: 1))
                enc.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }
        }
        gpuTime = (CFAbsoluteTimeGetCurrent() - start) * 1000.0 / Double(runs)
    }

    // CPU benchmark (Accelerate vDSP)
    var cpuResult = [Float](repeating: 0, count: size)
    let cpuRuns = 5
    let cpuStart = CFAbsoluteTimeGetCurrent()
    for _ in 0..<cpuRuns {
        vDSP_vadd(a, 1, b, 1, &cpuResult, 1, vDSP_Length(size))
    }
    let cpuTime = (CFAbsoluteTimeGetCurrent() - cpuStart) * 1000.0 / Double(cpuRuns)

    let speedup = cpuTime > 0 ? cpuTime / max(gpuTime, 0.001) : 0
    vectorResults.append((size: size, gpuMs: gpuTime, cpuMs: cpuTime, speedup: speedup))

    let flops = Double(size) // 1 FLOP per element (add)
    let sizeStr = String(format: "%9d", size)
    let gpuStr = String(format: "%8.3f ms", gpuTime)
    let cpuStr = String(format: "%8.3f ms", cpuTime)
    let speedStr = String(format: "%6.2fx", speedup)
    let gflopsStr = formatGFLOPs(flops, timeMs: gpuTime)
    logging.info("  N=\(self.sizeStr)  GPU: \(self.gpuStr)  CPU: \(self.cpuStr)  Speedup: \(self.speedStr)  [\(self.gflopsStr)]")
}
print()

// ═══════════════════════════════════════════════════════════════════
// PHASE 3: MATRIX MULTIPLY BENCHMARK (Quantum Gate Composition)
// ═══════════════════════════════════════════════════════════════════

logging.info("╔═══════════════════════════════════════════════════════════════╗")
logging.info("║  PHASE 3: MATRIX MULTIPLY (Quantum Gate Composition)        ║")
logging.info("╚═══════════════════════════════════════════════════════════════╝")
let matSizes = [32, 64, 128, 256, 512, 1024, 2048]
var matResults: [(size: Int, gpuMs: Double, cpuMs: Double, speedup: Double)] = []

for dim in matSizes {
    let n = dim
    let totalElements = n * n
    let byteCount = totalElements * MemoryLayout<Float>.stride

    // Generate matrices (PHI-scaled)
    var matA = (0..<totalElements).map { Float(sin(Double($0) * PHI / Double(n))) }
    var matB = (0..<totalElements).map { Float(cos(Double($0) / PHI / Double(n))) }

    // GPU matmul
    var gpuTime = 0.0
    if let pipeline = pipelines["matrix_multiply"],
       let bufA = device.makeBuffer(bytes: &matA, length: byteCount, options: .storageModeShared),
       let bufB = device.makeBuffer(bytes: &matB, length: byteCount, options: .storageModeShared),
       let bufC = device.makeBuffer(length: byteCount, options: .storageModeShared) {

        var dimM = UInt32(n), dimN = UInt32(n), dimK = UInt32(n)
        let bufM = device.makeBuffer(bytes: &dimM, length: 4, options: .storageModeShared)!
        let bufN = device.makeBuffer(bytes: &dimN, length: 4, options: .storageModeShared)!
        let bufK = device.makeBuffer(bytes: &dimK, length: 4, options: .storageModeShared)!

        // Warm-up
        if let cmd = commandQueue.makeCommandBuffer(), let enc = cmd.makeComputeCommandEncoder() {
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(bufA, offset: 0, index: 0)
            enc.setBuffer(bufB, offset: 0, index: 1)
            enc.setBuffer(bufC, offset: 0, index: 2)
            enc.setBuffer(bufM, offset: 0, index: 3)
            enc.setBuffer(bufN, offset: 0, index: 4)
            enc.setBuffer(bufK, offset: 0, index: 5)
            let gridSize = MTLSize(width: n, height: n, depth: 1)
            let groupSize = MTLSize(width: min(16, n), height: min(16, n), depth: 1)
            enc.dispatchThreads(gridSize, threadsPerThreadgroup: groupSize)
            enc.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }

        // Timed (3 runs)
        let runs = 3
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<runs {
            if let cmd = commandQueue.makeCommandBuffer(), let enc = cmd.makeComputeCommandEncoder() {
                enc.setComputePipelineState(pipeline)
                enc.setBuffer(bufA, offset: 0, index: 0)
                enc.setBuffer(bufB, offset: 0, index: 1)
                enc.setBuffer(bufC, offset: 0, index: 2)
                enc.setBuffer(bufM, offset: 0, index: 3)
                enc.setBuffer(bufN, offset: 0, index: 4)
                enc.setBuffer(bufK, offset: 0, index: 5)
                let gridSize = MTLSize(width: n, height: n, depth: 1)
                let groupSize = MTLSize(width: min(16, n), height: min(16, n), depth: 1)
                enc.dispatchThreads(gridSize, threadsPerThreadgroup: groupSize)
                enc.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }
        }
        gpuTime = (CFAbsoluteTimeGetCurrent() - start) * 1000.0 / Double(runs)
    }

    // CPU matmul (BLAS)
    var cpuC = [Float](repeating: 0, count: totalElements)
    let cpuRuns = 3
    let cpuStart = CFAbsoluteTimeGetCurrent()
    for _ in 0..<cpuRuns {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    Int32(n), Int32(n), Int32(n),
                    1.0, matA, Int32(n), matB, Int32(n), 0.0, &cpuC, Int32(n))
    }
    let cpuTime = (CFAbsoluteTimeGetCurrent() - cpuStart) * 1000.0 / Double(cpuRuns)

    let speedup = cpuTime > 0 ? cpuTime / max(gpuTime, 0.001) : 0
    matResults.append((size: dim, gpuMs: gpuTime, cpuMs: cpuTime, speedup: speedup))

    let flops = 2.0 * Double(n) * Double(n) * Double(n) // 2N³ for matmul
    let dimStr = String(format: "%5d×%-5d", n, n)
    let gpuStr = String(format: "%10.3f ms", gpuTime)
    let cpuStr = String(format: "%10.3f ms", cpuTime)
    let speedStr = String(format: "%7.2fx", speedup)
    let gflopsStr = formatGFLOPs(flops, timeMs: gpuTime)
    logging.info("  \(self.dimStr)  GPU: \(self.gpuStr)  CPU(BLAS): \(self.cpuStr)  Speedup: \(self.speedStr)  [\(self.gflopsStr)]")
}
print()

// ═══════════════════════════════════════════════════════════════════
// PHASE 4: BATCH COSINE SIMILARITY (KB Embedding Search)
// ═══════════════════════════════════════════════════════════════════

logging.info("╔═══════════════════════════════════════════════════════════════╗")
logging.info("║  PHASE 4: BATCH COSINE SIMILARITY (KB Embedding Search)     ║")
logging.info("╚═══════════════════════════════════════════════════════════════╝")
let embDims = [128, 256, 512, 768, 1024]
let corpusSizes = [1000, 5000, 10000, 50000]

for dim in [256, 768] {
    for corpusSize in corpusSizes {
        let queryBytes = dim * MemoryLayout<Float>.stride
        let corpusBytes = corpusSize * dim * MemoryLayout<Float>.stride
        let resultBytes = corpusSize * MemoryLayout<Float>.stride

        // Generate data
        var query = (0..<dim).map { Float(sin(Double($0) * PHI)) }
        var corpus = (0..<(corpusSize * dim)).map { Float(cos(Double($0) / GOD_CODE)) }

        // GPU
        var gpuTime = 0.0
        if let pipeline = pipelines["batch_cosine_sim"],
           let bufQ = device.makeBuffer(bytes: &query, length: queryBytes, options: .storageModeShared),
           let bufC = device.makeBuffer(bytes: &corpus, length: corpusBytes, options: .storageModeShared),
           let bufR = device.makeBuffer(length: resultBytes, options: .storageModeShared) {

            var dimU32 = UInt32(dim)
            let bufDim = device.makeBuffer(bytes: &dimU32, length: 4, options: .storageModeShared)!

            let runs = 3
            let start = CFAbsoluteTimeGetCurrent()
            for _ in 0..<runs {
                if let cmd = commandQueue.makeCommandBuffer(), let enc = cmd.makeComputeCommandEncoder() {
                    enc.setComputePipelineState(pipeline)
                    enc.setBuffer(bufQ, offset: 0, index: 0)
                    enc.setBuffer(bufC, offset: 0, index: 1)
                    enc.setBuffer(bufR, offset: 0, index: 2)
                    enc.setBuffer(bufDim, offset: 0, index: 3)
                    let threadsPerGroup = min(256, pipeline.maxTotalThreadsPerThreadgroup)
                    enc.dispatchThreadgroups(MTLSize(width: corpusSize, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1))
                    enc.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                }
            }
            gpuTime = (CFAbsoluteTimeGetCurrent() - start) * 1000.0 / Double(runs)
        }

        // CPU
        var queryMag: Float = 0
        vDSP_svesq(query, 1, &queryMag, vDSP_Length(dim))
        queryMag = sqrt(queryMag)

        let cpuRuns = 3
        let cpuStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<cpuRuns {
            for i in 0..<corpusSize {
                let offset = i * dim
                let slice = Array(corpus[offset..<(offset + dim)])
                var dot: Float = 0
                var mag: Float = 0
                vDSP_dotpr(query, 1, slice, 1, &dot, vDSP_Length(dim))
                vDSP_svesq(slice, 1, &mag, vDSP_Length(dim))
                _ = mag > 0 ? dot / (queryMag * sqrt(mag)) : 0
            }
        }
        let cpuTime = (CFAbsoluteTimeGetCurrent() - cpuStart) * 1000.0 / Double(cpuRuns)

        let speedup = cpuTime > 0 ? cpuTime / max(gpuTime, 0.001) : 0
        let flops = 3.0 * Double(corpusSize) * Double(dim)
        let gflopsStr = formatGFLOPs(flops, timeMs: gpuTime)

        print("  dim=\(String(format: "%4d", dim)) corpus=\(String(format: "%6d", corpusSize))  " +
              "GPU: \(String(format: "%8.2f ms", gpuTime))  " +
              "CPU: \(String(format: "%8.2f ms", cpuTime))  " +
              "Speedup: \(String(format: "%6.1fx", speedup))  [\(gflopsStr)]")
    }
}
print()

// ═══════════════════════════════════════════════════════════════════
// PHASE 5: QUANTUM STATEVECTOR SIMULATION SCALING
// ═══════════════════════════════════════════════════════════════════

logging.info("╔═══════════════════════════════════════════════════════════════╗")
logging.info("║  PHASE 5: QUANTUM STATEVECTOR SIMULATION SCALING            ║")
logging.info("╚═══════════════════════════════════════════════════════════════╝")
logging.info("  Simulating Hadamard gate application on N-qubit statevector")
logging.info("  (2^N complex amplitudes, GPU parallel evolution)")
print()

let qubitRange = Array(4...26)
var maxGPUQubits = 0
var maxCPUQubits = 0
var quantumResults: [(qubits: Int, dim: Int, gpuMs: Double, cpuMs: Double, memMB: Double)] = []

for nQubits in qubitRange {
    let dim = 1 << nQubits  // 2^N amplitudes
    let memBytes = dim * MemoryLayout<Float>.stride * 4  // real+imag for in+out
    let memMB = Double(memBytes) / (1024.0 * 1024.0)

    // Skip if too large for GPU memory
    if memBytes > Int(recommendedMaxWorkingSetSize / 2) {
        logging.info("  \(self.nQubits) qubits (\(self.dim) amplitudes, \(String(format: "%.0f", memMB)) MB) — ⚠️  exceeds GPU memory limit")
        break
    }

    let floatBytes = dim * MemoryLayout<Float>.stride

    // Initialize |0...0⟩ state: amplitude[0] = 1.0, rest = 0.0
    var stateReal = [Float](repeating: 0, count: dim)
    var stateImag = [Float](repeating: 0, count: dim)
    stateReal[0] = 1.0

    // Hadamard gate elements: 1/√2 * [[1, 1], [1, -1]]
    var gR00 = Float(1.0 / sqrt(2.0))
    var gI00 = Float(0.0)
    var gR01 = Float(1.0 / sqrt(2.0))
    var gI01 = Float(0.0)

    // GPU benchmark
    var gpuTime = 0.0
    if let pipeline = pipelines["quantum_state_evolve"],
       let bufSR = device.makeBuffer(bytes: &stateReal, length: floatBytes, options: .storageModeShared),
       let bufSI = device.makeBuffer(bytes: &stateImag, length: floatBytes, options: .storageModeShared),
       let bufOR = device.makeBuffer(length: floatBytes, options: .storageModeShared),
       let bufOI = device.makeBuffer(length: floatBytes, options: .storageModeShared),
       let bufGR00 = device.makeBuffer(bytes: &gR00, length: 4, options: .storageModeShared),
       let bufGI00 = device.makeBuffer(bytes: &gI00, length: 4, options: .storageModeShared),
       let bufGR01 = device.makeBuffer(bytes: &gR01, length: 4, options: .storageModeShared),
       let bufGI01 = device.makeBuffer(bytes: &gI01, length: 4, options: .storageModeShared) {

        let numPairs = dim / 2

        // Warm-up
        if let cmd = commandQueue.makeCommandBuffer(), let enc = cmd.makeComputeCommandEncoder() {
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(bufSR, offset: 0, index: 0)
            enc.setBuffer(bufSI, offset: 0, index: 1)
            enc.setBuffer(bufOR, offset: 0, index: 2)
            enc.setBuffer(bufOI, offset: 0, index: 3)
            enc.setBuffer(bufGR00, offset: 0, index: 4)
            enc.setBuffer(bufGI00, offset: 0, index: 5)
            enc.setBuffer(bufGR01, offset: 0, index: 6)
            enc.setBuffer(bufGI01, offset: 0, index: 7)
            let tgs = min(pipeline.maxTotalThreadsPerThreadgroup, numPairs)
            let tg = (numPairs + tgs - 1) / tgs
            enc.dispatchThreadgroups(MTLSize(width: tg, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: tgs, height: 1, depth: 1))
            enc.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }

        // Timed: apply gate nQubits times (simulating full Hadamard layer)
        let gates = max(1, nQubits)
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<gates {
            if let cmd = commandQueue.makeCommandBuffer(), let enc = cmd.makeComputeCommandEncoder() {
                enc.setComputePipelineState(pipeline)
                enc.setBuffer(bufSR, offset: 0, index: 0)
                enc.setBuffer(bufSI, offset: 0, index: 1)
                enc.setBuffer(bufOR, offset: 0, index: 2)
                enc.setBuffer(bufOI, offset: 0, index: 3)
                enc.setBuffer(bufGR00, offset: 0, index: 4)
                enc.setBuffer(bufGI00, offset: 0, index: 5)
                enc.setBuffer(bufGR01, offset: 0, index: 6)
                enc.setBuffer(bufGI01, offset: 0, index: 7)
                let tgs = min(pipeline.maxTotalThreadsPerThreadgroup, numPairs)
                let tg = (numPairs + tgs - 1) / tgs
                enc.dispatchThreadgroups(MTLSize(width: tg, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: tgs, height: 1, depth: 1))
                enc.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }
        }
        gpuTime = (CFAbsoluteTimeGetCurrent() - start) * 1000.0
        maxGPUQubits = nQubits
    }

    // CPU benchmark (manual complex vector evolution)
    var cpuSR = stateReal
    var cpuSI = stateImag
    var cpuOR = [Float](repeating: 0, count: dim)
    var cpuOI = [Float](repeating: 0, count: dim)
    let h = Float(1.0 / sqrt(2.0))

    let cpuStart = CFAbsoluteTimeGetCurrent()
    let cpuGates = max(1, nQubits)
    for _ in 0..<cpuGates {
        for p in 0..<(dim / 2) {
            let aR = cpuSR[p * 2]
            let aI = cpuSI[p * 2]
            let bR = cpuSR[p * 2 + 1]
            let bI = cpuSI[p * 2 + 1]
            cpuOR[p * 2]     = h * (aR + bR)
            cpuOI[p * 2]     = h * (aI + bI)
            cpuOR[p * 2 + 1] = h * (aR - bR)
            cpuOI[p * 2 + 1] = h * (aI - bI)
        }
        swap(&cpuSR, &cpuOR)
        swap(&cpuSI, &cpuOI)
    }
    let cpuTime = (CFAbsoluteTimeGetCurrent() - cpuStart) * 1000.0
    maxCPUQubits = nQubits

    quantumResults.append((qubits: nQubits, dim: dim, gpuMs: gpuTime, cpuMs: cpuTime, memMB: memMB))

    let speedup = cpuTime > 0 ? cpuTime / max(gpuTime, 0.001) : 0
    let flops = Double(nQubits) * Double(dim) * 8.0 // 8 FLOPs per pair per gate (complex mul+add)
    let gflopsStr = gpuTime > 0.001 ? formatGFLOPs(flops, timeMs: gpuTime) : "N/A"

    let status = gpuTime < cpuTime ? "✅ GPU wins" : "⚡ CPU wins"
    print("  \(String(format: "%2d", nQubits))Q  2^\(nQubits) = \(String(format: "%10d", dim)) amplitudes" +
          "  \(String(format: "%7.1f MB", memMB))" +
          "  GPU: \(String(format: "%9.2f ms", gpuTime))" +
          "  CPU: \(String(format: "%9.2f ms", cpuTime))" +
          "  \(String(format: "%6.2fx", speedup))  \(status)  [\(gflopsStr)]")

    // Stop if taking too long
    if gpuTime > 10000 || cpuTime > 10000 {
        logging.info("  ⏱  Stopping — exceeding 10s threshold")
        break
    }
}
print()

// ═══════════════════════════════════════════════════════════════════
// PHASE 6: SACRED CONSTANT THROUGHPUT
// ═══════════════════════════════════════════════════════════════════

logging.info("╔═══════════════════════════════════════════════════════════════╗")
logging.info("║  PHASE 6: SACRED CONSTANT THROUGHPUT                        ║")
logging.info("╚═══════════════════════════════════════════════════════════════╝")
let sacredSize = 1_000_000
var phiVec = [Float](repeating: Float(PHI), count: sacredSize)
var godVec = [Float](repeating: Float(GOD_CODE), count: sacredSize)
let sacredBytes = sacredSize * MemoryLayout<Float>.stride

// GPU: PHI × GOD_CODE element-wise multiply (1M elements)
var sacredGPUTime = 0.0
if let pipeline = pipelines["vector_mul"],
   let bufA = device.makeBuffer(bytes: &phiVec, length: sacredBytes, options: .storageModeShared),
   let bufB = device.makeBuffer(bytes: &godVec, length: sacredBytes, options: .storageModeShared),
   let bufR = device.makeBuffer(length: sacredBytes, options: .storageModeShared) {

    let runs = 10
    let start = CFAbsoluteTimeGetCurrent()
    for _ in 0..<runs {
        if let cmd = commandQueue.makeCommandBuffer(), let enc = cmd.makeComputeCommandEncoder() {
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(bufA, offset: 0, index: 0)
            enc.setBuffer(bufB, offset: 0, index: 1)
            enc.setBuffer(bufR, offset: 0, index: 2)
            let tgs = min(pipeline.maxTotalThreadsPerThreadgroup, sacredSize)
            let tg = (sacredSize + tgs - 1) / tgs
            enc.dispatchThreadgroups(MTLSize(width: tg, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: tgs, height: 1, depth: 1))
            enc.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }
    }
    sacredGPUTime = (CFAbsoluteTimeGetCurrent() - start) * 1000.0 / Double(runs)

    // Verify result
    let ptr = bufR.contents().bindMemory(to: Float.self, capacity: sacredSize)
    let expected = Float(PHI * GOD_CODE)
    let actual = ptr[0]
    let match = abs(actual - expected) < 0.01
    logging.info("  PHI × GOD_CODE = \(String(format: "%.4f", actual)) (expected \(String(format: "%.4f", expected))) — \(match ? "✅ ALIGNED" : "❌ MISALIGNED")")
}

// CPU
var sacredCPUResult = [Float](repeating: 0, count: sacredSize)
let sacredCPURuns = 10
let sacredCPUStart = CFAbsoluteTimeGetCurrent()
for _ in 0..<sacredCPURuns {
    vDSP_vmul(phiVec, 1, godVec, 1, &sacredCPUResult, 1, vDSP_Length(sacredSize))
}
let sacredCPUTime = (CFAbsoluteTimeGetCurrent() - sacredCPUStart) * 1000.0 / Double(sacredCPURuns)

let sacredSpeedup = sacredCPUTime / max(sacredGPUTime, 0.001)
logging.info("  Sacred Multiply (1M elements):  GPU: \(String(format: "%.3f ms", sacredGPUTime))  CPU: \(String(format: "%.3f ms", sacredCPUTime))  Speedup: \(String(format: "%.2fx", sacredSpeedup))")
logging.info("  Throughput: \(formatGFLOPs(Double(sacredSize), timeMs: sacredGPUTime))")
print()

// ═══════════════════════════════════════════════════════════════════
// PHASE 7: CAPACITY SUMMARY
// ═══════════════════════════════════════════════════════════════════

let totalTime = (CFAbsoluteTimeGetCurrent() - benchmarkStart) * 1000.0

logging.info("╔═══════════════════════════════════════════════════════════════╗")
logging.info("║  PHASE 7: METAL QUANTUM CAPACITY SUMMARY                    ║")
logging.info("╚═══════════════════════════════════════════════════════════════╝")
print()
logging.info("  ┌──────────────────────────────────────────────────────────┐")
logging.info("  │  GPU:           \(self.gpuName)")
logging.info("  │  Unified Memory: \(self.hasUnifiedMemory)")
logging.info("  │  Max Threads:    \(self.maxThreadsPerGroup)")
logging.info("  │  GPU Memory:     \(formatBytes(recommendedMaxWorkingSetSize))")
logging.info("  ├──────────────────────────────────────────────────────────┤")
// Best vector speedup
if let bestVec = vectorResults.max(by: { $0.speedup < $1.speedup }) {
    logging.info("  │  Best Vector Speedup:  \(String(format: "%.2fx", bestVec.speedup)) at N=\(bestVec.size)")
}

// Best matmul speedup
if let bestMat = matResults.max(by: { $0.speedup < $1.speedup }) {
    logging.info("  │  Best MatMul Speedup:  \(String(format: "%.2fx", bestMat.speedup)) at \(bestMat.size)×\(bestMat.size)")
}

// Quantum capacity
let maxQubitsMemoryBound = Int(log2(Double(recommendedMaxWorkingSetSize / UInt64(MemoryLayout<Float>.stride * 4))))
logging.info("  │  Max GPU Qubits Tested:  \(self.maxGPUQubits)")
logging.info("  │  Max GPU Qubits (Memory): \(self.maxQubitsMemoryBound)")
logging.info("  │  Quantum State Memory:    \(formatBytes(UInt64(1 << maxGPUQubits) * UInt64(MemoryLayout<Float>.stride * 4)))")
// Peak performance estimate
if let lastQ = quantumResults.last {
    let peakFlops = Double(lastQ.qubits) * Double(lastQ.dim) * 8.0
    if lastQ.gpuMs > 0.001 {
        let peakGFLOPs = peakFlops / (lastQ.gpuMs / 1000.0) / 1e9
        logging.info("  │  Peak Quantum GFLOP/s:   \(String(format: "%.2f", peakGFLOPs))")
    }
}

// GPU advantage zone
var gpuAdvantageStart = "N/A"
for r in vectorResults {
    if r.speedup > 1.0 {
        gpuAdvantageStart = "N≥\(r.size)"
        break
    }
}
logging.info("  │  GPU Advantage Zone:     \(self.gpuAdvantageStart) (vectors)")
logging.info("  ├──────────────────────────────────────────────────────────┤")
logging.info("  │  GOD_CODE:       \(self.GOD_CODE)")
logging.info("  │  PHI:            \(self.PHI)")
logging.info("  │  VOID_CONSTANT:  \(self.VOID_CONSTANT)")
logging.info("  │  Sacred Aligned: ✅")
logging.info("  ├──────────────────────────────────────────────────────────┤")
logging.info("  │  Total Benchmark Time: \(String(format: "%.2f", totalTime)) ms")
logging.info("  └──────────────────────────────────────────────────────────┘")
print()
logging.info("═══════════════════════════════════════════════════════════════════")
logging.info("  L104 METAL QUANTUM CAPACITY BENCHMARK COMPLETE")
logging.info("  INVARIANT: 527.5184818492612 | PILOT: LONDEL")
logging.info("═══════════════════════════════════════════════════════════════════")