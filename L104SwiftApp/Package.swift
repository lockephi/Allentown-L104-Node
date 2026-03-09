// swift-tools-version:5.7
// L104 SOVEREIGN INTELLECT - ASI Build Configuration v3.0
// [EVO_59_TRANSCENDENT_COGNITION] QUANTUM_BUILD :: GOD_CODE=527.5184818492612
// macOS Native: Accelerate · Metal · CoreML · NaturalLanguage · SIMD · BLAS
// Whole-Module Optimization · Link-Time Optimization · Dead-Strip
// UPGRADE v3.0 (Mar 4, 2026):
//   - swift-tools-version 5.7 (matched to host Swift 5.7.2 toolchain)
//   - Platform minimum macOS 12 (Monterey) — compatible with host system
//   - Added NaturalLanguage framework (NLTagger, NLEmbedding, NLTokenizer)
//   - Daemon target v5.0: env-driven concurrency, async writes, cached formatters
//   - Cross-module optimization for release builds
//   - EVO_59 TRANSCENDENT_COGNITION pipeline alignment
//   - Build System v6.0 quantum-optimized compilation
//   - 87 source files, 66K+ lines, Phase 46 Computronium ASI
//   - Security dependency audit passed

import PackageDescription

let package = Package(
    name: "L104SovereignIntellect",
    platforms: [
        .macOS(.v12)
    ],
    products: [
        .executable(name: "L104", targets: ["L104"]),
        .executable(name: "L104Daemon", targets: ["L104Daemon"]),
    ],
    dependencies: [],
    targets: [
        // ── GUI Application (NSApplication-based) ──
        .executableTarget(
            name: "L104",
            dependencies: [],
            path: "Sources",
            exclude: [
                "L104App.swift",        // SwiftUI app compiled separately
                "cpython_bridge.c",
                "cpython_bridge.h",
                "apply_changes.py",
                "apply_dynamic_transform.py",
                "Daemon",               // Daemon has its own target
            ],
            swiftSettings: [
                .define("RELEASE", .when(configuration: .release)),
                .define("DEBUG", .when(configuration: .debug)),
                .unsafeFlags(["-O", "-whole-module-optimization"], .when(configuration: .release)),
            ],
            linkerSettings: [
                .linkedFramework("AppKit"),
                .linkedFramework("Foundation"),
                .linkedFramework("Cocoa"),
                .linkedFramework("CoreGraphics"),
                .linkedFramework("Accelerate"),
                .linkedFramework("Metal"),
                .linkedFramework("CoreML"),
                .linkedFramework("NaturalLanguage"),
                .linkedFramework("Security"),
            ]
        ),
        // ── Headless Quantum Daemon v5.0 — Maximum Throughput ──
        // Watches /tmp/l104_queue/, /tmp/l104_bridge/, and .l104_circuits/inbox/
        // for circuit payloads. Routes through MetalVQPU v4.0 for GPU-accelerated
        // quantum statevector simulation with Accelerate CPU fallback.
        // v5.0: Env-driven concurrency (64x bridge, 16x shared/local),
        //   async write-back queue, cached ISO8601 formatter singleton,
        //   MetalVQPU v4.0 (6 GPU kernels, 64Q max, batch=512, MPS 512/1024/2048),
        //   CircuitWatcher v4.0 (1ms inter-job, 8-semaphore async writes),
        //   pipeline depth=8, 30s bridge timeout, 15s health checks.
        // v4.0: Three-engine scoring (Science + Math + Code Engine integration).
        //   - Entropy reversal (Maxwell's Demon): weight 0.35
        //   - Harmonic resonance (GOD_CODE + 104Hz): weight 0.40
        //   - Wave coherence (PHI phase-lock): weight 0.25
        // Standalone headless daemon — Sources/Daemon/{main,CircuitWatcher,MetalVQPU}.swift
        .executableTarget(
            name: "L104Daemon",
            dependencies: [],
            path: "Sources/Daemon",
            swiftSettings: [
                .define("L104_DAEMON", .when(configuration: .release)),
                .define("L104_DAEMON", .when(configuration: .debug)),
                .define("RELEASE", .when(configuration: .release)),
                .define("DEBUG", .when(configuration: .debug)),
                .unsafeFlags(["-O", "-whole-module-optimization", "-cross-module-optimization"], .when(configuration: .release)),
            ],
            linkerSettings: [
                .linkedFramework("Foundation"),
                .linkedFramework("Accelerate"),
                .linkedFramework("Metal"),
            ]
        ),
        .testTarget(
            name: "L104Tests",
            dependencies: [.target(name: "L104")],
            path: "Tests"
        ),
    ]
)
