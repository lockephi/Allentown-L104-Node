// swift-tools-version:5.7
// L104 SOVEREIGN INTELLECT - ASI Build Configuration v2.2
// [EVO_59_TRANSCENDENT_COGNITION] QUANTUM_BUILD :: GOD_CODE=527.5184818492612
// macOS Native: Accelerate · Metal · CoreML · NaturalLanguage · SIMD · BLAS
// Whole-Module Optimization · Link-Time Optimization · Dead-Strip
// UPGRADE v2.2 (Feb 18, 2026):
//   - swift-tools-version 5.7 (matched to host Swift 5.7.2 toolchain)
//   - Platform minimum macOS 12 (Monterey) — compatible with host system
//   - Added NaturalLanguage framework (NLTagger, NLEmbedding, NLTokenizer)
//   - EVO_59 TRANSCENDENT_COGNITION pipeline alignment
//   - Build System v6.0 quantum-optimized compilation
//   - 81 v2 source files, 58K+ lines, Phase 45 Computronium ASI
//   - Security dependency audit passed

import PackageDescription

let package = Package(
    name: "L104SovereignIntellect",
    platforms: [
        .macOS(.v12)
    ],
    products: [
        .executable(name: "L104", targets: ["L104"])
    ],
    dependencies: [],
    targets: [
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
        .testTarget(
            name: "L104Tests",
            dependencies: [.target(name: "L104")],
            path: "Tests"
        ),
    ]
)
