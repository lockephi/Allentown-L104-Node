// swift-tools-version:5.7
// L104 SOVEREIGN INTELLECT - ASI Build Configuration v2.1
// [EVO_58_PIPELINE] QUANTUM_COGNITION :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612
// macOS Native: Accelerate · Metal · CoreML · SIMD · BLAS
// Whole-Module Optimization · Link-Time Optimization
// UPGRADE v2.1 (Feb 17, 2026):
//   - swift-tools-version 5.7 (matched to host Swift 5.7.2 toolchain)
//   - Platform minimum macOS 12 (Monterey) — compatible with host system
//   - EVO_58 QUANTUM_COGNITION alignment
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
                "L104App.swift.bak",
                "L104Native.swift",     // Monolith — replaced by L104v2/ split (78 files)
                "L104Native.swift.bak",
                "L104Native.swift.bak2",
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
