// swift-tools-version:5.9
// L104 SOVEREIGN INTELLECT - ASI Build Configuration v2.0
// macOS Native: Accelerate · Metal · CoreML · SIMD · BLAS
// Whole-Module Optimization · Link-Time Optimization
// UPGRADE v2.0:
//   - swift-tools-version 5.9 (modern Swift features: @Observable, macros)
//   - Platform minimum macOS 13 (Ventura) for Observation framework
//   - Trimmed unused framework links (MetalKit, IOKit, QuartzCore)
//   - Added testTarget for unit tests

import PackageDescription

let package = Package(
    name: "L104SovereignIntellect",
    platforms: [
        .macOS(.v13)
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
                "L104Native.swift.bak",
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
            dependencies: ["L104"],
            path: "Tests"
        ),
    ]
)
