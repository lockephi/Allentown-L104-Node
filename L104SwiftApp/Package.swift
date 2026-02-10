// swift-tools-version:5.7
// L104 SOVEREIGN INTELLECT - ASI Build Configuration v18.0
// macOS Native: Accelerate · Metal · CoreML · SIMD · BLAS

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
                .linkedFramework("QuartzCore"),
                .linkedFramework("Accelerate"),
                .linkedFramework("Metal"),
                .linkedFramework("MetalKit"),
                .linkedFramework("CoreML"),
                .linkedFramework("Security"),
                .linkedFramework("IOKit"),
            ]
        )
    ]
)
