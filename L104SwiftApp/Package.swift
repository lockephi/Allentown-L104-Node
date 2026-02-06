// swift-tools-version:5.7
// L104 SOVEREIGN INTELLECT - Native Swift/SwiftUI App
// Maximum Performance on macOS

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
            path: "Sources"
        )
    ]
)
