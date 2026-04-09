import AppKit
import Foundation

// MARK: - Autonomous Process Helper

private class AutonomousProcess {
    let id: String
    let daemonName: String?
    let startDate: Date
    var isCompleted: Bool = false

    init(id: String, daemonName: String?) {
        self.id = id
        self.daemonName = daemonName
        self.startDate = Date()
    }
}