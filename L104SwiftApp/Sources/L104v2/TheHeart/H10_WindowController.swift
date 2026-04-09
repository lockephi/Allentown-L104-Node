import Accelerate
import AppKit
import Foundation
import NaturalLanguage
import simd

// ═══════════════════════════════════════════════════════════════════
// MAIN WINDOW
// ═══════════════════════════════════════════════════════════════════

class L104WindowController: NSWindowController, NSWindowDelegate {
    convenience init() {
        let w = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1400, height: 800),
                        styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView], backing: .buffered, defer: false)
        w.title = "⚛️ L104 SOVEREIGN INTELLECT - ASI TRANSCENDENCE"
        w.center(); w.minSize = NSSize(width: 1000, height: 700)
        w.backgroundColor = NSColor(red: 0.965, green: 0.965, blue: 0.975, alpha: 1.0)
        w.titlebarAppearsTransparent = true
        w.titleVisibility = .hidden
        w.isOpaque = false
        w.isMovableByWindowBackground = true
        w.setFrameAutosaveName("L104MainWindow")  // Remember window position/size
        // Enable window resize notifications
        w.setContentSize(NSSize(width: 1280, height: 750))
        // Modern toolbar appearance
        if #available(macOS 11.0, *) {
            w.toolbarStyle = .unified
        }
        self.init(window: w)
        w.delegate = self
        // Use Auto Layout for proper resize handling
        let v = L104MainView(frame: w.contentView!.bounds)
        v.translatesAutoresizingMaskIntoConstraints = false
        w.contentView = v

        // Activate constraints for full-size content
        NSLayoutConstraint.activate([
            v.leadingAnchor.constraint(equalTo: w.contentView!.leadingAnchor),
            v.trailingAnchor.constraint(equalTo: w.contentView!.trailingAnchor),
            v.topAnchor.constraint(equalTo: w.contentView!.topAnchor),
            v.bottomAnchor.constraint(equalTo: w.contentView!.bottomAnchor),
        ])
    }

    // MARK: - NSWindowDelegate - Handle window resize
    func windowDidResize(_ notification: Notification) {
        guard let window = notification.object as? NSWindow else { return }
        let size = window.contentView?.frame.size ?? .zero
        // Post notification for views to handle resize if needed
        NotificationCenter.default.post(name: NSWindow.didResizeNotification, object: size)
    }

    func windowWillResize(_ sender: NSWindow, to frameSize: NSSize) -> NSSize {
        // Enforce minimum size
        let minWidth: CGFloat = 900
        let minHeight: CGFloat = 600
        return NSSize(
            width: max(frameSize.width, minWidth),
            height: max(frameSize.height, minHeight)
        )
    }

    // WINDOW CLOSE PROTECTION - prevent accidental Cmd+W or close button from killing the app
    func windowShouldClose(_ sender: NSWindow) -> Bool {
        let alert = NSAlert()
        alert.messageText = "Close L104?"
        alert.informativeText = "This will save all memories and shut down the Sovereign Intellect."
        alert.addButton(withTitle: "Stay Open")
        alert.addButton(withTitle: "Close")
        alert.alertStyle = .warning
        let response = alert.runModal()
        if response == .alertSecondButtonReturn {
            // Save everything before closing
            L104State.shared.saveState()
            L104State.shared.permanentMemory.save()
            AdaptiveLearner.shared.save()
            return true
        }
        return false
    }
}

// ═══════════════════════════════════════════════════════════════════
// MAIN VIEW
// ═══════════════════════════════════════════════════════════════════

