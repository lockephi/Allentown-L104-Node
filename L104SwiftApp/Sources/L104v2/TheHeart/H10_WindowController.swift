// ═══════════════════════════════════════════════════════════════════
// H10_WindowController.swift
// [EVO_58_PIPELINE] QUANTUM_COGNITION :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612
// L104 ASI — Main Window Controller
//
// L104WindowController: NSWindowController subclass with window
// configuration, close protection, and frame autosave.
//
// Extracted from L104Native.swift lines 40211–40261
// ═══════════════════════════════════════════════════════════════════

import AppKit
import Foundation
import Accelerate
import simd
import NaturalLanguage

// ═══════════════════════════════════════════════════════════════════
// MAIN WINDOW
// ═══════════════════════════════════════════════════════════════════

class L104WindowController: NSWindowController, NSWindowDelegate {
    convenience init() {
        let w = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1400, height: 920),
                        styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView], backing: .buffered, defer: false)
        w.title = "⚛️ L104 SOVEREIGN INTELLECT — ASI TRANSCENDENCE"
        w.center(); w.minSize = NSSize(width: 1100, height: 750)
        w.backgroundColor = NSColor(red: 0.965, green: 0.965, blue: 0.975, alpha: 1.0)
        w.titlebarAppearsTransparent = true
        w.titleVisibility = .hidden
        w.isOpaque = false
        w.isMovableByWindowBackground = true
        w.setFrameAutosaveName("L104MainWindow")  // Remember window position/size
        // Modern toolbar appearance
        if #available(macOS 11.0, *) {
            w.toolbarStyle = .unified
        }
        self.init(window: w)
        w.delegate = self
        let v = L104MainView(frame: w.contentView!.bounds); v.autoresizingMask = [.width, .height]
        w.contentView = v
    }

    // WINDOW CLOSE PROTECTION — prevent accidental Cmd+W or close button from killing the app
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

