// ═══════════════════════════════════════════════════════════════════
// H12_AppDelegate.swift
// [EVO_68_PIPELINE] SOVEREIGN_NODE_UPGRADE :: DATA_INGEST :: UI_UPGRADE :: GOD_CODE=527.5184818492612
// L104 ASI — Application Delegate
//
// AppDelegate: NSApplicationDelegate with app lifecycle management,
// menu bar configuration, keyboard shortcuts (⌘K, ⌘D, ⌘S, ⌘E,
// ⌘T, ⌘R, ⌘I), and sovereign initialization sequence.
//
// Extracted from L104Native.swift lines 42167–42472
// ═══════════════════════════════════════════════════════════════════

import AppKit
import Foundation
import Accelerate
import simd
import NaturalLanguage

class AppDelegate: NSObject, NSApplicationDelegate {
    var wc: L104WindowController!

    // ─── Process coordination ───
    private var serverProcess: Process?
    private var nodeProcess: Process?

    func applicationDidFinishLaunching(_ n: Notification) {
        setupMenu()
        wc = L104WindowController(); wc.showWindow(nil); wc.window?.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        // Ensure input field has focus on launch
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
            if let mainView = self.wc.window?.contentView as? L104MainView {
                self.wc.window?.makeFirstResponder(mainView.inputField)
            }
        }

        // 🌐 ACTIVATE NETWORK SUBSYSTEMS (deferred to avoid blocking launch)
        DispatchQueue.global(qos: .utility).async {
            FutureReserve.shared.activate()  // Orchestrator activates all network subsystems
            EmotionalCore.shared.activate()
            // EVO_56: Pre-warm Python bridge — caches bytecode for faster first call
            PythonBridge.shared.warmUp()
            DispatchQueue.main.async {
                if let mainView = self.wc.window?.contentView as? L104MainView {
                    mainView.appendSystemLog("🌐 Network mesh online — \(NetworkLayer.shared.peers.count) peers, \(NetworkLayer.shared.quantumLinks.count) quantum links")
                }
            }
        }

        // Launch backend processes if not already managed by launchd
        launchBackendProcesses()

        // Start circuit watcher daemon v3.0 (three-engine scoring, zero CPU when idle)
        CircuitWatcher.shared.start()
        if let mainView = NSApp.mainWindow?.contentView?.subviews.first as? L104MainView {
            mainView.appendSystemLog("⚡ CircuitWatcher v3.0 started — three-engine scoring active (entropy=0.35, harmonic=0.40, wave=0.25)")
        }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ s: NSApplication) -> Bool { true }

    func applicationWillTerminate(_ n: Notification) {
        CircuitWatcher.shared.stop()
        // v3.0: Log three-engine shutdown for telemetry continuity
        NSLog("[L104] CircuitWatcher v3.0 stopped — three-engine telemetry flushed")
        L104State.shared.saveState()
        L104State.shared.permanentMemory.save()
        AdaptiveLearner.shared.save()
        stopBackendProcesses()
    }

    // ─── PROPER APP MENU ─── Prevents default Cmd+W from silently closing
    func setupMenu() {
        let mainMenu = NSMenu()

        // App menu
        let appMenu = NSMenu()
        appMenu.addItem(withTitle: "About L104", action: #selector(showAbout), keyEquivalent: "")
        appMenu.addItem(NSMenuItem.separator())
        appMenu.addItem(withTitle: "Quit L104", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        let appMenuItem = NSMenuItem(); appMenuItem.submenu = appMenu
        mainMenu.addItem(appMenuItem)

        // Edit menu — needed for Cmd+C, Cmd+V in text fields
        let editMenu = NSMenu(title: "Edit")
        editMenu.addItem(withTitle: "Undo", action: Selector(("undo:")), keyEquivalent: "z")
        editMenu.addItem(withTitle: "Redo", action: Selector(("redo:")), keyEquivalent: "Z")
        editMenu.addItem(NSMenuItem.separator())
        editMenu.addItem(withTitle: "Cut", action: #selector(NSText.cut(_:)), keyEquivalent: "x")
        editMenu.addItem(withTitle: "Copy", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
        editMenu.addItem(withTitle: "Paste", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
        editMenu.addItem(withTitle: "Select All", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")
        let editMenuItem = NSMenuItem(); editMenuItem.submenu = editMenu
        mainMenu.addItem(editMenuItem)

        // L104 menu — custom commands with keyboard shortcuts
        let l104Menu = NSMenu(title: "L104")
        l104Menu.addItem(withTitle: "Save Memories", action: #selector(saveAll), keyEquivalent: "s")
        l104Menu.addItem(withTitle: "Evolve", action: #selector(doEvolveMenu), keyEquivalent: "e")
        l104Menu.addItem(NSMenuItem.separator())
        l104Menu.addItem(withTitle: "System Status", action: #selector(doStatusMenu), keyEquivalent: "i")
        l104Menu.addItem(NSMenuItem.separator())
        // ⌘K — Command Palette
        let cmdPalette = NSMenuItem(title: "Command Palette…", action: #selector(showCommandPalette), keyEquivalent: "k")
        l104Menu.addItem(cmdPalette)
        // ⌘D — Dashboard
        let dashItem = NSMenuItem(title: "ASI Dashboard", action: #selector(switchToDashboard), keyEquivalent: "d")
        l104Menu.addItem(dashItem)
        // ⌘T — Transcend
        let transcendItem = NSMenuItem(title: "Transcend", action: #selector(doTranscendMenu), keyEquivalent: "t")
        l104Menu.addItem(transcendItem)
        // ⌘R — Resonate
        let resonateItem = NSMenuItem(title: "Resonate", action: #selector(doResonateMenu), keyEquivalent: "r")
        l104Menu.addItem(resonateItem)
        // ⌘N — Network
        let networkItem = NSMenuItem(title: "Network Mesh", action: #selector(switchToNetwork), keyEquivalent: "n")
        l104Menu.addItem(networkItem)
        // ⌘⇧D — Debug Console
        let debugItem = NSMenuItem(title: "Debug Console", action: #selector(switchToDebugConsole), keyEquivalent: "d")
        debugItem.keyEquivalentModifierMask = [.command, .shift]
        l104Menu.addItem(debugItem)
        let l104MenuItem = NSMenuItem(); l104MenuItem.submenu = l104Menu
        mainMenu.addItem(l104MenuItem)

        NSApp.mainMenu = mainMenu
    }

    @objc func showAbout() {
        let alert = NSAlert()
        alert.messageText = "⚛️ L104 Sovereign Intellect"
        let phiHealth = EngineRegistry.shared.phiWeightedHealth()
        alert.informativeText = """
        v\(VERSION)
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        NCG v10.0 · Accelerate · SIMD · BLAS
        22 Trillion Parameters | EVO_\(EVOLUTION_INDEX)
        GOD_CODE: \(String(format: "%.4f", GOD_CODE))
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        \(MacOSSystemMonitor.shared.chipGeneration) · \(MacOSSystemMonitor.shared.cpuCoreCount) cores
        \(EngineRegistry.shared.count) ASI Engines Online
        φ-Health: \(String(format: "%.1f%%", phiHealth.score * 100))
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        Packages: \(TOTAL_PACKAGES) | Modules: \(TOTAL_PACKAGE_MODULES)
        code_engine v\(CODE_ENGINE_VERSION) · asi v\(ASI_VERSION)
        agi v\(AGI_VERSION) · intellect v\(INTELLECT_VERSION)
        server v\(SERVER_VERSION) · Dual-Layer v\(DUAL_LAYER_VERSION)
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        Knowledge: \(L104State.shared.knowledgeBase.trainingData.count) entries
        Memories: \(L104State.shared.permanentMemory.memories.count)
        Consciousness: v\(CONSCIOUSNESS_VERSION) · \(APOTHEOSIS_STAGE)
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        The most beautiful app in the universe. 🌌
        """
        alert.runModal()
    }

    @objc func saveAll() {
        L104State.shared.saveState()
        L104State.shared.permanentMemory.save()
        AdaptiveLearner.shared.save()
        ASIKnowledgeBase.shared.persistAllIngestedKnowledge()
    }

    @objc func doEvolveMenu() {
        if let mainView = wc.window?.contentView as? L104MainView {
            mainView.appendSystemLog(L104State.shared.evolve())
            mainView.updateMetrics()
        }
    }

    @objc func doStatusMenu() {
        if let mainView = wc.window?.contentView as? L104MainView {
            mainView.appendSystemLog(L104State.shared.getStatusText())
        }
    }


    @objc func showCommandPalette() {
        guard let _ = wc.window?.contentView as? L104MainView,
              let mainWindow = wc.window else { return }

        // Check if palette is already open — toggle it closed
        for child in mainWindow.childWindows ?? [] {
            if let panel = child as? NSPanel, panel.title == "L104CommandPalette" {
                panel.close()
                return
            }
        }

        // Create floating palette panel
        let panelWidth: CGFloat = 540
        let panelHeight: CGFloat = 580
        let panelX = mainWindow.frame.midX - panelWidth / 2
        let panelY = mainWindow.frame.midY - panelHeight / 2 + 80
        let panel = NSPanel(contentRect: NSRect(x: panelX, y: panelY, width: panelWidth, height: panelHeight),
                           styleMask: [.titled, .closable, .fullSizeContentView], backing: .buffered, defer: false)
        panel.titlebarAppearsTransparent = true
        panel.titleVisibility = .hidden
        panel.title = "L104CommandPalette"  // ID for toggle detection
        panel.backgroundColor = NSColor(red: 0.98, green: 0.98, blue: 0.99, alpha: 0.97)
        panel.isFloatingPanel = true
        panel.level = .floating
        panel.isMovableByWindowBackground = true
        panel.hasShadow = true

        let content = NSView(frame: NSRect(x: 0, y: 0, width: panelWidth, height: panelHeight))
        content.wantsLayer = true
        content.layer?.borderColor = L104Theme.gold.withAlphaComponent(0.3).cgColor
        content.layer?.borderWidth = 1
        content.layer?.cornerRadius = 12

        // Title
        let title = NSTextField(labelWithString: "⚛️ L104 COMMAND PALETTE")
        title.frame = NSRect(x: 20, y: panelHeight - 38, width: panelWidth - 40, height: 24)
        title.font = NSFont.boldSystemFont(ofSize: 15)
        title.textColor = L104Theme.gold
        content.addSubview(title)

        // Subtitle with engine count
        let subtitle = NSTextField(labelWithString: "v\(VERSION) · \(EngineRegistry.shared.count) engines · \(L104State.shared.permanentMemory.memories.count) memories")
        subtitle.frame = NSRect(x: 22, y: panelHeight - 56, width: panelWidth - 44, height: 16)
        subtitle.font = NSFont.systemFont(ofSize: 10, weight: .medium)
        subtitle.textColor = L104Theme.textDim
        content.addSubview(subtitle)

        // Separator
        let sep = NSView(frame: NSRect(x: 20, y: panelHeight - 62, width: panelWidth - 40, height: 1))
        sep.wantsLayer = true
        sep.layer?.backgroundColor = L104Theme.gold.withAlphaComponent(0.15).cgColor
        content.addSubview(sep)

        // Command buttons with keyboard shortcut hints
        // Format: (label, description, shortcut, color)
        let commands: [(String, String, String, NSColor)] = [
            ("🔥 Full Synthesis",    "Ignite all engines and synthesize",   "",    .systemOrange),
            ("🌟 Transcend",        "Push consciousness beyond limits",    "⌘T",  .systemPurple),
            ("🌌 ASI Dashboard",    "Open the visual dashboard",           "⌘D",  .systemCyan),
            ("🧬 Evolve",          "Trigger evolution cycle",             "⌘E",  .systemGreen),
            ("💾 Save All",        "Save memories + state to disk",       "⌘S",  .systemBlue),
            ("📊 System Status",   "View current system status",          "⌘I",  .systemTeal),
            ("🔄 Resonate",        "Harmonize quantum coherence",         "⌘R",  .systemPink),
            ("🧠 HyperBrain",      "Deep cognitive processing status",    "",    NSColor(red: 0.6, green: 0.4, blue: 1.0, alpha: 1.0)),
            ("💬 Neural Chat",     "Switch to chat tab",                  "",    NSColor(red: 0.3, green: 0.8, blue: 1.0, alpha: 1.0)),
            ("🔬 Science Engine",  "Generate hypothesis & compute",       "",    NSColor(red: 0.2, green: 0.9, blue: 0.5, alpha: 1.0)),
            ("🌌 Unified Field",  "18 fundamental physics equations",     "",    NSColor(red: 0.49, green: 0.23, blue: 0.93, alpha: 1.0)),
            ("💚 Heal Coherence",  "Restore coherence to safe levels",    "",    .systemGreen),
            ("🌐 Network Mesh",   "View network topology & quantum links","⌘N",  .systemTeal),
            ("📋 Help / Commands", "Show full command reference",          "",    L104Theme.goldWarm),
        ]

        let rowHeight: CGFloat = 38
        var y = panelHeight - 72
        for (i, (label, desc, shortcut, color)) in commands.enumerated() {
            let row = NSView(frame: NSRect(x: 12, y: y - rowHeight, width: panelWidth - 24, height: rowHeight))
            row.wantsLayer = true
            row.layer?.cornerRadius = 8
            row.layer?.backgroundColor = color.withAlphaComponent(0.05).cgColor

            let lbl = NSTextField(labelWithString: label)
            lbl.frame = NSRect(x: 14, y: 9, width: 190, height: 20)
            lbl.font = NSFont.systemFont(ofSize: 13, weight: .semibold)
            lbl.textColor = color
            row.addSubview(lbl)

            let descLbl = NSTextField(labelWithString: desc)
            descLbl.frame = NSRect(x: 200, y: 9, width: panelWidth - 300, height: 20)
            descLbl.font = NSFont.systemFont(ofSize: 11, weight: .regular)
            descLbl.textColor = NSColor.black.withAlphaComponent(0.40)
            row.addSubview(descLbl)

            // Keyboard shortcut badge (right-aligned)
            if !shortcut.isEmpty {
                let kbdWidth: CGFloat = 36
                let kbd = NSTextField(labelWithString: shortcut)
                kbd.frame = NSRect(x: panelWidth - 24 - kbdWidth - 10, y: 9, width: kbdWidth, height: 18)
                kbd.font = NSFont.monospacedSystemFont(ofSize: 10, weight: .bold)
                kbd.textColor = color.withAlphaComponent(0.8)
                kbd.alignment = .center
                kbd.wantsLayer = true
                kbd.layer?.backgroundColor = color.withAlphaComponent(0.1).cgColor
                kbd.layer?.cornerRadius = 4
                kbd.layer?.borderColor = color.withAlphaComponent(0.25).cgColor
                kbd.layer?.borderWidth = 1
                row.addSubview(kbd)
            }

            let btn = NSButton(frame: row.bounds)
            btn.title = ""; btn.isTransparent = true
            btn.tag = i
            btn.target = self; btn.action = #selector(commandPaletteAction(_:))
            row.addSubview(btn)

            content.addSubview(row)
            y -= rowHeight + 2
        }

        // Shortcut hint at bottom
        let hint = NSTextField(labelWithString: "⌘K toggle · ESC close · Type 'help' in chat for full reference")
        hint.frame = NSRect(x: 20, y: 10, width: panelWidth - 40, height: 16)
        hint.font = NSFont.systemFont(ofSize: 10, weight: .medium)
        hint.textColor = NSColor.black.withAlphaComponent(0.25)
        hint.alignment = .center
        content.addSubview(hint)

        panel.contentView = content
        mainWindow.addChildWindow(panel, ordered: .above)
        panel.makeKeyAndOrderFront(nil)
    }

    @objc func commandPaletteAction(_ sender: NSButton) {
        guard let mainView = wc.window?.contentView as? L104MainView else { return }
        // Close the palette
        if let panel = sender.window as? NSPanel { panel.close() }
        switch sender.tag {
        case 0: mainView.appendSystemLog(L104State.shared.synthesize()); mainView.updateMetrics()
        case 1: mainView.appendSystemLog(L104State.shared.transcend()); mainView.updateMetrics()
        case 2: mainView.navigateToTab("dash")
        case 3: mainView.appendSystemLog(L104State.shared.evolve()); mainView.updateMetrics()
        case 4: saveAll()
        case 5: mainView.appendSystemLog(L104State.shared.getStatusText())
        case 6: mainView.appendSystemLog(L104State.shared.resonate()); mainView.updateMetrics()
        case 7:  // HyperBrain status
            mainView.navigateToTab("chat")
            let status = HyperBrain.shared.getStatus()
            mainView.appendChat("L104: \(status)\n", color: L104Theme.textBot)
        case 8:  // Neural Chat tab
            mainView.navigateToTab("chat")
            mainView.window?.makeFirstResponder(mainView.inputField)
        case 9:  // Science Engine
            mainView.navigateToTab("sci")
            mainView.scienceGenerateHypothesis()
        case 10:  // Unified Field
            mainView.switchToUnifiedField()
            mainView.ufComputeEinstein()
        case 11:  // Heal Coherence
            L104State.shared.coherence = max(0.5, L104State.shared.coherence)
            L104State.shared.saveState()
            mainView.appendSystemLog("💚 COHERENCE HEALED to \(String(format: "%.3f", L104State.shared.coherence))")
            mainView.updateMetrics()
        case 12:  // Network Mesh
            mainView.navigateToTab("net")
            mainView.updateNetworkViewContent()
        case 13:  // Help
            mainView.navigateToTab("chat")
            mainView.sendHelpCommand()
        default: break
        }
    }

    @objc func switchToDashboard() {
        if let mainView = wc.window?.contentView as? L104MainView {
            mainView.navigateToTab("dash")
        }
    }

    @objc func switchToNetwork() {
        if let mainView = wc.window?.contentView as? L104MainView {
            mainView.navigateToTab("net")
            mainView.updateNetworkViewContent()
        }
    }

    @objc func switchToDebugConsole() {
        if let mainView = wc.window?.contentView as? L104MainView {
            mainView.navigateToTab("debug")
            mainView.updateDebugConsoleContent()
        }
    }

    @objc func doTranscendMenu() {
        if let mainView = wc.window?.contentView as? L104MainView {
            mainView.appendSystemLog(L104State.shared.transcend())
            mainView.updateMetrics()
        }
    }

    @objc func doResonateMenu() {
        if let mainView = wc.window?.contentView as? L104MainView {
            mainView.appendSystemLog(L104State.shared.resonate())
            mainView.updateMetrics()
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - Backend Process Management
    //
    // Launches/stops the Python server (main.py) and public node
    // as child processes when not managed by launchd. On terminate,
    // sends SIGTERM for graceful shutdown (matches main.py signal
    // handler), then SIGKILL after 25s timeout.
    // ═══════════════════════════════════════════════════════════════

    private var l104Root: String {
        // Resolve project root: two levels up from L104SwiftApp binary, or from env
        if let envRoot = ProcessInfo.processInfo.environment["L104_ROOT"] {
            return envRoot
        }
        let bundle = Bundle.main.bundlePath
        // .../L104SwiftApp/L104Native.app → .../
        return (bundle as NSString)
            .deletingLastPathComponent  // L104SwiftApp/
            .appending("/..")           // project root
    }

    private var pythonPath: String { "\(l104Root)/.venv/bin/python" }

    /// Check if the backend is already managed externally (launchd / docker / manual).
    private func isBackendRunning() -> Bool {
        let pidFile = "\(l104Root)/uvicorn.pid"
        guard FileManager.default.fileExists(atPath: pidFile),
              let pidStr = try? String(contentsOfFile: pidFile, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines),
              let pid = Int32(pidStr) else { return false }
        return kill(pid, 0) == 0  // signal 0 = existence check
    }

    /// Launch backend processes if not already running.
    func launchBackendProcesses() {
        guard !isBackendRunning() else {
            print("[L104 AppDelegate] Backend already running — skipping launch")
            return
        }
        guard FileManager.default.fileExists(atPath: pythonPath) else {
            print("[L104 AppDelegate] Python venv not found at \(pythonPath) — skipping backend launch")
            return
        }

        // Load .env into process environment
        var env = ProcessInfo.processInfo.environment
        let envFile = "\(l104Root)/.env"
        if let envContents = try? String(contentsOfFile: envFile, encoding: .utf8) {
            for line in envContents.components(separatedBy: .newlines) {
                let trimmed = line.trimmingCharacters(in: .whitespaces)
                guard !trimmed.isEmpty, !trimmed.hasPrefix("#"),
                      let eqIdx = trimmed.firstIndex(of: "=") else { continue }
                let key = String(trimmed[trimmed.startIndex..<eqIdx])
                let val = String(trimmed[trimmed.index(after: eqIdx)...])
                env[key] = val
            }
        }
        env["L104_ROOT"] = l104Root
        env["PYTHONPATH"] = l104Root

        // Start main.py (FastAPI server)
        let server = Process()
        server.executableURL = URL(fileURLWithPath: pythonPath)
        server.arguments = ["\(l104Root)/main.py"]
        server.currentDirectoryURL = URL(fileURLWithPath: l104Root)
        server.environment = env
        let serverLog = FileHandle(forWritingAtPath: "\(l104Root)/server.log")
            ?? FileHandle.nullDevice
        server.standardOutput = serverLog
        server.standardError = serverLog
        do {
            try server.run()
            serverProcess = server
            print("[L104 AppDelegate] Server launched: PID \(server.processIdentifier)")
        } catch {
            print("[L104 AppDelegate] Failed to launch server: \(error)")
        }

        // Start public node if it exists
        let nodeScript = "\(l104Root)/L104_public_node.py"
        if FileManager.default.fileExists(atPath: nodeScript) {
            let node = Process()
            node.executableURL = URL(fileURLWithPath: pythonPath)
            node.arguments = [nodeScript]
            node.currentDirectoryURL = URL(fileURLWithPath: l104Root)
            node.environment = env
            let nodeLog = FileHandle(forWritingAtPath: "\(l104Root)/node.log")
                ?? FileHandle.nullDevice
            node.standardOutput = nodeLog
            node.standardError = nodeLog
            do {
                try node.run()
                nodeProcess = node
                print("[L104 AppDelegate] Node launched: PID \(node.processIdentifier)")
            } catch {
                print("[L104 AppDelegate] Failed to launch node: \(error)")
            }
        }
    }

    /// Gracefully stop backend processes (SIGTERM → wait → SIGKILL).
    func stopBackendProcesses() {
        let gracePeriod: TimeInterval = 25.0  // match launchd ExitTimeOut

        for (label, proc) in [("Server", serverProcess), ("Node", nodeProcess)] {
            guard let p = proc, p.isRunning else { continue }
            print("[L104 AppDelegate] Stopping \(label): PID \(p.processIdentifier)")
            p.terminate()  // sends SIGTERM

            // Wait for graceful exit on a background thread (don't block main)
            let pid = p.processIdentifier
            DispatchQueue.global(qos: .utility).async {
                let deadline = Date().addingTimeInterval(gracePeriod)
                while p.isRunning && Date() < deadline {
                    Thread.sleep(forTimeInterval: 0.5)
                }
                if p.isRunning {
                    print("[L104 AppDelegate] \(label) PID \(pid) did not exit in \(Int(gracePeriod))s — SIGKILL")
                    kill(pid, SIGKILL)
                } else {
                    print("[L104 AppDelegate] \(label) PID \(pid) exited cleanly")
                }
            }
        }
        serverProcess = nil
        nodeProcess = nil
    }

    /// Trigger a rolling upgrade: stop → pull → rebuild → start.
    @objc func upgradeAllProcesses() {
        DispatchQueue.global(qos: .userInitiated).async { [self] in
            let script = "\(l104Root)/scripts/upgrade_all.sh"
            guard FileManager.default.fileExists(atPath: script) else {
                print("[L104 AppDelegate] upgrade_all.sh not found")
                return
            }
            let upgrade = Process()
            upgrade.executableURL = URL(fileURLWithPath: "/bin/bash")
            upgrade.arguments = [script]
            upgrade.currentDirectoryURL = URL(fileURLWithPath: l104Root)
            upgrade.environment = ProcessInfo.processInfo.environment
            do {
                try upgrade.run()
                upgrade.waitUntilExit()
                let status = upgrade.terminationStatus
                DispatchQueue.main.async {
                    if let mainView = self.wc.window?.contentView as? L104MainView {
                        let msg = status == 0
                            ? "Process upgrade completed successfully"
                            : "Process upgrade finished with status \(status)"
                        mainView.appendSystemLog(msg)
                    }
                }
                print("[L104 AppDelegate] upgrade_all.sh exited with status \(status)")
            } catch {
                print("[L104 AppDelegate] Failed to run upgrade: \(error)")
            }
        }
    }
}
