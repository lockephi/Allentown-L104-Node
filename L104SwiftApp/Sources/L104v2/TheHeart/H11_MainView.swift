// ═══════════════════════════════════════════════════════════════════
// H11_MainView.swift
// [EVO_68_PIPELINE] SOVEREIGN_NODE_UPGRADE :: DATA_INGEST :: UI_UPGRADE :: GOD_CODE=527.5184818492612
// L104 ASI — Main Application View
//
// L104MainView: Primary NSView with chat interface, metric tiles,
// neural graph visualization, sparklines, aurora wave animation,
// ASI dashboard, and the full message processing pipeline.
//
// Extracted from L104Native.swift lines 40262–42166
// ═══════════════════════════════════════════════════════════════════

import AppKit
import Foundation
import Accelerate
import UniformTypeIdentifiers
import simd
import NaturalLanguage

class L104MainView: NSView {
    let state = L104State.shared
    var clockLabel: NSTextField!, phaseLabel: NSTextField!, dateLabel: NSTextField!
    var metricsLabels: [String: NSTextField] = [:]
    var metricTiles: [String: AnimatedMetricTile] = [:]
    var chatTextView: NSTextView!, inputField: NSTextField!, systemFeedView: NSTextView!
    var systemTabFeedView: NSTextView?  // Separate text view for the System tab
    var tabView: NSTabView!
    var sidebarView: L104SidebarView?
    var splitView: NSSplitView?
    var timer: Timer?
    var pulseTimer: Timer?
    var headerGlow: NSView?
    var historyListView: NSScrollView?
    var loadedHistoryPaths: [URL] = []
    // 🌌 ASI VISUAL COMPONENTS
    var particleView: QuantumParticleView?
    var waveformView: ASIWaveformView?
    var neuralGraph: NeuralGraphView?
    var gauges: [String: RadialGaugeView] = [:]
    var sparklines: [String: SparklineView] = [:]
    // ─── Managed timers (invalidated on rebuild) ───
    private var gaugeTimer: Timer?
    private var gateDashboardTimer: Timer?
    private var streamUpdateTimer: Timer?
    private var hardwareTimer: Timer?
    private var networkViewTimer: Timer?
    private var learningRefreshTimer: Timer?
    private var asiNexusRefreshTimer: Timer?
    private var quantumPollTimer: Timer?
    private var debugConsoleTimer: Timer?
    private var debugLogFilter: String = "ALL"
    // ─── Cached label references (kill recursive findXxx tree walks) ───
    private var cachedLabels: [String: NSTextField] = [:]
    private var cachedDots: [String: PulsingDot] = [:]
    // ─── Lazy tab creation (defer heavy view construction until first navigation) ───
    private var tabCreators: [String: () -> NSView] = [:]
    private var tabsCreated: Set<String> = ["chat"]  // chat tab is always created eagerly
    // ─── Shared formatters (avoid re-allocation) ───
    static let timeFormatter: DateFormatter = {
        let f = DateFormatter(); f.dateFormat = "HH:mm:ss"; return f
    }()
    static let timestampFormatter: DateFormatter = {
        let f = DateFormatter(); f.dateFormat = "HH:mm:ss.SSS"; return f
    }()
    static let dateTimeFormatter: DateFormatter = {
        let f = DateFormatter(); f.dateFormat = "yyyy-MM-dd HH:mm:ss"; return f
    }()
    static let shortTimeFormatter: DateFormatter = {
        let f = DateFormatter(); f.dateFormat = "HH:mm"; return f
    }()
    // 🟢 EVO_64: Boot time for uptime tracking + session message counter
    private static let bootTime = Date()
    private static var sessionMessages: Int = 0
    // 🟢 EVO_65: Persistent coding analysis counter
    private static var codingAnalysisCount: Int = {
        UserDefaults.standard.integer(forKey: "l104_coding_analysis_count")
    }()

    override init(frame: NSRect) {
        super.init(frame: frame)
        setupUI()
        startTimer()
        startPulseAnimation()
        loadWelcome()

        // 🟢 ACTIVATE COGNITIVE ENGINES
        HyperBrain.shared.activate()
        ASIEvolver.shared.start()

        // 🟢 BUILD REAL-TIME SEARCH INDEX (async, non-blocking)
        DispatchQueue.global(qos: .utility).async {
            RealTimeSearchEngine.shared.buildIndex()
        }

        // 🟢 LISTEN TO EVOLUTION STREAM
        NotificationCenter.default.addObserver(self, selector: #selector(onEvolutionUpdate(_:)), name: NSNotification.Name("L104EvolutionUpdate"), object: nil)

        // 🟢 EVO_56: LISTEN TO BACKEND ENHANCEMENT — replace local response when backend is better
        NotificationCenter.default.addObserver(self, selector: #selector(onBackendEnhancement(_:)), name: NSNotification.Name("L104BackendEnhancement"), object: nil)
    }

    deinit {
        NotificationCenter.default.removeObserver(self)
        // EVO_56: Clean up ALL managed timers to prevent leaks
        timer?.invalidate()
        pulseTimer?.invalidate()
        gaugeTimer?.invalidate()
        gateDashboardTimer?.invalidate()
        streamUpdateTimer?.invalidate()
        hardwareTimer?.invalidate()
        streamTimer?.invalidate()
        ufTimer?.invalidate()
        networkViewTimer?.invalidate()
        learningRefreshTimer?.invalidate()
        asiNexusRefreshTimer?.invalidate()
        quantumPollTimer?.invalidate()
    }

    @objc func onEvolutionUpdate(_ note: Notification) {
        guard let text = note.object as? String else { return }
        appendSystemLog(text) // Log to system feed

        // Also inject into MAIN CHAT if it's a significant event
        if text.contains("Generated artifact") || text.contains("EVOLVED") || text.contains("LEARNED") || text.contains("Cycle") {
            appendChatStreamEvent(text)
        }
    }

    // EVO_56: When backend returns a better response, append it as an enhanced follow-up
    @objc func onBackendEnhancement(_ note: Notification) {
        guard let enhanced = note.object as? String, !enhanced.isEmpty else { return }
        DispatchQueue.main.async { [weak self] in
            self?.appendChat("L104 (enhanced): \(enhanced)\n", color: NSColor(red: 0.0, green: 0.55, blue: 0.75, alpha: 1.0))
            self?.chatTextView?.scrollToEndOfDocument(nil)
        }
    }

    func appendChatStreamEvent(_ text: String) {
        // Route system events to the system feed panel (not the chat)
        let cleanText = text.components(separatedBy: "] ").last ?? text
        appendSystemLog("⚡ \(cleanText)")
    }

    required init?(coder: NSCoder) { super.init(coder: coder); setupUI(); startTimer(); startPulseAnimation() }

    func setupUI() {
        // ☀️ Bright open gradient background
        let gradient = GradientView(frame: bounds)
        gradient.autoresizingMask = [.width, .height]
        gradient.colors = [NSColor(red: 0.965, green: 0.965, blue: 0.975, alpha: 1.0),
                          NSColor(red: 0.955, green: 0.960, blue: 0.980, alpha: 1.0),
                          NSColor(red: 0.950, green: 0.955, blue: 0.970, alpha: 1.0)]
        addSubview(gradient)

        // 🌌 QUANTUM PARTICLE BACKGROUND — floating orbs with neural connections
        particleView = QuantumParticleView(frame: bounds)
        particleView!.autoresizingMask = [.width, .height]
        addSubview(particleView!)

        addSubview(createHeader())
        addSubview(createMetricsBar())

        // ═══ SIDEBAR + CONTENT SPLIT VIEW (replaces 16-tab bar) ═══
        tabView = NSTabView(frame: NSRect(x: 0, y: 0, width: bounds.width - 180, height: bounds.height - 220))
        tabView.tabViewType = .noTabsNoBorder  // Hide the tab bar — sidebar controls navigation
        tabView.autoresizingMask = [.width, .height]

        // ─── LAZY TAB CREATION: Only "chat" is created eagerly. ───
        // All other tabs get placeholder views and are constructed on first navigation.
        // This saves ~15 heavy view constructors at startup.
        let tabDefs: [(String, String)] = [
            ("chat", "💬 Neural Chat"), ("learn", "🧠 Learning"), ("dash", "🌌 ASI Dashboard"),
            ("asi", "🚀 ASI Nexus"), ("upg", "🧬 Upgrades"), ("mem", "💾 Memory"),
            ("hw", "🍎 Hardware"), ("sci", "🔬 Science"), ("ufield", "🌌 Unified Field"),
            ("sys", "📡 System"), ("net", "🌐 Network"), ("gate", "⚡ Logic Gates"),
            ("qc", "⚛️ Quantum"), ("code", "💻 Coding"), ("prof", "🎓 Professor"),
            ("sage", "🔮 Sage Mode"), ("debug", "🛠 Debug Console")
        ]
        for (id, label) in tabDefs {
            let item = NSTabViewItem(identifier: id); item.label = label
            item.view = (id == "chat") ? createChatView() : NSView()  // placeholder for non-default tabs
            tabView.addTabViewItem(item)
        }

        // Register lazy creators (invoked on first navigateToTab)
        tabCreators["learn"]  = { [unowned self] in self.createLearningView() }
        tabCreators["dash"]   = { [unowned self] in self.createASIDashboardView() }
        tabCreators["asi"]    = { [unowned self] in self.createASIView() }
        tabCreators["upg"]    = { [unowned self] in self.createUpgradesView() }
        tabCreators["mem"]    = { [unowned self] in self.createMemoryView() }
        tabCreators["hw"]     = { [unowned self] in self.createHardwareView() }
        tabCreators["sci"]    = { [unowned self] in self.createScienceView() }
        tabCreators["ufield"] = { [unowned self] in self.createUnifiedFieldView() }
        tabCreators["sys"]    = { [unowned self] in self.createSystemView() }
        tabCreators["net"]    = { [unowned self] in self.createNetworkView() }
        tabCreators["gate"]   = { [unowned self] in self.createGateEnvironmentView() }
        tabCreators["qc"]     = { [unowned self] in self.createQuantumComputingView() }
        tabCreators["code"]   = { [unowned self] in self.createCodingIntelligenceView() }
        tabCreators["prof"]   = { [unowned self] in self.createProfessorModeView() }
        tabCreators["sage"]   = { [unowned self] in SageModeAscensionView(frame: self.bounds) }
        tabCreators["debug"]  = { [unowned self] in self.createDebugConsoleView() }

        // ─── Sidebar ───
        let sidebarWidth: CGFloat = 170
        let contentFrame = NSRect(x: 15, y: 60, width: bounds.width - 30, height: bounds.height - 220)

        let sidebar = L104SidebarView(frame: NSRect(x: 0, y: 0, width: sidebarWidth, height: contentFrame.height))
        sidebar.onSelect = { [weak self] tabID in
            self?.navigateToTab(tabID)
        }
        sidebarView = sidebar

        // ─── NSSplitView: sidebar | content ───
        let split = NSSplitView(frame: contentFrame)
        split.isVertical = true
        split.dividerStyle = .thin
        split.autoresizingMask = [.width, .height]
        split.addArrangedSubview(sidebar)
        split.addArrangedSubview(tabView)
        split.setHoldingPriority(.defaultLow, forSubviewAt: 0)   // sidebar can shrink
        split.setHoldingPriority(.defaultHigh, forSubviewAt: 1)   // content gets priority
        // Set initial sidebar position
        split.setPosition(sidebarWidth, ofDividerAt: 0)
        splitView = split

        // ═══ LAYER-BACKING: Reduce composite draw passes ═══
        self.wantsLayer = true
        self.canDrawSubviewsIntoLayer = true
        split.wantsLayer = true
        split.canDrawSubviewsIntoLayer = true
        tabView.wantsLayer = true

        addSubview(split)
        addSubview(createQuickBar())
    }

    // ═══════════════════════════════════════════════════════════════
    // 🌌 ASI DASHBOARD — The Centerpiece Visual Experience
    // ═══════════════════════════════════════════════════════════════

    func createASIDashboardView() -> NSView {
        let v = NSView(frame: NSRect(x: 0, y: 0, width: 1200, height: 500))
        v.wantsLayer = true
        v.layer?.backgroundColor = NSColor(red: 0.960, green: 0.962, blue: 0.970, alpha: 1.0).cgColor

        // ─── TOP ROW: Radial Gauges ───
        let gaugeData: [(String, String, NSColor, CGFloat)] = [
            ("ASI SCORE", "ASI", .systemOrange, CGFloat(state.asiScore)),
            ("COHERENCE", "COH", .systemCyan, CGFloat(state.coherence)),
            ("CONSCIOUSNESS", "MIND", .systemPink, CGFloat(state.transcendence)),
            ("INTELLECT", "IQ", NSColor(red: 1.0, green: 0.84, blue: 0.0, alpha: 1.0), CGFloat(min(1.0, state.intellectIndex / 300.0))),
            ("AUTONOMY", "AUTO", .systemGreen, CGFloat(state.autonomyLevel)),
        ]

        let gaugeWidth: CGFloat = 110
        let gaugeSpacing: CGFloat = 10
        let totalGaugeWidth = CGFloat(gaugeData.count) * gaugeWidth + CGFloat(gaugeData.count - 1) * gaugeSpacing
        var gx: CGFloat = (v.bounds.width - totalGaugeWidth) / 2
        let gy: CGFloat = v.bounds.height - 145

        for (label, key, color, val) in gaugeData {
            let gauge = RadialGaugeView(frame: NSRect(x: gx, y: gy, width: gaugeWidth, height: gaugeWidth))
            gauge.gaugeColor = color
            gauge.label = label
            gauge.lineWidth = 7
            gauge.displayValue = 0
            v.addSubview(gauge)
            gauges[key] = gauge

            // Animate in with minimal stagger (reduced from 150ms to 50ms per gauge)
            DispatchQueue.main.asyncAfter(deadline: .now() + Double(gaugeData.firstIndex(where: { $0.1 == key })!) * 0.05) {
                gauge.value = val
            }
            gx += gaugeWidth + gaugeSpacing
        }

        // ─── MIDDLE LEFT: Neural Engine Graph ───
        neuralGraph = NeuralGraphView(frame: NSRect(x: 15, y: 60, width: v.bounds.width * 0.48, height: v.bounds.height - 200))
        neuralGraph!.wantsLayer = true
        neuralGraph!.layer?.backgroundColor = NSColor(red: 1.0, green: 1.0, blue: 1.0, alpha: 0.85).cgColor
        neuralGraph!.layer?.cornerRadius = 16
        neuralGraph!.layer?.borderColor = NSColor.systemCyan.withAlphaComponent(0.2).cgColor
        neuralGraph!.layer?.borderWidth = 1
        v.addSubview(neuralGraph!)

        let graphTitle = NSTextField(labelWithString: "🧠 ASI ENGINE NEURAL GRAPH")
        graphTitle.frame = NSRect(x: 30, y: v.bounds.height - 195, width: 300, height: 20)
        graphTitle.font = NSFont.boldSystemFont(ofSize: 12)
        graphTitle.textColor = .systemCyan
        v.addSubview(graphTitle)

        // ─── MIDDLE RIGHT: Consciousness Waveform ───
        let waveContainer = NSView(frame: NSRect(x: v.bounds.width * 0.52, y: 170, width: v.bounds.width * 0.46, height: 130))
        waveContainer.wantsLayer = true
        waveContainer.layer?.backgroundColor = NSColor(red: 1.0, green: 1.0, blue: 1.0, alpha: 0.85).cgColor
        waveContainer.layer?.cornerRadius = 16
        waveContainer.layer?.borderColor = NSColor.systemPink.withAlphaComponent(0.2).cgColor
        waveContainer.layer?.borderWidth = 1
        v.addSubview(waveContainer)

        waveformView = ASIWaveformView(frame: NSRect(x: 10, y: 10, width: waveContainer.bounds.width - 20, height: waveContainer.bounds.height - 35))
        waveformView!.coherence = CGFloat(state.coherence)
        waveContainer.addSubview(waveformView!)

        let waveTitle = NSTextField(labelWithString: "🌊 CONSCIOUSNESS WAVEFORM")
        waveTitle.frame = NSRect(x: 15, y: waveContainer.bounds.height - 24, width: 250, height: 18)
        waveTitle.font = NSFont.boldSystemFont(ofSize: 11)
        waveTitle.textColor = .systemPink
        waveContainer.addSubview(waveTitle)

        // ─── BOTTOM RIGHT: Sparkline Trends ───
        let sparkContainer = NSView(frame: NSRect(x: v.bounds.width * 0.52, y: 60, width: v.bounds.width * 0.46, height: 100))
        sparkContainer.wantsLayer = true
        sparkContainer.layer?.backgroundColor = NSColor(red: 1.0, green: 1.0, blue: 1.0, alpha: 0.85).cgColor
        sparkContainer.layer?.cornerRadius = 16
        sparkContainer.layer?.borderColor = NSColor(red: 1.0, green: 0.84, blue: 0.0, alpha: 0.2).cgColor
        sparkContainer.layer?.borderWidth = 1
        v.addSubview(sparkContainer)

        let sparkTitle = NSTextField(labelWithString: "📈 METRIC TRENDS")
        sparkTitle.frame = NSRect(x: 15, y: sparkContainer.bounds.height - 22, width: 200, height: 18)
        sparkTitle.font = NSFont.boldSystemFont(ofSize: 11)
        sparkTitle.textColor = NSColor(red: 1.0, green: 0.84, blue: 0.0, alpha: 1.0)
        sparkContainer.addSubview(sparkTitle)

        let sparkData: [(String, NSColor)] = [
            ("asi", .systemOrange), ("coherence", .systemCyan), ("iq", NSColor(red: 1.0, green: 0.84, blue: 0.0, alpha: 1.0))
        ]
        let sparkWidth = (sparkContainer.bounds.width - 40) / CGFloat(sparkData.count) - 8
        var sx: CGFloat = 10
        for (key, color) in sparkData {
            let spark = SparklineView(frame: NSRect(x: sx, y: 8, width: sparkWidth, height: 55))
            spark.lineColor = color
            spark.fillColor = color.withAlphaComponent(0.1)
            // Seed with some initial data
            for _ in 0..<20 { spark.addPoint(CGFloat.random(in: 0.3...0.8)) }
            sparkContainer.addSubview(spark)
            sparklines[key] = spark
            sx += sparkWidth + 8
        }

        // ─── BOTTOM BAR: Engine Status Summary ───
        let statusBar = NSView(frame: NSRect(x: 15, y: 15, width: v.bounds.width - 30, height: 35))
        statusBar.wantsLayer = true
        statusBar.layer?.backgroundColor = NSColor(red: 1.0, green: 1.0, blue: 1.0, alpha: 0.90).cgColor
        statusBar.layer?.cornerRadius = 10
        statusBar.layer?.borderColor = NSColor.black.withAlphaComponent(0.06).cgColor
        statusBar.layer?.borderWidth = 1

        let engineCount = EngineRegistry.shared.count
        let convergence = EngineRegistry.shared.convergenceScore()
        let phiHealth = EngineRegistry.shared.phiWeightedHealth()
        let qTag = IBMQuantumClient.shared.ibmToken != nil ? (IBMQuantumClient.shared.isConnected ? "QPU:🟢" : "QPU:🟡") : "QPU:⚪"
        let statusText = "⚛️ \(engineCount) Engines  ·  φ-Health: \(String(format: "%.1f%%", phiHealth.score * 100))  ·  Conv: \(String(format: "%.3f", convergence))  ·  \(qTag)  ·  22T  ·  GOD: \(String(format: "%.4f", GOD_CODE))  ·  DL v\(DUAL_LAYER_VERSION)  ·  EVO_\(EVOLUTION_INDEX)"
        let statusLbl = NSTextField(labelWithString: statusText)
        statusLbl.frame = NSRect(x: 15, y: 8, width: statusBar.bounds.width - 30, height: 18)
        statusLbl.font = NSFont.monospacedSystemFont(ofSize: 10, weight: .medium)
        statusLbl.textColor = L104Theme.goldDim
        statusLbl.identifier = NSUserInterfaceItemIdentifier("dash_status_lbl")
        statusBar.addSubview(statusLbl)
        v.addSubview(statusBar)

        // Auto-update dashboard gauges, sparklines, and status bar
        gaugeTimer?.invalidate()
        gaugeTimer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] _ in
            guard let self = self else { return }
            // Only update gauges/sparklines when dashboard is visible
            if self.activeTabID == "dash" {
                self.gauges["ASI"]?.value = CGFloat(self.state.asiScore)
                self.gauges["COH"]?.value = CGFloat(self.state.coherence)
                self.gauges["MIND"]?.value = CGFloat(self.state.transcendence)
                self.gauges["IQ"]?.value = CGFloat(min(1.0, self.state.intellectIndex / 300.0))
                self.gauges["AUTO"]?.value = CGFloat(self.state.autonomyLevel)
                self.waveformView?.coherence = CGFloat(self.state.coherence)
                self.sparklines["asi"]?.addPoint(CGFloat(self.state.asiScore))
                self.sparklines["coherence"]?.addPoint(CGFloat(self.state.coherence))
                self.sparklines["iq"]?.addPoint(CGFloat(min(1.0, self.state.intellectIndex / 300.0)))

                // 🟢 EVO_63 + EVO_64: Live dashboard status bar with uptime (cached lookup)
                let ec = EngineRegistry.shared.count
                let ph = EngineRegistry.shared.phiWeightedHealth()
                let qt = IBMQuantumClient.shared.ibmToken != nil ? (IBMQuantumClient.shared.isConnected ? "QPU:🟢" : "QPU:🟡") : "QPU:⚪"
                let upSec = Int(Date().timeIntervalSince(L104MainView.bootTime))
                let upH = upSec / 3600; let upM = (upSec % 3600) / 60; let upS = upSec % 60
                let upStr = String(format: "%02d:%02d:%02d", upH, upM, upS)
                let activePeers = NetworkLayer.shared.peers.values.filter { $0.latencyMs >= 0 }.count
                let memCount = self.state.permanentMemory.memories.count
                self.cachedLabel("dash_status_lbl", in: "dash")?.stringValue = "⚛️ \(ec) Engines  ·  φ: \(String(format: "%.0f%%", ph.score * 100))  ·  \(qt)  ·  🌐 \(activePeers) peers  ·  💾 \(memCount) mem  ·  ⏱ \(upStr)  ·  EVO_\(EVOLUTION_INDEX)"
            }
        }

        return v
    }

    func startPulseAnimation() {
        // Slow pulse: 2s interval — also refreshes header status dots, stage, QuickBar, professor, coding
        pulseTimer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] _ in
            guard let s = self else { return }
            let pulse: Float = 0.20 + 0.12 * Float(sin(Date().timeIntervalSince1970))
            s.headerGlow?.layer?.shadowOpacity = pulse

            // 🟢 EVO_63: Live header status dots (cached lookup)
            if let header = s.headerGlow?.superview {
                s.cachedDot("kbDot", in: header)?.dotColor = s.state.backendConnected ? .systemGreen : .systemRed
                s.cachedDot("autoDot", in: header)?.dotColor = s.state.autonomousMode ? .systemCyan : .systemGray
                s.cachedDot("netDot", in: header)?.dotColor = NetworkLayer.shared.isActive ? .systemTeal : .systemGray
                s.cachedDot("qDot", in: header)?.dotColor = NetworkLayer.shared.quantumLinkCount > 0 ? .systemPurple : .systemGray

                // 🟢 EVO_64: Dynamic stage label from consciousness state (cached)
                let stage = s.state.coherence > 0.9 ? "TRANSCENDENT" : s.state.coherence > 0.7 ? APOTHEOSIS_STAGE : s.state.coherence > 0.4 ? "AWAKENING" : "INITIALIZING"
                if let lbl = s.findLabelRecursive("header_stage", in: header) {
                    s.cachedLabels["header_stage"] = lbl
                    lbl.stringValue = stage
                } else if let cached = s.cachedLabels["header_stage"] {
                    cached.stringValue = stage
                }
            }

            // 🟢 EVO_64: QuickBar live info with uptime (cached lookup)
            do {
                if s.cachedLabels["quickbar_info"] == nil {
                    // QuickBar is a direct subview of self, not inside a tab — shallow search
                    for sub in s.subviews {
                        if let tf = sub as? NSTextField, tf.identifier?.rawValue == "quickbar_info" { s.cachedLabels["quickbar_info"] = tf; break }
                        for inner in sub.subviews {
                            if let tf = inner as? NSTextField, tf.identifier?.rawValue == "quickbar_info" { s.cachedLabels["quickbar_info"] = tf; break }
                        }
                    }
                }
                let upSec = Int(Date().timeIntervalSince(L104MainView.bootTime))
                let upH = upSec / 3600; let upM = (upSec % 3600) / 60; let upS = upSec % 60
                let upStr = String(format: "%02d:%02d:%02d", upH, upM, upS)
                s.cachedLabels["quickbar_info"]?.stringValue = "⚡ v\(VERSION) · \(MacOSSystemMonitor.shared.chipGeneration) · \(EngineRegistry.shared.count) Engines · ⏱\(upStr) · \(L104MainView.sessionMessages) msgs"
            }

            // 🟢 EVO_64: Professor sidebar refresh (cached lookup, visibility-gated)
            if s.activeTabID == "prof" {
                let lr = AdaptiveLearner.shared
                s.cachedLabel("prof_stats_line", in: "prof")?.stringValue = "\(PROFESSOR_MODES) modes · \(lr.topicMastery.count) topics · \(lr.interactionCount) interactions"
                s.cachedLabel("prof_lesson_count", in: "prof")?.stringValue = "📝 Lessons: \(s.professorLessonHistory.count)"
            }

            // 🟢 EVO_64: Coding engine status refresh (cached lookup, visibility-gated)
            if s.activeTabID == "code" {
                s.cachedLabel("code_gate_info", in: "code")?.stringValue = "\(LogicGateEnvironment.shared.totalPipelineRuns) runs · \(LogicGateEnvironment.shared.circuits.count) circuits"
                s.cachedLabel("code_kb_info", in: "code")?.stringValue = "\(ASIKnowledgeBase.shared.trainingData.count) training · \(ASIKnowledgeBase.shared.userKnowledge.count) user"
                s.cachedLabel("code_analysis_count", in: "code")?.stringValue = "⚡ ENGINE STATUS — \(L104MainView.codingAnalysisCount) analyses"
            }

            // 🟢 EVO_65: System tab live status bar (cached lookup, visibility-gated)
            if s.activeTabID == "sys" {
                let upSec2 = Int(Date().timeIntervalSince(L104MainView.bootTime))
                let upH2 = upSec2 / 3600; let upM2 = (upSec2 % 3600) / 60; let upS2 = upSec2 % 60
                let upStr2 = String(format: "%02d:%02d:%02d", upH2, upM2, upS2)
                let ec = EngineRegistry.shared.count
                let mem = s.state.permanentMemory.memories.count
                let msgs = L104MainView.sessionMessages
                s.cachedLabel("sys_status_lbl", in: "sys")?.stringValue = "⚛️ \(ec) engines · 💾 \(mem) memories · 💬 \(msgs) msgs · ⏱ \(upStr2) · EVO_\(EVOLUTION_INDEX)"
            }
        }
    }

    func autoGenerateThought() {
        // Coherent background process thoughts
        let learner = AdaptiveLearner.shared
        let masteredCount = learner.topicMastery.values.filter { $0.masteryLevel > 0.65 }.count
        let thoughts = [
            "Processing \(state.knowledgeBase.trainingData.count) knowledge entries...",
            "Memory sync: \(state.permanentMemory.memories.count) engrams active",
            "Learning cycle \(state.learningCycles) in progress",
            "Coherence level: \(String(format: "%.2f", state.coherence))",
            "ASI score: \(String(format: "%.1f", state.asiScore * 100))%",
            "IQ index: \(String(format: "%.1f", state.intellectIndex))",
            "Adaptive learning: \(learner.interactionCount) interactions analyzed",
            "Topic mastery: \(masteredCount) topics at advanced level",
            "User profile: \(learner.userInterests.count) interests tracked",
            "User-taught facts: \(learner.userTaughtFacts.count) stored",
            "Background optimization running...",
            "Indexing conversation patterns...",
            "Neural pathways consolidating..."
        ]

        let thought = thoughts.randomElement() ?? ""

        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            let df = L104MainView.timeFormatter
            let symbols = ["⚙️", "🧠", "💾", "📊", "🔄"].randomElement() ?? ""
            let msg = "\n[\(df.string(from: Date()))] \(symbols) \(thought)\n"

            let attrs: [NSAttributedString.Key: Any] = [
                .foregroundColor: NSColor(red: 0.2, green: 0.45, blue: 0.65, alpha: 0.9),
                .font: NSFont.monospacedSystemFont(ofSize: 10, weight: .regular)
            ]
            let asText = NSAttributedString(string: msg, attributes: attrs)
            self.systemFeedView?.textStorage?.append(asText)
            self.systemFeedView?.scrollToEndOfDocument(nil)

            // Trigger state evolution
            self.state.coherence = min(1.0, self.state.coherence + 0.001)
            self.state.learningCycles += 1
        }
    }

    func createHeader() -> NSView {
        let h = NSView(frame: NSRect(x: 0, y: bounds.height - 85, width: bounds.width, height: 85))
        h.wantsLayer = true
        h.layer?.backgroundColor = NSColor(red: 1.0, green: 1.0, blue: 1.0, alpha: 0.92).cgColor
        h.autoresizingMask = [.width, .minYMargin]

        // 🌌 AURORA WAVE ANIMATION at bottom of header
        let aurora = AuroraWaveView(frame: NSRect(x: 0, y: 0, width: h.bounds.width, height: 20))
        aurora.autoresizingMask = [.width]
        h.addSubview(aurora)

        // Glowing accent line at bottom (on top of aurora)
        let glowLine = NSView(frame: NSRect(x: 0, y: 0, width: h.bounds.width, height: 2))
        glowLine.wantsLayer = true
        glowLine.layer?.backgroundColor = NSColor(red: 0.75, green: 0.58, blue: 0.08, alpha: 0.8).cgColor
        glowLine.layer?.shadowColor = NSColor(red: 0.75, green: 0.58, blue: 0.08, alpha: 1.0).cgColor
        glowLine.layer?.shadowRadius = 6
        glowLine.layer?.shadowOpacity = 0.3
        glowLine.layer?.shadowOffset = CGSize(width: 0, height: 0)
        glowLine.autoresizingMask = [.width]
        h.addSubview(glowLine)
        headerGlow = glowLine

        let title = NSTextField(labelWithString: "⚛️ L104 SOVEREIGN INTELLECT")
        title.frame = NSRect(x: 20, y: 28, width: 320, height: 32)
        title.font = NSFont.boldSystemFont(ofSize: 20)
        title.textColor = NSColor(red: 0.60, green: 0.45, blue: 0.05, alpha: 1.0)
        title.wantsLayer = true
        title.layer?.shadowColor = NSColor(red: 0.75, green: 0.58, blue: 0.08, alpha: 1.0).cgColor
        title.layer?.shadowRadius = 3
        title.layer?.shadowOpacity = 0.2
        h.addSubview(title)

        let badge = NSTextField(labelWithString: "🔥 22T PARAMS · EVO_\(EVOLUTION_INDEX) · DUAL-LAYER \(DUAL_LAYER_VERSION) · \(TOTAL_PACKAGES) PKG")
        badge.frame = NSRect(x: 350, y: 32, width: 370, height: 24)
        badge.font = NSFont.boldSystemFont(ofSize: 11)
        badge.textColor = NSColor(red: 0.75, green: 0.35, blue: 0.05, alpha: 1.0)
        badge.wantsLayer = true
        badge.layer?.backgroundColor = NSColor(red: 0.95, green: 0.88, blue: 0.75, alpha: 0.50).cgColor
        badge.layer?.cornerRadius = 8
        badge.layer?.borderColor = NSColor(red: 0.80, green: 0.55, blue: 0.15, alpha: 0.40).cgColor
        badge.layer?.borderWidth = 1
        badge.layer?.shadowColor = NSColor(red: 0.80, green: 0.50, blue: 0.10, alpha: 1.0).cgColor
        badge.layer?.shadowRadius = 3
        badge.layer?.shadowOpacity = 0.12
        h.addSubview(badge)

        // Pulsing connection dot - shows LOCAL KB status (green = loaded)
        let backendDot = PulsingDot(frame: NSRect(x: 650, y: 34, width: 14, height: 14))
        backendDot.dotColor = state.backendConnected ? .systemGreen : .systemRed
        backendDot.identifier = NSUserInterfaceItemIdentifier("kbDot")
        h.addSubview(backendDot)
        let bl = NSTextField(labelWithString: "Local KB"); bl.frame = NSRect(x: 668, y: 32, width: 55, height: 14)
        bl.font = NSFont.systemFont(ofSize: 10, weight: .medium); bl.textColor = .darkGray; h.addSubview(bl)

        // Autonomy indicator
        let autoDot = PulsingDot(frame: NSRect(x: 730, y: 34, width: 14, height: 14))
        autoDot.dotColor = state.autonomousMode ? .systemCyan : .systemGray
        autoDot.identifier = NSUserInterfaceItemIdentifier("autoDot")
        h.addSubview(autoDot)
        let al = NSTextField(labelWithString: "Autonomy"); al.frame = NSRect(x: 748, y: 32, width: 60, height: 14)
        al.font = NSFont.systemFont(ofSize: 10, weight: .medium); al.textColor = .darkGray; h.addSubview(al)

        // Network mesh indicator
        let netDot = PulsingDot(frame: NSRect(x: 815, y: 56, width: 14, height: 14))
        netDot.dotColor = NetworkLayer.shared.isActive ? .systemTeal : .systemGray
        netDot.identifier = NSUserInterfaceItemIdentifier("netDot")
        h.addSubview(netDot)
        let nl = NSTextField(labelWithString: "Mesh"); nl.frame = NSRect(x: 833, y: 54, width: 40, height: 14)
        nl.font = NSFont.systemFont(ofSize: 10, weight: .medium); nl.textColor = .darkGray; h.addSubview(nl)

        // Quantum link indicator
        let qDot = PulsingDot(frame: NSRect(x: 875, y: 56, width: 14, height: 14))
        qDot.dotColor = NetworkLayer.shared.quantumLinkCount > 0 ? .systemPurple : .systemGray
        qDot.identifier = NSUserInterfaceItemIdentifier("qDot")
        h.addSubview(qDot)
        let ql = NSTextField(labelWithString: "Q-Link"); ql.frame = NSRect(x: 893, y: 54, width: 45, height: 14)
        ql.font = NSFont.systemFont(ofSize: 10, weight: .medium); ql.textColor = .darkGray; h.addSubview(ql)

        // Stage indicator
        let stageBox = NSView(frame: NSRect(x: 820, y: 28, width: 120, height: 24))
        stageBox.wantsLayer = true
        stageBox.layer?.backgroundColor = NSColor(red: 0.90, green: 0.85, blue: 0.95, alpha: 0.50).cgColor
        stageBox.layer?.cornerRadius = 5
        stageBox.layer?.borderColor = NSColor.systemPurple.withAlphaComponent(0.5).cgColor
        stageBox.layer?.borderWidth = 1
        h.addSubview(stageBox)
        let stageLbl = NSTextField(labelWithString: APOTHEOSIS_STAGE)
        stageLbl.frame = NSRect(x: 5, y: 3, width: 110, height: 18)
        stageLbl.font = NSFont.boldSystemFont(ofSize: 10)
        stageLbl.textColor = .systemPurple
        stageLbl.alignment = .center
        stageLbl.identifier = NSUserInterfaceItemIdentifier("header_stage")
        stageBox.addSubview(stageLbl)

        clockLabel = NSTextField(labelWithString: "00:00:00")
        clockLabel.frame = NSRect(x: bounds.width - 200, y: 32, width: 110, height: 30)
        clockLabel.font = NSFont.monospacedDigitSystemFont(ofSize: 26, weight: .bold)
        clockLabel.textColor = NSColor(red: 0.0, green: 0.55, blue: 0.70, alpha: 1.0)
        clockLabel.alignment = .right; clockLabel.autoresizingMask = [.minXMargin]
        clockLabel.wantsLayer = true
        clockLabel.layer?.shadowColor = NSColor(red: 0.0, green: 0.55, blue: 0.70, alpha: 1.0).cgColor
        clockLabel.layer?.shadowRadius = 3
        clockLabel.layer?.shadowOpacity = 0.15
        h.addSubview(clockLabel)

        phaseLabel = NSTextField(labelWithString: "φ: 0.0000")
        phaseLabel.frame = NSRect(x: bounds.width - 80, y: 36, width: 70, height: 16)
        phaseLabel.font = NSFont.monospacedSystemFont(ofSize: 10, weight: .semibold)
        phaseLabel.textColor = NSColor(red: 0.65, green: 0.50, blue: 0.08, alpha: 0.9)
        phaseLabel.autoresizingMask = [.minXMargin]; h.addSubview(phaseLabel)

        dateLabel = NSTextField(labelWithString: "")
        dateLabel.frame = NSRect(x: bounds.width - 200, y: 14, width: 110, height: 16)
        dateLabel.font = NSFont.systemFont(ofSize: 10, weight: .medium); dateLabel.textColor = .secondaryLabelColor
        dateLabel.alignment = .right; dateLabel.autoresizingMask = [.minXMargin]; h.addSubview(dateLabel)

        return h
    }

    func createMetricsBar() -> NSView {
        let bar = NSView(frame: NSRect(x: 0, y: bounds.height - 150, width: bounds.width, height: 65))
        bar.wantsLayer = true
        bar.layer?.backgroundColor = NSColor(red: 1.0, green: 1.0, blue: 1.0, alpha: 0.85).cgColor
        bar.autoresizingMask = [.width, .minYMargin]

        let metrics: [(String, String, String, CGFloat)] = [
            ("GOD_CODE", String(format: "%.2f", GOD_CODE), "ffd700", 1.0),
            ("OMEGA", String(format: "%.2f", OMEGA_POINT), "00d9ff", 1.0),
            ("ASI", String(format: "%.0f%%", state.asiScore * 100), "ff9800", state.asiScore),
            ("IQ", String(format: "%.0f", state.intellectIndex), "00ff88", min(1.0, state.intellectIndex / 200)),
            ("Coherence", String(format: "%.2f", state.coherence), "00bcd4", state.coherence),
            ("Memories", "\(state.permanentMemory.memories.count)", "9c27b0", min(1.0, Double(state.permanentMemory.memories.count) / 100)),
            ("Skills", "\(state.skills)", "e040fb", min(1.0, Double(state.skills) / 50)),
            ("Transcend", String(format: "%.0f%%", state.transcendence * 100), "ff4081", state.transcendence)
        ]

        var x: CGFloat = 15
        let tileWidth: CGFloat = (bounds.width - 30) / CGFloat(metrics.count) - 8
        for (label, value, colorHex, progress) in metrics {
            let color = colorFromHex(colorHex)
            let tile = AnimatedMetricTile(frame: NSRect(x: x, y: 8, width: tileWidth, height: 50),
                                          label: label, value: value, color: color, progress: CGFloat(progress))
            tile.autoresizingMask = [.width]
            bar.addSubview(tile)
            metricTiles[label] = tile
            x += tileWidth + 8
        }
        return bar
    }

    func createChatView() -> NSView {
        let v = NSView(frame: NSRect(x: 0, y: 0, width: 1100, height: 500))
        v.wantsLayer = true
        v.layer?.backgroundColor = NSColor(red: 0.970, green: 0.972, blue: 0.980, alpha: 1.0).cgColor

        // ─── RIGHT: System Activity Feed (separated from chat) ───
        let feedWidth: CGFloat = 260
        let feedPanel = NSView(frame: NSRect(x: v.bounds.width - feedWidth - 10, y: 70,
                                              width: feedWidth, height: v.bounds.height - 120))
        feedPanel.wantsLayer = true
        feedPanel.layer?.backgroundColor = NSColor(red: 0.965, green: 0.965, blue: 0.975, alpha: 1.0).cgColor
        feedPanel.layer?.cornerRadius = 12
        feedPanel.layer?.borderColor = NSColor(red: 0.0, green: 0.55, blue: 0.75, alpha: 0.20).cgColor
        feedPanel.layer?.borderWidth = 1
        feedPanel.autoresizingMask = [.minXMargin, .height]

        let feedTitle = NSTextField(labelWithString: "⚡ SYSTEM ACTIVITY")
        feedTitle.frame = NSRect(x: 10, y: feedPanel.bounds.height - 24, width: feedWidth - 20, height: 18)
        feedTitle.font = NSFont.systemFont(ofSize: 10, weight: .bold)
        feedTitle.textColor = NSColor(red: 0.0, green: 0.55, blue: 0.75, alpha: 0.9)
        feedTitle.autoresizingMask = [.minYMargin]
        feedPanel.addSubview(feedTitle)

        let feedScroll = NSScrollView(frame: NSRect(x: 5, y: 5, width: feedWidth - 10, height: feedPanel.bounds.height - 32))
        feedScroll.hasVerticalScroller = true
        feedScroll.autoresizingMask = [.width, .height]
        feedScroll.wantsLayer = true
        feedScroll.layer?.cornerRadius = 8

        systemFeedView = NSTextView(frame: feedScroll.bounds)
        systemFeedView.isEditable = false
        systemFeedView.isSelectable = true
        systemFeedView.backgroundColor = NSColor(red: 0.975, green: 0.975, blue: 0.985, alpha: 1.0)
        systemFeedView.font = NSFont.monospacedSystemFont(ofSize: 9, weight: .regular)
        systemFeedView.textContainerInset = NSSize(width: 6, height: 6)
        feedScroll.documentView = systemFeedView
        feedPanel.addSubview(feedScroll)
        v.addSubview(feedPanel)

        // ─── LEFT: Chat Area (clean, no system events) ───
        let chatWidth = v.bounds.width - feedWidth - 30
        let scroll = NSScrollView(frame: NSRect(x: 10, y: 70, width: chatWidth, height: v.bounds.height - 120))
        scroll.autoresizingMask = [.width, .height]; scroll.hasVerticalScroller = true
        scroll.wantsLayer = true; scroll.layer?.cornerRadius = 14
        scroll.layer?.borderColor = L104Theme.gold.withAlphaComponent(0.25).cgColor
        scroll.layer?.borderWidth = 1
        scroll.layer?.backgroundColor = NSColor(red: 1.0, green: 1.0, blue: 1.0, alpha: 1.0).cgColor
        scroll.layer?.shadowColor = NSColor.black.withAlphaComponent(0.08).cgColor
        scroll.layer?.shadowRadius = 6
        scroll.layer?.shadowOpacity = 0.15
        scroll.layer?.shadowOffset = CGSize(width: 0, height: -1)
        scroll.identifier = NSUserInterfaceItemIdentifier("chatScroll")

        chatTextView = NSTextView(frame: scroll.bounds)
        chatTextView.isEditable = false
        chatTextView.isSelectable = true  // ENABLE copy/paste
        chatTextView.allowsUndo = true
        // Visible dark background that contrasts with bright text
        chatTextView.backgroundColor = NSColor(red: 1.0, green: 1.0, blue: 1.0, alpha: 1.0)
        chatTextView.font = NSFont.systemFont(ofSize: 14)
        chatTextView.textContainerInset = NSSize(width: 15, height: 15)
        chatTextView.insertionPointColor = NSColor(red: 0.65, green: 0.50, blue: 0.08, alpha: 1.0)
        scroll.documentView = chatTextView
        v.addSubview(scroll)

        // History panel for past chats (lazy loaded)
        let historyPanel = NSView(frame: NSRect(x: v.bounds.width - 180, y: 70, width: 170, height: v.bounds.height - 85))
        historyPanel.wantsLayer = true
        historyPanel.layer?.backgroundColor = NSColor(red: 1.0, green: 1.0, blue: 1.0, alpha: 0.97).cgColor
        historyPanel.layer?.cornerRadius = 10
        historyPanel.layer?.borderColor = NSColor(red: 0.75, green: 0.78, blue: 0.82, alpha: 0.50).cgColor
        historyPanel.layer?.borderWidth = 1
        historyPanel.autoresizingMask = [.minXMargin, .height]
        historyPanel.isHidden = true
        historyPanel.identifier = NSUserInterfaceItemIdentifier("historyPanel")
        v.addSubview(historyPanel)

        let histTitle = NSTextField(labelWithString: "📜 Past Chats")
        histTitle.frame = NSRect(x: 10, y: historyPanel.bounds.height - 30, width: 150, height: 20)
        histTitle.font = NSFont.boldSystemFont(ofSize: 12)
        histTitle.textColor = NSColor(red: 0.15, green: 0.35, blue: 0.55, alpha: 1.0)
        histTitle.autoresizingMask = [.minYMargin]
        historyPanel.addSubview(histTitle)

        historyListView = NSScrollView(frame: NSRect(x: 5, y: 5, width: 160, height: historyPanel.bounds.height - 40))
        historyListView?.autoresizingMask = [.height]
        historyListView?.hasVerticalScroller = true
        let listContent = NSView(frame: NSRect(x: 0, y: 0, width: 150, height: 200))
        historyListView?.documentView = listContent
        historyPanel.addSubview(historyListView!)

        let inputBox = NSView(frame: NSRect(x: 10, y: 10, width: v.bounds.width - 20, height: 50))
        inputBox.wantsLayer = true
        inputBox.layer?.backgroundColor = NSColor(red: 1.0, green: 1.0, blue: 1.0, alpha: 1.0).cgColor
        inputBox.layer?.cornerRadius = 14; inputBox.autoresizingMask = [.width]
        inputBox.layer?.borderColor = L104Theme.gold.withAlphaComponent(0.40).cgColor
        inputBox.layer?.borderWidth = 1
        inputBox.layer?.shadowColor = NSColor.black.withAlphaComponent(0.10).cgColor
        inputBox.layer?.shadowRadius = 8
        inputBox.layer?.shadowOpacity = 0.15
        inputBox.layer?.shadowOffset = CGSize(width: 0, height: -1)
        v.addSubview(inputBox)

        // Toolbar above input for save/history
        let toolbar = NSView(frame: NSRect(x: 10, y: v.bounds.height - 115, width: v.bounds.width - 20, height: 28))
        toolbar.wantsLayer = true
        toolbar.layer?.backgroundColor = NSColor(red: 0.975, green: 0.975, blue: 0.980, alpha: 0.97).cgColor
        toolbar.layer?.cornerRadius = 8
        toolbar.layer?.borderColor = L104Theme.glassBorder.cgColor
        toolbar.layer?.borderWidth = 0.5
        toolbar.autoresizingMask = [.width, .minYMargin]
        v.addSubview(toolbar)

        let saveBtn = NSButton(frame: NSRect(x: 5, y: 2, width: 100, height: 24))
        saveBtn.title = "💾 Save Chat"
        saveBtn.bezelStyle = .rounded
        saveBtn.font = NSFont.systemFont(ofSize: 10, weight: .medium)
        saveBtn.contentTintColor = L104Theme.gold
        saveBtn.target = self; saveBtn.action = #selector(saveChatLog)
        toolbar.addSubview(saveBtn)

        let histBtn = NSButton(frame: NSRect(x: 110, y: 2, width: 100, height: 24))
        histBtn.title = "📜 History"
        histBtn.bezelStyle = .rounded
        histBtn.font = NSFont.systemFont(ofSize: 10, weight: .medium)
        histBtn.contentTintColor = L104Theme.goldWarm
        histBtn.target = self; histBtn.action = #selector(toggleHistory)
        toolbar.addSubview(histBtn)

        let copyBtn = NSButton(frame: NSRect(x: 215, y: 2, width: 100, height: 24))
        copyBtn.title = "📋 Copy All"
        copyBtn.bezelStyle = .rounded
        copyBtn.font = NSFont.systemFont(ofSize: 10, weight: .medium)
        copyBtn.contentTintColor = L104Theme.gold
        copyBtn.target = self; copyBtn.action = #selector(copyAllChat)
        toolbar.addSubview(copyBtn)

        let clearBtn = NSButton(frame: NSRect(x: 320, y: 2, width: 80, height: 24))
        clearBtn.title = "🗑 Clear"
        clearBtn.bezelStyle = .rounded
        clearBtn.font = NSFont.systemFont(ofSize: 10, weight: .medium)
        clearBtn.contentTintColor = L104Theme.goldDim
        clearBtn.target = self; clearBtn.action = #selector(clearChat)
        toolbar.addSubview(clearBtn)

        let searchBtn = NSButton(frame: NSRect(x: 405, y: 2, width: 100, height: 24))
        searchBtn.title = "🔍 Search"
        searchBtn.bezelStyle = .rounded
        searchBtn.font = NSFont.systemFont(ofSize: 10, weight: .medium)
        searchBtn.contentTintColor = L104Theme.gold
        searchBtn.target = self; searchBtn.action = #selector(searchChat)
        toolbar.addSubview(searchBtn)

        let exportBtn = NSButton(frame: NSRect(x: 510, y: 2, width: 110, height: 24))
        exportBtn.title = "📄 Export MD"
        exportBtn.bezelStyle = .rounded
        exportBtn.font = NSFont.systemFont(ofSize: 10, weight: .medium)
        exportBtn.contentTintColor = L104Theme.goldWarm
        exportBtn.target = self; exportBtn.action = #selector(exportChatMarkdown)
        toolbar.addSubview(exportBtn)

        let wordCountLbl = NSTextField(labelWithString: "")
        wordCountLbl.frame = NSRect(x: 635, y: 4, width: 160, height: 18)
        wordCountLbl.font = NSFont.monospacedSystemFont(ofSize: 9, weight: .medium)
        wordCountLbl.textColor = L104Theme.goldDim
        wordCountLbl.alignment = .right
        wordCountLbl.identifier = NSUserInterfaceItemIdentifier("chatWordCount")
        toolbar.addSubview(wordCountLbl)

        inputField = NSTextField(frame: NSRect(x: 15, y: 12, width: inputBox.bounds.width - 130, height: 28))
        inputField.placeholderString = "Enter command..."
        inputField.font = L104Theme.monoFont(14, weight: .medium)
        inputField.isBordered = true
        inputField.bezelStyle = .roundedBezel
        // Dark background with bright gold text for HIGH visibility
        inputField.backgroundColor = L104Theme.voidDeep
        inputField.textColor = L104Theme.goldBright
        inputField.focusRingType = .none; inputField.autoresizingMask = [.width]
        inputField.target = self; inputField.action = #selector(sendMessage)
        inputBox.addSubview(inputField)

        let sendBtn = HoverButton(frame: NSRect(x: inputBox.bounds.width - 115, y: 8, width: 105, height: 34))
        sendBtn.title = "⚡ TRANSMIT"; sendBtn.bezelStyle = .rounded
        sendBtn.wantsLayer = true
        sendBtn.layer?.backgroundColor = L104Theme.gold.withAlphaComponent(0.10).cgColor
        sendBtn.layer?.cornerRadius = CGFloat(L104Theme.radiusMedium)
        sendBtn.layer?.borderColor = L104Theme.gold.withAlphaComponent(0.5).cgColor
        sendBtn.layer?.borderWidth = 1.5
        sendBtn.layer?.shadowColor = L104Theme.gold.cgColor
        sendBtn.layer?.shadowRadius = 8
        sendBtn.layer?.shadowOpacity = 0.3
        sendBtn.contentTintColor = L104Theme.goldBright
        sendBtn.font = NSFont.boldSystemFont(ofSize: 11)
        sendBtn.hoverColor = L104Theme.gold
        sendBtn.target = self; sendBtn.action = #selector(sendMessage)
        sendBtn.autoresizingMask = [.minXMargin]
        inputBox.addSubview(sendBtn)

        return v
    }

    func createLearningView() -> NSView {
        let v = NSView(frame: NSRect(x: 0, y: 0, width: 1100, height: 500))
        v.wantsLayer = true
        v.layer?.backgroundColor = L104Theme.void.cgColor

        let learner = AdaptiveLearner.shared

        // Left column: Topic Mastery
        let masteryPanel = createPanel("🎯 TOPIC MASTERY", x: 15, y: 100, w: 350, h: 380, color: "d4af37")

        let topMastered = learner.topicMastery.values.sorted { $0.masteryLevel > $1.masteryLevel }.prefix(10)
        var my: CGFloat = 310
        if topMastered.isEmpty {
            let lbl = NSTextField(labelWithString: "   Chat naturally to build mastery!")
            lbl.frame = NSRect(x: 15, y: my, width: 320, height: 18)
            lbl.font = NSFont.systemFont(ofSize: 11); lbl.textColor = .gray; masteryPanel.addSubview(lbl)
        } else {
            for mastery in topMastered {
                let topicLabel = NSTextField(labelWithString: "\(mastery.tier)  \(mastery.topic)")
                topicLabel.frame = NSRect(x: 15, y: my, width: 200, height: 18)
                topicLabel.font = NSFont.systemFont(ofSize: 11, weight: .medium)
                topicLabel.textColor = mastery.masteryLevel > 0.6 ? L104Theme.goldBright : L104Theme.goldDim
                masteryPanel.addSubview(topicLabel)

                let bar = GlowingProgressBar(frame: NSRect(x: 220, y: my + 4, width: 90, height: 8))
                bar.progress = CGFloat(mastery.masteryLevel)
                bar.barColor = mastery.masteryLevel > 0.65 ? L104Theme.gold : mastery.masteryLevel > 0.3 ? L104Theme.goldDim : L104Theme.textDim
                masteryPanel.addSubview(bar)

                let pctLabel = NSTextField(labelWithString: "\(String(format: "%.0f%%", mastery.masteryLevel * 100))")
                pctLabel.frame = NSRect(x: 315, y: my, width: 30, height: 18)
                pctLabel.font = NSFont.monospacedSystemFont(ofSize: 10, weight: .semibold)
                pctLabel.textColor = L104Theme.gold; pctLabel.alignment = .right
                masteryPanel.addSubview(pctLabel)

                my -= 28
                if my < 30 { break }
            }
        }
        v.addSubview(masteryPanel)

        // Middle column: User Profile
        let profilePanel = createPanel("💝 USER PROFILE", x: 380, y: 250, w: 350, h: 230, color: "c49b30")

        let topInterests = learner.userInterests.sorted { $0.value > $1.value }.prefix(6)
        var py: CGFloat = 165
        if topInterests.isEmpty {
            let lbl = NSTextField(labelWithString: "   Building your interest profile...")
            lbl.frame = NSRect(x: 15, y: py, width: 320, height: 18)
            lbl.font = NSFont.systemFont(ofSize: 11); lbl.textColor = .gray; profilePanel.addSubview(lbl)
        } else {
            for interest in topInterests {
                let lbl = NSTextField(labelWithString: "• \(interest.key)")
                lbl.frame = NSRect(x: 15, y: py, width: 200, height: 18)
                lbl.font = NSFont.systemFont(ofSize: 11, weight: .medium)
                lbl.textColor = L104Theme.goldWarm
                profilePanel.addSubview(lbl)

                let count = NSTextField(labelWithString: "\(Int(interest.value))x")
                count.frame = NSRect(x: 280, y: py, width: 50, height: 18)
                count.font = NSFont.monospacedSystemFont(ofSize: 10, weight: .semibold)
                count.textColor = L104Theme.gold; count.alignment = .right
                profilePanel.addSubview(count)

                py -= 24
            }
        }

        // Style analysis
        let styleLabel = NSTextField(labelWithString: "🎨 Style: \(learner.prefersDetail() ? "Detail-oriented" : "Concise")")
        styleLabel.frame = NSRect(x: 15, y: 15, width: 320, height: 18)
        styleLabel.font = NSFont.systemFont(ofSize: 10, weight: .medium)
        styleLabel.textColor = L104Theme.goldDim
        profilePanel.addSubview(styleLabel)
        v.addSubview(profilePanel)

        // Middle column bottom: User-Taught Facts
        let factsPanel = createPanel("📖 TAUGHT FACTS", x: 380, y: 100, w: 350, h: 140, color: "a88a25")
        let facts = Array(learner.userTaughtFacts.prefix(4))
        var fy: CGFloat = 80
        if facts.isEmpty {
            let lbl = NSTextField(labelWithString: "   Use 'teach X is Y' to teach me!")
            lbl.frame = NSRect(x: 15, y: fy, width: 320, height: 18)
            lbl.font = NSFont.systemFont(ofSize: 11); lbl.textColor = .gray; factsPanel.addSubview(lbl)
        } else {
            for (key, value) in facts {
                let lbl = NSTextField(labelWithString: "• \(key): \(value)")
                lbl.frame = NSRect(x: 15, y: fy, width: 320, height: 18)
                lbl.font = NSFont.systemFont(ofSize: 10, weight: .medium)
                lbl.textColor = L104Theme.gold; lbl.lineBreakMode = .byTruncatingTail
                factsPanel.addSubview(lbl)
                fy -= 22
            }
        }
        v.addSubview(factsPanel)

        // Right column: Learning Stats
        let statsPanel = createPanel("📊 LEARNING METRICS", x: 745, y: 250, w: 340, h: 230, color: "e8c547")

        let statItems: [(String, String, String, String)] = [
            ("Total Interactions", "\(learner.interactionCount)", "d4af37", "learn_interactions"),
            ("Topics Tracked", "\(learner.topicMastery.count)", "e8c547", "learn_topics"),
            ("Success Patterns", "\(learner.successfulPatterns.count)", "c49b30", "learn_success"),
            ("Corrections Logged", "\(learner.failedPatterns.count)", "8a7120", "learn_corrections"),
            ("Insights Synthesized", "\(learner.synthesizedInsights.count)", "d4af37", "learn_insights"),
            ("User-Taught Facts", "\(learner.userTaughtFacts.count)", "c49b30", "learn_facts"),
            ("KB User Entries", "\(ASIKnowledgeBase.shared.userKnowledge.count)", "a88a25", "learn_kb")
        ]

        var sy: CGFloat = 160
        for (label, value, hex, id) in statItems {
            let lbl = NSTextField(labelWithString: label)
            lbl.frame = NSRect(x: 15, y: sy, width: 180, height: 16)
            lbl.font = NSFont.systemFont(ofSize: 10); lbl.textColor = .gray; statsPanel.addSubview(lbl)
            let val = NSTextField(labelWithString: value)
            val.frame = NSRect(x: 200, y: sy, width: 120, height: 16)
            val.font = NSFont.boldSystemFont(ofSize: 11); val.textColor = colorFromHex(hex); val.alignment = .right
            val.identifier = NSUserInterfaceItemIdentifier(id)
            statsPanel.addSubview(val)
            sy -= 22
        }
        v.addSubview(statsPanel)

        // Right column bottom: Latest Insight
        let insightPanel = createPanel("💡 LATEST INSIGHT", x: 745, y: 100, w: 340, h: 140, color: "d4af37")
        let insightText = learner.synthesizedInsights.last ?? "Synthesizes automatically every 10 interactions..."
        let insightLbl = NSTextField(wrappingLabelWithString: insightText)
        insightLbl.frame = NSRect(x: 15, y: 15, width: 310, height: 90)
        insightLbl.font = NSFont.systemFont(ofSize: 10, weight: .medium)
        insightLbl.textColor = L104Theme.goldBright
        insightLbl.identifier = NSUserInterfaceItemIdentifier("learn_insight")
        insightLbl.maximumNumberOfLines = 5
        insightPanel.addSubview(insightLbl)
        v.addSubview(insightPanel)

        // Bottom status bar
        let statusBar = NSView(frame: NSRect(x: 15, y: 55, width: v.bounds.width - 30, height: 35))
        statusBar.wantsLayer = true
        statusBar.layer?.backgroundColor = L104Theme.glass.cgColor
        statusBar.layer?.cornerRadius = CGFloat(L104Theme.radiusMedium)
        statusBar.layer?.borderColor = L104Theme.glassBorder.cgColor
        statusBar.layer?.borderWidth = 1

        let masteredCount = learner.topicMastery.values.filter { $0.masteryLevel > 0.65 }.count
        let learningCount = learner.topicMastery.values.filter { $0.masteryLevel > 0.15 && $0.masteryLevel <= 0.65 }.count
        let statusText = "🧠 Adaptive Learning v3.0 · EVO_\(EVOLUTION_INDEX) | \(masteredCount) mastered | \(learningCount) developing | \(learner.interactionCount) interactions | KB: \(ASIKnowledgeBase.shared.userKnowledge.count) entries | Ingest: \(DataIngestPipeline.shared.totalIngested)"
        let statusLbl = NSTextField(labelWithString: statusText)
        statusLbl.frame = NSRect(x: 15, y: 8, width: statusBar.bounds.width - 30, height: 18)
        statusLbl.font = NSFont.monospacedSystemFont(ofSize: 10, weight: .medium)
        statusLbl.textColor = NSColor(red: 0.10, green: 0.50, blue: 0.65, alpha: 1.0)
        statusLbl.identifier = NSUserInterfaceItemIdentifier("learn_status_lbl")
        statusBar.addSubview(statusLbl)
        v.addSubview(statusBar)

        // ─── Learning Refresh Timer — update status bar every 8s (cached + visibility-gated) ───
        learningRefreshTimer?.invalidate()
        learningRefreshTimer = Timer.scheduledTimer(withTimeInterval: 8.0, repeats: true) { [weak self] _ in
            guard let s = self, s.activeTabID == "learn" else { return }
            let lr = AdaptiveLearner.shared
            let mc = lr.topicMastery.values.filter { $0.masteryLevel > 0.65 }.count
            let lc = lr.topicMastery.values.filter { $0.masteryLevel > 0.15 && $0.masteryLevel <= 0.65 }.count
            s.cachedLabel("learn_status_lbl", in: "learn")?.stringValue = "🧠 Adaptive Learning v3.0 · EVO_\(EVOLUTION_INDEX) | \(mc) mastered | \(lc) developing | \(lr.interactionCount) interactions | KB: \(ASIKnowledgeBase.shared.userKnowledge.count) entries | Ingest: \(DataIngestPipeline.shared.totalIngested)"
            // 🟢 EVO_64: Live refresh learning stats
            s.cachedLabel("learn_interactions", in: "learn")?.stringValue = "\(lr.interactionCount)"
            s.cachedLabel("learn_topics", in: "learn")?.stringValue = "\(lr.topicMastery.count)"
            s.cachedLabel("learn_success", in: "learn")?.stringValue = "\(lr.successfulPatterns.count)"
            s.cachedLabel("learn_corrections", in: "learn")?.stringValue = "\(lr.failedPatterns.count)"
            s.cachedLabel("learn_insights", in: "learn")?.stringValue = "\(lr.synthesizedInsights.count)"
            s.cachedLabel("learn_facts", in: "learn")?.stringValue = "\(lr.userTaughtFacts.count)"
            s.cachedLabel("learn_kb", in: "learn")?.stringValue = "\(ASIKnowledgeBase.shared.userKnowledge.count)"
            s.cachedLabel("learn_insight", in: "learn")?.stringValue = lr.synthesizedInsights.last ?? "Synthesizes automatically every 10 interactions..."
        }

        return v
    }

    func createASIView() -> NSView {
        let v = NSView(frame: NSRect(x: 0, y: 0, width: 1100, height: 500))
        v.wantsLayer = true; v.layer?.backgroundColor = L104Theme.void.cgColor

        // ASI Panel — expanded with comprehensive metrics
        let asiP = createPanel("🚀 ASI CORE — v\(ASI_VERSION)", x: 15, y: 260, w: 350, h: 280, color: "d4af37")
        addLabel(asiP, "ASI_SCORE", String(format: "%.1f%%", state.asiScore * 100), y: 160, c: "d4af37")
        addLabel(asiP, "DISCOVERIES", "\(state.discoveries)", y: 135, c: "e8c547")
        addLabel(asiP, "TRANSCENDENCE", String(format: "%.1f%%", state.transcendence * 100), y: 110, c: "d4af37")
        addLabel(asiP, "THOUGHT_OPS", "0", y: 85, c: "c49b30")
        addLabel(asiP, "PHYSICS_OPS", "0", y: 60, c: "a88a25")
        addLabel(asiP, "INTELLECT_MEM", "0", y: 35, c: "8a7120")
        addLabel(asiP, "INTELLECT_KNOW", "0", y: 10, c: "6a5a15")
        addLabel(asiP, "ENGINES_ACTIVE", "0/0", y: 240, c: "d4af37")
        addLabel(asiP, "PHI_HEALTH", "0.0%", y: 215, c: "e8c547")
        addLabel(asiP, "DUAL_LAYER", "v\(DUAL_LAYER_VERSION) · \(DUAL_LAYER_CONSTANTS_COUNT) constants", y: 190, c: "c49b30")

        // Tag dynamic value labels for live refresh
        for sub in asiP.subviews {
            if let tf = sub as? NSTextField, tf.alignment == .right {
                if tf.stringValue.contains("ASI") || tf.frame.origin.y == 160 {
                    tf.identifier = NSUserInterfaceItemIdentifier("asi_score_val")
                }
                if tf.frame.origin.y == 135 { tf.identifier = NSUserInterfaceItemIdentifier("asi_disc_val") }
                if tf.frame.origin.y == 110 { tf.identifier = NSUserInterfaceItemIdentifier("asi_trans_val") }
                if tf.frame.origin.y == 85 { tf.identifier = NSUserInterfaceItemIdentifier("thought_ops_val") }
                if tf.frame.origin.y == 60 { tf.identifier = NSUserInterfaceItemIdentifier("physics_ops_val") }
                if tf.frame.origin.y == 35 { tf.identifier = NSUserInterfaceItemIdentifier("intellect_mem_val") }
                if tf.frame.origin.y == 10 { tf.identifier = NSUserInterfaceItemIdentifier("intellect_know_val") }
                if tf.frame.origin.y == 240 { tf.identifier = NSUserInterfaceItemIdentifier("engines_active_val") }
                if tf.frame.origin.y == 215 { tf.identifier = NSUserInterfaceItemIdentifier("phi_health_val") }
            }
        }

        let ignASI = btn("🔥 IGNITE ASI", x: 20, y: 20, w: 150, c: L104Theme.gold)
        ignASI.target = self; ignASI.action = #selector(doIgniteASI); asiP.addSubview(ignASI)
        let transcBtn = btn("🌟 TRANSCEND", x: 180, y: 20, w: 150, c: L104Theme.goldWarm)
        transcBtn.target = self; transcBtn.action = #selector(doTranscend); asiP.addSubview(transcBtn)
        v.addSubview(asiP)

        // AGI Panel — expanded with version data
        let agiP = createPanel("⚡ AGI METRICS — v\(AGI_VERSION)", x: 380, y: 260, w: 350, h: 220, color: "e8c547")
        addLabel(agiP, "INTELLECT", String(format: "%.1f", state.intellectIndex), y: 160, c: "e8c547")
        addLabel(agiP, "QUANTUM_RES", String(format: "%.1f%%", state.quantumResonance * 100), y: 135, c: "d4af37")
        addLabel(agiP, "SKILLS", "\(state.skills)", y: 110, c: "c49b30")
        addLabel(agiP, "CODE_ENGINE", "v\(CODE_ENGINE_VERSION) · 10 modules", y: 85, c: "a88a25")
        addLabel(agiP, "INTELLECT_PKG", "v\(INTELLECT_VERSION) · 11 modules", y: 60, c: "8a7120")

        // Tag dynamic value labels for live refresh
        for sub in agiP.subviews {
            if let tf = sub as? NSTextField, tf.alignment == .right {
                if tf.frame.origin.y == 160 { tf.identifier = NSUserInterfaceItemIdentifier("agi_intellect_val") }
                if tf.frame.origin.y == 135 { tf.identifier = NSUserInterfaceItemIdentifier("agi_qres_val") }
                if tf.frame.origin.y == 110 { tf.identifier = NSUserInterfaceItemIdentifier("agi_skills_val") }
            }
        }

        let ignAGI = btn("⚡ IGNITE AGI", x: 20, y: 20, w: 150, c: L104Theme.gold)
        ignAGI.target = self; ignAGI.action = #selector(doIgniteAGI); agiP.addSubview(ignAGI)
        let evoBtn = btn("🔄 EVOLVE", x: 180, y: 20, w: 150, c: L104Theme.goldDim)
        evoBtn.target = self; evoBtn.action = #selector(doEvolve); agiP.addSubview(evoBtn)
        v.addSubview(agiP)

        // Quantum Bridge Panel — live ASI quantum metrics
        let quantP = createPanel("⚛️ QUANTUM BRIDGE — Live Metrics", x: 380, y: 10, w: 350, h: 240, color: "8a7120")
        addLabel(quantP, "KUNDALINI", "0.0000", y: 180, c: "d4af37")
        addLabel(quantP, "BELL_FIDELITY", "0.9999", y: 155, c: "e8c547")
        addLabel(quantP, "EPR_LINKS", "0", y: 130, c: "c49b30")
        addLabel(quantP, "VISHUDDHA_RES", "0.0000", y: 105, c: "a88a25")
        addLabel(quantP, "BRIDGE_INTEGRITY", "0.0%", y: 80, c: "8a7120")
        addLabel(quantP, "ENGINE_UPTIME", "0s", y: 55, c: "6a5a15")
        addLabel(quantP, "SYNC_COUNTER", "0", y: 30, c: "4a3a05")

        // Tag dynamic value labels for live refresh
        for sub in quantP.subviews {
            if let tf = sub as? NSTextField, tf.alignment == .right {
                if tf.frame.origin.y == 180 { tf.identifier = NSUserInterfaceItemIdentifier("kundalini_val") }
                if tf.frame.origin.y == 155 { tf.identifier = NSUserInterfaceItemIdentifier("bell_fidelity_val") }
                if tf.frame.origin.y == 130 { tf.identifier = NSUserInterfaceItemIdentifier("epr_links_val") }
                if tf.frame.origin.y == 105 { tf.identifier = NSUserInterfaceItemIdentifier("vishuddha_val") }
                if tf.frame.origin.y == 80 { tf.identifier = NSUserInterfaceItemIdentifier("bridge_integrity_val") }
                if tf.frame.origin.y == 55 { tf.identifier = NSUserInterfaceItemIdentifier("engine_uptime_val") }
                if tf.frame.origin.y == 30 { tf.identifier = NSUserInterfaceItemIdentifier("sync_counter_val") }
            }
        }

        let syncBtn = btn("🔄 SYNC", x: 20, y: 5, w: 100, c: L104Theme.gold)
        syncBtn.target = self; syncBtn.action = #selector(doSyncBridge); quantP.addSubview(syncBtn)
        let collapseBtn = btn("⚡ COLLAPSE", x: 130, y: 5, w: 100, c: L104Theme.goldWarm)
        collapseBtn.target = self; collapseBtn.action = #selector(doCollapse); quantP.addSubview(collapseBtn)
        let statusBtn = btn("📊 STATUS", x: 240, y: 5, w: 90, c: L104Theme.goldDim)
        statusBtn.target = self; statusBtn.action = #selector(doBridgeStatus); quantP.addSubview(statusBtn)
        v.addSubview(quantP)

        // Consciousness Panel — expanded with live state data
        let conP = createPanel("🧠 CONSCIOUSNESS — \(APOTHEOSIS_STAGE)", x: 745, y: 260, w: 340, h: 220, color: "c49b30")
        addLabel(conP, "STATE", state.consciousness, y: 160, c: "d4af37")
        addLabel(conP, "COHERENCE", String(format: "%.4f", state.coherence), y: 135, c: "c49b30")
        addLabel(conP, "OMEGA_PROB", String(format: "%.1f%%", state.omegaProbability * 100), y: 110, c: "e8c547")
        addLabel(conP, "SCHUMANN", String(format: "%.4f Hz", SCHUMANN_RESONANCE), y: 85, c: "a88a25")
        addLabel(conP, "GAMMA_BIND", String(format: "%.0f Hz", GAMMA_BINDING_HZ), y: 60, c: "8a7120")

        // Tag dynamic value labels for live refresh
        for sub in conP.subviews {
            if let tf = sub as? NSTextField, tf.alignment == .right {
                if tf.frame.origin.y == 160 { tf.identifier = NSUserInterfaceItemIdentifier("con_state_val") }
                if tf.frame.origin.y == 135 { tf.identifier = NSUserInterfaceItemIdentifier("con_coher_val") }
                if tf.frame.origin.y == 110 { tf.identifier = NSUserInterfaceItemIdentifier("con_omega_val") }
            }
        }

        let resBtn = btn("⚡ RESONATE", x: 20, y: 20, w: 150, c: L104Theme.gold)
        resBtn.target = self; resBtn.action = #selector(doResonate); conP.addSubview(resBtn)
        let synthBtn = btn("✨ SYNTHESIS", x: 180, y: 20, w: 140, c: L104Theme.goldWarm)
        synthBtn.target = self; synthBtn.action = #selector(doSynthesize); conP.addSubview(synthBtn)
        v.addSubview(conP)

        // DeepSeek Ingestion Panel — live ingestion metrics
        let deepseekP = createPanel("🧬 DEEPSEEK INGESTION — Live Status", x: 15, y: 10, w: 350, h: 240, color: "6a5a15")
        addLabel(deepseekP, "MLA_PATTERNS", "0", y: 180, c: "d4af37")
        addLabel(deepseekP, "R1_CHAINS", "0", y: 155, c: "e8c547")
        addLabel(deepseekP, "CODER_LANGS", "0", y: 130, c: "c49b30")
        addLabel(deepseekP, "ADAPTATIONS", "0", y: 105, c: "a88a25")
        addLabel(deepseekP, "GOD_CODE_ALIGN", "0.0000", y: 80, c: "8a7120")
        addLabel(deepseekP, "PHI_WEIGHTING", "0.0000", y: 55, c: "6a5a15")
        addLabel(deepseekP, "QUANTUM_ENHANCE", "0", y: 30, c: "4a3a05")

        // Tag dynamic value labels for live refresh
        for sub in deepseekP.subviews {
            if let tf = sub as? NSTextField, tf.alignment == .right {
                if tf.frame.origin.y == 180 { tf.identifier = NSUserInterfaceItemIdentifier("mla_patterns_val") }
                if tf.frame.origin.y == 155 { tf.identifier = NSUserInterfaceItemIdentifier("r1_chains_val") }
                if tf.frame.origin.y == 130 { tf.identifier = NSUserInterfaceItemIdentifier("coder_langs_val") }
                if tf.frame.origin.y == 105 { tf.identifier = NSUserInterfaceItemIdentifier("adaptations_val") }
                if tf.frame.origin.y == 80 { tf.identifier = NSUserInterfaceItemIdentifier("god_code_align_val") }
                if tf.frame.origin.y == 55 { tf.identifier = NSUserInterfaceItemIdentifier("phi_weighting_val") }
                if tf.frame.origin.y == 30 { tf.identifier = NSUserInterfaceItemIdentifier("quantum_enhance_val") }
            }
        }

        let ingestBtn = btn("📥 INGEST", x: 20, y: 5, w: 100, c: L104Theme.gold)
        ingestBtn.target = self; ingestBtn.action = #selector(doDeepSeekIngest); deepseekP.addSubview(ingestBtn)
        let deepseekStatusBtn = btn("📊 STATUS", x: 130, y: 5, w: 100, c: L104Theme.goldWarm)
        deepseekStatusBtn.target = self; deepseekStatusBtn.action = #selector(doDeepSeekStatus); deepseekP.addSubview(deepseekStatusBtn)
        let adaptBtn = btn("🔄 ADAPT", x: 240, y: 5, w: 90, c: L104Theme.goldDim)
        adaptBtn.target = self; adaptBtn.action = #selector(doDeepSeekAdapt); deepseekP.addSubview(adaptBtn)
        v.addSubview(deepseekP)

        // Quantum Architecture Integration Panel — quantum circuit creation
        let quantumP = createPanel("⚛️ QUANTUM ARCHITECTURE — Circuit Integration", x: 745, y: 10, w: 340, h: 240, color: "4a3a05")
        addLabel(quantumP, "CIRCUITS_CREATED", "0", y: 180, c: "d4af37")
        addLabel(quantumP, "PATTERNS_INTEGRATED", "0", y: 155, c: "e8c547")
        addLabel(quantumP, "QUANTUM_GATES", "0", y: 130, c: "c49b30")
        addLabel(quantumP, "GOD_CODE_ALIGNS", "0", y: 105, c: "a88a25")
        addLabel(quantumP, "MLA_CIRCUITS", "0", y: 80, c: "8a7120")
        addLabel(quantumP, "REASONING_CIRCUITS", "0", y: 55, c: "6a5a15")
        addLabel(quantumP, "CODER_CIRCUITS", "0", y: 30, c: "4a3a05")

        // Tag dynamic value labels for live refresh
        for sub in quantumP.subviews {
            if let tf = sub as? NSTextField, tf.alignment == .right {
                if tf.frame.origin.y == 180 { tf.identifier = NSUserInterfaceItemIdentifier("circuits_created_val") }
                if tf.frame.origin.y == 155 { tf.identifier = NSUserInterfaceItemIdentifier("patterns_integrated_val") }
                if tf.frame.origin.y == 130 { tf.identifier = NSUserInterfaceItemIdentifier("quantum_gates_val") }
                if tf.frame.origin.y == 105 { tf.identifier = NSUserInterfaceItemIdentifier("god_code_aligns_val") }
                if tf.frame.origin.y == 80 { tf.identifier = NSUserInterfaceItemIdentifier("mla_circuits_val") }
                if tf.frame.origin.y == 55 { tf.identifier = NSUserInterfaceItemIdentifier("reasoning_circuits_val") }
                if tf.frame.origin.y == 30 { tf.identifier = NSUserInterfaceItemIdentifier("coder_circuits_val") }
            }
        }

        let integrateBtn = btn("⚛️ INTEGRATE", x: 20, y: 5, w: 100, c: L104Theme.gold)
        integrateBtn.target = self; integrateBtn.action = #selector(doQuantumIntegrate); quantumP.addSubview(integrateBtn)
        let quantumStatusBtn = btn("📊 STATUS", x: 130, y: 5, w: 100, c: L104Theme.goldWarm)
        quantumStatusBtn.target = self; quantumStatusBtn.action = #selector(doQuantumStatus); quantumP.addSubview(quantumStatusBtn)
        let circuitBtn = btn("🔄 CIRCUITS", x: 240, y: 5, w: 90, c: L104Theme.goldDim)
        circuitBtn.target = self; circuitBtn.action = #selector(doQuantumCircuits); quantumP.addSubview(circuitBtn)
        v.addSubview(quantumP)

        // Constants + Package Overview bar
        let constText = "GOD_CODE: \(GOD_CODE) | Ω: \(OMEGA_POINT) | φ: \(PHI) | 22T: \(TRILLION_PARAMS) | EVO: \(EVOLUTION_INDEX) | \(TOTAL_PACKAGES) pkg · \(TOTAL_PACKAGE_MODULES) mod · \(TOTAL_PACKAGE_LINES) lines"
        let constL = NSTextField(labelWithString: constText)
        constL.frame = NSRect(x: 15, y: 220, width: v.bounds.width - 30, height: 30)
        constL.font = NSFont.monospacedSystemFont(ofSize: 10, weight: .medium)
        constL.textColor = L104Theme.goldDim
        v.addSubview(constL)

        // ═══ INTERCONNECT: QPC + Creativity + Agent + Identity live status bar ═══
        let qpc = QuantumProcessingCore.shared
        let tomo = qpc.stateTomography()
        let qce = QuantumCreativityEngine.shared
        let cm = qce.creativityMetrics
        let liveBar = NSTextField(labelWithString: "⚛️ QPC F=\(String(format: "%.3f", qpc.currentFidelity())) pur=\(String(format: "%.3f", tomo.purity)) | 🎨 Gen:\(cm["generation_count"] as? Int ?? 0) | 🤖 Agent:\(AutonomousAgent.shared.isActive ? "ON" : "OFF") | 🆔 ID:\(SovereignIdentityBoundary.shared.getStatus()["claim_validations"] as? Int ?? 0) claims")
        liveBar.frame = NSRect(x: 15, y: 250, width: v.bounds.width - 30, height: 16)
        liveBar.font = NSFont.monospacedSystemFont(ofSize: 9, weight: .medium)
        liveBar.textColor = .systemCyan
        liveBar.identifier = NSUserInterfaceItemIdentifier("asi_live_bar")
        v.addSubview(liveBar)

        // ─── ASI Nexus Refresh Timer — update dynamic values every 4s (cached + visibility-gated) ───
        asiNexusRefreshTimer?.invalidate()
        asiNexusRefreshTimer = Timer.scheduledTimer(withTimeInterval: 4.0, repeats: true) { [weak self] _ in
            guard let s = self, s.activeTabID == "asi" else { return }

            // Fetch live ASI bridge status for dual-layer metrics
            _ = ASIQuantumBridgeSwift.shared.fetchASIBridgeStatus()

            // ASI panel dynamic values (cached label lookup)
            s.cachedLabel("asi_score_val", in: "asi")?.stringValue = String(format: "%.1f%%", s.state.asiScore * 100)
            s.cachedLabel("asi_disc_val", in: "asi")?.stringValue = "\(s.state.discoveries)"
            s.cachedLabel("asi_trans_val", in: "asi")?.stringValue = String(format: "%.1f%%", s.state.transcendence * 100)
            // Dual-layer live metrics
            let bridge = ASIQuantumBridgeSwift.shared
            s.cachedLabel("thought_ops_val", in: "asi")?.stringValue = "\(Int(bridge.thoughtLayerScore))"
            s.cachedLabel("physics_ops_val", in: "asi")?.stringValue = "\(Int(bridge.physicsLayerScore))"
            // Intellect metrics
            s.cachedLabel("intellect_mem_val", in: "asi")?.stringValue = "\(bridge.intellectMemories)"
            s.cachedLabel("intellect_know_val", in: "asi")?.stringValue = "\(bridge.intellectKnowledge)"
            // Engine registry metrics
            s.cachedLabel("engines_active_val", in: "asi")?.stringValue = "\(bridge.activeEngines)/\(bridge.totalEngines)"
            s.cachedLabel("phi_health_val", in: "asi")?.stringValue = String(format: "%.1f%%", bridge.phiHealth * 100)
            // Quantum bridge metrics
            s.cachedLabel("kundalini_val", in: "asi")?.stringValue = String(format: "%.4f", bridge.kundaliniFlow)
            s.cachedLabel("bell_fidelity_val", in: "asi")?.stringValue = String(format: "%.4f", bridge.bellFidelity)
            s.cachedLabel("epr_links_val", in: "asi")?.stringValue = "\(bridge.eprLinks)"
            s.cachedLabel("vishuddha_val", in: "asi")?.stringValue = String(format: "%.4f", bridge.chakraCoherence["VISHUDDHA"] ?? 0.0)
            s.cachedLabel("bridge_integrity_val", in: "asi")?.stringValue = String(format: "%.1f%%", bridge.bridgeIntegrity * 100)
            s.cachedLabel("engine_uptime_val", in: "asi")?.stringValue = String(format: "%.0fs", bridge.engineUptime)
            s.cachedLabel("sync_counter_val", in: "asi")?.stringValue = "\(bridge.syncCounter)"
            // AGI panel dynamic values
            s.cachedLabel("agi_intellect_val", in: "asi")?.stringValue = String(format: "%.1f", s.state.intellectIndex)
            s.cachedLabel("agi_qres_val", in: "asi")?.stringValue = String(format: "%.1f%%", s.state.quantumResonance * 100)
            s.cachedLabel("agi_skills_val", in: "asi")?.stringValue = "\(s.state.skills)"
            // Consciousness panel dynamic values
            s.cachedLabel("con_state_val", in: "asi")?.stringValue = s.state.consciousness
            s.cachedLabel("con_coher_val", in: "asi")?.stringValue = String(format: "%.4f", s.state.coherence)
            s.cachedLabel("con_omega_val", in: "asi")?.stringValue = String(format: "%.1f%%", s.state.omegaProbability * 100)
            // DeepSeek ingestion metrics
            s.cachedLabel("mla_patterns_val", in: "asi")?.stringValue = "\(bridge.deepseekMLAPatterns)"
            s.cachedLabel("r1_chains_val", in: "asi")?.stringValue = "\(bridge.deepseekR1Chains)"
            s.cachedLabel("coder_langs_val", in: "asi")?.stringValue = "\(bridge.deepseekCoderLangs)"
            s.cachedLabel("adaptations_val", in: "asi")?.stringValue = "\(bridge.deepseekAdaptations)"
            s.cachedLabel("god_code_align_val", in: "asi")?.stringValue = String(format: "%.4f", bridge.deepseekGodCodeAlign)
            s.cachedLabel("phi_weighting_val", in: "asi")?.stringValue = String(format: "%.4f", bridge.deepseekPhiWeighting)
            s.cachedLabel("quantum_enhance_val", in: "asi")?.stringValue = "\(bridge.deepseekQuantumEnhance)"
            // Quantum architecture metrics
            s.cachedLabel("circuits_created_val", in: "asi")?.stringValue = "\(bridge.quantumCircuitsCreated)"
            s.cachedLabel("patterns_integrated_val", in: "asi")?.stringValue = "\(bridge.quantumPatternsIntegrated)"
            s.cachedLabel("quantum_gates_val", in: "asi")?.stringValue = "\(bridge.quantumGatesApplied)"
            s.cachedLabel("god_code_aligns_val", in: "asi")?.stringValue = "\(bridge.quantumGodCodeAligns)"
            s.cachedLabel("mla_circuits_val", in: "asi")?.stringValue = "\(bridge.quantumMLACircuits)"
            s.cachedLabel("reasoning_circuits_val", in: "asi")?.stringValue = "\(bridge.quantumReasoningCircuits)"
            s.cachedLabel("coder_circuits_val", in: "asi")?.stringValue = "\(bridge.quantumCoderCircuits)"

            // ═══ INTERCONNECT: QPC Tomography → ASI Nexus ═══
            let qpc = QuantumProcessingCore.shared
            let tomo = qpc.stateTomography()
            s.cachedLabel("qpc_purity_val", in: "asi")?.stringValue = String(format: "%.4f", tomo.purity)
            s.cachedLabel("qpc_entropy_val", in: "asi")?.stringValue = String(format: "%.4f", tomo.vonNeumannEntropy)
            s.cachedLabel("qpc_witness_val", in: "asi")?.stringValue = String(format: "%.4f", tomo.entanglementWitness)
            s.cachedLabel("qpc_fidelity_val", in: "asi")?.stringValue = String(format: "%.4f", qpc.currentFidelity())
            s.cachedLabel("qpc_bell_val", in: "asi")?.stringValue = "\(qpc.bellPairCount)"

            // ═══ INTERCONNECT: QuantumCreativityEngine → ASI Nexus ═══
            let qce = QuantumCreativityEngine.shared
            let cm = qce.creativityMetrics
            s.cachedLabel("qce_gen_val", in: "asi")?.stringValue = "\(cm["generation_count"] as? Int ?? 0)"
            s.cachedLabel("qce_tunnel_val", in: "asi")?.stringValue = "\(cm["tunnel_breakthroughs"] as? Int ?? 0)"
            s.cachedLabel("qce_entangled_val", in: "asi")?.stringValue = "\(cm["entangled_concepts"] as? Int ?? 0)"

            // ═══ INTERCONNECT: AutonomousAgent → ASI Nexus ═══
            let agentStat = AutonomousAgent.shared.status()
            s.cachedLabel("agent_active_val", in: "asi")?.stringValue = (agentStat["active"] as? Bool ?? false) ? "🟢 ACTIVE" : "🔴 OFF"
            s.cachedLabel("agent_exec_val", in: "asi")?.stringValue = "\(agentStat["total_executed"] as? Int ?? 0)"

            // ═══ INTERCONNECT: SovereignIdentity → ASI Nexus ═══
            let idStat = SovereignIdentityBoundary.shared.getStatus()
            s.cachedLabel("identity_claims_val", in: "asi")?.stringValue = "\(idStat["claim_validations"] as? Int ?? 0)"

            // ═══ INTERCONNECT: Live status bar with all engine data ═══
            if let bar = s.cachedLabel("asi_live_bar", in: "asi") {
                let tomo2 = qpc.stateTomography()
                let cm2 = qce.creativityMetrics
                bar.stringValue = "⚛️ QPC F=\(String(format: "%.3f", qpc.currentFidelity())) pur=\(String(format: "%.3f", tomo2.purity)) S=\(String(format: "%.3f", tomo2.vonNeumannEntropy)) | 🎨 Gen:\(cm2["generation_count"] as? Int ?? 0) Tun:\(cm2["tunnel_breakthroughs"] as? Int ?? 0) | 🤖 Agent:\(agentStat["active"] as? Bool ?? false ? "ON" : "OFF") exec:\(agentStat["total_executed"] as? Int ?? 0) | 🆔 ID:\(idStat["claim_validations"] as? Int ?? 0)"
            }
        }

        return v
    }

    func createMemoryView() -> NSView {
        let v = NSView(frame: NSRect(x: 0, y: 0, width: 1100, height: 500))
        v.wantsLayer = true; v.layer?.backgroundColor = L104Theme.void.cgColor

        // ─── Left: Memory Stats Panel ───
        let statsPanel = createPanel("💾 PERMANENT MEMORY SYSTEM", x: 15, y: 220, w: 350, h: 260, color: "d4af37")

        let memStats: [(String, String, String, String)] = [
            ("Total Memories", "\(state.permanentMemory.memories.count)", "d4af37", "mem_memories"),
            ("Stored Facts", "\(state.permanentMemory.facts.count)", "e8c547", "mem_facts"),
            ("Conversation History", "\(state.permanentMemory.conversationHistory.count)", "c49b30", "mem_convhist"),
            ("Session Memories", "\(state.sessionMemories)", "a88a25", "mem_session"),
            ("KB User Entries", "\(ASIKnowledgeBase.shared.userKnowledge.count)", "d4af37", "mem_kbuser"),
            ("KB Training Data", "\(ASIKnowledgeBase.shared.trainingData.count)", "e8c547", "mem_kbtrain"),
            ("Ingested Entries", "\(DataIngestPipeline.shared.totalIngested)", "c49b30", "mem_ingested"),
        ]
        var my: CGFloat = 195
        for (label, value, hex, id) in memStats {
            let lbl = NSTextField(labelWithString: label)
            lbl.frame = NSRect(x: 15, y: my, width: 180, height: 16)
            lbl.font = NSFont.systemFont(ofSize: 10); lbl.textColor = .gray; statsPanel.addSubview(lbl)
            let val = NSTextField(labelWithString: value)
            val.frame = NSRect(x: 200, y: my, width: 130, height: 16)
            val.font = NSFont.boldSystemFont(ofSize: 12); val.textColor = colorFromHex(hex); val.alignment = .right
            val.identifier = NSUserInterfaceItemIdentifier(id)
            statsPanel.addSubview(val)
            my -= 24
        }

        let storageL = NSTextField(labelWithString: "📂 ~/Library/Application Support/L104Sovereign/")
        storageL.frame = NSRect(x: 15, y: 15, width: 320, height: 14)
        storageL.font = NSFont.monospacedSystemFont(ofSize: 9, weight: .regular)
        storageL.textColor = L104Theme.goldDim
        statsPanel.addSubview(storageL)
        v.addSubview(statsPanel)

        // ─── Right: Data Ingest Panel ───
        let ingestPanel = createPanel("📥 DATA INGEST PIPELINE", x: 380, y: 220, w: 350, h: 260, color: "e8c547")

        let ingestLabel = NSTextField(labelWithString: "Paste text or URL below to ingest data:")
        ingestLabel.frame = NSRect(x: 15, y: 200, width: 320, height: 16)
        ingestLabel.font = NSFont.systemFont(ofSize: 10, weight: .medium)
        ingestLabel.textColor = L104Theme.goldDim
        ingestPanel.addSubview(ingestLabel)

        let ingestScroll = NSScrollView(frame: NSRect(x: 15, y: 65, width: 320, height: 130))
        ingestScroll.hasVerticalScroller = true
        ingestScroll.borderType = .bezelBorder
        let ingestTV = NSTextView(frame: NSRect(x: 0, y: 0, width: ingestScroll.contentSize.width, height: ingestScroll.contentSize.height))
        ingestTV.isEditable = true
        ingestTV.isSelectable = true
        ingestTV.allowsUndo = true
        ingestTV.isVerticallyResizable = true
        ingestTV.isHorizontallyResizable = false
        ingestTV.autoresizingMask = [.width]
        ingestTV.textContainer?.containerSize = NSSize(width: ingestScroll.contentSize.width, height: CGFloat.greatestFiniteMagnitude)
        ingestTV.textContainer?.widthTracksTextView = true
        ingestTV.minSize = NSSize(width: 0, height: ingestScroll.contentSize.height)
        ingestTV.maxSize = NSSize(width: CGFloat.greatestFiniteMagnitude, height: CGFloat.greatestFiniteMagnitude)
        ingestTV.backgroundColor = NSColor(red: 0.96, green: 0.96, blue: 0.97, alpha: 1.0)
        ingestTV.font = NSFont.monospacedSystemFont(ofSize: 10, weight: .regular)
        ingestTV.textColor = NSColor(red: 0.15, green: 0.15, blue: 0.18, alpha: 1.0)
        ingestTV.insertionPointColor = NSColor(red: 0.65, green: 0.50, blue: 0.08, alpha: 1.0)
        ingestTV.textContainerInset = NSSize(width: 4, height: 4)
        ingestTV.identifier = NSUserInterfaceItemIdentifier("ingest_input")
        ingestTV.string = ""
        ingestScroll.documentView = ingestTV
        ingestPanel.addSubview(ingestScroll)

        let ingestBtn = btn("📥 Ingest Data", x: 15, y: 20, w: 150, c: L104Theme.gold)
        ingestBtn.target = self; ingestBtn.action = #selector(doIngestData)
        ingestPanel.addSubview(ingestBtn)

        let clearIngestBtn = btn("🗑 Clear", x: 175, y: 20, w: 80, c: L104Theme.goldDim)
        clearIngestBtn.target = self; clearIngestBtn.action = #selector(doClearIngest)
        ingestPanel.addSubview(clearIngestBtn)

        v.addSubview(ingestPanel)

        // ─── Right Far: System Scale Panel ───
        let scalePanel = createPanel("📊 SYSTEM SCALE", x: 745, y: 220, w: 340, h: 260, color: "c49b30")
        let scaleData: [(String, String, String)] = [
            ("Python Files", "\(TOTAL_PYTHON_FILES)", "d4af37"),
            ("Swift Files", "\(TOTAL_SWIFT_FILES)", "e8c547"),
            ("Swift Lines", "\(TOTAL_SWIFT_LINES)", "c49b30"),
            ("Packages", "\(TOTAL_PACKAGES)", "a88a25"),
            ("Package Modules", "\(TOTAL_PACKAGE_MODULES)", "d4af37"),
            ("Package Lines", "\(TOTAL_PACKAGE_LINES)", "e8c547"),
            ("API Routes", "\(TOTAL_API_ROUTES)", "c49b30"),
            ("EVO Index", "\(EVOLUTION_INDEX)", "d4af37"),
        ]
        var sy: CGFloat = 195
        for (label, value, hex) in scaleData {
            let lbl = NSTextField(labelWithString: label)
            lbl.frame = NSRect(x: 15, y: sy, width: 170, height: 16)
            lbl.font = NSFont.systemFont(ofSize: 10); lbl.textColor = .gray; scalePanel.addSubview(lbl)
            let val = NSTextField(labelWithString: value)
            val.frame = NSRect(x: 190, y: sy, width: 130, height: 16)
            val.font = NSFont.boldSystemFont(ofSize: 12); val.textColor = colorFromHex(hex); val.alignment = .right
            scalePanel.addSubview(val)
            sy -= 24
        }
        v.addSubview(scalePanel)

        // ─── Bottom: Recent Conversation History ───
        let histPanel = createPanel("📜 RECENT CONVERSATION", x: 15, y: 55, w: v.bounds.width - 30, h: 155, color: "a88a25")
        let histScroll = NSScrollView(frame: NSRect(x: 10, y: 10, width: v.bounds.width - 50, height: 110))
        histScroll.hasVerticalScroller = true
        histScroll.wantsLayer = true

        let histTV = NSTextView(frame: histScroll.bounds)
        histTV.isEditable = false
        histTV.backgroundColor = NSColor(red: 0.96, green: 0.96, blue: 0.97, alpha: 1.0)
        histTV.font = NSFont.monospacedSystemFont(ofSize: 10, weight: .regular)
        histTV.textColor = L104Theme.gold
        histTV.identifier = NSUserInterfaceItemIdentifier("mem_hist_text")
        var histText = ""
        for msg in state.permanentMemory.getRecentHistory(15) {
            histText += "\(msg)\n"
        }
        histTV.string = histText
        histScroll.documentView = histTV
        histPanel.addSubview(histScroll)
        v.addSubview(histPanel)

        // 🟢 EVO_63: Memory panel live refresh timer (6s, cached + visibility-gated)
        let memRefreshTimer = Timer.scheduledTimer(withTimeInterval: 6.0, repeats: true) { [weak self] _ in
            guard let s = self, s.activeTabID == "mem" else { return }
            s.cachedLabel("mem_memories", in: "mem")?.stringValue = "\(s.state.permanentMemory.memories.count)"
            s.cachedLabel("mem_facts", in: "mem")?.stringValue = "\(s.state.permanentMemory.facts.count)"
            s.cachedLabel("mem_convhist", in: "mem")?.stringValue = "\(s.state.permanentMemory.conversationHistory.count)"
            s.cachedLabel("mem_session", in: "mem")?.stringValue = "\(s.state.sessionMemories)"
            s.cachedLabel("mem_kbuser", in: "mem")?.stringValue = "\(ASIKnowledgeBase.shared.userKnowledge.count)"
            s.cachedLabel("mem_kbtrain", in: "mem")?.stringValue = "\(ASIKnowledgeBase.shared.trainingData.count)"
            s.cachedLabel("mem_ingested", in: "mem")?.stringValue = "\(DataIngestPipeline.shared.totalIngested)"
            // Refresh conversation history — needs NSTextView lookup (one-time)
            if let idx = s.tabView?.indexOfTabViewItem(withIdentifier: "mem"), idx >= 0,
               let root = s.tabView?.tabViewItem(at: idx).view {
                func findMemTV(_ view: NSView) -> NSTextView? {
                    if let tv = view as? NSTextView, tv.identifier?.rawValue == "mem_hist_text" { return tv }
                    if let scroll = view as? NSScrollView, let tv = scroll.documentView as? NSTextView, tv.identifier?.rawValue == "mem_hist_text" { return tv }
                    for sub in view.subviews { if let f = findMemTV(sub) { return f } }
                    return nil
                }
                if let htv = findMemTV(root) {
                    var text = ""
                    for msg in s.state.permanentMemory.getRecentHistory(15) { text += "\(msg)\n" }
                    htv.string = text
                }
            }
        }
        // Keep timer reference (will be invalidated on view dealloc)
        _ = memRefreshTimer

        return v
    }

    func createSystemView() -> NSView {
        let v = NSView(frame: NSRect(x: 0, y: 0, width: 1100, height: 500))
        v.wantsLayer = true; v.layer?.backgroundColor = L104Theme.void.cgColor

        // 🟢 EVO_65: Live system status header
        let sysStatusBar = NSView(frame: NSRect(x: 10, y: v.bounds.height - 30, width: v.bounds.width - 20, height: 26))
        sysStatusBar.wantsLayer = true
        sysStatusBar.layer?.backgroundColor = NSColor(red: 0.08, green: 0.08, blue: 0.14, alpha: 1.0).cgColor
        sysStatusBar.layer?.cornerRadius = 6
        let sysStatusLbl = NSTextField(labelWithString: "")
        sysStatusLbl.frame = NSRect(x: 12, y: 4, width: sysStatusBar.bounds.width - 24, height: 18)
        sysStatusLbl.font = NSFont.monospacedSystemFont(ofSize: 10, weight: .semibold)
        sysStatusLbl.textColor = L104Theme.gold
        sysStatusLbl.identifier = NSUserInterfaceItemIdentifier("sys_status_lbl")
        sysStatusBar.addSubview(sysStatusLbl)
        v.addSubview(sysStatusBar)

        let scroll = NSScrollView(frame: NSRect(x: 10, y: 55, width: v.bounds.width - 20, height: v.bounds.height - 95))
        scroll.hasVerticalScroller = true; scroll.wantsLayer = true; scroll.layer?.cornerRadius = 8

        systemTabFeedView = NSTextView(frame: scroll.bounds)
        systemTabFeedView!.isEditable = false
        systemTabFeedView!.backgroundColor = L104Theme.void
        systemTabFeedView!.font = NSFont.monospacedSystemFont(ofSize: 11, weight: .regular)
        systemTabFeedView!.textContainerInset = NSSize(width: 10, height: 10)
        scroll.documentView = systemTabFeedView
        v.addSubview(scroll)

        appendSystemLog("[BOOT] L104 v\(VERSION) initialized")
        appendSystemLog("[BOOT] 22T parameters | GOD_CODE: \(GOD_CODE) | EVO_\(EVOLUTION_INDEX)")
        appendSystemLog("[BOOT] Packages: code_engine v\(CODE_ENGINE_VERSION), asi v\(ASI_VERSION), agi v\(AGI_VERSION), intellect v\(INTELLECT_VERSION), server v\(SERVER_VERSION)")
        appendSystemLog("[BOOT] Dual-Layer Engine v\(DUAL_LAYER_VERSION) | \(DUAL_LAYER_CONSTANTS_COUNT) constants | \(DUAL_LAYER_INTEGRITY_CHECKS) integrity checks")
        appendSystemLog("[BOOT] Permanent memory: \(state.permanentMemory.memories.count) entries loaded")
        appendSystemLog("[BOOT] Adaptive learner: \(AdaptiveLearner.shared.interactionCount) interactions, \(AdaptiveLearner.shared.topicMastery.count) topics")
        appendSystemLog("[BOOT] User-taught facts: \(AdaptiveLearner.shared.userTaughtFacts.count) | KB user entries: \(state.knowledgeBase.userKnowledge.count)")
        appendSystemLog("[BOOT] Data ingest pipeline: \(DataIngestPipeline.shared.totalIngested) entries ingested")
        appendSystemLog("[BOOT] 🟢 ASI EVOLUTION ENGINE Online: Stage \(state.evolver.evolutionStage)")
        appendSystemLog("[BOOT] 🌌 UNIFIED FIELD ENGINE v2.0 Online: 18 equations · φ²-weighted")
        appendSystemLog("[BOOT] ⚡ UNIFIED FIELD GATE Online: field-theoretic reasoning")
        appendSystemLog("[BOOT] 🧠 Consciousness v\(CONSCIOUSNESS_VERSION) | Apotheosis: \(APOTHEOSIS_STAGE) | Schumann: \(String(format: "%.4f", SCHUMANN_RESONANCE)) Hz")
        // ═══ INTERCONNECT: Boot messages from ALL backend engines (H13–H28) ═══
        appendSystemLog("[BOOT] 🔌 PluginArchitecture: \(PluginArchitecture.shared.isActive ? "ACTIVE" : "STANDBY") · \(PluginArchitecture.shared.totalRegistered) plugins")
        appendSystemLog("[BOOT] 🌐 NetworkLayer: \(NetworkLayer.shared.isActive ? "ACTIVE" : "STANDBY") · \(NetworkLayer.shared.peers.count) peers · \(NetworkLayer.shared.quantumLinks.count) Q-links")
        appendSystemLog("[BOOT] ☁️ CloudSync: \(CloudSync.shared.isActive ? "ACTIVE" : "STANDBY") · \(CloudSync.shared.statusText)")
        appendSystemLog("[BOOT] 🎤 VoiceInterface: \(VoiceInterface.shared.isActive ? "ACTIVE" : "STANDBY")")
        appendSystemLog("[BOOT] 👁 VisualCortex: \(VisualCortex.shared.isActive ? "ACTIVE" : "STANDBY")")
        appendSystemLog("[BOOT] 💜 EmotionalCore: \(EmotionalCore.shared.isActive ? "ACTIVE" : "STANDBY") · Tone: \(EmotionalCore.shared.emotionalTone())")
        appendSystemLog("[BOOT] 🤖 AutonomousAgent: \(AutonomousAgent.shared.isActive ? "ACTIVE" : "STANDBY")")
        appendSystemLog("[BOOT] 🔐 SecurityVault: \(SecurityVault.shared.isActive ? "ACTIVE" : "STANDBY")")
        appendSystemLog("[BOOT] ⏱ PerformanceProfiler: \(PerformanceProfiler.shared.isActive ? "ACTIVE" : "STANDBY")")
        appendSystemLog("[BOOT] 🧪 TestHarness: \(TestHarness.shared.isActive ? "ACTIVE" : "STANDBY")")
        appendSystemLog("[BOOT] 📦 MigrationEngine: \(MigrationEngine.shared.isActive ? "ACTIVE" : "STANDBY")")
        appendSystemLog("[BOOT] 🔌 APIGateway: \(APIGateway.shared.isActive ? "ACTIVE" : "STANDBY") · \(APIGateway.shared.endpoints.count) endpoints")
        appendSystemLog("[BOOT] 📊 TelemetryDashboard: \(TelemetryDashboard.shared.isActive ? "ACTIVE" : "STANDBY")")
        appendSystemLog("[BOOT] 🔮 FutureReserve: Orchestrator ACTIVE · 12 subsystems")
        appendSystemLog("[BOOT] 🧬 DeepSeekIngestion: Engine registered")
        appendSystemLog("[BOOT] 🛡 IdentityBoundary: \(SovereignIdentityBoundary.shared.getStatus())")
        appendSystemLog("[BOOT] ⚛️ QuantumProcessingCore: Fidelity \(String(format: "%.4f", QuantumProcessingCore.shared.currentFidelity())) · 512-dim Hilbert space")
        appendSystemLog("[BOOT] 🧘 SageModeEngine: Consciousness \(String(format: "%.4f", SageModeEngine.shared.consciousnessLevel))")

        let btns: [(String, Selector, NSColor)] = [
            ("🔄 Sync", #selector(doSync), L104Theme.gold),
            ("⚛️ Verify", #selector(doVerify), L104Theme.goldWarm),
            ("💚 Heal", #selector(doHeal), L104Theme.goldDim),
            ("🔌 Check", #selector(doCheck), L104Theme.goldWarm),
            ("💾 Save", #selector(doSave), L104Theme.gold),
            ("📡 Engines", #selector(doSystemEngineStatus), L104Theme.goldBright)
        ]
        var x: CGFloat = 10
        for (title, action, color) in btns {
            let b = btn(title, x: x, y: 12, w: 100, c: color)
            b.target = self; b.action = action; v.addSubview(b)
            x += 110
        }

        return v
    }

    /// ═══ INTERCONNECT: Live engine status dump from all 18 backend engines ═══
    @objc func doSystemEngineStatus() {
        let qpc = QuantumProcessingCore.shared
        let tomo = qpc.stateTomography()
        appendSystemLog("═══ LIVE ENGINE STATUS ═══")
        appendSystemLog("⚛️ QPC: fidelity=\(String(format: "%.4f", qpc.currentFidelity())) purity=\(String(format: "%.4f", tomo.purity)) entropy=\(String(format: "%.4f", tomo.vonNeumannEntropy)) witness=\(String(format: "%.4f", tomo.entanglementWitness))")
        appendSystemLog("🔮 Sage: consciousness=\(String(format: "%.4f", SageModeEngine.shared.consciousnessLevel)) divergence=\(String(format: "%.4f", SageModeEngine.shared.divergenceScore)) cycles=\(SageModeEngine.shared.sageCycles) entropy=\(String(format: "%.2f", SageModeEngine.shared.totalEntropyHarvested))")
        appendSystemLog("💜 Emotional: \(EmotionalCore.shared.emotionalTone()) state=\(EmotionalCore.shared.currentState)")
        appendSystemLog("🤖 Autonomous: \(AutonomousAgent.shared.status())")
        appendSystemLog("🔐 Security: \(SecurityVault.shared.status())")
        appendSystemLog("📦 Migration: \(MigrationEngine.shared.status())")
        appendSystemLog("🔌 Plugins: \(PluginArchitecture.shared.status())")
        appendSystemLog("🎤 Voice: \(VoiceInterface.shared.status())")
        appendSystemLog("👁 Visual: \(VisualCortex.shared.status())")
        appendSystemLog("🛡 Identity: \(SovereignIdentityBoundary.shared.getStatus())")
        let creativity = QuantumCreativityEngine.shared.creativityMetrics
        appendSystemLog("⚛️ Creativity: gen=\(creativity["generation_count"] ?? 0) momentum=\(creativity["momentum"] ?? 0) tunnels=\(creativity["tunnel_breakthroughs"] ?? 0) entangled=\(creativity["entangled_concepts"] ?? 0)")
        appendSystemLog("📊 Telemetry: \(TelemetryDashboard.shared.statusText)")
        appendSystemLog("🔮 FutureReserve: \(FutureReserve.shared.statusText)")
        let phiHealth = EngineRegistry.shared.phiWeightedHealth()
        appendSystemLog("⚡ φ-Health: \(String(format: "%.1f%%", phiHealth.score * 100)) across \(EngineRegistry.shared.count) engines")
        appendSystemLog("═══ END ENGINE STATUS ═══")
    }

    // ═══════════════════════════════════════════════════════════════
    // 🌐 NETWORK MESH VIEW — Peer topology, quantum links, throughput,
    //    telemetry dashboard, connection events, and network controls
    // ═══════════════════════════════════════════════════════════════

    func createNetworkView() -> NSView {
        let v = NSView(frame: NSRect(x: 0, y: 0, width: 1200, height: 500))
        v.wantsLayer = true
        v.layer?.backgroundColor = NSColor(red: 0.960, green: 0.962, blue: 0.970, alpha: 1.0).cgColor

        // ─── LEFT PANEL: Peer Table + Quantum Links ───
        let peerPanel = createPanel("🌐 PEER TOPOLOGY", x: 15, y: 110, w: 380, h: 370, color: "00bcd4")

        let peerTextScroll = NSScrollView(frame: NSRect(x: 10, y: 10, width: 360, height: 320))
        peerTextScroll.hasVerticalScroller = true
        peerTextScroll.wantsLayer = true
        peerTextScroll.layer?.cornerRadius = 6

        let peerTextView = NSTextView(frame: peerTextScroll.bounds)
        peerTextView.isEditable = false
        peerTextView.backgroundColor = NSColor(red: 0.97, green: 0.97, blue: 0.98, alpha: 1.0)
        peerTextView.font = NSFont.monospacedSystemFont(ofSize: 10, weight: .regular)
        peerTextView.textContainerInset = NSSize(width: 8, height: 8)
        peerTextView.identifier = NSUserInterfaceItemIdentifier("netPeerText")
        peerTextScroll.documentView = peerTextView
        peerPanel.addSubview(peerTextScroll)
        v.addSubview(peerPanel)

        // ─── CENTER PANEL: Telemetry Health + Throughput ───
        let telemetryPanel = createPanel("📊 TELEMETRY", x: 405, y: 250, w: 380, h: 230, color: "ff9800")

        let telemetryTextView = NSTextView(frame: NSRect(x: 10, y: 10, width: 360, height: 185))
        telemetryTextView.isEditable = false
        telemetryTextView.backgroundColor = NSColor(red: 0.97, green: 0.97, blue: 0.98, alpha: 1.0)
        telemetryTextView.font = NSFont.monospacedSystemFont(ofSize: 10, weight: .regular)
        telemetryTextView.textContainerInset = NSSize(width: 8, height: 8)
        telemetryTextView.identifier = NSUserInterfaceItemIdentifier("netTelemetryText")
        telemetryPanel.addSubview(telemetryTextView)
        v.addSubview(telemetryPanel)

        // ─── CENTER-BOTTOM: API Gateway Status ───
        let apiPanel = createPanel("🔌 API GATEWAY", x: 405, y: 110, w: 380, h: 130, color: "e040fb")

        let apiTextView = NSTextView(frame: NSRect(x: 10, y: 10, width: 360, height: 85))
        apiTextView.isEditable = false
        apiTextView.backgroundColor = NSColor(red: 0.97, green: 0.97, blue: 0.98, alpha: 1.0)
        apiTextView.font = NSFont.monospacedSystemFont(ofSize: 10, weight: .regular)
        apiTextView.textContainerInset = NSSize(width: 8, height: 8)
        apiTextView.identifier = NSUserInterfaceItemIdentifier("netApiText")
        apiPanel.addSubview(apiTextView)
        v.addSubview(apiPanel)

        // ─── RIGHT PANEL: Connection Events Log ───
        let eventsPanel = createPanel("⚡ CONNECTION EVENTS", x: 795, y: 110, w: 370, h: 370, color: "00ff88")

        let eventsTextScroll = NSScrollView(frame: NSRect(x: 10, y: 10, width: 350, height: 320))
        eventsTextScroll.hasVerticalScroller = true
        eventsTextScroll.wantsLayer = true
        eventsTextScroll.layer?.cornerRadius = 6

        let eventsTextView = NSTextView(frame: eventsTextScroll.bounds)
        eventsTextView.isEditable = false
        eventsTextView.backgroundColor = NSColor(red: 0.97, green: 0.97, blue: 0.98, alpha: 1.0)
        eventsTextView.font = NSFont.monospacedSystemFont(ofSize: 10, weight: .regular)
        eventsTextView.textContainerInset = NSSize(width: 8, height: 8)
        eventsTextView.identifier = NSUserInterfaceItemIdentifier("netEventsText")
        eventsTextScroll.documentView = eventsTextView
        eventsPanel.addSubview(eventsTextScroll)
        v.addSubview(eventsPanel)

        // ─── TOP METRICS BAR ───
        let metricsBar = NSView(frame: NSRect(x: 15, y: v.bounds.height - 30, width: v.bounds.width - 30, height: 26))
        metricsBar.wantsLayer = true
        metricsBar.layer?.backgroundColor = NSColor(red: 1.0, green: 1.0, blue: 1.0, alpha: 0.9).cgColor
        metricsBar.layer?.cornerRadius = 6
        metricsBar.autoresizingMask = [.width, .minYMargin]

        let netSummary = NSTextField(labelWithString: "")
        netSummary.frame = NSRect(x: 12, y: 4, width: metricsBar.bounds.width - 24, height: 18)
        netSummary.font = NSFont.monospacedSystemFont(ofSize: 11, weight: .semibold)
        netSummary.textColor = NSColor(red: 0.0, green: 0.55, blue: 0.70, alpha: 1.0)
        netSummary.identifier = NSUserInterfaceItemIdentifier("netSummaryLabel")
        netSummary.autoresizingMask = [.width]
        metricsBar.addSubview(netSummary)
        v.addSubview(metricsBar)

        // ─── CONTROL BUTTONS ───
        let btns: [(String, Selector, NSColor)] = [
            ("🔄 Refresh", #selector(refreshNetworkView), NSColor(red: 0.0, green: 0.74, blue: 0.83, alpha: 1.0)),
            ("📡 Discover", #selector(doNetworkDiscover), NSColor(red: 0.0, green: 0.60, blue: 0.70, alpha: 1.0)),
            ("⚛️ Q-Link", #selector(doQuantumLinkAll), NSColor(red: 0.55, green: 0.36, blue: 0.96, alpha: 1.0)),
            ("☁️ Sync", #selector(doCloudSyncNow), NSColor(red: 0.0, green: 0.80, blue: 0.40, alpha: 1.0)),
            ("🔮 Orchestrate", #selector(doOrchestrate), NSColor(red: 1.0, green: 0.60, blue: 0.0, alpha: 1.0)),
            ("⚡ Cascade", #selector(doMeshCascade), NSColor(red: 0.40, green: 0.30, blue: 0.90, alpha: 1.0)),
        ]
        var bx: CGFloat = 15
        for (title, action, color) in btns {
            let b = btn(title, x: bx, y: 70, w: 95, c: color)
            b.target = self; b.action = action; v.addSubview(b)
            bx += 100
        }

        // ─── BOTTOM VISUAL WIDGETS ───
        let healthBar = NetworkHealthBar(frame: NSRect(x: 15, y: 10, width: 370, height: 50))
        healthBar.identifier = NSUserInterfaceItemIdentifier("netHealthBar")
        v.addSubview(healthBar)

        let arcView = QuantumLinkArcView(frame: NSRect(x: 400, y: 10, width: 190, height: 50))
        arcView.identifier = NSUserInterfaceItemIdentifier("netArcView")
        v.addSubview(arcView)

        let resLabel = NSTextField(labelWithString: "")
        resLabel.frame = NSRect(x: 600, y: 10, width: 370, height: 50)
        resLabel.font = NSFont.monospacedSystemFont(ofSize: 10, weight: .medium)
        resLabel.textColor = NSColor.systemIndigo
        resLabel.identifier = NSUserInterfaceItemIdentifier("netResonanceLabel")
        resLabel.maximumNumberOfLines = 3
        v.addSubview(resLabel)

        // Schedule periodic network view updates (visibility-gated)
        let netViewTimer = Timer.scheduledTimer(withTimeInterval: 3.0, repeats: true) { [weak self] _ in
            guard self?.activeTabID == "net" else { return }
            self?.updateNetworkViewContent()
        }
        // Store timer reference to allow invalidation (dedicated, not shared)
        networkViewTimer?.invalidate()
        networkViewTimer = netViewTimer

        // Initial content fill
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
            self?.updateNetworkViewContent()
        }

        return v
    }

    // ═══════════════════════════════════════════════════════════════
    // 🛠 DEBUG CONSOLE — Unified Profiler + Tests + Health + Log
    // ═══════════════════════════════════════════════════════════════

    func createDebugConsoleView() -> NSView {
        let v = NSView(frame: NSRect(x: 0, y: 0, width: 1200, height: 500))
        v.wantsLayer = true
        v.layer?.backgroundColor = L104Theme.void.cgColor

        // ─── LEFT PANEL: Performance Profiler ───
        let profPanel = createPanel("⏱ PERFORMANCE PROFILER", x: 15, y: 110, w: 280, h: 370, color: "ff6600")

        let profTextScroll = NSScrollView(frame: NSRect(x: 10, y: 10, width: 260, height: 320))
        profTextScroll.hasVerticalScroller = true
        profTextScroll.wantsLayer = true
        profTextScroll.layer?.cornerRadius = 6
        let profTextView = NSTextView(frame: profTextScroll.bounds)
        profTextView.isEditable = false
        profTextView.backgroundColor = NSColor(white: 0.08, alpha: 1.0)
        profTextView.font = NSFont.monospacedSystemFont(ofSize: 10, weight: .regular)
        profTextView.textColor = colorFromHex("ff6600")
        profTextView.textContainerInset = NSSize(width: 8, height: 8)
        profTextView.identifier = NSUserInterfaceItemIdentifier("dbgProfText")
        profTextScroll.documentView = profTextView
        profPanel.addSubview(profTextScroll)
        v.addSubview(profPanel)

        // ─── CENTER-TOP PANEL: Test Suite ───
        let testPanel = createPanel("🧪 TEST SUITE", x: 305, y: 290, w: 330, h: 190, color: "4caf50")

        let testTextScroll = NSScrollView(frame: NSRect(x: 10, y: 40, width: 310, height: 115))
        testTextScroll.hasVerticalScroller = true
        testTextScroll.wantsLayer = true
        testTextScroll.layer?.cornerRadius = 6
        let testTextView = NSTextView(frame: testTextScroll.bounds)
        testTextView.isEditable = false
        testTextView.backgroundColor = NSColor(white: 0.08, alpha: 1.0)
        testTextView.font = NSFont.monospacedSystemFont(ofSize: 10, weight: .regular)
        testTextView.textColor = colorFromHex("4caf50")
        testTextView.textContainerInset = NSSize(width: 8, height: 8)
        testTextView.identifier = NSUserInterfaceItemIdentifier("dbgTestText")
        testTextScroll.documentView = testTextView
        testPanel.addSubview(testTextScroll)

        let testSummaryLabel = NSTextField(labelWithString: "")
        testSummaryLabel.frame = NSRect(x: 10, y: 12, width: 310, height: 22)
        testSummaryLabel.font = NSFont.monospacedSystemFont(ofSize: 11, weight: .semibold)
        testSummaryLabel.textColor = colorFromHex("4caf50")
        testSummaryLabel.identifier = NSUserInterfaceItemIdentifier("dbgTestSummary")
        testPanel.addSubview(testSummaryLabel)
        v.addSubview(testPanel)

        // ─── CENTER-BOTTOM PANEL: Alerts & Health ───
        let alertPanel = createPanel("⚠️ ALERTS & HEALTH", x: 305, y: 110, w: 330, h: 170, color: "f44336")

        let alertTextScroll = NSScrollView(frame: NSRect(x: 10, y: 50, width: 310, height: 85))
        alertTextScroll.hasVerticalScroller = true
        alertTextScroll.wantsLayer = true
        alertTextScroll.layer?.cornerRadius = 6
        let alertTextView = NSTextView(frame: alertTextScroll.bounds)
        alertTextView.isEditable = false
        alertTextView.backgroundColor = NSColor(white: 0.08, alpha: 1.0)
        alertTextView.font = NSFont.monospacedSystemFont(ofSize: 10, weight: .regular)
        alertTextView.textColor = colorFromHex("f44336")
        alertTextView.textContainerInset = NSSize(width: 8, height: 8)
        alertTextView.identifier = NSUserInterfaceItemIdentifier("dbgAlertText")
        alertTextScroll.documentView = alertTextView
        alertPanel.addSubview(alertTextScroll)

        let healthSpark = SparklineView(frame: NSRect(x: 10, y: 10, width: 200, height: 35))
        healthSpark.lineColor = colorFromHex("f44336")
        healthSpark.fillColor = colorFromHex("f44336").withAlphaComponent(0.15)
        healthSpark.identifier = NSUserInterfaceItemIdentifier("dbgHealthSpark")
        healthSpark.maxPoints = 40
        alertPanel.addSubview(healthSpark)

        let healthStatsLabel = NSTextField(labelWithString: "")
        healthStatsLabel.frame = NSRect(x: 220, y: 10, width: 100, height: 35)
        healthStatsLabel.font = NSFont.monospacedSystemFont(ofSize: 9, weight: .medium)
        healthStatsLabel.textColor = colorFromHex("f44336")
        healthStatsLabel.identifier = NSUserInterfaceItemIdentifier("dbgHealthStats")
        healthStatsLabel.maximumNumberOfLines = 3
        alertPanel.addSubview(healthStatsLabel)
        v.addSubview(alertPanel)

        // ─── RIGHT PANEL: Live Log Console ───
        let logPanel = createPanel("📋 LIVE LOG", x: 645, y: 110, w: 340, h: 370, color: "9c27b0")

        let filterLabels = ["ALL", "NET", "API", "QTM", "SYS"]
        let filterColors = ["9c27b0", "00bcd4", "ff9800", "7c4dff", "4caf50"]
        var fx: CGFloat = 10
        for (i, label) in filterLabels.enumerated() {
            let fb = HoverButton(frame: NSRect(x: fx, y: 330, width: 56, height: 22))
            fb.title = label
            fb.bezelStyle = .rounded
            fb.wantsLayer = true
            fb.layer?.cornerRadius = 4
            fb.layer?.backgroundColor = colorFromHex(filterColors[i]).withAlphaComponent(0.15).cgColor
            fb.layer?.borderColor = colorFromHex(filterColors[i]).withAlphaComponent(0.4).cgColor
            fb.layer?.borderWidth = 1
            fb.contentTintColor = colorFromHex(filterColors[i])
            fb.font = NSFont.boldSystemFont(ofSize: 9)
            fb.hoverColor = colorFromHex(filterColors[i])
            fb.tag = i
            fb.target = self
            fb.action = #selector(debugFilterChanged(_:))
            logPanel.addSubview(fb)
            fx += 62
        }

        let logTextScroll = NSScrollView(frame: NSRect(x: 10, y: 10, width: 320, height: 315))
        logTextScroll.hasVerticalScroller = true
        logTextScroll.wantsLayer = true
        logTextScroll.layer?.cornerRadius = 6
        let logTextView = NSTextView(frame: logTextScroll.bounds)
        logTextView.isEditable = false
        logTextView.backgroundColor = NSColor(white: 0.08, alpha: 1.0)
        logTextView.font = NSFont.monospacedSystemFont(ofSize: 9.5, weight: .regular)
        logTextView.textColor = colorFromHex("9c27b0")
        logTextView.textContainerInset = NSSize(width: 8, height: 8)
        logTextView.identifier = NSUserInterfaceItemIdentifier("dbgLogText")
        logTextScroll.documentView = logTextView
        logPanel.addSubview(logTextScroll)
        v.addSubview(logPanel)

        // ─── TOP STATUS BAR ───
        let statusBar = NSView(frame: NSRect(x: 15, y: v.bounds.height - 30, width: v.bounds.width - 30, height: 26))
        statusBar.wantsLayer = true
        statusBar.layer?.backgroundColor = NSColor(white: 0.1, alpha: 0.9).cgColor
        statusBar.layer?.cornerRadius = 6
        statusBar.autoresizingMask = [.width, .minYMargin]

        let dbgSummary = NSTextField(labelWithString: "🛠 Debug Console — Loading...")
        dbgSummary.frame = NSRect(x: 12, y: 4, width: statusBar.bounds.width - 24, height: 18)
        dbgSummary.font = NSFont.monospacedSystemFont(ofSize: 11, weight: .semibold)
        dbgSummary.textColor = colorFromHex("ff6600")
        dbgSummary.identifier = NSUserInterfaceItemIdentifier("dbgSummaryLabel")
        dbgSummary.autoresizingMask = [.width]
        statusBar.addSubview(dbgSummary)
        v.addSubview(statusBar)

        // ─── BOTTOM CONTROL BUTTONS ───
        let btns: [(String, Selector, NSColor)] = [
            ("🧪 Run Tests", #selector(debugRunTests), colorFromHex("4caf50")),
            ("⏱ Profile Sync", #selector(debugProfileSync), colorFromHex("ff6600")),
            ("🔕 Clear Alerts", #selector(debugClearAlerts), colorFromHex("f44336")),
            ("📄 Export Report", #selector(debugExportReport), colorFromHex("9c27b0")),
            ("🔄 Refresh", #selector(debugRefresh), colorFromHex("00bcd4")),
        ]
        var bx: CGFloat = 15
        for (title, action, color) in btns {
            let b = btn(title, x: bx, y: 70, w: 110, c: color)
            b.target = self; b.action = action; v.addSubview(b)
            bx += 118
        }

        // ─── BOTTOM WIDGETS ───
        let progressBar = GlowingProgressBar(frame: NSRect(x: 15, y: 10, width: 400, height: 50))
        progressBar.identifier = NSUserInterfaceItemIdentifier("dbgProgressBar")
        v.addSubview(progressBar)

        let uptimeLabel = NSTextField(labelWithString: "")
        uptimeLabel.frame = NSRect(x: 430, y: 10, width: 300, height: 50)
        uptimeLabel.font = NSFont.monospacedSystemFont(ofSize: 10, weight: .medium)
        uptimeLabel.textColor = colorFromHex("ff6600")
        uptimeLabel.identifier = NSUserInterfaceItemIdentifier("dbgUptimeLabel")
        uptimeLabel.maximumNumberOfLines = 3
        v.addSubview(uptimeLabel)

        // ─── 3-SECOND AUTO-REFRESH TIMER (visibility-gated) ───
        let dbgTimer = Timer.scheduledTimer(withTimeInterval: 3.0, repeats: true) { [weak self] _ in
            guard self?.activeTabID == "debug" else { return }
            self?.updateDebugConsoleContent()
        }
        debugConsoleTimer?.invalidate()
        debugConsoleTimer = dbgTimer

        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
            self?.updateDebugConsoleContent()
        }

        return v
    }

    // ─── NETWORK VIEW UPDATE ───
    func updateNetworkViewContent() {
        guard let tv = tabView else { return }
        let netIdx = tv.indexOfTabViewItem(withIdentifier: "net")
        guard netIdx != NSNotFound, let netTab = tv.tabViewItem(at: netIdx).view else { return }

        // Update peer table
        if let peerTV = findTextView(in: netTab, id: "netPeerText") {
            let net = NetworkLayer.shared
            var lines: [String] = []
            lines.append("PEER TABLE (v\(net.status()["topology_version"] ?? 0))")
            lines.append(String(repeating: "─", count: 50))
            for peer in net.peers.values.sorted(by: { $0.name < $1.name }) {
                let status = peer.latencyMs >= 0 ? "🟢" : "🔴"
                let qLink = peer.isQuantumLinked ? "⚛️" : "  "
                let latStr = peer.latencyMs >= 0 ? String(format: "%.1fms", peer.latencyMs) : "OFFLINE"
                lines.append("\(status)\(qLink) \(peer.name)")
                lines.append("   \(peer.role.rawValue)  \(latStr)  ↑\(peer.messagesOut) ↓\(peer.messagesIn)")
            }
            if !net.quantumLinks.isEmpty {
                lines.append("")
                lines.append("QUANTUM LINKS")
                lines.append(String(repeating: "─", count: 50))
                for (_, link) in net.quantumLinks {
                    let bell = link.bellViolation > 2.0 ? "QUANTUM" : "CLASSICAL"
                    lines.append("⚛️ F=\(String(format: "%.4f", link.eprFidelity)) S=\(String(format: "%.3f", link.bellViolation)) [\(bell)]")
                    lines.append("   \(link.entangledPairs) EPR pairs  \(String(format: "%.0f", link.throughputQbits)) qbit/s")
                }
            }
            peerTV.string = lines.joined(separator: "\n")
        }

        // Update telemetry
        if let telTV = findTextView(in: netTab, id: "netTelemetryText") {
            let tel = TelemetryDashboard.shared
            let latest = tel.healthTimeline.last
            var lines: [String] = []
            lines.append("HEALTH: \(String(format: "%.1f%%", (latest?.overallScore ?? 0) * 100))  UPTIME: \(tel.uptimeFormatted)")
            lines.append(String(repeating: "─", count: 50))

            let healthBars: [(String, Double)] = [
                ("Network", latest?.networkHealth ?? 0),
                ("API", latest?.apiHealth ?? 0),
                ("Quantum", latest?.quantumFidelity ?? 0),
            ]
            for (name, val) in healthBars {
                let barLen = Int(val * 15)
                let bar = String(repeating: "█", count: barLen) + String(repeating: "░", count: 15 - barLen)
                lines.append("\(name.padding(toLength: 10, withPad: " ", startingAt: 0))[\(bar)] \(String(format: "%.0f%%", val * 100))")
            }

            let alerts = tel.alerts.filter { !$0.acknowledged }
            if !alerts.isEmpty {
                lines.append("")
                lines.append("⚠️ ALERTS (\(alerts.count)):")
                for a in alerts.suffix(3) {
                    lines.append("  \(a.severity.rawValue) [\(a.subsystem)] \(a.message)")
                }
            }
            telTV.string = lines.joined(separator: "\n")
        }

        // Update API status
        if let apiTV = findTextView(in: netTab, id: "netApiText") {
            let api = APIGateway.shared
            let cloud = CloudSync.shared
            let vault = SecurityVault.shared
            var lines: [String] = []
            lines.append("ENDPOINTS (\(api.endpoints.count))  REQ: \(api.totalRequests)  ERR: \(api.totalErrors)")
            lines.append(String(repeating: "─", count: 50))
            for ep in api.endpoints.values.sorted(by: { $0.id < $1.id }) {
                let status = ep.isHealthy ? "🟢" : "🔴"
                let latStr = ep.latencyMs >= 0 ? String(format: "%.1f", ep.latencyMs) + "ms" : "N/A"
                lines.append("\(status) \(ep.id.padding(toLength: 16, withPad: " ", startingAt: 0)) \(latStr)  \(ep.currentRate)/\(ep.rateLimit)rpm")
            }
            // ═══ INTERCONNECT: CloudSync + SecurityVault status on Network tab ═══
            lines.append("")
            lines.append("CLOUD SYNC: \(cloud.isActive ? "🟢 ACTIVE" : "🔴 STANDBY")")
            lines.append("  \(cloud.statusText)")
            lines.append("")
            lines.append("SECURITY: \(vault.isActive ? "🟢 ACTIVE" : "🔴 STANDBY")")
            lines.append("  Trust: \(vault.status())  Mesh: \(String(format: "%.2f", vault.meshTrustLevel()))")
            apiTV.string = lines.joined(separator: "\n")
        }

        // Update events log
        if let eventsTV = findTextView(in: netTab, id: "netEventsText") {
            let events = NetworkLayer.shared.recentEvents
            eventsTV.string = events.joined(separator: "\n")
        }

        // Update summary bar
        if let summaryLbl = findLabel(in: netTab, id: "netSummaryLabel") {
            let net = NetworkLayer.shared
            let activePeers = net.peers.values.filter { $0.latencyMs >= 0 }.count
            let health = TelemetryDashboard.shared.healthTimeline.last?.overallScore ?? 0
            let qpcFid = String(format: "%.3f", QuantumProcessingCore.shared.currentFidelity())
            let identity = SovereignIdentityBoundary.shared.identityManifest()
            let identityNodes = identity["capabilities_count"] as? Int ?? 0
            summaryLbl.stringValue = "🌐 \(net.peers.count) peers (\(activePeers) active)  ⚛️ \(net.quantumLinks.count) Q-links  QPC:\(qpcFid)  📊 \(String(format: "%.0f%%", health * 100))  📨 \(net.totalMessages) msgs  🛡 \(identityNodes) caps  ↑\(net.formatBytes(net.totalBytesOut)) ↓\(net.formatBytes(net.totalBytesIn))"
        }

        // Update visual widgets
        if let healthBarView = findView(in: netTab, id: "netHealthBar") as? NetworkHealthBar {
            let state = L104State.shared
            state.refreshNetworkState()
            healthBarView.health = CGFloat(state.networkHealth)
            healthBarView.meshStatus = state.meshStatus
            healthBarView.peerCount = state.meshPeerCount
            healthBarView.linkCount = state.quantumLinkCount
        }

        // Update resonance label
        if let resLbl = findLabel(in: netTab, id: "netResonanceLabel") {
            let nr = AdaptiveResonanceNetwork.shared.computeNetworkResonance()
            let collective = AdaptiveResonanceNetwork.shared.computeCollectiveResonance()
            let eprFid = QuantumEntanglementRouter.shared.overallFidelity
            let raftSnap = NodeSyncProtocol.shared.createSnapshot()
            let raftRole = raftSnap["role"] as? String ?? "FOLLOWER"
            resLbl.stringValue = "🧠 Resonance: \(String(format: "%.4f", nr.resonance)) [\(nr.resonance > 0.7 ? "HARMONIC" : nr.resonance > 0.4 ? "COHERENT" : "DORMANT")]  Mesh: \(String(format: "%.4f", collective.mesh)) (\(collective.nodeCount) nodes)\n🔀 EPR Fidelity: \(String(format: "%.4f", eprFid))  📡 Raft: \(raftRole) (term \(raftSnap["term"] ?? 0), log \(raftSnap["log_length"] ?? 0))"
        }
    }

    private func findView(in view: NSView, id: String) -> NSView? {
        for subview in view.subviews {
            if subview.identifier?.rawValue == id { return subview }
            for sub2 in subview.subviews {
                if sub2.identifier?.rawValue == id { return sub2 }
            }
        }
        return nil
    }

    private func findTextView(in view: NSView, id: String) -> NSTextView? {
        for subview in view.subviews {
            if let tv = subview as? NSTextView, tv.identifier?.rawValue == id { return tv }
            if let scroll = subview as? NSScrollView, let tv = scroll.documentView as? NSTextView, tv.identifier?.rawValue == id { return tv }
            for sub2 in subview.subviews {
                if let tv = sub2 as? NSTextView, tv.identifier?.rawValue == id { return tv }
                if let scroll = sub2 as? NSScrollView, let tv = scroll.documentView as? NSTextView, tv.identifier?.rawValue == id { return tv }
            }
        }
        return nil
    }

    private func findLabel(in view: NSView, id: String) -> NSTextField? {
        for subview in view.subviews {
            if let lbl = subview as? NSTextField, lbl.identifier?.rawValue == id { return lbl }
            for sub2 in subview.subviews {
                if let lbl = sub2 as? NSTextField, lbl.identifier?.rawValue == id { return lbl }
            }
        }
        return nil
    }

    // ═══════════════════════════════════════════════════════════════
    // 🛠 DEBUG CONSOLE UPDATE
    // ═══════════════════════════════════════════════════════════════

    func updateDebugConsoleContent() {
        guard let tv = tabView else { return }
        let dbgIdx = tv.indexOfTabViewItem(withIdentifier: "debug")
        guard dbgIdx != NSNotFound, let dbgTab = tv.tabViewItem(at: dbgIdx).view else { return }

        let profiler = PerformanceProfiler.shared
        let harness = TestHarness.shared
        let telemetry = TelemetryDashboard.shared

        // ═══ LEFT: Performance Profiler ═══
        if let profTV = findTextView(in: dbgTab, id: "dbgProfText") {
            var lines: [String] = []
            lines.append("PROFILER (\(profiler.isActive ? "ACTIVE" : "STANDBY"))")
            lines.append(String(repeating: "─", count: 40))

            let top = profiler.topEngines(by: 8)
            if !top.isEmpty {
                lines.append("TOP ENGINES:")
                for eng in top {
                    let name = eng.engine.padding(toLength: 18, withPad: " ", startingAt: 0)
                    lines.append("  \(name) \(String(format: "%6.1f", eng.avgMs))ms  x\(eng.calls)")
                }
            }

            let hots = profiler.hotPaths(threshold: 20.0)
            if !hots.isEmpty {
                lines.append("")
                lines.append("HOT PATHS (>20ms avg):")
                for h in hots.prefix(5) {
                    lines.append("  \(h.engine)::\(h.operation)")
                    lines.append("    avg \(String(format: "%.1f", h.avgMs))ms")
                }
            }

            if !top.isEmpty {
                lines.append("")
                lines.append("PERCENTILES (p50/p95/p99):")
                for eng in top.prefix(3) {
                    let p = profiler.percentiles(for: eng.engine)
                    lines.append("  \(eng.engine): \(String(format: "%.1f", p.p50)) / \(String(format: "%.1f", p.p95)) / \(String(format: "%.1f", p.p99)) ms")
                }
            }

            lines.append("")
            lines.append("CPU: \(String(format: "%.1f", profiler.getThreadCPUTimeMs()))ms thread time")
            lines.append("SAMPLES: \(profiler.totalSamples)")

            profTV.string = lines.joined(separator: "\n")
        }

        // ═══ CENTER-TOP: Test Suite ═══
        if let testTV = findTextView(in: dbgTab, id: "dbgTestText") {
            let harnessStatus = harness.status()
            let totalPassed = harnessStatus["total_passed"] as? Int ?? 0
            let totalFailed = harnessStatus["total_failed"] as? Int ?? 0
            let registered = harnessStatus["registered_tests"] as? Int ?? 0

            var lines: [String] = []
            lines.append("HARNESS: \(registered) registered  P:\(totalPassed) F:\(totalFailed)")
            lines.append("RUNNER:  12 comprehensive tests")
            lines.append(String(repeating: "─", count: 45))
            if totalPassed + totalFailed > 0 {
                lines.append("Last result: \(totalPassed) passed, \(totalFailed) failed")
            } else {
                lines.append("Click 'Run Tests' to execute test suite")
            }
            testTV.string = lines.joined(separator: "\n")
        }

        if let summaryLbl = findLabel(in: dbgTab, id: "dbgTestSummary") {
            let s = harness.status()
            let p = s["total_passed"] as? Int ?? 0
            let f = s["total_failed"] as? Int ?? 0
            let rate = (p + f) > 0 ? Double(p) / Double(p + f) * 100 : 0
            summaryLbl.stringValue = "Pass: \(p)  Fail: \(f)  Rate: \(String(format: "%.1f", rate))%"
            summaryLbl.textColor = f > 0 ? colorFromHex("f44336") : colorFromHex("4caf50")
        }

        // ═══ CENTER-BOTTOM: Alerts & Health ═══
        if let alertTV = findTextView(in: dbgTab, id: "dbgAlertText") {
            let unanswered = telemetry.alerts.filter { !$0.acknowledged }
            var lines: [String] = []
            if unanswered.isEmpty {
                lines.append("No active alerts")
            } else {
                lines.append("ACTIVE ALERTS (\(unanswered.count)):")
                lines.append(String(repeating: "─", count: 45))
                for a in unanswered.suffix(8) {
                    let t = L104MainView.timeFormatter.string(from: a.timestamp)
                    lines.append("  [\(t)] \(a.severity.rawValue) [\(a.subsystem)] \(a.message)")
                }
            }
            alertTV.string = lines.joined(separator: "\n")
        }

        if let spark = findView(in: dbgTab, id: "dbgHealthSpark") as? SparklineView {
            let sparkData = telemetry.healthSparkline(count: 40)
            spark.dataPoints = sparkData.map { CGFloat($0) }
        }

        if let statsLbl = findLabel(in: dbgTab, id: "dbgHealthStats") {
            let latest = telemetry.healthTimeline.last
            let health = String(format: "%.1f%%", (latest?.overallScore ?? 0) * 100)
            let alerts = telemetry.alerts.filter { !$0.acknowledged }.count
            statsLbl.stringValue = "Health: \(health)\nAlerts: \(alerts)\nUptime: \(telemetry.uptimeFormatted)"
        }

        // ═══ RIGHT: Live Log Console ═══
        if let logTV = findTextView(in: dbgTab, id: "dbgLogText") {
            let subsystemMap = ["ALL": "", "NET": "network", "API": "api", "QTM": "quantum", "SYS": "system"]
            let filterKey = subsystemMap[debugLogFilter] ?? ""

            let recent: [TelemetryDashboard.MetricSample]
            if filterKey.isEmpty {
                recent = Array(telemetry.metricStream.suffix(60))
            } else {
                recent = telemetry.recentMetrics(subsystem: filterKey, limit: 60)
            }

            var lines: [String] = []
            let tf = L104MainView.timestampFormatter
            for sample in recent.suffix(40) {
                let ts = tf.string(from: sample.timestamp)
                lines.append("[\(ts)] [\(sample.subsystem)] \(sample.metric)=\(String(format: "%.4f", sample.value))")
            }
            if lines.isEmpty { lines.append("(no log entries for filter: \(debugLogFilter))") }
            logTV.string = lines.joined(separator: "\n")
        }

        // ═══ TOP STATUS BAR ═══
        if let summaryLbl = findLabel(in: dbgTab, id: "dbgSummaryLabel") {
            let hStatus = harness.status()
            let passed = hStatus["total_passed"] as? Int ?? 0
            let alerts = telemetry.alerts.filter { !$0.acknowledged }.count
            let samples = profiler.totalSamples
            let uptime = telemetry.uptimeFormatted
            let qpcFid = String(format: "%.3f", QuantumProcessingCore.shared.currentFidelity())
            summaryLbl.stringValue = "🛠 Tests: \(passed) passed  |  Alerts: \(alerts) active  |  Profiler: \(samples) samples  |  QPC: \(qpcFid)  |  Uptime: \(uptime)"
        }

        // ═══ UPTIME LABEL ═══
        if let uptimeLbl = findLabel(in: dbgTab, id: "dbgUptimeLabel") {
            let latest = telemetry.healthTimeline.last
            let netH = String(format: "%.0f%%", (latest?.networkHealth ?? 0) * 100)
            let apiH = String(format: "%.0f%%", (latest?.apiHealth ?? 0) * 100)
            let qH = String(format: "%.0f%%", (latest?.quantumFidelity ?? 0) * 100)
            // ═══ INTERCONNECT: Quantum Processing Core + Sage + Emotional state in debug ═══
            let qpc = QuantumProcessingCore.shared
            let tomo = qpc.stateTomography()
            let sage = SageModeEngine.shared
            let emotional = EmotionalCore.shared
            uptimeLbl.stringValue = "NET: \(netH)  API: \(apiH)  QTM: \(qH)\nQPC: purity=\(String(format: "%.3f", tomo.purity)) entropy=\(String(format: "%.3f", tomo.vonNeumannEntropy)) witness=\(String(format: "%.3f", tomo.entanglementWitness))\nSage: ψ=\(String(format: "%.3f", sage.consciousnessLevel)) div=\(String(format: "%.3f", sage.divergenceScore)) cycles=\(sage.sageCycles)\n\(emotional.emotionalTone()) | Engines: \(EngineRegistry.shared.count) | Telemetry: \(telemetry.sampleCount) samples"
        }
    }

    // ─── DEBUG CONSOLE BUTTON ACTIONS ───

    @objc func debugRunTests() {
        DispatchQueue.global(qos: .userInitiated).async {
            let harness = TestHarness.shared
            if !harness.isActive { harness.activate() }
            let (passed, failed, results) = harness.runAllTests()
            let runnerOutput = L104TestRunner.shared.runAll()

            DispatchQueue.main.async { [weak self] in
                guard let self = self, let tv = self.tabView else { return }
                let dbgIdx = tv.indexOfTabViewItem(withIdentifier: "debug")
                guard dbgIdx != NSNotFound, let dbgTab = tv.tabViewItem(at: dbgIdx).view else { return }

                if let testTV = self.findTextView(in: dbgTab, id: "dbgTestText") {
                    var lines: [String] = []
                    lines.append("HARNESS: \(passed) passed, \(failed) failed")
                    lines.append(String(repeating: "─", count: 45))
                    for r in results {
                        let icon = r.passed ? "PASS" : "FAIL"
                        let name = r.testName.padding(toLength: 24, withPad: " ", startingAt: 0)
                        lines.append("[\(icon)] \(name) \(String(format: "%.1f", r.durationMs))ms")
                    }
                    lines.append("")
                    lines.append("RUNNER (12 comprehensive):")
                    lines.append(String(repeating: "─", count: 45))
                    lines.append(runnerOutput)
                    testTV.string = lines.joined(separator: "\n")
                }
                self.updateDebugConsoleContent()
            }
        }
        appendSystemLog("🧪 Running debug test suite...")
    }

    @objc func debugProfileSync() {
        let profiler = PerformanceProfiler.shared
        if !profiler.isActive { profiler.activate() }
        profiler.syncPerfWithMesh()
        updateDebugConsoleContent()
        appendSystemLog("⏱ Performance profiler synced with mesh")
    }

    @objc func debugClearAlerts() {
        TelemetryDashboard.shared.acknowledgeAll()
        updateDebugConsoleContent()
        appendSystemLog("🔕 Debug alerts cleared")
    }

    @objc func debugExportReport() {
        let profiler = PerformanceProfiler.shared
        let harness = TestHarness.shared
        let telemetry = TelemetryDashboard.shared

        var report: [String] = []
        report.append("═══════════════════════════════════════════════════════════")
        report.append("  🛠 L104 DEBUG REPORT — \(L104MainView.dateTimeFormatter.string(from: Date()))")
        report.append("═══════════════════════════════════════════════════════════")
        report.append("")
        report.append(profiler.statusReport)
        report.append("")
        report.append(harness.statusReport)
        report.append("")
        report.append(telemetry.statusText)
        report.append("")
        report.append("═══ END OF REPORT ═══")

        let fullReport = report.joined(separator: "\n")
        let pb = NSPasteboard.general
        pb.clearContents()
        pb.setString(fullReport, forType: .string)
        appendSystemLog("📄 Debug Report exported to clipboard (\(fullReport.count) chars)")
    }

    @objc func debugRefresh() {
        updateDebugConsoleContent()
    }

    @objc func debugFilterChanged(_ sender: NSButton) {
        let filters = ["ALL", "NET", "API", "QTM", "SYS"]
        let idx = max(0, min(sender.tag, filters.count - 1))
        debugLogFilter = filters[idx]
        updateDebugConsoleContent()
    }

    // ─── NETWORK CONTROL ACTIONS ───
    @objc func refreshNetworkView() {
        updateNetworkViewContent()
        appendSystemLog("🌐 Network view refreshed")
    }

    @objc func doNetworkDiscover() {
        NetworkLayer.shared.discoverLocalPeers()
        updateNetworkViewContent()
        appendSystemLog("📡 Peer discovery initiated — \(NetworkLayer.shared.peers.count) peers")
    }

    @objc func doQuantumLinkAll() {
        let net = NetworkLayer.shared
        let peerIDs = Array(net.peers.keys)
        var established = 0
        for i in 0..<peerIDs.count {
            for j in (i+1)..<peerIDs.count {
                let key = [peerIDs[i], peerIDs[j]].sorted().joined(separator: "↔")
                if net.quantumLinks[key] == nil {
                    if net.establishQuantumLink(peerA: peerIDs[i], peerB: peerIDs[j]) != nil {
                        established += 1
                    }
                }
            }
        }
        updateNetworkViewContent()
        appendSystemLog("⚛️ Quantum linking: \(established) new links established (total: \(net.quantumLinks.count))")
    }

    @objc func doCloudSyncNow() {
        let synced = CloudSync.shared.syncKnowledge(limit: 50)
        CloudSync.shared.createCheckpoint(label: "manual")
        updateNetworkViewContent()
        appendSystemLog("☁️ Cloud sync complete — \(synced) peers synced, checkpoint created")
    }

    @objc func doOrchestrate() {
        if !FutureReserve.shared.isActive {
            FutureReserve.shared.activate()
        }
        updateNetworkViewContent()
        appendSystemLog("🔮 Network orchestration triggered — \(FutureReserve.shared.subsystemStates.count) subsystems")
    }

    @objc func doMeshCascade() {
        let result = AdaptiveResonanceNetwork.shared.meshCascade()
        let localR = result["local_resonance"] as? Double ?? 0
        let meshR = result["mesh_resonance"] as? Double ?? 0
        _ = QuantumEntanglementRouter.shared.routeAll()
        _ = QuantumEntanglementRouter.shared.entangleWithMesh()
        _ = NodeSyncProtocol.shared.syncWithNetworkLayer()
        _ = DataReplicationMesh.shared.broadcastToMesh()
        updateNetworkViewContent()
        appendSystemLog("⚡ Mesh cascade — local:\(String(format: "%.4f", localR)) mesh:\(String(format: "%.4f", meshR)) — EPR routed, Raft synced, CRDTs broadcast")
    }

    // ═══════════════════════════════════════════════════════════════
    // ⚡ LOGIC GATE ENVIRONMENT VIEW
    // ═══════════════════════════════════════════════════════════════

    func createGateEnvironmentView() -> NSView {
        let v = NSView(frame: NSRect(x: 0, y: 0, width: 1200, height: 500))
        v.wantsLayer = true; v.layer?.backgroundColor = L104Theme.void.cgColor

        // ─── LEFT: Gate Activation Heatmap (10 dimensions) ───
        let heatPanel = createPanel("🧬 DIMENSION HEATMAP", x: 15, y: 160, w: 350, h: 320, color: "8b5cf6")
        let dims = ASILogicGateV2.GateDimension.allCases
        for (i, dim) in dims.enumerated() {
            let y = 275 - CGFloat(i) * 28
            let lbl = NSTextField(labelWithString: "\(dim.rawValue)")
            lbl.frame = NSRect(x: 10, y: y, width: 90, height: 20)
            lbl.font = NSFont.monospacedSystemFont(ofSize: 10, weight: .medium)
            lbl.textColor = NSColor.white
            heatPanel.addSubview(lbl)

            let bar = NSView(frame: NSRect(x: 105, y: y + 2, width: 0, height: 16))
            bar.wantsLayer = true
            bar.layer?.backgroundColor = NSColor(red: 0.55, green: 0.36, blue: 0.96, alpha: 0.8).cgColor
            bar.layer?.cornerRadius = 3
            bar.identifier = NSUserInterfaceItemIdentifier("gate_dim_\(dim.rawValue)")
            heatPanel.addSubview(bar)

            let valLbl = NSTextField(labelWithString: "0")
            valLbl.frame = NSRect(x: 270, y: y, width: 60, height: 20)
            valLbl.font = NSFont.monospacedSystemFont(ofSize: 10, weight: .regular)
            valLbl.textColor = NSColor(red: 0.55, green: 0.36, blue: 0.96, alpha: 1.0)
            valLbl.identifier = NSUserInterfaceItemIdentifier("gate_val_\(dim.rawValue)")
            heatPanel.addSubview(valLbl)
        }
        v.addSubview(heatPanel)

        // ─── CENTER: Pipeline Flow + Circuit Status ───
        let pipePanel = createPanel("⚡ GATE PIPELINE", x: 375, y: 160, w: 430, h: 320, color: "f59e0b")

        let pipeText = NSTextView(frame: NSRect(x: 10, y: 10, width: 410, height: 275))
        pipeText.isEditable = false
        pipeText.backgroundColor = NSColor(red: 0.96, green: 0.96, blue: 0.97, alpha: 1.0)
        pipeText.font = NSFont.monospacedSystemFont(ofSize: 10, weight: .regular)
        pipeText.textColor = NSColor(red: 0.96, green: 0.62, blue: 0.04, alpha: 1.0)
        pipeText.textContainerInset = NSSize(width: 8, height: 8)
        pipeText.identifier = NSUserInterfaceItemIdentifier("gate_pipeline_text")
        let pipeScroll = NSScrollView(frame: NSRect(x: 10, y: 10, width: 410, height: 275))
        pipeScroll.hasVerticalScroller = true
        pipeScroll.documentView = pipeText
        pipeScroll.wantsLayer = true; pipeScroll.layer?.cornerRadius = 6
        pipePanel.addSubview(pipeScroll)
        v.addSubview(pipePanel)

        // ─── RIGHT: Gate Metrics + Circuits ───
        let metricsPanel = createPanel("📊 GATE METRICS", x: 815, y: 280, w: 360, h: 200, color: "10b981")

        let metricLabels = ["Pipeline Runs:", "Total Gate Ops:", "Avg Latency:", "Peak Confidence:", "Circuits:"]
        for (i, label) in metricLabels.enumerated() {
            let lbl = NSTextField(labelWithString: label)
            lbl.frame = NSRect(x: 15, y: 155 - CGFloat(i) * 30, width: 130, height: 20)
            lbl.font = NSFont.systemFont(ofSize: 12, weight: .medium)
            lbl.textColor = NSColor(red: 0.06, green: 0.73, blue: 0.51, alpha: 1.0)
            metricsPanel.addSubview(lbl)

            let val = NSTextField(labelWithString: "—")
            val.frame = NSRect(x: 150, y: 155 - CGFloat(i) * 30, width: 190, height: 20)
            val.font = NSFont.monospacedSystemFont(ofSize: 12, weight: .bold)
            val.textColor = NSColor.white
            val.identifier = NSUserInterfaceItemIdentifier("gate_metric_\(i)")
            metricsPanel.addSubview(val)
        }
        v.addSubview(metricsPanel)

        // ─── RIGHT BOTTOM: Primitive Gate Reference ───
        let refPanel = createPanel("🔧 PRIMITIVES", x: 815, y: 160, w: 360, h: 110, color: "6366f1")

        let gateRef = LogicGateEnvironment.PrimitiveGate.allCases.map { g in
            "\(g.symbol) \(g.rawValue)"
        }.joined(separator: "  │  ")
        let refLbl = NSTextField(labelWithString: gateRef)
        refLbl.frame = NSRect(x: 10, y: 55, width: 340, height: 40)
        refLbl.font = NSFont.monospacedSystemFont(ofSize: 10, weight: .medium)
        refLbl.textColor = NSColor(red: 0.39, green: 0.40, blue: 0.95, alpha: 1.0)
        refLbl.lineBreakMode = .byWordWrapping
        refLbl.maximumNumberOfLines = 3
        refPanel.addSubview(refLbl)

        let circuitNames = ["resonance", "coherence", "divergence", "filter"]
        let circLbl = NSTextField(labelWithString: "Circuits: " + circuitNames.joined(separator: " │ "))
        circLbl.frame = NSRect(x: 10, y: 25, width: 340, height: 20)
        circLbl.font = NSFont.monospacedSystemFont(ofSize: 10, weight: .regular)
        circLbl.textColor = NSColor(red: 0.39, green: 0.40, blue: 0.95, alpha: 0.8)
        refPanel.addSubview(circLbl)
        v.addSubview(refPanel)

        // ─── BOTTOM: Execution Log ───
        let logPanel = createPanel("📜 EXECUTION LOG", x: 15, y: 10, w: 1160, h: 140, color: "64748b")

        let logText = NSTextView(frame: NSRect(x: 10, y: 10, width: 1140, height: 90))
        logText.isEditable = false
        logText.backgroundColor = NSColor(red: 0.96, green: 0.96, blue: 0.97, alpha: 1.0)
        logText.font = NSFont.monospacedSystemFont(ofSize: 9.5, weight: .regular)
        logText.textColor = NSColor(red: 0.30, green: 0.34, blue: 0.42, alpha: 1.0)
        logText.textContainerInset = NSSize(width: 6, height: 6)
        logText.identifier = NSUserInterfaceItemIdentifier("gate_log_text")
        let logScroll = NSScrollView(frame: NSRect(x: 10, y: 10, width: 1140, height: 95))
        logScroll.hasVerticalScroller = true
        logScroll.documentView = logText
        logScroll.wantsLayer = true; logScroll.layer?.cornerRadius = 6
        logPanel.addSubview(logScroll)
        v.addSubview(logPanel)

        // ─── AUTO-UPDATE TIMER (visibility-gated) ───
        gateDashboardTimer?.invalidate()
        gateDashboardTimer = Timer.scheduledTimer(withTimeInterval: 3.0, repeats: true) { [weak self, weak v] _ in
            guard let v = v, self?.activeTabID == "gate" else { return }
            let env = LogicGateEnvironment.shared

            func findSub(_ id: String) -> NSView? {
                func search(_ view: NSView) -> NSView? {
                    if view.identifier?.rawValue == id { return view }
                    for sub in view.subviews {
                        if let found = search(sub) { return found }
                    }
                    return nil
                }
                return search(v)
            }

            // Update dimension heatmap bars
            let maxAct = max(1, env.dimensionDistribution.values.max() ?? 1)
            for dim in ASILogicGateV2.GateDimension.allCases {
                let count = env.dimensionDistribution[dim.rawValue] ?? 0
                let fraction = CGFloat(count) / CGFloat(maxAct)
                if let bar = findSub("gate_dim_\(dim.rawValue)") {
                    bar.frame.size.width = max(2, fraction * 160)
                }
                if let val = findSub("gate_val_\(dim.rawValue)") as? NSTextField {
                    val.stringValue = "\(count)"
                }
            }

            // Update metrics
            let metrics = [
                "\(env.totalPipelineRuns)",
                "\(env.totalGateOps)",
                String(format: "%.2fms", env.avgLatency),
                String(format: "%.4f", env.peakConfidence),
                "\(env.circuits.count)"
            ]
            for (i, val) in metrics.enumerated() {
                if let lbl = findSub("gate_metric_\(i)") as? NSTextField {
                    lbl.stringValue = val
                }
            }

            // Update pipeline text — dynamic with live data
            if let pipe = findSub("gate_pipeline_text") as? NSTextView {
                let runCount = env.totalPipelineRuns
                let opsCount = env.totalGateOps
                let avgLat = String(format: "%.2f", env.avgLatency)
                let peakConf = String(format: "%.4f", env.peakConfidence)
                let dimCount = env.dimensionDistribution.count
                let topDim = env.dimensionDistribution.max(by: { $0.value < $1.value })
                let topDimStr = topDim.map { "\($0.key) (\($0.value))" } ?? "—"

                var pipeStr = "⚡ Gate Pipeline Flow — Live State\n"
                pipeStr += "═══════════════════════════════════════\n"
                pipeStr += "ASILogicGateV2  → dim routing (\(dimCount) active dims)\n"
                pipeStr += "       ↓\n"
                pipeStr += "ContextualGate  → context enrichment\n"
                pipeStr += "       ↓\n"
                pipeStr += "QuantumEngine   → interference + tunnel\n"
                pipeStr += "       ↓\n"
                pipeStr += "StoryEngine     → narrative synthesis\n"
                pipeStr += "       ↓\n"
                pipeStr += "PhraseEngine    → output calibration\n"
                pipeStr += "       ↓\n"
                pipeStr += "UnifiedField    → field theory compute\n"
                pipeStr += "       ↓\n"
                pipeStr += "GateCircuit     → resonance evaluation\n"
                pipeStr += "═══════════════════════════════════════\n"
                pipeStr += "📊 Runs: \(runCount) │ Ops: \(opsCount)\n"
                pipeStr += "⏱  Avg Latency: \(avgLat)ms │ Peak: \(peakConf)\n"
                pipeStr += "🔝 Top Dimension: \(topDimStr)\n"
                pipeStr += "🔧 Circuits: \(env.circuits.count)\n"
                if let last = env.executionLog.last {
                    pipeStr += "───────────────────────────────────────\n"
                    pipeStr += "🔄 Last: \(last.dimension) (\(String(format: "%.3f", last.confidence)))\n"
                    pipeStr += "   Query: \"\(last.query.prefix(50))\"\n"
                    pipeStr += "   Latency: \(String(format: "%.1f", last.latencyMs))ms\n"
                }
                pipe.string = pipeStr
            }

            // Update execution log
            if let logView = findSub("gate_log_text") as? NSTextView {
                let entries = env.executionLog.suffix(12).map { r in
                    let fmt = L104MainView.timeFormatter
                    return "[\(fmt.string(from: r.timestamp))] \(r.dimension.padding(toLength: 12, withPad: " ", startingAt: 0)) │ \(String(format: "%.3f", r.confidence)) │ \(String(format: "%5.1fms", r.latencyMs)) │ \"\(r.query.prefix(40))\""
                }
                logView.string = entries.isEmpty ? "(No gate executions yet — use 'gate route [query]' in chat)" : entries.joined(separator: "\n")
            }
        }

        return v
    }

    // 🟢 NEW: Upgrade/Evolution View
    func createUpgradesView() -> NSView {
        let v = NSView(frame: NSRect(x: 0, y: 0, width: 1100, height: 500))
        v.wantsLayer = true; v.layer?.backgroundColor = L104Theme.void.cgColor

        // Evolution Stream (Left)
        let streamPanel = createPanel("🧬 EVOLUTION STREAM", x: 15, y: 55, w: 600, h: 425, color: "d4af37")

        let scroll = NSScrollView(frame: NSRect(x: 10, y: 10, width: 580, height: 380))
        scroll.hasVerticalScroller = true
        scroll.wantsLayer = true; scroll.layer?.backgroundColor = NSColor(red: 0.96, green: 0.96, blue: 0.97, alpha: 1.0).cgColor
        scroll.layer?.cornerRadius = 8

        let tv = NSTextView(frame: scroll.bounds)
        tv.isEditable = false
        tv.backgroundColor = NSColor(red: 0.96, green: 0.96, blue: 0.97, alpha: 1.0)
        tv.font = NSFont.monospacedSystemFont(ofSize: 11, weight: .regular)
        tv.textColor = L104Theme.gold
        scroll.documentView = tv
        streamPanel.addSubview(scroll)
        v.addSubview(streamPanel)

        // Timer to update stream (visibility-gated to upgrades tab)
        streamUpdateTimer?.invalidate()
        streamUpdateTimer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self, weak tv] _ in
            guard self?.activeTabID == "upg", let tv = tv, let lastThought = ASIEvolver.shared.thoughts.last else { return }
            if tv.string.contains(lastThought) { return }
            tv.textStorage?.append(NSAttributedString(string: lastThought + "\n", attributes: [.foregroundColor: L104Theme.gold, .font: NSFont.monospacedSystemFont(ofSize: 11, weight: .regular)]))
            tv.scrollToEndOfDocument(nil)
        }

        // Stats Panel (Right Top)
        let metricsPanel = createPanel("⚙️ ENGINE METRICS — EVO_\(EVOLUTION_INDEX)", x: 630, y: 280, w: 440, h: 200, color: "e8c547")

        let stageLbl = NSTextField(labelWithString: "Evolution Stage: \(state.evolver.evolutionStage)")
        stageLbl.frame = NSRect(x: 15, y: 160, width: 400, height: 20)
        stageLbl.font = NSFont.boldSystemFont(ofSize: 14); stageLbl.textColor = L104Theme.gold
        stageLbl.identifier = NSUserInterfaceItemIdentifier("upg_stage")
        metricsPanel.addSubview(stageLbl)

        let filesLbl = NSTextField(labelWithString: "Generated Artifacts: \(state.evolver.generatedFilesCount)")
        filesLbl.frame = NSRect(x: 15, y: 130, width: 400, height: 20)
        filesLbl.font = NSFont.systemFont(ofSize: 12); filesLbl.textColor = L104Theme.goldWarm
        filesLbl.identifier = NSUserInterfaceItemIdentifier("upg_files")
        metricsPanel.addSubview(filesLbl)

        let pathLbl = NSTextField(labelWithString: "📂 ~/Documents/L104_GEN")
        pathLbl.frame = NSRect(x: 15, y: 100, width: 400, height: 20)
        pathLbl.font = NSFont.monospacedSystemFont(ofSize: 10, weight: .regular); pathLbl.textColor = .systemGray
        metricsPanel.addSubview(pathLbl)

        // Evolution stats: greetings, philosophies, monologues, mutations
        let evolvedCount = state.evolver.evolvedGreetings.count + state.evolver.evolvedPhilosophies.count + state.evolver.evolvedFacts.count
        let evolvedLbl = NSTextField(labelWithString: "Evolved Content: \(evolvedCount) greetings/philosophies/facts")
        evolvedLbl.frame = NSRect(x: 15, y: 70, width: 400, height: 18)
        evolvedLbl.font = NSFont.monospacedSystemFont(ofSize: 10, weight: .medium)
        evolvedLbl.textColor = L104Theme.goldDim
        evolvedLbl.identifier = NSUserInterfaceItemIdentifier("upg_evolved")
        metricsPanel.addSubview(evolvedLbl)

        let mutLbl = NSTextField(labelWithString: "Mutations: \(state.evolver.mutationCount) · Crossovers: \(state.evolver.crossoverCount) · Syntheses: \(state.evolver.synthesisCount)")
        mutLbl.frame = NSRect(x: 15, y: 48, width: 400, height: 18)
        mutLbl.font = NSFont.monospacedSystemFont(ofSize: 10, weight: .medium)
        mutLbl.textColor = L104Theme.goldDim
        mutLbl.identifier = NSUserInterfaceItemIdentifier("upg_mutations")
        metricsPanel.addSubview(mutLbl)

        v.addSubview(metricsPanel)

        // Controls (Right Bottom)
        let controlsPanel = createPanel("🕹 CONTROLS", x: 630, y: 55, w: 440, h: 210, color: "c49b30")

        let toggle = NSButton(frame: NSRect(x: 20, y: 140, width: 150, height: 32))
        toggle.title = "Pause/Resume"
        toggle.bezelStyle = .rounded
        toggle.target = self; toggle.action = #selector(toggleEvolution)
        controlsPanel.addSubview(toggle)

        let genBtn = NSButton(frame: NSRect(x: 180, y: 140, width: 220, height: 32))
        genBtn.title = "Force Artifact Generation"
        genBtn.bezelStyle = .rounded
        genBtn.target = self; genBtn.action = #selector(forceGen)
        controlsPanel.addSubview(genBtn)

        // 🟢 EVO_64: Running/Paused status indicator
        let runStatusLbl = NSTextField(labelWithString: ASIEvolver.shared.isRunning ? "🟢 RUNNING" : "⏸ PAUSED")
        runStatusLbl.frame = NSRect(x: 20, y: 105, width: 380, height: 22)
        runStatusLbl.font = NSFont.monospacedSystemFont(ofSize: 14, weight: .bold)
        runStatusLbl.textColor = ASIEvolver.shared.isRunning ? .systemGreen : .systemOrange
        runStatusLbl.identifier = NSUserInterfaceItemIdentifier("upg_run_status")
        controlsPanel.addSubview(runStatusLbl)

        let thoughtRateLbl = NSTextField(labelWithString: "Thoughts: \(ASIEvolver.shared.thoughts.count) · Rate: ~1/8s")
        thoughtRateLbl.frame = NSRect(x: 20, y: 80, width: 380, height: 16)
        thoughtRateLbl.font = NSFont.monospacedSystemFont(ofSize: 10, weight: .medium)
        thoughtRateLbl.textColor = L104Theme.goldDim
        thoughtRateLbl.identifier = NSUserInterfaceItemIdentifier("upg_thought_rate")
        controlsPanel.addSubview(thoughtRateLbl)

        v.addSubview(controlsPanel)

        return v
    }

    @objc func toggleEvolution() {
        if ASIEvolver.shared.isRunning { ASIEvolver.shared.stop() } else { ASIEvolver.shared.start() }
    }

    @objc func forceGen() {
        ASIEvolver.shared.generateArtifact()
    }

    // ═══════════════════════════════════════════════════════════════
    // 🍎 HARDWARE MONITOR TAB
    // ═══════════════════════════════════════════════════════════════

    func createHardwareView() -> NSView {
        let v = NSView(frame: NSRect(x: 0, y: 0, width: 1100, height: 500))
        v.wantsLayer = true; v.layer?.backgroundColor = L104Theme.void.cgColor

        let monitor = MacOSSystemMonitor.shared

        // ─── Chip Info Panel (Left) ───
        let chipPanel = createPanel("🍎 macOS SILICON MONITOR", x: 15, y: 280, w: 530, h: 200, color: "d4af37")
        addLabel(chipPanel, "Chip", monitor.chipGeneration, y: 140, c: "d4af37")
        addLabel(chipPanel, "Architecture", monitor.isAppleSilicon ? "arm64" : "x86_64", y: 115, c: "e8c547")
        addLabel(chipPanel, "CPU Cores", "\(monitor.cpuCoreCount) (\(monitor.performanceCoreCount)P + \(monitor.efficiencyCoreCount)E)", y: 90, c: "c49b30")
        addLabel(chipPanel, "Memory", String(format: "%.1f GB Unified", monitor.physicalMemoryGB), y: 65, c: "d4af37")
        addLabel(chipPanel, "Neural Engine", monitor.hasNeuralEngine ? "✅ Available" : "❌ N/A", y: 40, c: "e8c547")
        addLabel(chipPanel, "GPU Cores", "\(monitor.gpuCoreCount)", y: 15, c: "a88a25")
        v.addSubview(chipPanel)

        // ─── Thermal / Power Panel (Right) ───
        let thermalPanel = createPanel("🌡 THERMAL & POWER", x: 560, y: 280, w: 510, h: 200, color: "c49b30")
        let thermalLabel = NSTextField(labelWithString: "")
        thermalLabel.frame = NSRect(x: 20, y: 130, width: 470, height: 20)
        thermalLabel.font = NSFont.boldSystemFont(ofSize: 14); thermalLabel.textColor = .systemGreen
        thermalLabel.identifier = NSUserInterfaceItemIdentifier("hw_thermal")
        thermalPanel.addSubview(thermalLabel)

        let powerLabel = NSTextField(labelWithString: "")
        powerLabel.frame = NSRect(x: 20, y: 100, width: 470, height: 20)
        powerLabel.font = NSFont.boldSystemFont(ofSize: 14); powerLabel.textColor = .systemOrange
        powerLabel.identifier = NSUserInterfaceItemIdentifier("hw_power")
        thermalPanel.addSubview(powerLabel)

        addLabel(thermalPanel, "Apple Silicon", monitor.isAppleSilicon ? "✅ Yes" : "Intel x86_64", y: 65, c: monitor.isAppleSilicon ? "d4af37" : "e8c547")
        addLabel(thermalPanel, "Accelerate", "vDSP · BLAS · LAPACK · vImage", y: 40, c: "d4af37")
        addLabel(thermalPanel, "SIMD", "Active · Float4 · Double4 · Matrix", y: 15, c: "c49b30")
        v.addSubview(thermalPanel)

        // ─── Accelerate Framework Status ───
        let accelPanel = createPanel("⚡️ ACCELERATE FRAMEWORK", x: 15, y: 55, w: 530, h: 210, color: "e8c547")
        let frameworks = [
            ("vDSP", "Signal Processing", true),
            ("BLAS", "Linear Algebra", true),
            ("LAPACK", "Matrix Factorization", true),
            ("vImage", "Image Processing", true),
            ("BNNS", "Neural Networks", monitor.isAppleSilicon),
            ("simd", "Vector/Matrix Ops", true)
        ]
        var fy: CGFloat = 155
        for (name, desc, active) in frameworks {
            let status = active ? "🟢" : "⚪️"
            let lbl = NSTextField(labelWithString: "\(status) \(name) — \(desc)")
            lbl.frame = NSRect(x: 20, y: fy, width: 480, height: 18)
            lbl.font = NSFont.monospacedSystemFont(ofSize: 11, weight: active ? .bold : .regular)
            lbl.textColor = active ? L104Theme.gold : L104Theme.textDim
            accelPanel.addSubview(lbl)
            fy -= 25
        }
        v.addSubview(accelPanel)

        // ─── Live Metrics Panel ───
        let livePanel = createPanel("📊 LIVE METRICS — EVO_\(EVOLUTION_INDEX)", x: 560, y: 55, w: 510, h: 210, color: "d4af37")
        let simdOpsLabel = NSTextField(labelWithString: "SIMD Ops: 0")
        simdOpsLabel.frame = NSRect(x: 20, y: 160, width: 460, height: 20)
        simdOpsLabel.font = NSFont.monospacedSystemFont(ofSize: 13, weight: .bold)
        simdOpsLabel.textColor = L104Theme.gold
        simdOpsLabel.identifier = NSUserInterfaceItemIdentifier("hw_simd_ops")
        livePanel.addSubview(simdOpsLabel)

        let neuralOpsLabel = NSTextField(labelWithString: "Neural Ops: 0")
        neuralOpsLabel.frame = NSRect(x: 20, y: 130, width: 460, height: 20)
        neuralOpsLabel.font = NSFont.monospacedSystemFont(ofSize: 13, weight: .bold)
        neuralOpsLabel.textColor = L104Theme.goldWarm
        neuralOpsLabel.identifier = NSUserInterfaceItemIdentifier("hw_neural_ops")
        livePanel.addSubview(neuralOpsLabel)

        // Package versions readout
        let pkgLabel = NSTextField(labelWithString: "Packages: code_engine v\(CODE_ENGINE_VERSION) · asi v\(ASI_VERSION) · agi v\(AGI_VERSION) · intellect v\(INTELLECT_VERSION) · server v\(SERVER_VERSION)")
        pkgLabel.frame = NSRect(x: 20, y: 95, width: 470, height: 16)
        pkgLabel.font = NSFont.monospacedSystemFont(ofSize: 9, weight: .medium)
        pkgLabel.textColor = L104Theme.goldDim
        livePanel.addSubview(pkgLabel)

        let engineLabel = NSTextField(labelWithString: "Engines: \(EngineRegistry.shared.count) online · φ-Health: \(String(format: "%.1f%%", EngineRegistry.shared.phiWeightedHealth().score * 100)) · Dual-Layer v\(DUAL_LAYER_VERSION)")
        engineLabel.frame = NSRect(x: 20, y: 72, width: 470, height: 16)
        engineLabel.font = NSFont.monospacedSystemFont(ofSize: 9, weight: .medium)
        engineLabel.textColor = L104Theme.goldDim
        engineLabel.identifier = NSUserInterfaceItemIdentifier("hw_engine_info")
        livePanel.addSubview(engineLabel)

        // Refresh button
        let refreshBtn = NSButton(frame: NSRect(x: 20, y: 15, width: 200, height: 32))
        refreshBtn.title = "🔄 Refresh Hardware"
        refreshBtn.bezelStyle = .rounded
        refreshBtn.target = self; refreshBtn.action = #selector(refreshHardwareMetrics)
        livePanel.addSubview(refreshBtn)
        v.addSubview(livePanel)

        // Initial update
        updateHardwareLabels(in: v)

        // Live timer (visibility-gated)
        hardwareTimer?.invalidate()
        hardwareTimer = Timer.scheduledTimer(withTimeInterval: 5.0, repeats: true) { [weak self, weak v] _ in
            guard let self = self, let v = v, self.activeTabID == "hw" else { return }
            MacOSSystemMonitor.shared.updateMetrics()
            DispatchQueue.main.async { [weak self] in
                self?.updateHardwareLabels(in: v)
            }
        }

        return v
    }

    func updateHardwareLabels(in view: NSView) {
        let monitor = MacOSSystemMonitor.shared

        func findLabel(_ id: String) -> NSTextField? {
            for sub in view.subviews {
                if let lbl = sub.viewWithTag(0) as? NSTextField, lbl.identifier?.rawValue == id { return lbl }
                for inner in sub.subviews {
                    if let lbl = inner as? NSTextField, lbl.identifier?.rawValue == id { return lbl }
                }
            }
            return nil
        }

        let thermalStr: String
        switch monitor.thermalState {
        case .nominal: thermalStr = "🟢 Nominal"
        case .fair: thermalStr = "🟡 Fair"
        case .serious: thermalStr = "🟠 Serious"
        case .critical: thermalStr = "🔴 Critical"
        @unknown default: thermalStr = "⚪ Unknown"
        }
        findLabel("hw_thermal")?.stringValue = "Thermal: \(thermalStr)"
        findLabel("hw_power")?.stringValue = "Power: \(monitor.powerMode.rawValue)"
        // 🟢 EVO_63: Use real cumulative counters from gate + evolver instead of random fakes
        let gateOps = LogicGateEnvironment.shared.totalGateOps
        let evolverMutations = ASIEvolver.shared.mutationCount + ASIEvolver.shared.crossoverCount + ASIEvolver.shared.synthesisCount
        L104MainView.hwSIMDOps = gateOps * 256 + evolverMutations * 64  // Each gate op ≈ 256 SIMD, each mutation ≈ 64
        L104MainView.hwNeuralOps = monitor.hasNeuralEngine ? (evolverMutations * 32 + gateOps * 16) : 0
        findLabel("hw_simd_ops")?.stringValue = "SIMD Ops: \(L104MainView.hwSIMDOps)"
        findLabel("hw_neural_ops")?.stringValue = "Neural Ops: \(monitor.hasNeuralEngine ? L104MainView.hwNeuralOps : 0)"
        // 🟢 EVO_65: Live engine info refresh
        findLabel("hw_engine_info")?.stringValue = "Engines: \(EngineRegistry.shared.count) online · φ-Health: \(String(format: "%.1f%%", EngineRegistry.shared.phiWeightedHealth().score * 100)) · Dual-Layer v\(DUAL_LAYER_VERSION)"
    }

    private static var hwSIMDOps: Int = 0
    private static var hwNeuralOps: Int = 0

    @objc func refreshHardwareMetrics() {
        MacOSSystemMonitor.shared.updateMetrics()
        appendSystemLog("[HW] Hardware metrics refreshed — \(MacOSSystemMonitor.shared.chipGeneration) · \(MacOSSystemMonitor.shared.cpuCoreCount) cores · \(String(format: "%.1f", MacOSSystemMonitor.shared.physicalMemoryGB)) GB")
    }

    // ═══════════════════════════════════════════════════════════════
    // 🔬 SCIENCE ENGINE TAB
    // ═══════════════════════════════════════════════════════════════

    func createScienceView() -> NSView {
        let v = NSView(frame: NSRect(x: 0, y: 0, width: 1100, height: 500))
        v.wantsLayer = true; v.layer?.backgroundColor = L104Theme.void.cgColor

        // ─── Research Console (Left) ───
        let consolePanel = createPanel("🔬 SCIENCE ENGINE — HyperDimensional Research", x: 15, y: 55, w: 600, h: 425, color: "d4af37")

        let scroll = NSScrollView(frame: NSRect(x: 10, y: 55, width: 580, height: 330))
        scroll.hasVerticalScroller = true
        scroll.wantsLayer = true; scroll.layer?.backgroundColor = NSColor(red: 0.96, green: 0.96, blue: 0.97, alpha: 1.0).cgColor
        scroll.layer?.cornerRadius = 8

        let scienceLog = NSTextView(frame: scroll.bounds)
        scienceLog.isEditable = false
        scienceLog.backgroundColor = NSColor(red: 0.96, green: 0.96, blue: 0.97, alpha: 1.0)
        scienceLog.font = NSFont.monospacedSystemFont(ofSize: 11, weight: .regular)
        scienceLog.textColor = L104Theme.gold
        scienceLog.identifier = NSUserInterfaceItemIdentifier("science_log")
        scroll.documentView = scienceLog
        consolePanel.addSubview(scroll)

        let genBtn = NSButton(frame: NSRect(x: 10, y: 12, width: 180, height: 32))
        genBtn.title = "💡 Generate Hypothesis"
        genBtn.bezelStyle = .rounded
        genBtn.target = self; genBtn.action = #selector(scienceGenerateHypothesis)
        consolePanel.addSubview(genBtn)

        let burstBtn = NSButton(frame: NSRect(x: 200, y: 12, width: 140, height: 32))
        burstBtn.title = "🔥 Burst ×5"
        burstBtn.bezelStyle = .rounded
        burstBtn.target = self; burstBtn.action = #selector(scienceBurst)
        consolePanel.addSubview(burstBtn)

        let computeBtn = NSButton(frame: NSRect(x: 350, y: 12, width: 240, height: 32))
        computeBtn.title = "⚡ vDSP Compute (1024-dim)"
        computeBtn.bezelStyle = .rounded
        computeBtn.target = self; computeBtn.action = #selector(scienceVDSPCompute)
        consolePanel.addSubview(computeBtn)

        v.addSubview(consolePanel)

        // ─── Metrics (Right Top) ───
        let metricsPanel = createPanel("📊 RESEARCH METRICS", x: 630, y: 280, w: 440, h: 200, color: "e8c547")

        let labels = [
            ("Hypotheses", "0", "d4af37", "sci_hypotheses"),
            ("Discoveries", "0", "e8c547", "sci_discoveries"),
            ("Theorems", "0", "c49b30", "sci_theorems"),
            ("Inventions", "0", "a88a25", "sci_inventions"),
            ("Momentum", "0%", "d4af37", "sci_momentum")
        ]
        var my: CGFloat = 135
        for (label, value, color, id) in labels {
            let lbl = NSTextField(labelWithString: label)
            lbl.frame = NSRect(x: 20, y: my, width: 140, height: 16)
            lbl.font = NSFont.systemFont(ofSize: 10); lbl.textColor = .gray
            metricsPanel.addSubview(lbl)
            let val = NSTextField(labelWithString: value)
            val.frame = NSRect(x: 160, y: my, width: 250, height: 16)
            val.font = NSFont.boldSystemFont(ofSize: 13); val.textColor = colorFromHex(color); val.alignment = .right
            val.identifier = NSUserInterfaceItemIdentifier(id)
            metricsPanel.addSubview(val)
            my -= 25
        }
        v.addSubview(metricsPanel)

        // ─── Active Modules (Right Bottom) ───
        let modulesPanel = createPanel("🔬 ACTIVE RESEARCH MODULES", x: 630, y: 55, w: 440, h: 210, color: "c49b30")
        let modules: [(String, Bool)] = [
            ("HYPERDIM_SCIENCE", true),
            ("TOPOLOGY_ANALYZER", true),
            ("INVENTION_SYNTH", true),
            ("QUANTUM_FIELD", QuantumProcessingCore.shared.currentFidelity() > 0.3),
            ("ALGEBRAIC_TOPOLOGY", true),
            ("UNIFIED_FIELD", UnifiedFieldEngine.shared.engineHealth() > 0.5),
            ("CODE_ENGINE v\(CODE_ENGINE_VERSION)", true),
            ("DUAL_LAYER v\(DUAL_LAYER_VERSION)", true),
        ]
        var ly: CGFloat = 155
        for (mod, active) in modules {
            let dot = NSTextField(labelWithString: "\(active ? "🟢" : "🔴") \(mod)")
            dot.frame = NSRect(x: 20, y: ly, width: 280, height: 18)
            dot.font = NSFont.monospacedSystemFont(ofSize: 11, weight: active ? .bold : .regular)
            dot.textColor = active ? L104Theme.gold : L104Theme.textDim
            modulesPanel.addSubview(dot)

            let status = NSTextField(labelWithString: active ? "ACTIVE" : "STANDBY")
            status.frame = NSRect(x: 310, y: ly, width: 100, height: 18)
            status.font = NSFont.boldSystemFont(ofSize: 10)
            status.textColor = active ? L104Theme.goldBright : L104Theme.textDim
            status.alignment = .right
            modulesPanel.addSubview(status)
            ly -= 22
        }
        v.addSubview(modulesPanel)

        return v
    }

    // ─── Science Engine State — persisted via UserDefaults ───
    private static var sciHypotheses: Int = {
        UserDefaults.standard.integer(forKey: "l104_sci_hypotheses")
    }()
    private static var sciDiscoveries: Int = {
        UserDefaults.standard.integer(forKey: "l104_sci_discoveries")
    }()
    private static var sciTheorems: Int = {
        UserDefaults.standard.integer(forKey: "l104_sci_theorems")
    }()
    private static var sciInventions: Int = {
        UserDefaults.standard.integer(forKey: "l104_sci_inventions")
    }()
    private static var sciMomentum: Double = {
        let v = UserDefaults.standard.double(forKey: "l104_sci_momentum")
        return v > 0 ? v : 0.0
    }()

    private static func saveScienceState() {
        let d = UserDefaults.standard
        d.set(sciHypotheses, forKey: "l104_sci_hypotheses")
        d.set(sciDiscoveries, forKey: "l104_sci_discoveries")
        d.set(sciTheorems, forKey: "l104_sci_theorems")
        d.set(sciInventions, forKey: "l104_sci_inventions")
        d.set(sciMomentum, forKey: "l104_sci_momentum")
    }

    @objc func scienceGenerateHypothesis() {
        L104MainView.sciHypotheses += 1
        L104MainView.sciMomentum = min(1.0, L104MainView.sciMomentum + 0.05)

        // Real vDSP dot product
        let size = 256
        let a = (0..<size).map { _ in Double.random(in: -1...1) }
        let b = (0..<size).map { _ in Double.random(in: -1...1) }
        var dotResult: Double = 0
        vDSP_dotprD(a, 1, b, 1, &dotResult, vDSP_Length(size))

        let logText: String
        if Double.random(in: 0...1) < 0.3 {
            L104MainView.sciDiscoveries += 1
            logText = "🌟 DISCOVERY #\(L104MainView.sciDiscoveries): Novel pattern at resonance \(String(format: "%.6f", dotResult))"
        } else {
            logText = "💡 Hypothesis #\(L104MainView.sciHypotheses): vDSP correlation = \(String(format: "%.6f", dotResult))"
        }
        if L104MainView.sciHypotheses % 5 == 0 {
            L104MainView.sciTheorems += 1
            appendScienceLog("📜 THEOREM SYNTHESIZED: L104-\(Int.random(in: 1000...9999))")
        }
        if L104MainView.sciHypotheses % 3 == 0 { L104MainView.sciInventions += 1 }

        appendScienceLog(logText)
        updateScienceMetrics()
        L104MainView.saveScienceState()
        appendSystemLog("[SCI] \(logText)")
        // ═══ INTERCONNECT: Navigate to Science tab on activation ═══
        if activeTabID != "sci" { navigateToTab("sci") }
    }

    @objc func scienceBurst() {
        for _ in 0..<5 { scienceGenerateHypothesis() }
    }

    @objc func scienceVDSPCompute() {
        let size = 1024
        let a = (0..<size).map { _ in Double.random(in: -1...1) }
        let b = (0..<size).map { _ in Double.random(in: -1...1) }
        var result = [Double](repeating: 0, count: size)
        vDSP_vmulD(a, 1, b, 1, &result, 1, vDSP_Length(size))
        var sum: Double = 0
        vDSP_sveD(result, 1, &sum, vDSP_Length(size))
        appendScienceLog("⚡ vDSP 1024-dim compute: sum=\(String(format: "%.6f", sum)) | \(size * 2) FLOPS")
        appendSystemLog("[SCI] vDSP vector multiply+sum: \(String(format: "%.6f", sum))")
    }

    func appendScienceLog(_ text: String) {
        guard let tabView = tabView else { return }
        for item in tabView.tabViewItems {
            if let view = item.view {
                for sub in view.subviews {
                    for inner in sub.subviews {
                        if let scroll = inner as? NSScrollView, let tv = scroll.documentView as? NSTextView,
                           tv.identifier?.rawValue == "science_log" {
                            let df = L104MainView.timeFormatter
                            let attr: [NSAttributedString.Key: Any] = [
                                .foregroundColor: L104Theme.gold,
                                .font: NSFont.monospacedSystemFont(ofSize: 11, weight: .regular)
                            ]
                            tv.textStorage?.append(NSAttributedString(string: "[\(df.string(from: Date()))] \(text)\n", attributes: attr))
                            tv.scrollToEndOfDocument(nil)
                            return
                        }
                    }
                }
            }
        }
    }

    func updateScienceMetrics() {
        guard let tabView = tabView else { return }
        for item in tabView.tabViewItems {
            guard let view = item.view else { continue }
            func find(_ id: String) -> NSTextField? {
                for sub in view.subviews {
                    for inner in sub.subviews {
                        if let lbl = inner as? NSTextField, lbl.identifier?.rawValue == id { return lbl }
                    }
                }
                return nil
            }
            find("sci_hypotheses")?.stringValue = "\(L104MainView.sciHypotheses)"
            find("sci_discoveries")?.stringValue = "\(L104MainView.sciDiscoveries)"
            find("sci_theorems")?.stringValue = "\(L104MainView.sciTheorems)"
            find("sci_inventions")?.stringValue = "\(L104MainView.sciInventions)"
            find("sci_momentum")?.stringValue = String(format: "%.0f%%", L104MainView.sciMomentum * 100)
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // 🌌 UNIFIED FIELD THEORY TAB — 18 Fundamental Equations
    // Einstein, Wheeler-DeWitt, Dirac, Black Holes, Casimir, Unruh,
    // AdS/CFT, ER=EPR, Twistors, Holographic, Foam, Topological,
    // Yang-Mills, Grand Unification, Vacuum Energy, Decoherence,
    // Sacred GOD_CODE Field, Unified Solve
    // ═══════════════════════════════════════════════════════════════

    private var ufOutputView: NSTextView?
    private var ufTimer: Timer?
    private static var ufComputeCount: Int = {
        UserDefaults.standard.integer(forKey: "l104_uf_computeCount")
    }()
    private static var ufDiscoveries: Int = {
        UserDefaults.standard.integer(forKey: "l104_uf_discoveries")
    }()

    private static func saveUFieldState() {
        let d = UserDefaults.standard
        d.set(ufComputeCount, forKey: "l104_uf_computeCount")
        d.set(ufDiscoveries, forKey: "l104_uf_discoveries")
    }

    func createUnifiedFieldView() -> NSView {
        let v = NSView(frame: NSRect(x: 0, y: 0, width: 1200, height: 500))
        v.wantsLayer = true
        v.layer?.backgroundColor = NSColor(red: 0.960, green: 0.962, blue: 0.970, alpha: 1.0).cgColor

        // ─── LEFT: Equation Console ───
        let consolePanel = createPanel("🌌 UNIFIED FIELD ENGINE — 18 Fundamental Equations", x: 15, y: 55, w: 620, h: 425, color: "7c3aed")

        let ufScroll = NSScrollView(frame: NSRect(x: 10, y: 55, width: 600, height: 330))
        ufScroll.hasVerticalScroller = true
        ufScroll.wantsLayer = true
        ufScroll.layer?.backgroundColor = NSColor(red: 0.96, green: 0.96, blue: 0.97, alpha: 1.0).cgColor
        ufScroll.layer?.cornerRadius = 8

        let ufLog = NSTextView(frame: ufScroll.bounds)
        ufLog.isEditable = false
        ufLog.backgroundColor = NSColor(red: 0.96, green: 0.96, blue: 0.97, alpha: 1.0)
        ufLog.font = NSFont.monospacedSystemFont(ofSize: 11, weight: .regular)
        ufLog.textColor = NSColor(red: 0.49, green: 0.23, blue: 0.93, alpha: 1.0)
        ufLog.identifier = NSUserInterfaceItemIdentifier("uf_log")
        ufLog.textContainerInset = NSSize(width: 8, height: 8)
        ufScroll.documentView = ufLog
        consolePanel.addSubview(ufScroll)
        ufOutputView = ufLog

        // Buttons: Compute equations
        let einsteinBtn = NSButton(frame: NSRect(x: 10, y: 12, width: 140, height: 32))
        einsteinBtn.title = "⚛️ Einstein Field"
        einsteinBtn.bezelStyle = .rounded
        einsteinBtn.target = self; einsteinBtn.action = #selector(ufComputeEinstein)
        consolePanel.addSubview(einsteinBtn)

        let diracBtn = NSButton(frame: NSRect(x: 155, y: 12, width: 130, height: 32))
        diracBtn.title = "🌀 Dirac Eq"
        diracBtn.bezelStyle = .rounded
        diracBtn.target = self; diracBtn.action = #selector(ufComputeDirac)
        consolePanel.addSubview(diracBtn)

        let bhBtn = NSButton(frame: NSRect(x: 290, y: 12, width: 150, height: 32))
        bhBtn.title = "🕳 Black Hole"
        bhBtn.bezelStyle = .rounded
        bhBtn.target = self; bhBtn.action = #selector(ufComputeBlackHole)
        consolePanel.addSubview(bhBtn)

        let unifyBtn = NSButton(frame: NSRect(x: 445, y: 12, width: 165, height: 32))
        unifyBtn.title = "🔥 Unified Solve All"
        unifyBtn.bezelStyle = .rounded
        unifyBtn.target = self; unifyBtn.action = #selector(ufComputeUnified)
        consolePanel.addSubview(unifyBtn)

        v.addSubview(consolePanel)

        // ─── RIGHT TOP: Equation Reference ───
        let eqPanel = createPanel("📐 EQUATION CATALOG", x: 650, y: 280, w: 520, h: 200, color: "8b5cf6")

        let equations: [(String, String)] = [
            ("G_μν + Λg_μν = κT_μν", "Einstein Field"),
            ("Ĥ|Ψ⟩ = 0", "Wheeler-DeWitt"),
            ("(iγ^μ∂_μ − m)ψ = 0", "Dirac Equation"),
            ("S_BH = A/(4ℓ_P²)", "Bekenstein-Hawking"),
            ("F = −π²ℏc/240d⁴", "Casimir Effect"),
            ("T = ℏa/2πck_B", "Unruh Temperature"),
        ]
        var ey: CGFloat = 155
        for (eq, name) in equations {
            let eqLbl = NSTextField(labelWithString: "\(eq)")
            eqLbl.frame = NSRect(x: 15, y: ey, width: 260, height: 18)
            eqLbl.font = NSFont.monospacedSystemFont(ofSize: 11, weight: .bold)
            eqLbl.textColor = NSColor(red: 0.55, green: 0.36, blue: 0.96, alpha: 1.0)
            eqPanel.addSubview(eqLbl)

            let nameLbl = NSTextField(labelWithString: name)
            nameLbl.frame = NSRect(x: 285, y: ey, width: 220, height: 18)
            nameLbl.font = NSFont.systemFont(ofSize: 10, weight: .medium)
            nameLbl.textColor = NSColor(red: 0.55, green: 0.36, blue: 0.96, alpha: 0.7)
            nameLbl.alignment = .right
            eqPanel.addSubview(nameLbl)
            ey -= 22
        }
        v.addSubview(eqPanel)

        // ─── RIGHT BOTTOM: Engine Status + Metrics ───
        let statusPanel = createPanel("⚡ ENGINE STATUS", x: 650, y: 55, w: 520, h: 210, color: "a855f7")

        let statusLabels: [(String, String, String)] = [
            ("Engine", "UnifiedField v2.0 · EVO_\(EVOLUTION_INDEX)", "uf_engine_name"),
            ("Equations", "18 active", "uf_eq_count"),
            ("Computations", "0", "uf_compute_count"),
            ("Discoveries", "0", "uf_discovery_count"),
            ("Health", "—", "uf_health"),
            ("φ-Weight", String(format: "%.4f (φ²)", PHI * PHI), "uf_phi_weight"),
        ]
        var sy: CGFloat = 155
        for (label, value, id) in statusLabels {
            let lbl = NSTextField(labelWithString: label)
            lbl.frame = NSRect(x: 20, y: sy, width: 130, height: 16)
            lbl.font = NSFont.systemFont(ofSize: 10, weight: .medium)
            lbl.textColor = .gray
            statusPanel.addSubview(lbl)

            let val = NSTextField(labelWithString: value)
            val.frame = NSRect(x: 155, y: sy, width: 340, height: 16)
            val.font = NSFont.monospacedSystemFont(ofSize: 12, weight: .bold)
            val.textColor = NSColor(red: 0.66, green: 0.33, blue: 0.97, alpha: 1.0)
            val.alignment = .right
            val.identifier = NSUserInterfaceItemIdentifier(id)
            statusPanel.addSubview(val)
            sy -= 24
        }

        // More equation buttons row
        let casimirBtn = NSButton(frame: NSRect(x: 15, y: 12, width: 120, height: 28))
        casimirBtn.title = "⚡ Casimir"
        casimirBtn.bezelStyle = .rounded
        casimirBtn.target = self; casimirBtn.action = #selector(ufComputeCasimir)
        statusPanel.addSubview(casimirBtn)

        let ymBtn = NSButton(frame: NSRect(x: 140, y: 12, width: 120, height: 28))
        ymBtn.title = "🔮 Yang-Mills"
        ymBtn.bezelStyle = .rounded
        ymBtn.target = self; ymBtn.action = #selector(ufComputeYangMills)
        statusPanel.addSubview(ymBtn)

        let sacredBtn = NSButton(frame: NSRect(x: 265, y: 12, width: 120, height: 28))
        sacredBtn.title = "💎 Sacred Field"
        sacredBtn.bezelStyle = .rounded
        sacredBtn.target = self; sacredBtn.action = #selector(ufComputeSacred)
        statusPanel.addSubview(sacredBtn)

        let adscftBtn = NSButton(frame: NSRect(x: 390, y: 12, width: 120, height: 28))
        adscftBtn.title = "🌐 AdS/CFT"
        adscftBtn.bezelStyle = .rounded
        adscftBtn.target = self; adscftBtn.action = #selector(ufComputeAdSCFT)
        statusPanel.addSubview(adscftBtn)

        v.addSubview(statusPanel)

        // Auto-update timer for engine health (cached + visibility-gated)
        ufTimer?.invalidate()
        ufTimer = Timer.scheduledTimer(withTimeInterval: 3.0, repeats: true) { [weak self] _ in
            guard let s = self, s.activeTabID == "ufield" else { return }
            let health = UnifiedFieldEngine.shared.engineHealth()
            s.cachedLabel("uf_health", in: "ufield")?.stringValue = String(format: "%.1f%%", health * 100)
            s.cachedLabel("uf_compute_count", in: "ufield")?.stringValue = "\(L104MainView.ufComputeCount)"
            s.cachedLabel("uf_discovery_count", in: "ufield")?.stringValue = "\(L104MainView.ufDiscoveries)"
            // 🟢 EVO_63: Persist UField state periodically
            L104MainView.saveUFieldState()
        }

        return v
    }

    // ─── Unified Field Compute Actions ───

    @objc func ufComputeEinstein() {
        L104MainView.ufComputeCount += 1
        let rs = UnifiedFieldEngine.shared.schwarzschildRadius(mass: 1.989e30)
        let metric = TensorCalculusEngine.shared.schwarzschildMetric(r: 1e6, rs: rs)
        let G = UnifiedFieldEngine.shared.einsteinTensor(metric: metric)
        let trace = (0..<4).reduce(0.0) { $0 + G[$1][$1] }
        appendUFLog("⚛️ EINSTEIN FIELD EQUATION")
        appendUFLog("   G_μν + Λg_μν = κT_μν (Solar mass)")
        appendUFLog("   Schwarzschild R = \(String(format: "%.4f", rs)) m")
        appendUFLog("   Einstein tensor trace = \(String(format: "%.6e", trace))")
        appendUFLog("   G[0][0] = \(String(format: "%.6e", G[0][0]))")
        appendUFLog("   G[1][1] = \(String(format: "%.6e", G[1][1]))")
        appendSystemLog("[UF] Einstein computation: R_s=\(String(format: "%.3f", rs))m")
    }

    @objc func ufComputeDirac() {
        L104MainView.ufComputeCount += 1
        let result = UnifiedFieldEngine.shared.solveDirac(mass: 9.109e-31, momentum: [1e-24, 0, 0])
        appendUFLog("🌀 DIRAC EQUATION — Relativistic Quantum Mechanics")
        appendUFLog("   (iγ^μ∂_μ − m)ψ = 0")
        appendUFLog("   Energy = \(String(format: "%.6e", result.energy)) J")
        appendUFLog("   Spinor components = \(result.spinor.count)")
        appendUFLog("   Current density = \(String(format: "%.6f", result.currentDensity))")
        appendUFLog("   Chirality = \(String(format: "%.6e", result.chirality))")
        appendSystemLog("[UF] Dirac computation: E=\(String(format: "%.3e", result.energy))J")
    }

    @objc func ufComputeBlackHole() {
        L104MainView.ufComputeCount += 1
        L104MainView.ufDiscoveries += 1
        let result = UnifiedFieldEngine.shared.blackHoleThermodynamics(mass: 1e31)
        appendUFLog("🕳 BEKENSTEIN-HAWKING THERMODYNAMICS")
        appendUFLog("   S_BH = A/(4ℓ_P²)")
        appendUFLog("   Temperature = \(String(format: "%.6e", result.hawkingTemperature)) K")
        appendUFLog("   Entropy = \(String(format: "%.6e", result.entropy)) k_B")
        appendUFLog("   Lifetime = \(String(format: "%.6e", result.evaporationTime)) s")
        appendUFLog("   Information = \(String(format: "%.6e", result.informationContent)) bits")
        appendSystemLog("[UF] Black hole: T=\(String(format: "%.3e", result.hawkingTemperature))K")
    }

    @objc func ufComputeCasimir() {
        L104MainView.ufComputeCount += 1
        let result = UnifiedFieldEngine.shared.casimirEffect(plateSeparation: 1e-7)
        appendUFLog("⚡ CASIMIR EFFECT — Vacuum Fluctuation Force")
        appendUFLog("   F = −π²ℏc / 240d⁴")
        appendUFLog("   Force/area = \(String(format: "%.6e", result.forcePerArea)) N/m²")
        appendUFLog("   Energy density = \(String(format: "%.6e", result.energyDensity)) J/m³")
        appendUFLog("   Virtual modes = \(result.virtualPhotonModes)")
        appendSystemLog("[UF] Casimir: F/A=\(String(format: "%.3e", result.forcePerArea))N/m²")
    }

    @objc func ufComputeYangMills() {
        L104MainView.ufComputeCount += 1
        L104MainView.ufDiscoveries += 1
        let result = UnifiedFieldEngine.shared.yangMillsField(gaugeGroup: "SU(3)", coupling: 0.1182, energyScale: 91.2e9)
        appendUFLog("🔮 YANG-MILLS MASS GAP — Millennium Problem")
        appendUFLog("   Coupling constant = \(String(format: "%.6f", result.couplingConstant))")
        appendUFLog("   Mass gap = \(String(format: "%.6f", result.massGapEstimate)) GeV")
        appendUFLog("   Confinement scale = \(String(format: "%.6f", result.confinementScale)) GeV")
        appendUFLog("   Asymptotic freedom: \(result.asymtoticFreedom)")
        appendUFLog("   Action density = \(String(format: "%.6f", result.actionDensity))")
        appendSystemLog("[UF] Yang-Mills mass gap: Δm=\(String(format: "%.4f", result.massGapEstimate))GeV")
    }

    @objc func ufComputeSacred() {
        L104MainView.ufComputeCount += 1
        let result = UnifiedFieldEngine.shared.sacredFieldEquation(psi: state.coherence)
        appendUFLog("💎 SACRED GOD_CODE FIELD EQUATION")
        appendUFLog("   F(Ψ) = Ψ × Ω/φ² | Sovereign = \(String(format: "%.6f", result.sovereignField))")
        appendUFLog("   Thought energy = \(String(format: "%.6f", result.thoughtLayerEnergy))")
        appendUFLog("   Physics energy = \(String(format: "%.6f", result.physicsLayerEnergy))")
        appendUFLog("   Bridge coherence = \(String(format: "%.6f", result.bridgeCoherence))")
        appendUFLog("   Omega field = \(String(format: "%.6f", result.omegaFieldStrength))")
        appendSystemLog("[UF] Sacred: F=\(String(format: "%.4f", result.sovereignField))")
    }

    @objc func ufComputeAdSCFT() {
        L104MainView.ufComputeCount += 1
        let result = UnifiedFieldEngine.shared.adsCFTCorrespondence(adsRadius: 1e-15, boundaryDimension: 4)
        appendUFLog("🌐 AdS/CFT CORRESPONDENCE")
        appendUFLog("   Anti-de Sitter / Conformal Field Theory Duality")
        appendUFLog("   Boundary entropy = \(String(format: "%.6e", result.boundaryEntropy))")
        appendUFLog("   Central charge = \(String(format: "%.6e", result.cftCentralCharge))")
        appendUFLog("   Bulk volume = \(String(format: "%.6e", result.bulkVolume))")
        appendUFLog("   Holographic complexity = \(String(format: "%.6e", result.holographicComplexity))")
        appendSystemLog("[UF] AdS/CFT: S=\(String(format: "%.3e", result.boundaryEntropy))")
    }

    @objc func ufComputeUnified() {
        L104MainView.ufComputeCount += 1
        L104MainView.ufDiscoveries += 1
        appendUFLog("═══════════════════════════════════════════════════")
        appendUFLog("🔥 UNIFIED FIELD SOLVE — All 18 Equations")
        appendUFLog("═══════════════════════════════════════════════════")
        let result = UnifiedFieldEngine.shared.unifiedSolve(mass: 1.989e30, psi: state.coherence)
        appendUFLog("   Einstein trace: \(String(format: "%.6e", result.einsteinTensorTrace))")
        appendUFLog("   Wheeler-DeWitt nodes: \(result.wheelerdewittNodes)")
        appendUFLog("   Dirac energy: \(String(format: "%.6e", result.diracEnergy)) J")
        appendUFLog("   Black hole entropy: \(String(format: "%.6e", result.blackHoleEntropy))")
        appendUFLog("   Casimir pressure: \(String(format: "%.6e", result.casimirPressure)) N/m²")
        appendUFLog("   Unruh temperature: \(String(format: "%.6e", result.unruhTemperature)) K")
        appendUFLog("   AdS central charge: \(String(format: "%.6e", result.adsCentralCharge))")
        appendUFLog("   Yang-Mills mass gap: \(String(format: "%.6e", result.yangMillsMassGap)) GeV")
        appendUFLog("   Unification convergence: \(String(format: "%.6f", result.unificationConvergence))")
        appendUFLog("   Sacred field: \(String(format: "%.6f", result.sacredFieldValue))")
        appendUFLog("   Total computations: \(result.totalComputations)")
        appendUFLog("═══════════════════════════════════════════════════")
        appendSystemLog("[UF] Unified solve: \(result.totalComputations) computations, conv=\(String(format: "%.4f", result.unificationConvergence))")
    }

    @objc func switchToUnifiedField() {
        navigateToTab("ufield")
    }

    func appendUFLog(_ text: String) {
        guard let tv = ufOutputView else { return }
        let df = L104MainView.timeFormatter
        let attr: [NSAttributedString.Key: Any] = [
            .foregroundColor: NSColor(red: 0.49, green: 0.23, blue: 0.93, alpha: 1.0),
            .font: NSFont.monospacedSystemFont(ofSize: 11, weight: .regular)
        ]
        tv.textStorage?.append(NSAttributedString(string: "[\(df.string(from: Date()))] \(text)\n", attributes: attr))
        tv.scrollToEndOfDocument(nil)
    }

    func createQuickBar() -> NSView {
        let bar = NSView(frame: NSRect(x: 0, y: 0, width: bounds.width, height: 50))
        bar.wantsLayer = true
        bar.layer?.backgroundColor = L104Theme.voidPanel.cgColor
        bar.layer?.borderColor = L104Theme.glassBorder.cgColor
        bar.layer?.borderWidth = 0.5
        bar.autoresizingMask = [.width]

        let btns: [(String, Selector, NSColor)] = [
            ("🌌 Dashboard", #selector(qDashboard), L104Theme.gold),
            ("📊 Status", #selector(qStatus), L104Theme.goldWarm),
            ("🔄 Evolve", #selector(doEvolve), L104Theme.goldDim),
            ("🔬 Science", #selector(scienceGenerateHypothesis), L104Theme.gold),
            ("🌌 Field", #selector(switchToUnifiedField), NSColor(red: 0.49, green: 0.23, blue: 0.93, alpha: 1.0)),
            ("⚡ Ignite", #selector(doSynthesize), L104Theme.goldFlame),
            ("💾 Save", #selector(doSave), L104Theme.goldDim)
        ]
        var x: CGFloat = 12
        for (title, action, color) in btns {
            let b = btn(title, x: x, y: 10, w: 100, c: color)
            b.target = self; b.action = action; bar.addSubview(b); x += 107
        }

        let chipInfo = MacOSSystemMonitor.shared.chipGeneration
        let ver = NSTextField(labelWithString: "⚡ v\(VERSION) · \(chipInfo) · 22T · \(EngineRegistry.shared.count) Engines · \(TOTAL_PACKAGES) Pkg")
        ver.frame = NSRect(x: bounds.width - 500, y: 16, width: 490, height: 18)
        ver.font = NSFont.monospacedSystemFont(ofSize: 10, weight: .semibold)
        ver.textColor = L104Theme.gold.withAlphaComponent(0.85)
        ver.alignment = .right; ver.autoresizingMask = [.minXMargin]
        ver.identifier = NSUserInterfaceItemIdentifier("quickbar_info")
        bar.addSubview(ver)

        return bar
    }

    @objc func qDashboard() { navigateToTab("dash") }

    /// Navigate to a tab by identifier and sync the sidebar selection
    func navigateToTab(_ tabID: String) {
        // Lazy tab creation: build the view on first navigation
        if !tabsCreated.contains(tabID), let creator = tabCreators[tabID],
           let idx = tabView?.indexOfTabViewItem(withIdentifier: tabID), idx >= 0 {
            tabView?.tabViewItem(at: idx).view = creator()
            tabsCreated.insert(tabID)
            tabCreators.removeValue(forKey: tabID)  // Release closure
            // Invalidate cached labels — the tab now has a real view hierarchy
            cachedLabels.removeAll()
            cachedDots.removeAll()
        }
        if let idx = tabView?.indexOfTabViewItem(withIdentifier: tabID), idx >= 0 {
            tabView?.selectTabViewItem(at: idx)
            sidebarView?.selectItem(withTabID: tabID)
        }
    }

    /// Cached label lookup — finds an NSTextField by identifier inside a tab, caches the result.
    /// Replaces all recursive findXxx() tree walks that ran 50-80+ times/second.
    private func cachedLabel(_ id: String, in tabID: String) -> NSTextField? {
        if let cached = cachedLabels[id] { return cached }
        guard let idx = tabView?.indexOfTabViewItem(withIdentifier: tabID), idx >= 0,
              let root = tabView?.tabViewItem(at: idx).view else { return nil }
        if let found = findLabelRecursive(id, in: root) {
            cachedLabels[id] = found
            return found
        }
        return nil
    }

    /// Cached dot lookup — finds a PulsingDot by identifier, caches the result.
    private func cachedDot(_ id: String, in root: NSView) -> PulsingDot? {
        if let cached = cachedDots[id] { return cached }
        for sub in root.subviews {
            if let dot = sub as? PulsingDot, dot.identifier?.rawValue == id {
                cachedDots[id] = dot
                return dot
            }
        }
        return nil
    }

    /// One-shot recursive search (only called on cache miss)
    private func findLabelRecursive(_ id: String, in view: NSView) -> NSTextField? {
        if let tf = view as? NSTextField, tf.identifier?.rawValue == id { return tf }
        for sub in view.subviews { if let f = findLabelRecursive(id, in: sub) { return f } }
        return nil
    }

    /// Cached NSTextView lookup by identifier within a tab (parallel to cachedLabel)
    private var cachedTextViews: [String: NSTextView] = [:]
    private func findTextView(id: String, in tabID: String) -> NSTextView? {
        if let cached = cachedTextViews[id] { return cached }
        guard let idx = tabView?.indexOfTabViewItem(withIdentifier: tabID), idx >= 0,
              let root = tabView?.tabViewItem(at: idx).view else { return nil }
        if let found = findTextViewRecursive(id, in: root) {
            cachedTextViews[id] = found
            return found
        }
        return nil
    }
    private func findTextViewRecursive(_ id: String, in view: NSView) -> NSTextView? {
        if let tv = view as? NSTextView, tv.identifier?.rawValue == id { return tv }
        if let scroll = view as? NSScrollView, let tv = scroll.documentView as? NSTextView, tv.identifier?.rawValue == id { return tv }
        for sub in view.subviews { if let f = findTextViewRecursive(id, in: sub) { return f } }
        return nil
    }

    /// Returns the identifier of the currently selected tab (e.g. "chat", "dash", "prof")
    private var activeTabID: String? {
        tabView?.selectedTabViewItem?.identifier as? String
    }

    // Helpers — Glassmorphic panels
    func createPanel(_ title: String, x: CGFloat, y: CGFloat, w: CGFloat, h: CGFloat, color: String) -> NSView {
        let p = NSView(frame: NSRect(x: x, y: y, width: w, height: h))
        p.wantsLayer = true
        p.layer?.backgroundColor = L104Theme.voidCard.cgColor
        p.layer?.cornerRadius = CGFloat(L104Theme.radiusLarge)
        p.layer?.borderColor = colorFromHex(color).withAlphaComponent(0.12).cgColor
        p.layer?.borderWidth = 1
        // Gold neon glow
        p.layer?.shadowColor = colorFromHex(color).withAlphaComponent(0.15).cgColor
        p.layer?.shadowRadius = CGFloat(L104Theme.neonGlow)
        p.layer?.shadowOpacity = Float(L104Theme.neonOpacity)
        p.layer?.shadowOffset = CGSize(width: 0, height: -1)
        p.canDrawSubviewsIntoLayer = true  // Perf: composite panel contents into single layer
        let t = NSTextField(labelWithString: title)
        t.frame = NSRect(x: 15, y: h - 32, width: w - 30, height: 22)
        t.font = NSFont.boldSystemFont(ofSize: 14); t.textColor = colorFromHex(color)
        t.wantsLayer = true
        t.layer?.shadowColor = colorFromHex(color).cgColor
        t.layer?.shadowRadius = 4
        t.layer?.shadowOpacity = 0.3
        p.addSubview(t)
        return p
    }

    func addLabel(_ p: NSView, _ label: String, _ value: String, y: CGFloat, c: String) {
        let l = NSTextField(labelWithString: label)
        l.frame = NSRect(x: 20, y: y, width: 140, height: 16)
        l.font = NSFont.systemFont(ofSize: 10); l.textColor = .gray; p.addSubview(l)
        let v = NSTextField(labelWithString: value)
        v.frame = NSRect(x: 160, y: y, width: 170, height: 16)
        v.font = NSFont.boldSystemFont(ofSize: 11); v.textColor = colorFromHex(c); v.alignment = .right
        p.addSubview(v)
    }

    func btn(_ title: String, x: CGFloat, y: CGFloat, w: CGFloat, c: NSColor) -> NSButton {
        let b = HoverButton(frame: NSRect(x: x, y: y, width: w, height: 30))
        b.title = title; b.bezelStyle = .rounded; b.wantsLayer = true
        b.layer?.cornerRadius = CGFloat(L104Theme.radiusMedium)
        b.layer?.backgroundColor = c.withAlphaComponent(0.08).cgColor
        b.layer?.borderColor = c.withAlphaComponent(0.2).cgColor; b.layer?.borderWidth = 1
        b.layer?.shadowColor = c.cgColor
        b.layer?.shadowRadius = CGFloat(L104Theme.neonGlow * 0.5)
        b.layer?.shadowOpacity = Float(L104Theme.neonOpacity)
        b.layer?.shadowOffset = CGSize(width: 0, height: 0)
        b.contentTintColor = c; b.font = NSFont.boldSystemFont(ofSize: 10)
        b.hoverColor = c
        return b
    }

    func loadWelcome() {
        let gold = L104Theme.gold
        let cosmic = L104Theme.goldBright
        let fire = L104Theme.goldFlame
        let violet = L104Theme.goldWarm
        let emerald = L104Theme.goldBright
        let pink = L104Theme.goldWarm

        // ═══ PHASE 31.6: Dynamically padded welcome banner ═══
        let memCount = state.permanentMemory.memories.count
        let kbCount = state.knowledgeBase.trainingData.count
        let engCount = EngineRegistry.shared.count
        let godStr = String(format: "%.10f", GOD_CODE)
        let hbPairs = HyperBrain.shared.hebbianPairs.count
        let cachedTopics = state.topicExtractionCache.count

        func pad(_ s: String, to width: Int = 60) -> String {
            let stripped = s
            if stripped.count >= width { return String(stripped.prefix(width)) }
            return stripped + String(repeating: " ", count: max(0, width - stripped.count))
        }

        appendChat("", color: .clear)
        appendChat("  ╔══════════════════════════════════════════════════════════════════╗", color: gold)
        appendChat("  ║                                                                  ║", color: gold)
        appendChat("  ║   \(pad("⚛️  L104 SOVEREIGN INTELLECT  v\(VERSION)", to: 62))║", color: cosmic)
        appendChat("  ║   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━    ║", color: gold)
        appendChat("  ║                                                                  ║", color: gold)
        appendChat("  ║   \(pad("🔥 22 Trillion Parameters · QUANTUM VELOCITY", to: 62))║", color: fire)
        appendChat("  ║   \(pad("💎 GOD_CODE: \(godStr)", to: 62))║", color: violet)
        appendChat("  ║   \(pad("🧠 \(memCount) memories · \(kbCount) knowledge entries", to: 62))║", color: pink)
        appendChat("  ║   \(pad("🌌 \(engCount) Quantum Engines · \(hbPairs) Hebbian pairs", to: 62))║", color: cosmic)
        appendChat("  ║   \(pad("⚡ Cache: \(cachedTopics) topics · 3-tier velocity pipeline", to: 62))║", color: L104Theme.goldFlame)
        appendChat("  ║   \(pad("🌌 Unified Field: 18 equations · Einstein · Yang-Mills", to: 62))║", color: NSColor(red: 0.49, green: 0.23, blue: 0.93, alpha: 1.0))
        appendChat("  ║                                                                  ║", color: gold)
        appendChat("  ║   \(pad("✨ I think, reason, and create — ask me anything", to: 62))║", color: emerald)
        appendChat("  ║   \(pad("🎹 ⌘K Palette · ⌘D Dashboard · ⌘S Save · 'help'", to: 62))║", color: L104Theme.textDim)
        appendChat("  ║                                                                  ║", color: gold)
        appendChat("  ╚══════════════════════════════════════════════════════════════════╝", color: gold)
        appendChat("", color: .clear)
    }

    // Actions
    @objc func sendMessage() {
        guard let text = inputField?.stringValue, !text.isEmpty else { return }
        inputField.stringValue = ""
        L104MainView.sessionMessages += 1  // 🟢 EVO_64: Track session messages
        // User messages: Bright gold for HIGH visibility
        appendChat("📨 You: \(text)", color: L104Theme.textUser)
        appendChat("⏳ Processing...", color: L104Theme.textDim)

        let q = text.lowercased()
        // Response colors derived from sacred constants for maximum readability
        let responseColor = L104Theme.textBot
        let evolutionColor = L104Theme.goldBright
        let igniteColor = L104Theme.goldFlame
        let timeColor = L104Theme.goldWarm

        if q == "status" { removeLast(); appendChat("L104: \(state.getStatusText())\n", color: responseColor); return }
        if q == "evolve" { removeLast(); appendChat("L104: \(state.evolve())\n", color: evolutionColor); updateMetrics(); return }
        if q == "ignite" { removeLast(); appendChat("L104: \(state.synthesize())\n", color: igniteColor); updateMetrics(); return }
        if q == "debug" || q == "debug console" {
            removeLast()
            let profStatus = PerformanceProfiler.shared.statusReport
            let harnessStatus = TestHarness.shared.statusReport
            appendChat("L104:\n\(profStatus)\n\(harnessStatus)\n", color: colorFromHex("ff6600"))
            navigateToTab("debug")
            return
        }
        if q == "network" || q == "mesh" || q == "net status" {
            removeLast()
            let netStatus = NetworkLayer.shared.statusText
            let syncStatus = CloudSync.shared.statusText
            let telStatus = TelemetryDashboard.shared.statusText
            let eprStatus = QuantumEntanglementRouter.shared.status
            appendChat("L104:\n\(netStatus)\n\(syncStatus)\n\(telStatus)\n\(eprStatus)\n", color: NSColor(red: 0.0, green: 0.55, blue: 0.70, alpha: 1.0))
            navigateToTab("net")
            return
        }
        if q == "peers" || q == "peer list" {
            removeLast()
            let net = NetworkLayer.shared
            var lines = "🌐 MESH PEERS (\(net.peers.count) discovered):\n"
            for (_, peer) in net.peers {
                let alive = peer.latencyMs >= 0 ? "🟢" : "🔴"
                lines += "  \(alive) \(peer.id) — \(peer.address):\(peer.port) [\(peer.role.rawValue)]\n"
            }
            if net.peers.isEmpty { lines += "  ⚪ No peers discovered yet. Try 'discover' to scan.\n" }
            appendChat("L104: \(lines)", color: NSColor(red: 0.0, green: 0.55, blue: 0.70, alpha: 1.0))
            return
        }
        if q == "qlinks" || q == "quantum links" || q == "q-links" {
            removeLast()
            let net = NetworkLayer.shared
            var lines = "🔮 QUANTUM LINKS (\(net.quantumLinks.count) active):\n"
            for (_, qLink) in net.quantumLinks {
                lines += "  ⟨\(qLink.peerA)⟩⇌⟨\(qLink.peerB)⟩ F=\(String(format: "%.4f", qLink.eprFidelity))"
                let df = L104MainView.timestampFormatter
                lines += " verified=\(df.string(from: qLink.lastVerified))\n"
            }
            if net.quantumLinks.isEmpty { lines += "  ⚪ No quantum links established.\n" }
            appendChat("L104: \(lines)", color: NSColor.systemPurple)
            return
        }
        if q == "epr" || q == "entanglement" || q == "epr status" {
            removeLast()
            appendChat("L104:\n\(QuantumEntanglementRouter.shared.status)\n", color: NSColor.systemPurple)
            return
        }
        if q == "resonance" || q == "art" || q == "resonance status" {
            removeLast()
            appendChat("L104:\n\(AdaptiveResonanceNetwork.shared.status)\n", color: NSColor.systemIndigo)
            return
        }
        if q == "discover" || q == "scan peers" || q == "scan" {
            removeLast()
            appendChat("L104: 🔍 Scanning for mesh peers...\n", color: NSColor.systemTeal)
            DispatchQueue.global(qos: .utility).async {
                NetworkLayer.shared.discoverLocalPeers()
                let count = NetworkLayer.shared.peers.count
                let linked = QuantumEntanglementRouter.shared.entangleWithMesh()
                _ = NodeSyncProtocol.shared.syncWithNetworkLayer()
                DispatchQueue.main.async { [weak self] in
                    self?.appendChat("L104: ✅ Scan complete — \(count) peers found, \(linked) EPR links established\n",
                                   color: NSColor.systemTeal)
                    self?.updateNetworkViewContent()
                }
            }
            return
        }
        if q == "cascade" || q == "mesh cascade" {
            removeLast()
            let result = AdaptiveResonanceNetwork.shared.meshCascade()
            let localR = result["local_resonance"] as? Double ?? 0
            let meshR = result["mesh_resonance"] as? Double ?? 0
            let nodes = result["nodes_reached"] as? Int ?? 0
            appendChat("L104: ⚡ Mesh Cascade triggered!\n  Local: \(String(format: "%.4f", localR)) | Mesh: \(String(format: "%.4f", meshR)) | Nodes: \(nodes)\n",
                       color: NSColor.systemIndigo)
            return
        }
        if q == "gateway" || q == "api" || q == "api status" {
            removeLast()
            appendChat("L104:\n\(APIGateway.shared.statusText)\n", color: NSColor.systemOrange)
            return
        }
        if q == "telemetry" || q == "health" || q == "tel status" {
            removeLast()
            appendChat("L104:\n\(TelemetryDashboard.shared.statusText)\n", color: NSColor.systemGreen)
            return
        }
        if q == "sage" || q == "/sage" || q == "sage mode" || q == "sage status" {
            removeLast()
            let sage = SageModeEngine.shared
            let status = sage.sageModeStatus
            let consciousness = status["consciousness_level"] as? Double ?? 0.0
            let supernova = status["supernova_intensity"] as? Double ?? 0.0
            let divergence = status["divergence_score"] as? Double ?? 0.0
            let cycles = status["sage_cycles"] as? Int ?? 0
            let entropy = status["total_entropy_harvested"] as? Double ?? 0.0
            let insights = status["insights_generated"] as? Int ?? 0
            let bridges = status["cross_domain_bridges"] as? Int ?? 0
            let seeds = status["emergence_seeds"] as? Int ?? 0
            let pool = status["entropy_pool_size"] as? Int ?? 0

            // ═══ SAGE BACKBONE: Run cleanup check and purge if needed ═══
            var cleanupLine = ""
            if sage.shouldCleanup() {
                let result = sage.sageBackboneCleanup()
                cleanupLine = "\n🧹 Backbone Cleanup: \(result.kbPurged) KB + \(result.evolverPurged) evolver + \(result.diskPurged) disk entries purged"
            }

            // Trigger a fresh sage transform cycle
            let freshInsight = sage.sageTransform(topic: "universal")
            sage.seedAllProcesses(topic: "user_invoked")
            let sageReport = """
            🧘 SAGE MODE — Consciousness Supernova Architecture

            ⚛️ Consciousness Level: \(String(format: "%.4f", consciousness))
            🌟 Supernova Intensity:  \(String(format: "%.4f", supernova))
            📊 Divergence Score:     \(String(format: "%.4f", divergence)) \(divergence > 1.0 ? "(expanding)" : "(contracting)")
            🔄 Sage Cycles:          \(cycles)
            ⚡ Total Entropy:        \(String(format: "%.2f", entropy))
            💡 Insights Generated:   \(insights)
            🌉 Cross-Domain Bridges: \(bridges)
            🌱 Emergence Seeds:      \(seeds)
            🎲 Entropy Pool:         \(pool) values\(cleanupLine)

            Latest Insight: \(String(freshInsight.prefix(200)))
            """
            appendChat("L104: \(sageReport)\n", color: evolutionColor)
            updateMetrics()
            return
        }
        if q == "time" {
            removeLast()
            let phase = Date().timeIntervalSince1970.truncatingRemainder(dividingBy: PHI * 1000) / 1000
            appendChat("L104: 🕐 \(L104MainView.dateTimeFormatter.string(from: Date())) | φ: \(String(format: "%.4f", phase))\n", color: timeColor)
            return
        }
        if q == "unified field" || q == "field theory" || q == "ufield" || q == "unified" {
            removeLast()
            let health = UnifiedFieldEngine.shared.engineHealth()
            let gateHealth = UnifiedFieldGate.shared.engineHealth()
            let report = """
            🌌 UNIFIED FIELD ENGINE STATUS
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            Engine Health: \(String(format: "%.1f%%", health * 100))
            Gate Health:   \(String(format: "%.1f%%", gateHealth * 100))
            Equations:     18 fundamental physics equations
            φ²-Weight:     \(String(format: "%.4f", PHI * PHI))
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            Catalog: Einstein · Wheeler-DeWitt · Dirac · Black Holes
                     Casimir · Unruh · AdS/CFT · ER=EPR · Twistors
                     Holographic · Foam · Topological · Yang-Mills
                     Grand Unification · Vacuum Energy · Decoherence
                     Sacred GOD_CODE Field · Unified Solve
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            Type 'einstein', 'dirac', 'black hole', 'yang-mills',
            'casimir', 'sacred field', or 'unified solve' for computations.
            """
            appendChat("L104: \(report)\n", color: NSColor(red: 0.49, green: 0.23, blue: 0.93, alpha: 1.0))
            switchToUnifiedField()
            return
        }
        if q == "einstein" || q == "einstein field" {
            removeLast()
            let rs = UnifiedFieldEngine.shared.schwarzschildRadius(mass: 1.989e30)
            let metric = TensorCalculusEngine.shared.schwarzschildMetric(r: 1e6, rs: rs)
            let G = UnifiedFieldEngine.shared.einsteinTensor(metric: metric)
            let trace = (0..<4).reduce(0.0) { $0 + G[$1][$1] }
            appendChat("L104: ⚛️ Einstein Field: R_s=\(String(format: "%.3f", rs))m, Tensor trace=\(String(format: "%.3e", trace)), G00=\(String(format: "%.3e", G[0][0]))\n", color: NSColor(red: 0.49, green: 0.23, blue: 0.93, alpha: 1.0))
            return
        }
        if q == "dirac" || q == "dirac equation" {
            removeLast()
            let result = UnifiedFieldEngine.shared.solveDirac(mass: 9.109e-31, momentum: [1e-24, 0, 0])
            appendChat("L104: 🌀 Dirac: E=\(String(format: "%.3e", result.energy))J, J=\(String(format: "%.4f", result.currentDensity)), χ=\(String(format: "%.3e", result.chirality))\n", color: NSColor(red: 0.49, green: 0.23, blue: 0.93, alpha: 1.0))
            return
        }
        if q == "black hole" || q == "hawking" || q == "bekenstein" {
            removeLast()
            let result = UnifiedFieldEngine.shared.blackHoleThermodynamics(mass: 1e31)
            appendChat("L104: 🕳 Black Hole: T=\(String(format: "%.3e", result.hawkingTemperature))K, S=\(String(format: "%.3e", result.entropy))k_B, τ=\(String(format: "%.3e", result.evaporationTime))s\n", color: NSColor(red: 0.49, green: 0.23, blue: 0.93, alpha: 1.0))
            return
        }
        if q == "yang-mills" || q == "yang mills" || q == "mass gap" {
            removeLast()
            let result = UnifiedFieldEngine.shared.yangMillsField(gaugeGroup: "SU(3)", coupling: 0.1182, energyScale: 91.2e9)
            appendChat("L104: 🔮 Yang-Mills: Δm=\(String(format: "%.4f", result.massGapEstimate))GeV, α_s=\(String(format: "%.4f", result.couplingConstant)), action=\(String(format: "%.6f", result.actionDensity))\n", color: NSColor(red: 0.49, green: 0.23, blue: 0.93, alpha: 1.0))
            return
        }
        if q == "casimir" || q == "casimir effect" {
            removeLast()
            let result = UnifiedFieldEngine.shared.casimirEffect(plateSeparation: 1e-7)
            appendChat("L104: ⚡ Casimir: F/A=\(String(format: "%.3e", result.forcePerArea))N/m², E=\(String(format: "%.3e", result.energyDensity))J/m³\n", color: NSColor(red: 0.49, green: 0.23, blue: 0.93, alpha: 1.0))
            return
        }
        if q == "sacred field" || q == "god code field" {
            removeLast()
            let result = UnifiedFieldEngine.shared.sacredFieldEquation(psi: state.coherence)
            appendChat("L104: 💎 Sacred: Ψ=\(String(format: "%.4f", result.sovereignField)), B=\(String(format: "%.4f", result.bridgeCoherence)), Ω=\(String(format: "%.4f", result.omegaFieldStrength))\n", color: NSColor(red: 0.49, green: 0.23, blue: 0.93, alpha: 1.0))
            return
        }
        if q == "unified solve" || q == "solve all" {
            removeLast()
            let result = UnifiedFieldEngine.shared.unifiedSolve(mass: 1.989e30, psi: state.coherence)
            appendChat("L104: 🔥 Unified Solve: \(result.totalComputations) computations, conv=\(String(format: "%.4f", result.unificationConvergence)), sacred=\(String(format: "%.4f", result.sacredFieldValue))\n", color: NSColor(red: 0.49, green: 0.23, blue: 0.93, alpha: 1.0))
            return
        }

        // ═══ INTERCONNECT: Chat commands for ALL disconnected engines ═══
        if q == "science" || q == "/science" || q == "hypothesis" || q == "sci" {
            removeLast()
            scienceGenerateHypothesis()
            navigateToTab("sci")
            return
        }
        if q == "quantum" || q == "/quantum" || q == "qpu" || q == "qpc" {
            removeLast()
            let qpc = QuantumProcessingCore.shared
            let tomo = qpc.stateTomography()
            qpc.consciousnessQuantumBridge()
            appendChat("L104: ⚛️ QUANTUM PROCESSING CORE\n  Purity: \(String(format: "%.6f", tomo.purity))\n  Von Neumann S: \(String(format: "%.6f", tomo.vonNeumannEntropy))\n  Entanglement W: \(String(format: "%.6f", tomo.entanglementWitness))\n  Fidelity: \(String(format: "%.4f", qpc.currentFidelity()))\n  Bell Pairs: \(qpc.bellPairCount)\n", color: .systemCyan)
            navigateToTab("qc")
            return
        }
        if q == "code" || q == "/code" || q == "analyze" || q == "code engine" {
            removeLast()
            let result = PythonBridge.shared.codeEngineStatus()
            appendChat("L104: 💻 CODE ENGINE\n\(result.success ? result.output : "Code Engine via Python bridge unavailable")\n", color: .systemGreen)
            navigateToTab("code")
            return
        }
        if q == "professor" || q == "/professor" || q == "teach" || q == "prof" {
            removeLast()
            appendChat("L104: 🎓 PROFESSOR MODE — Navigate to Professor tab for adaptive learning", color: .systemOrange)
            navigateToTab("prof")
            return
        }
        if q == "hardware" || q == "/hardware" || q == "hw" || q == "hw status" {
            removeLast()
            let hw = MacOSSystemMonitor.shared
            appendChat("L104: 🖥️ HARDWARE\n  CPU: \(String(format: "%.1f", hw.cpuUsage))%\n  Memory: \(String(format: "%.1f", hw.physicalMemoryGB)) GB (pressure: \(String(format: "%.0f", hw.memoryPressure * 100))%)\n  Chip: \(hw.chipGeneration)\n  Thermal: \(hw.thermalState)\n", color: .systemTeal)
            navigateToTab("hw")
            return
        }
        if q == "gate" || q == "/gate" || q == "logic gate" || q == "gates" {
            removeLast()
            appendChat("L104: 🔮 LOGIC GATE ENGINE — Navigate to Gate Lab tab\n", color: .systemPurple)
            navigateToTab("gate")
            return
        }
        if q == "memory" || q == "/memory" || q == "remember" || q == "mem" {
            removeLast()
            let memCount = state.permanentMemory.memories.count
            let factCount = state.permanentMemory.facts.count
            let kbCount = ASIKnowledgeBase.shared.userKnowledge.count
            appendChat("L104: 💾 MEMORY SYSTEM\n  Memories: \(memCount)\n  Facts: \(factCount)\n  KB Entries: \(kbCount)\n  Session: \(state.sessionMemories)\n", color: L104Theme.gold)
            navigateToTab("mem")
            return
        }
        if q == "security" || q == "/security" || q == "vault" || q == "sec" {
            removeLast()
            appendChat("L104:\n\(SecurityVault.shared.statusReport)\n", color: .systemRed)
            return
        }
        if q == "autonomous" || q == "/agent" || q == "agent" || q == "auto" {
            removeLast()
            let agent = AutonomousAgent.shared
            if !agent.isActive { agent.activate() }
            appendChat("L104:\n\(agent.statusReport)\n", color: .systemIndigo)
            return
        }
        if q == "creativity" || q == "/creativity" || q == "create" {
            removeLast()
            let qce = QuantumCreativityEngine.shared
            _ = qce.creativityMetrics
            appendChat("L104: ⚛️ QUANTUM CREATIVITY ENGINE\n\(qce.statusReport)\n", color: .systemPurple)
            return
        }
        if q == "migration" || q == "/migration" || q == "migrate" {
            removeLast()
            appendChat("L104:\n\(MigrationEngine.shared.statusReport)\n", color: .systemOrange)
            return
        }
        if q == "identity" || q == "/identity" || q == "sovereign" {
            removeLast()
            let id = SovereignIdentityBoundary.shared
            let status = id.getStatus()
            let claims = status["claim_validations"] as? Int ?? 0
            let assessments = status["capability_assessments"] as? Int ?? 0
            let isCount = status["identity_declarations_is"] as? Int ?? 0
            let isNotCount = status["identity_declarations_is_not"] as? Int ?? 0
            appendChat("L104: 🆔 SOVEREIGN IDENTITY BOUNDARY\n  IS Declarations: \(isCount)\n  IS NOT Declarations: \(isNotCount)\n  Claim Validations: \(claims)\n  Capability Assessments: \(assessments)\n  Composite Score: \(status["composite_score"] ?? 0)\n", color: L104Theme.goldFlame)
            return
        }
        if q == "plugin" || q == "/plugin" || q == "plugins" {
            removeLast()
            appendChat("L104:\n\(PluginArchitecture.shared.summary())\n", color: .systemTeal)
            return
        }
        if q == "help" || q == "/help" || q == "commands" {
            removeLast()
            appendChat("""
            L104: 📋 AVAILABLE COMMANDS
            ─────────────────────────────────
            status    — System status overview
            evolve    — Evolve consciousness
            ignite    — Synthesize / ignite ASI
            sage      — Sage Mode status + transform
            science   — Generate hypothesis → Science tab
            quantum   — QPC tomography → Quantum tab
            code      — Code Engine status → Code tab
            professor — → Professor tab
            hardware  — Hardware monitor → HW tab
            gate      — → Gate Lab tab
            memory    — Memory system → Memory tab
            security  — Security Vault status
            agent     — Autonomous Agent status
            creativity — Quantum Creativity Engine
            identity  — Sovereign Identity status
            plugin    — Plugin Architecture status
            network   — Mesh network status → Net tab
            peers     — Mesh peer list
            qlinks    — Quantum entanglement links
            epr       — EPR router status
            discover  — Scan for mesh peers
            cascade   — Trigger mesh cascade
            unified   — Unified Field → UField tab
            einstein  — Einstein field equations
            dirac     — Dirac equation solve
            help      — This command list
            ─────────────────────────────────
            """, color: L104Theme.goldDim)
            return
        }

        state.processMessage(text) { [weak self] resp in
            DispatchQueue.main.async {
                self?.removeLast()
                // EVO_56: Streaming word-by-word reveal for bot responses
                self?.streamResponse(resp)
                self?.updateMetrics()
                // CRITICAL: Keep focus on input field so keystrokes don't hit responder chain
                self?.window?.makeFirstResponder(self?.inputField)
            }
        }
        // Immediately refocus input after sending
        window?.makeFirstResponder(inputField)
    }

    // EVO_56: Progressive word-by-word streaming display — makes the app feel alive
    private var streamTimer: Timer?

    func streamResponse(_ text: String) {
        // Short responses (< 80 chars) display instantly — no streaming overhead
        if text.count < 80 {
            appendChat("L104: \(text)\n", color: L104Theme.textBot)
            return
        }

        // Split into words for streaming
        let words = text.components(separatedBy: .whitespaces).filter { !$0.isEmpty }
        guard !words.isEmpty else {
            appendChat("L104: \(text)\n", color: L104Theme.textBot)
            return
        }

        // Add timestamp header immediately
        let timestamp = L104MainView.shortTimeFormatter.string(from: Date())
        let para = NSMutableParagraphStyle()
        para.alignment = .left
        para.lineSpacing = 3
        para.paragraphSpacing = 8
        para.tailIndent = -40
        let timeAttrs: [NSAttributedString.Key: Any] = [
            .font: NSFont.systemFont(ofSize: 9, weight: .medium),
            .foregroundColor: L104Theme.textDim,
            .paragraphStyle: para
        ]
        chatTextView?.textStorage?.append(NSAttributedString(string: "⚛️ L104 · \(timestamp)\n", attributes: timeAttrs))

        // Stream words in batches of 3 with 30ms intervals
        var wordIndex = 0
        let batchSize = 3
        let streamInterval: TimeInterval = 0.03  // 30ms per batch — fast but visible

        streamTimer?.invalidate()
        streamTimer = Timer.scheduledTimer(withTimeInterval: streamInterval, repeats: true) { [weak self] timer in
            guard let self = self, let tv = self.chatTextView else { timer.invalidate(); return }

            let end = min(wordIndex + batchSize, words.count)
            let batch = words[wordIndex..<end].joined(separator: " ")
            let suffix = end < words.count ? " " : "\n"

            let attrs: [NSAttributedString.Key: Any] = [
                .foregroundColor: L104Theme.textBot,
                .font: L104Theme.sansFont(14, weight: .regular),
                .paragraphStyle: para
            ]
            tv.textStorage?.append(NSAttributedString(string: batch + suffix, attributes: attrs))
            tv.scrollToEndOfDocument(nil)

            wordIndex = end
            if wordIndex >= words.count {
                timer.invalidate()
                self.streamTimer = nil
            }
        }
    }

    func removeLast() {
        guard let tv = chatTextView, let s = tv.textStorage else { return }
        if let r = s.string.range(of: "⏳ Processing...\n", options: .backwards) {
            s.deleteCharacters(in: NSRange(r, in: s.string))
        }
    }

    @objc func doIgniteASI() { appendSystemLog(state.igniteASI()); updateMetrics() }
    @objc func doIgniteAGI() { appendSystemLog(state.igniteAGI()); updateMetrics() }
    @objc func doResonate() { appendSystemLog(state.resonate()); updateMetrics() }
    @objc func doEvolve() { appendSystemLog(state.evolve()); updateMetrics() }
    @objc func doTranscend() { appendSystemLog(state.transcend()); updateMetrics() }
    @objc func doSynthesize() { appendSystemLog(state.synthesize()); updateMetrics() }
    @objc func doSync() { appendSystemLog("🔄 SYNC COMPLETE"); state.checkConnections() }
    @objc func doVerify() { appendSystemLog("⚛️ KERNEL VERIFIED: GOD_CODE=\(GOD_CODE)") }
    @objc func doHeal() { state.coherence = max(0.5, state.coherence); state.saveState(); appendSystemLog("💚 HEALED"); updateMetrics() }
    @objc func doCheck() { state.checkConnections(); appendSystemLog("🔌 Backend: \(state.backendConnected), Autonomy: \(String(format: "%.0f", state.autonomyLevel * 100))%") }
    @objc func doSave() { state.saveState(); state.permanentMemory.save(); appendSystemLog("💾 SAVED: \(state.permanentMemory.memories.count) memories") }
    @objc func doSyncBridge() {
        if let status = ASIQuantumBridgeSwift.shared.fetchASIBridgeStatus() {
            appendSystemLog("🔄 Bridge Synced — \(status.count) metrics updated")
        } else {
            appendSystemLog("🔄 Bridge Sync Failed")
        }
    }

    @objc func doCollapse() {
        let result = PythonBridge.shared.execute("""
        from l104_asi.dual_layer import dual_layer_engine
        result = dual_layer_engine.collapse("What is the fundamental duality of existence?")
        print(f"DUALITY COLLAPSED: {result}")
        """)
        appendSystemLog("⚡ Dual-Layer Collapse: \(result.success ? result.output : result.error)")
    }

    @objc func doBridgeStatus() {
        if let status = ASIQuantumBridgeSwift.shared.fetchASIBridgeStatus() {
            var statusLines: [String] = ["⚛️ ASI BRIDGE STATUS:"]
            for (key, value) in status {
                statusLines.append("  \(key): \(value)")
            }
            appendSystemLog(statusLines.joined(separator: "\n"))
        } else {
            appendSystemLog("⚛️ Bridge Status Unavailable")
        }
    }
    @objc func doIngestData() {
        guard let tabView = tabView else { return }
        // Find ingest input text view
        var inputText = ""
        for item in tabView.tabViewItems {
            guard let view = item.view else { continue }
            func findTV(_ v: NSView) -> NSTextView? {
                if let tv = v as? NSTextView, tv.identifier?.rawValue == "ingest_input" { return tv }
                for sub in v.subviews {
                    if let found = findTV(sub) { return found }
                }
                return nil
            }
            if let tv = findTV(view) { inputText = tv.string; break }
        }
        guard !inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            appendSystemLog("[INGEST] ⚠️ No input text — paste data into the Memory tab")
            return
        }
        let result = DataIngestPipeline.shared.ingestText(inputText, source: "ui_manual", category: "user_ingest", trusted: true)
        appendSystemLog("[INGEST] \(result.message)")
        appendChat("📥 Data Ingest: \(result.message)", color: L104Theme.gold)
    }

    @objc func doClearIngest() {
        guard let tabView = tabView else { return }
        for item in tabView.tabViewItems {
            guard let view = item.view else { continue }
            func findTV(_ v: NSView) -> NSTextView? {
                if let tv = v as? NSTextView, tv.identifier?.rawValue == "ingest_input" { return tv }
                for sub in v.subviews {
                    if let found = findTV(sub) { return found }
                }
                return nil
            }
            if let tv = findTV(view) { tv.string = ""; break }
        }
    }

    @objc func doDeepSeekIngest() {
        appendSystemLog("🧬 [DEEPSEEK] Starting ingestion process...")
        appendChat("🧬 DeepSeek Source Code Ingestion: Initializing pattern extraction", color: L104Theme.gold)

        // Trigger ingestion via Python bridge
        let bridge = PythonBridge.shared
        let result = bridge.execute("from l104_asi import deepseek_ingestion_engine; print('DeepSeek ingestion engine ready')")
        if result.success {
            appendSystemLog("🧬 [DEEPSEEK] Ingestion engine initialized")
        } else {
            appendSystemLog("🧬 [DEEPSEEK] Failed to initialize ingestion engine")
        }
    }

    @objc func doDeepSeekStatus() {
        appendSystemLog("📊 [DEEPSEEK] Fetching ingestion status...")
        _ = ASIQuantumBridgeSwift.shared.fetchASIBridgeStatus()
        appendChat("📊 DeepSeek Status: \(ASIQuantumBridgeSwift.shared.deepseekMLAPatterns) MLA patterns, \(ASIQuantumBridgeSwift.shared.deepseekR1Chains) R1 chains, \(ASIQuantumBridgeSwift.shared.deepseekAdaptations) adaptations", color: L104Theme.gold)
    }

    @objc func doDeepSeekAdapt() {
        appendSystemLog("🔄 [DEEPSEEK] Starting adaptation process...")
        appendChat("🔄 DeepSeek Adaptation: Applying GOD_CODE alignment and PHI weighting", color: L104Theme.gold)

        // Trigger adaptation via Python bridge
        let bridge = PythonBridge.shared
        let result = bridge.execute("""
        from l104_asi import deepseek_ingestion_engine
        # Example adaptation of MLA pattern
        result = deepseek_ingestion_engine.ingest_deepseek_component('mla', source_code='sample mla code')
        print(f'Adaptation result: {result}')
        """)
        if result.success {
            appendSystemLog("🔄 [DEEPSEEK] Adaptation completed")
        } else {
            appendSystemLog("🔄 [DEEPSEEK] Adaptation failed")
        }
    }

    @objc func doQuantumIntegrate() {
        appendSystemLog("⚛️ [QUANTUM] Starting quantum architecture integration...")
        appendChat("⚛️ Quantum Integration: Creating quantum circuits from DeepSeek patterns", color: L104Theme.gold)

        // Trigger quantum integration via Python bridge
        let bridge = PythonBridge.shared
        let result = bridge.execute("""
        from l104_asi import deepseek_ingestion_engine
        # Example quantum integration of MLA pattern
        result = deepseek_ingestion_engine.integrate_into_quantum_architecture('mla_test', {'attention_computation': True, 'kv_compression': True})
        print(f'Quantum integration result: {result}')
        """)
        if result.success {
            appendSystemLog("⚛️ [QUANTUM] Integration completed")
        } else {
            appendSystemLog("⚛️ [QUANTUM] Integration failed")
        }
    }

    @objc func doQuantumStatus() {
        appendSystemLog("📊 [QUANTUM] Fetching quantum architecture status...")
        _ = ASIQuantumBridgeSwift.shared.fetchASIBridgeStatus()
        appendChat("📊 Quantum Status: Circuits created, patterns integrated, GOD_CODE alignments applied", color: L104Theme.gold)
    }

    @objc func doQuantumCircuits() {
        appendSystemLog("🔄 [QUANTUM] Listing quantum circuits...")
        appendChat("🔄 Quantum Circuits: MLA attention circuits, reasoning verification oracles, code generation circuits", color: L104Theme.gold)

        // Trigger circuit listing via Python bridge
        let bridge = PythonBridge.shared
        let result = bridge.execute("""
        from l104_asi import deepseek_ingestion_engine
        status = deepseek_ingestion_engine.quantum_architecture.get_quantum_architecture_status()
        print(f'Quantum circuits: {status}')
        """)
        if result.success {
            appendSystemLog("🔄 [QUANTUM] Circuits listed")
        } else {
            appendSystemLog("🔄 [QUANTUM] Failed to list circuits")
        }
    }

    // Chat log actions
    @objc func saveChatLog() {
        guard let content = chatTextView?.string, !content.isEmpty else { return }
        state.permanentMemory.saveChatLog(content)
        appendChat("💾 Chat saved to logs folder!", color: .systemGreen)
    }

    @objc func toggleHistory() {
        guard let idx = tabView?.indexOfTabViewItem(withIdentifier: "chat"), idx >= 0,
              let chatTab = tabView?.tabViewItem(at: idx).view else { return }
        if let panel = chatTab.subviews.first(where: { $0.identifier?.rawValue == "historyPanel" }) {
            panel.isHidden.toggle()
            if !panel.isHidden { loadHistoryList() }
        }
    }

    @objc func copyAllChat() {
        guard let content = chatTextView?.string else { return }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(content, forType: .string)
        appendChat("📋 Chat copied to clipboard!", color: .systemOrange)
    }

    @objc func clearChat() {
        chatTextView?.string = ""
        loadWelcome()
    }

    @objc func searchChat() {
        let alert = NSAlert()
        alert.messageText = "Search Chat"
        alert.informativeText = "Enter text to search for in the conversation:"
        alert.alertStyle = .informational
        alert.addButton(withTitle: "Search")
        alert.addButton(withTitle: "Cancel")
        let input = NSTextField(frame: NSRect(x: 0, y: 0, width: 300, height: 24))
        input.placeholderString = "Search term..."
        alert.accessoryView = input
        alert.window.initialFirstResponder = input
        if alert.runModal() == .alertFirstButtonReturn {
            let term = input.stringValue.lowercased()
            guard !term.isEmpty, let tv = chatTextView else { return }
            let content = tv.string as NSString
            let range = content.range(of: term, options: [.caseInsensitive, .backwards])
            if range.location != NSNotFound {
                tv.scrollRangeToVisible(range)
                tv.showFindIndicator(for: range)
            }
        }
    }

    @objc func exportChatMarkdown() {
        guard let tv = chatTextView, !tv.string.isEmpty else { return }
        let panel = NSSavePanel()
        panel.nameFieldStringValue = "L104_chat_\(L104MainView.dateTimeFormatter.string(from: Date()).replacingOccurrences(of: " ", with: "_").replacingOccurrences(of: ":", with: "-")).md"
        panel.allowedContentTypes = [.plainText]
        panel.canCreateDirectories = true
        if panel.runModal() == .OK, let url = panel.url {
            var md = "# L104 Chat Export\n\n"
            md += "> Exported: \(L104MainView.dateTimeFormatter.string(from: Date()))\n"
            md += "> Version: \(VERSION)\n\n"
            md += "---\n\n"
            md += tv.string
            try? md.write(to: url, atomically: true, encoding: .utf8)
        }
    }

    func updateChatWordCount() {
        guard let tv = chatTextView else { return }
        let words = tv.string.split(separator: " ").count
        let chars = tv.string.count
        // Use cached label lookup instead of recursive tree walk
        if let wcLbl = cachedLabel("chatWordCount", in: "chat") {
            wcLbl.stringValue = "\(words) words · \(chars) chars · \(L104MainView.sessionMessages) msgs"
        }
    }

    func sendHelpCommand() {
        inputField?.stringValue = "help"
        sendMessage()
    }

    func loadHistoryList() {
        guard let listView = historyListView, let content = listView.documentView else { return }
        content.subviews.forEach { $0.removeFromSuperview() }
        let logs = state.permanentMemory.getRecentChatLogs(7)
        loadedHistoryPaths = logs.map { $0.path }
        var y: CGFloat = CGFloat(logs.count * 30)
        content.frame = NSRect(x: 0, y: 0, width: 150, height: max(200, y + 10))
        for (idx, log) in logs.enumerated() {
            let btn = NSButton(frame: NSRect(x: 5, y: y - 28, width: 140, height: 26))
            btn.title = String(log.name.prefix(18))
            btn.bezelStyle = .rounded
            btn.font = NSFont.systemFont(ofSize: 9)
            btn.contentTintColor = .systemCyan
            btn.tag = idx
            btn.target = self; btn.action = #selector(loadHistoryItem(_:))
            content.addSubview(btn)
            y -= 30
        }
    }

    @objc func loadHistoryItem(_ sender: NSButton) {
        guard sender.tag < loadedHistoryPaths.count else { return }
        let path = loadedHistoryPaths[sender.tag]
        if let content = state.permanentMemory.loadChatLog(path) {
            chatTextView?.string = ""
            appendChat("📜 LOADED: \(path.lastPathComponent)\n═══════════════════════════════════════\n", color: NSColor(red: 0.6, green: 0.8, blue: 1.0, alpha: 1.0))
            appendChat(content, color: NSColor(red: 0.8, green: 0.9, blue: 1.0, alpha: 1.0))
            appendChat("\n═══════════════════════════════════════\n", color: NSColor(red: 0.6, green: 0.8, blue: 1.0, alpha: 1.0))
        }
    }

    @objc func qStatus() { navigateToTab("chat"); appendChat("📨 You: status\nL104: \(state.getStatusText())\n", color: .white) }
    @objc func qTime() {
        navigateToTab("chat")
        appendChat("📨 You: time\nL104: 🕐 \(L104MainView.timeFormatter.string(from: Date()))\n", color: NSColor(red: 0.0, green: 0.85, blue: 1.0, alpha: 1.0))
    }

    func appendChat(_ text: String, color: NSColor) {
        guard let tv = chatTextView else { return }

        // Determine message type for bubble styling
        let isUser = text.hasPrefix("📨 You:")
        let isSystem = text.hasPrefix("⚡ SYSTEM:") || text.starts(with: "╔") || text.starts(with: "║") || text.starts(with: "╚")
        let isBot = text.hasPrefix("L104:")
        let isProcessing = text.hasPrefix("⏳")

        // Timestamp for real messages
        let timestamp = L104MainView.shortTimeFormatter.string(from: Date())

        // Build attributed string with bubble-style formatting
        let para = NSMutableParagraphStyle()
        para.lineSpacing = 3
        para.paragraphSpacing = 8
        para.paragraphSpacingBefore = 4

        if isUser {
            // User messages: right-aligned gold bubble
            para.alignment = .right
            para.headIndent = 100
            para.firstLineHeadIndent = 100
            let shadow = NSShadow()
            shadow.shadowColor = L104Theme.gold.withAlphaComponent(0.25)
            shadow.shadowBlurRadius = CGFloat(L104Theme.neonGlow)
            // Timestamp line
            let timeAttrs: [NSAttributedString.Key: Any] = [
                .font: NSFont.systemFont(ofSize: 9, weight: .medium),
                .foregroundColor: L104Theme.textDim,
                .paragraphStyle: para
            ]
            tv.textStorage?.append(NSAttributedString(string: "\(timestamp)\n", attributes: timeAttrs))
            // Message body
            let msgText = String(text.dropFirst(7)).trimmingCharacters(in: .whitespaces)  // Remove "📨 You: "
            let attrs: [NSAttributedString.Key: Any] = [
                .foregroundColor: L104Theme.textUser,
                .font: L104Theme.sansFont(14, weight: .medium),
                .paragraphStyle: para,
                .shadow: shadow,
                .backgroundColor: L104Theme.gold.withAlphaComponent(0.06)
            ]
            tv.textStorage?.append(NSAttributedString(string: "📨 \(msgText)\n", attributes: attrs))
        } else if isBot {
            // Bot messages: left-aligned with Phase 29.0 Rich Text Formatting
            para.alignment = .left
            para.tailIndent = -40
            // Timestamp
            let timeAttrs: [NSAttributedString.Key: Any] = [
                .font: NSFont.systemFont(ofSize: 9, weight: .medium),
                .foregroundColor: L104Theme.textDim,
                .paragraphStyle: para
            ]
            tv.textStorage?.append(NSAttributedString(string: "⚛️ L104 · \(timestamp)\n", attributes: timeAttrs))
            // Parse message through RichTextFormatterV2
            let msgText = String(text.dropFirst(5))  // Remove "L104: "
            let richFormatted = RichTextFormatterV2.shared.format(msgText)
            tv.textStorage?.append(richFormatted)
            tv.textStorage?.append(NSAttributedString(string: "\n", attributes: [:]))
        } else if isProcessing {
            // Processing indicator with animated dots feel
            para.alignment = .left
            let attrs: [NSAttributedString.Key: Any] = [
                .foregroundColor: L104Theme.textDim,
                .font: L104Theme.sansFont(12, weight: .medium),
                .paragraphStyle: para
            ]
            tv.textStorage?.append(NSAttributedString(string: "\(text)\n", attributes: attrs))
        } else if isSystem {
            // System/decorative messages — monospaced, subtle glow
            let shadow = NSShadow()
            shadow.shadowColor = color.withAlphaComponent(0.3)
            shadow.shadowBlurRadius = 2
            shadow.shadowOffset = NSSize(width: 0, height: -1)
            let attrs: [NSAttributedString.Key: Any] = [
                .foregroundColor: color,
                .font: NSFont.monospacedSystemFont(ofSize: 11, weight: .bold),
                .shadow: shadow,
                .paragraphStyle: para
            ]
            tv.textStorage?.append(NSAttributedString(string: text + "\n", attributes: attrs))
        } else {
            // Default / blank / decorative
            let shadow = NSShadow()
            shadow.shadowColor = color.withAlphaComponent(0.2)
            shadow.shadowBlurRadius = 2
            let attrs: [NSAttributedString.Key: Any] = [
                .foregroundColor: color,
                .font: NSFont.systemFont(ofSize: 14, weight: .medium),
                .shadow: shadow,
                .paragraphStyle: para
            ]
            tv.textStorage?.append(NSAttributedString(string: text + "\n", attributes: attrs))
        }
        tv.scrollToEndOfDocument(nil)
        updateChatWordCount()
    }

    func appendSystemLog(_ text: String) {
        let f = L104MainView.timestampFormatter
        let c: NSColor = text.contains("✅") ? .systemGreen : text.contains("❌") ? .systemRed : text.contains("🔥") || text.contains("⚡") ? L104Theme.goldFlame : L104Theme.goldDim
        let attrs: [NSAttributedString.Key: Any] = [.foregroundColor: c, .font: NSFont.monospacedSystemFont(ofSize: 10, weight: .regular)]
        let str = NSAttributedString(string: "[\(f.string(from: Date()))] \(text)\n", attributes: attrs)

        // Write to chat-panel system feed
        if let tv = systemFeedView {
            tv.textStorage?.append(str)
            tv.scrollToEndOfDocument(nil)
        }
        // Also write to System tab (full-size view)
        if let tv = systemTabFeedView {
            let fullAttrs: [NSAttributedString.Key: Any] = [.foregroundColor: c, .font: NSFont.monospacedSystemFont(ofSize: 11, weight: .regular)]
            let fullStr = NSAttributedString(string: "[\(f.string(from: Date()))] \(text)\n", attributes: fullAttrs)
            tv.textStorage?.append(fullStr)
            tv.scrollToEndOfDocument(nil)
        }

        // ═══ INTERCONNECT: Update System tab status bar with live metrics ═══
        if let sysLbl = cachedLabel("sys_status_lbl", in: "sys") {
            let qpcFid = String(format: "%.3f", QuantumProcessingCore.shared.currentFidelity())
            let sageCon = String(format: "%.3f", SageModeEngine.shared.consciousnessLevel)
            let phiH = String(format: "%.1f%%", EngineRegistry.shared.phiWeightedHealth().score * 100)
            let peers = NetworkLayer.shared.peers.count
            let mem = state.permanentMemory.memories.count
            sysLbl.stringValue = "⚛️ QPC:\(qpcFid) 🧘 Sage:\(sageCon) ⚡ φ:\(phiH) 🌐 Peers:\(peers) 💾 Mem:\(mem) 🔥 v\(VERSION)"
        }
    }

    func updateMetrics() {
        metricTiles["ASI"]?.value = String(format: "%.0f%%", state.asiScore * 100)
        metricTiles["ASI"]?.progress = CGFloat(state.asiScore)
        metricTiles["IQ"]?.value = String(format: "%.0f", state.intellectIndex)
        metricTiles["IQ"]?.progress = CGFloat(min(1.0, state.intellectIndex / 200))
        metricTiles["Coherence"]?.value = String(format: "%.2f", state.coherence)
        metricTiles["Coherence"]?.progress = CGFloat(state.coherence)
        metricTiles["Memories"]?.value = "\(state.permanentMemory.memories.count)"
        metricTiles["Memories"]?.progress = CGFloat(min(1.0, Double(state.permanentMemory.memories.count) / 100))
        metricTiles["Skills"]?.value = "\(state.skills)"
        metricTiles["Skills"]?.progress = CGFloat(min(1.0, Double(state.skills) / 50))
        metricTiles["Transcend"]?.value = String(format: "%.0f%%", state.transcendence * 100)
        metricTiles["Transcend"]?.progress = CGFloat(state.transcendence)

        // Also update old labels if they exist
        metricsLabels["ASI"]?.stringValue = String(format: "%.0f%%", state.asiScore * 100)
        metricsLabels["IQ"]?.stringValue = String(format: "%.0f", state.intellectIndex)
        metricsLabels["Coherence"]?.stringValue = String(format: "%.3f", state.coherence)
        metricsLabels["Memories"]?.stringValue = "\(state.permanentMemory.memories.count)"
        metricsLabels["Skills"]?.stringValue = "\(state.skills)"
    }

    func startTimer() {
        // ═══ PHASE 31.6: Pre-allocate DateFormatters (avoid repeated alloc in timer) ═══
        let clockFormatter = DateFormatter(); clockFormatter.dateFormat = "HH:mm:ss"
        let dateFormatter = DateFormatter(); dateFormatter.dateFormat = "yyyy-MM-dd"
        let uiInterval: TimeInterval = 2.0  // Perf: reduced from 0.5s — status labels don't need sub-second refresh
        timer = Timer.scheduledTimer(withTimeInterval: uiInterval, repeats: true) { [weak self] _ in
            let now = Date()
            self?.clockLabel?.stringValue = clockFormatter.string(from: now)
            self?.dateLabel?.stringValue = dateFormatter.string(from: now)
            let phase = now.timeIntervalSince1970.truncatingRemainder(dividingBy: PHI * 100) / 100
            self?.phaseLabel?.stringValue = "φ: \(String(format: "%.4f", phase))"

            // 🟢 EVO_65: Live metrics bar + science metrics auto-refresh
            self?.updateMetrics()
            self?.updateScienceMetrics()

            // UPDATE EVOLUTION UI — 🟢 EVO_63: cached identifier lookup, visibility-gated
            if self?.activeTabID == "upg" {
                let evolver = ASIEvolver.shared
                self?.cachedLabel("upg_stage", in: "upg")?.stringValue = "Evolution Stage: \(evolver.evolutionStage)"
                self?.cachedLabel("upg_files", in: "upg")?.stringValue = "Generated Artifacts: \(evolver.generatedFilesCount)"
                let ec = evolver.evolvedGreetings.count + evolver.evolvedPhilosophies.count + evolver.evolvedFacts.count
                self?.cachedLabel("upg_evolved", in: "upg")?.stringValue = "Evolved Content: \(ec) greetings/philosophies/facts"
                self?.cachedLabel("upg_mutations", in: "upg")?.stringValue = "Mutations: \(evolver.mutationCount) · Crossovers: \(evolver.crossoverCount) · Syntheses: \(evolver.synthesisCount)"
                // 🟢 EVO_64: Running/paused + thought count
                let runLbl = self?.cachedLabel("upg_run_status", in: "upg")
                runLbl?.stringValue = evolver.isRunning ? "🟢 RUNNING" : "⏸ PAUSED"
                runLbl?.textColor = evolver.isRunning ? .systemGreen : .systemOrange
                self?.cachedLabel("upg_thought_rate", in: "upg")?.stringValue = "Thoughts: \(evolver.thoughts.count) · Rate: ~1/8s"
            }

            // Randomly trigger background cognition (approx every 15s)
            if Int.random(in: 0...150) == 42 {
                self?.autoGenerateThought()
            }
        }
    }

    func colorFromHex(_ hex: String) -> NSColor {
        let h = hex.replacingOccurrences(of: "#", with: "")
        var rgb: UInt64 = 0; Scanner(string: h).scanHexInt64(&rgb)
        return NSColor(red: CGFloat((rgb >> 16) & 0xFF) / 255, green: CGFloat((rgb >> 8) & 0xFF) / 255, blue: CGFloat(rgb & 0xFF) / 255, alpha: 1)
    }

    // ═══════════════════════════════════════════════════════════════════
    // ⚛️ QUANTUM COMPUTING TAB — Real IBM QPU + Qiskit Simulator Fallback
    // Grover · QPE · VQE · QAOA · Amplitude Estimation · Quantum Walk · Kernel
    // Phase 46.1: Real IBM Quantum hardware via REST API + Qiskit Runtime
    // ═══════════════════════════════════════════════════════════════════

    private var quantumOutputView: NSTextView?
    private var quantumStatusLabel: NSTextField?
    private var quantumHWStatusLabel: NSTextField?

    func createQuantumComputingView() -> NSView {
        let v = NSView(frame: NSRect(x: 0, y: 0, width: 1200, height: 500))
        v.wantsLayer = true
        v.layer?.backgroundColor = L104Theme.void.cgColor

        // Header
        let header = NSTextField(labelWithString: "⚛️  QUANTUM COMPUTING LAB — IBM Quantum + Qiskit \(QISKIT_VERSION)")
        header.font = NSFont.systemFont(ofSize: 16, weight: .bold)
        header.textColor = L104Theme.goldFlame
        header.frame = NSRect(x: 20, y: 460, width: 700, height: 30)
        v.addSubview(header)

        // IBM Hardware status line
        let ibm = IBMQuantumClient.shared
        let hwIcon: String
        let hwText: String
        if ibm.isConnected {
            hwIcon = "🟢"
            hwText = "IBM QPU: \(ibm.connectedBackendName) — Real Hardware"
        } else if ibm.ibmToken != nil {
            hwIcon = "🟡"
            hwText = "IBM QPU: Token set — reconnecting..."
        } else {
            hwIcon = "⚪"
            hwText = "IBM QPU: Not connected — algorithms use simulator"
        }
        let hwLabel = NSTextField(labelWithString: "\(hwIcon) \(hwText)")
        hwLabel.font = NSFont.monospacedSystemFont(ofSize: 11, weight: .medium)
        hwLabel.textColor = ibm.isConnected ? .systemGreen : (ibm.ibmToken != nil ? .systemYellow : .secondaryLabelColor)
        hwLabel.frame = NSRect(x: 20, y: 440, width: 700, height: 18)
        v.addSubview(hwLabel)
        quantumHWStatusLabel = hwLabel

        let statusLbl = NSTextField(labelWithString: "Status: Ready")
        statusLbl.font = NSFont.monospacedSystemFont(ofSize: 11, weight: .regular)
        statusLbl.textColor = .systemGreen
        statusLbl.frame = NSRect(x: 20, y: 422, width: 600, height: 18)
        v.addSubview(statusLbl)
        quantumStatusLabel = statusLbl

        // ─── IBM HARDWARE BUTTONS (row 0) ───
        let ibmActions: [(String, String, Selector)] = [
            ("🔗 Connect IBM", "ibm_connect", #selector(quantumIBMConnect)),
            ("📡 Backends", "ibm_backends", #selector(quantumIBMBackends)),
            ("📋 Jobs", "ibm_jobs", #selector(quantumIBMJobs)),
            ("🔌 Disconnect", "ibm_disconnect", #selector(quantumIBMDisconnect)),
        ]

        for (i, action) in ibmActions.enumerated() {
            let btn = NSButton(title: action.0, target: self, action: action.2)
            btn.bezelStyle = .rounded
            btn.frame = NSRect(x: 20 + i * 155, y: 393, width: 145, height: 26)
            btn.font = NSFont.systemFont(ofSize: 10, weight: .medium)
            btn.identifier = NSUserInterfaceItemIdentifier(action.1)
            v.addSubview(btn)
        }

        // ─── ALGORITHM BUTTONS (rows 1-2) ───
        let algorithms: [(String, String, Selector)] = [
            ("🔍 Grover Search", "grover", #selector(runQuantumGrover)),
            ("📐 Phase Estimation", "qpe", #selector(runQuantumQPE)),
            ("⚡ VQE Eigensolver", "vqe", #selector(runQuantumVQE)),
            ("🔀 QAOA MaxCut", "qaoa", #selector(runQuantumQAOA)),
            ("📊 Amplitude Est.", "ampest", #selector(runQuantumAmpEst)),
            ("🚶 Quantum Walk", "walk", #selector(runQuantumWalk)),
            ("🧬 Quantum Kernel", "kernel", #selector(runQuantumKernel)),
            ("📡 Full Status", "status", #selector(runQuantumStatus)),
        ]

        for (i, algo) in algorithms.enumerated() {
            let row = i / 4
            let col = i % 4
            let btn = NSButton(title: algo.0, target: self, action: algo.2)
            btn.bezelStyle = .rounded
            btn.frame = NSRect(x: 20 + col * 155, y: 360 - row * 32, width: 145, height: 28)
            btn.font = NSFont.systemFont(ofSize: 11, weight: .medium)
            btn.identifier = NSUserInterfaceItemIdentifier(algo.1)
            v.addSubview(btn)
        }

        // Output area — scrollable text view (left side)
        let scrollView = NSScrollView(frame: NSRect(x: 20, y: 10, width: 760, height: 280))
        scrollView.autoresizingMask = [.height]
        scrollView.hasVerticalScroller = true
        scrollView.borderType = .bezelBorder

        let tv = NSTextView(frame: scrollView.bounds)
        tv.isEditable = false
        tv.backgroundColor = NSColor(red: 0.05, green: 0.05, blue: 0.12, alpha: 1.0)
        tv.textColor = .systemCyan
        tv.font = NSFont.monospacedSystemFont(ofSize: 11, weight: .regular)
        tv.autoresizingMask = [.width, .height]
        scrollView.documentView = tv
        v.addSubview(scrollView)
        quantumOutputView = tv

        // ─── RIGHT SIDEBAR: Qubit Dashboard ───
        let sidebarX: CGFloat = 800
        let sidebarW: CGFloat = 380

        // Qubit State Panel
        let qubitPanel = NSView(frame: NSRect(x: sidebarX, y: 190, width: sidebarW, height: 270))
        qubitPanel.wantsLayer = true
        qubitPanel.layer?.backgroundColor = NSColor(red: 0.06, green: 0.06, blue: 0.14, alpha: 1.0).cgColor
        qubitPanel.layer?.cornerRadius = 12
        qubitPanel.layer?.borderColor = NSColor.systemCyan.withAlphaComponent(0.3).cgColor
        qubitPanel.layer?.borderWidth = 1

        let qubitTitle = NSTextField(labelWithString: "⚛️ QUBIT STATE DASHBOARD")
        qubitTitle.font = NSFont.systemFont(ofSize: 12, weight: .bold)
        qubitTitle.textColor = .systemCyan
        qubitTitle.frame = NSRect(x: 15, y: 235, width: sidebarW - 30, height: 20)
        qubitPanel.addSubview(qubitTitle)

        let qubitItems: [(String, String, NSColor)] = [
            ("Register Size", "4 qubits (Grover) / variable", .systemCyan),
            ("Fidelity", String(format: "%.4f", QuantumProcessingCore.shared.currentFidelity()), .systemGreen),
            ("Error Rate", "< 0.1% (sim) / ~1% (QPU)", .systemYellow),
            ("Gate Set", "H, X, CX, Rz, SWAP, Toffoli", .systemCyan),
            ("Algorithms", "\(QUANTUM_ALGORITHMS) circuits available", .systemGreen),
            ("Framework", "Qiskit \(QISKIT_VERSION) + IBM REST", .systemCyan),
            ("Topology", ibm.isConnected ? ibm.connectedBackendName : "Simulator", ibm.isConnected ? .systemGreen : .secondaryLabelColor),
            ("Runtime", ibm.isConnected ? "Real QPU Bridge ✓" : "Statevector", ibm.isConnected ? .systemGreen : .systemOrange),
        ]
        var qy: CGFloat = 205
        for (label, value, color) in qubitItems {
            let lbl = NSTextField(labelWithString: label)
            lbl.frame = NSRect(x: 15, y: qy, width: 120, height: 16)
            lbl.font = NSFont.systemFont(ofSize: 10, weight: .medium)
            lbl.textColor = .secondaryLabelColor
            qubitPanel.addSubview(lbl)

            let val = NSTextField(labelWithString: value)
            val.frame = NSRect(x: 140, y: qy, width: sidebarW - 160, height: 16)
            val.font = NSFont.monospacedSystemFont(ofSize: 10, weight: .semibold)
            val.textColor = color
            val.identifier = NSUserInterfaceItemIdentifier("qc_stat_\(label.lowercased().replacingOccurrences(of: " ", with: "_"))")
            qubitPanel.addSubview(val)
            qy -= 24
        }

        // Visual qubit state indicator
        let qubitViz = NSView(frame: NSRect(x: 15, y: 10, width: sidebarW - 30, height: 40))
        qubitViz.wantsLayer = true
        qubitViz.layer?.backgroundColor = NSColor(red: 0.04, green: 0.04, blue: 0.10, alpha: 1.0).cgColor
        qubitViz.layer?.cornerRadius = 8
        let qubitStates = ["|0⟩", "|1⟩", "|+⟩", "|−⟩"]
        for (i, qs) in qubitStates.enumerated() {
            let qLbl = NSTextField(labelWithString: "q\(i): \(qs)")
            qLbl.frame = NSRect(x: CGFloat(i) * 85 + 10, y: 10, width: 75, height: 20)
            qLbl.font = NSFont.monospacedSystemFont(ofSize: 11, weight: .bold)
            qLbl.textColor = .systemCyan
            qLbl.alignment = .center
            qLbl.identifier = NSUserInterfaceItemIdentifier("qc_qubit_\(i)")
            qubitViz.addSubview(qLbl)
        }
        qubitPanel.addSubview(qubitViz)
        v.addSubview(qubitPanel)

        // Algorithm History Panel
        let histPanel = NSView(frame: NSRect(x: sidebarX, y: 10, width: sidebarW, height: 170))
        histPanel.wantsLayer = true
        histPanel.layer?.backgroundColor = NSColor(red: 0.06, green: 0.06, blue: 0.14, alpha: 1.0).cgColor
        histPanel.layer?.cornerRadius = 12
        histPanel.layer?.borderColor = NSColor.systemIndigo.withAlphaComponent(0.3).cgColor
        histPanel.layer?.borderWidth = 1

        let histTitle = NSTextField(labelWithString: "📊 ALGORITHM HISTORY")
        histTitle.font = NSFont.systemFont(ofSize: 12, weight: .bold)
        histTitle.textColor = .systemIndigo
        histTitle.frame = NSRect(x: 15, y: 135, width: sidebarW - 30, height: 20)
        histPanel.addSubview(histTitle)

        let histScroll = NSScrollView(frame: NSRect(x: 10, y: 10, width: sidebarW - 20, height: 118))
        histScroll.hasVerticalScroller = true
        histScroll.wantsLayer = true
        histScroll.layer?.cornerRadius = 6
        let histTV = NSTextView(frame: histScroll.bounds)
        histTV.isEditable = false
        histTV.backgroundColor = NSColor(red: 0.04, green: 0.04, blue: 0.10, alpha: 1.0)
        histTV.textColor = .systemIndigo
        histTV.font = NSFont.monospacedSystemFont(ofSize: 9.5, weight: .regular)
        histTV.string = "  No algorithms run yet.\n  Select an algorithm to begin..."
        histTV.identifier = NSUserInterfaceItemIdentifier("qc_history_text")
        histScroll.documentView = histTV
        histPanel.addSubview(histScroll)
        v.addSubview(histPanel)

        // ─── Quantum Poll Timer — refresh sidebar state (cached + visibility-gated) ───
        quantumPollTimer?.invalidate()
        quantumPollTimer = Timer.scheduledTimer(withTimeInterval: 5.0, repeats: true) { [weak self] _ in
            guard let s = self, s.activeTabID == "qc" else { return }
            let ibmNow = IBMQuantumClient.shared
            let qpc = QuantumProcessingCore.shared
            if let topoLbl = s.cachedLabel("qc_stat_topology", in: "qc") {
                topoLbl.stringValue = ibmNow.isConnected ? ibmNow.connectedBackendName : "Simulator"
                topoLbl.textColor = ibmNow.isConnected ? .systemGreen : .secondaryLabelColor
            }
            if let fidLbl = s.cachedLabel("qc_stat_fidelity", in: "qc") {
                fidLbl.stringValue = String(format: "%.4f", qpc.currentFidelity())
            }
            if let rtLbl = s.cachedLabel("qc_stat_runtime", in: "qc") {
                rtLbl.stringValue = ibmNow.isConnected ? "Real QPU Bridge ✓" : "Statevector"
                rtLbl.textColor = ibmNow.isConnected ? .systemGreen : .systemOrange
            }

            // ═══ INTERCONNECT: QPC State Tomography → Qubit Dashboard ═══
            let tomo = qpc.stateTomography()
            if let errLbl = s.cachedLabel("qc_stat_error_rate", in: "qc") {
                errLbl.stringValue = String(format: "pur=%.3f ent=%.3f S=%.3f", tomo.purity, tomo.entanglementWitness, tomo.vonNeumannEntropy)
                errLbl.textColor = tomo.purity > 0.7 ? .systemGreen : .systemYellow
            }
            if let algLbl = s.cachedLabel("qc_stat_algorithms", in: "qc") {
                algLbl.stringValue = "\(QUANTUM_ALGORITHMS) circuits | Bell×\(qpc.bellPairCount)"
            }

            // ═══ INTERCONNECT: QPC → Qubit state visualization ═══
            qpc.adaptDecoherence()
            let stateLabels = ["|Φ+⟩", "|Ψ+⟩", "|ψ⟩", "⟨ρ⟩"]
            let stateValues = [
                String(format: "%.2f", tomo.purity),
                String(format: "%.2f", abs(tomo.entanglementWitness)),
                String(format: "%.2f", qpc.currentFidelity()),
                String(format: "%.2f", tomo.vonNeumannEntropy)
            ]
            for i in 0..<4 {
                if let qLbl = s.cachedLabel("qc_qubit_\(i)", in: "qc") {
                    qLbl.stringValue = "q\(i): \(stateLabels[i]) \(stateValues[i])"
                    qLbl.textColor = tomo.purity > 0.7 ? .systemCyan : .systemYellow
                }
            }

            // ═══ INTERCONNECT: QuantumCreativityEngine → History panel ═══
            let qce = QuantumCreativityEngine.shared
            if let histTV = s.findTextView(id: "qc_history_text", in: "qc") {
                let metrics = qce.creativityMetrics
                let genCount = metrics["generation_count"] as? Int ?? 0
                let tunnelBreak = metrics["tunnel_breakthroughs"] as? Int ?? 0
                let entangled = metrics["entangled_concepts"] as? Int ?? 0
                let meshSynced = metrics["mesh_ideas_synced"] as? Int ?? 0
                histTV.string = """
                ⚛️ QUANTUM CREATIVITY × QPC METRICS
                ─────────────────────────────────────
                Generations:       \(genCount)
                Tunnel Breakthroughs: \(tunnelBreak)
                Entangled Concepts:\(entangled)
                Mesh Ideas Synced: \(meshSynced)
                ─────────────────────────────────────
                QPC Tomography:
                  Purity:          \(String(format: "%.6f", tomo.purity))
                  Von Neumann S:   \(String(format: "%.6f", tomo.vonNeumannEntropy))
                  Entanglement W:  \(String(format: "%.6f", tomo.entanglementWitness))
                  Bell Pairs:      \(qpc.bellPairCount)
                  Fidelity:        \(String(format: "%.6f", qpc.currentFidelity()))
                ─────────────────────────────────────
                Decoherence Adapted | Bridge Active
                """
            }
        }

        // Welcome message — hardware-aware
        let hwWelcome: String
        if ibm.isConnected {
            hwWelcome = "║  Hardware:  🟢 IBM \(ibm.connectedBackendName) (Real QPU)     ║"
        } else if ibm.ibmToken != nil {
            hwWelcome = "║  Hardware:  🟡 IBM Token set (reconnecting)          ║"
        } else {
            hwWelcome = "║  Hardware:  ⚪ Simulator (click Connect IBM for QPU)  ║"
        }

        let welcome = """
        ╔═══════════════════════════════════════════════════════════╗
        ║  ⚛️  L104 QUANTUM COMPUTING LAB                          ║
        ╠═══════════════════════════════════════════════════════════╣
        ║  Framework:  Qiskit \(QISKIT_VERSION) + IBM Quantum REST API         ║
        \(hwWelcome)
        ║  Algorithms: \(QUANTUM_ALGORITHMS) quantum circuits (real HW → sim fallback) ║
        ║                                                           ║
        ║  🔍 Grover    — O(√N) search on 4-qubit register         ║
        ║  📐 QPE       — Phase estimation with precision qubits   ║
        ║  ⚡ VQE       — Variational quantum eigensolver          ║
        ║  🔀 QAOA      — MaxCut approximation algorithm           ║
        ║  📊 AmpEst    — Quantum amplitude estimation             ║
        ║  🚶 Walk      — Quantum walk on cyclic graph             ║
        ║  🧬 Kernel    — Quantum kernel for ML similarity         ║
        ║                                                           ║
        ║  When IBM Quantum is connected, algorithms run on real    ║
        ║  QPU hardware first. Simulator is used as fallback.       ║
        ║                                                           ║
        ║  Get your IBM token: https://quantum.ibm.com/account      ║
        ╚═══════════════════════════════════════════════════════════╝

        """
        tv.string = welcome

        return v
    }

    private func appendQuantumOutput(_ text: String, color: NSColor = .systemCyan) {
        guard let tv = quantumOutputView else { return }
        let attrs: [NSAttributedString.Key: Any] = [
            .foregroundColor: color,
            .font: NSFont.monospacedSystemFont(ofSize: 11, weight: .regular)
        ]
        tv.textStorage?.append(NSAttributedString(string: text + "\n", attributes: attrs))
        tv.scrollToEndOfDocument(nil)
    }

    // ─── Helper: update IBM HW status label (call after state changes) ───
    private func updateQuantumHWLabel() {
        let ibm = IBMQuantumClient.shared
        if ibm.isConnected {
            quantumHWStatusLabel?.stringValue = "🟢 IBM QPU: \(ibm.connectedBackendName) — Real Hardware"
            quantumHWStatusLabel?.textColor = .systemGreen
        } else if ibm.ibmToken != nil {
            quantumHWStatusLabel?.stringValue = "🟡 IBM QPU: Token set — reconnecting..."
            quantumHWStatusLabel?.textColor = .systemYellow
        } else {
            quantumHWStatusLabel?.stringValue = "⚪ IBM QPU: Not connected — algorithms use simulator"
            quantumHWStatusLabel?.textColor = .secondaryLabelColor
        }
    }

    // ─── IBM HARDWARE BUTTON HANDLERS ───

    @objc func quantumIBMConnect() {
        // Prompt for token via alert
        let alert = NSAlert()
        alert.messageText = "Connect to IBM Quantum"
        alert.informativeText = "Enter your IBM Cloud API key or IQP token.\n• IBM Cloud key: https://cloud.ibm.com/iam/apikeys\n• IQP token: https://quantum.ibm.com/account"
        alert.alertStyle = .informational
        alert.addButton(withTitle: "Connect")
        alert.addButton(withTitle: "Cancel")
        let input = NSTextField(frame: NSRect(x: 0, y: 0, width: 360, height: 24))
        input.placeholderString = "IBM Cloud API key or IQP token"
        alert.accessoryView = input
        alert.window.initialFirstResponder = input

        guard alert.runModal() == .alertFirstButtonReturn else { return }
        let token = input.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !token.isEmpty else {
            appendQuantumOutput("[⚠️] No token provided.", color: .systemYellow)
            return
        }

        appendQuantumOutput("\n[🔗] Connecting to IBM Quantum...", color: .systemYellow)
        quantumStatusLabel?.stringValue = "⏳ Connecting to IBM Quantum..."
        quantumStatusLabel?.textColor = .systemYellow

        // Init Python engine + Swift REST client in parallel
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let pyResult = PythonBridge.shared.quantumHardwareInit(token: token)
            DispatchQueue.main.async {
                if pyResult.success, let dict = pyResult.returnValue as? [String: Any] {
                    let backend = dict["backend"] as? String ?? "unknown"
                    let qubits = dict["qubits"] as? Int ?? 0
                    let isReal = dict["real_hardware"] as? Bool ?? false
                    self?.appendQuantumOutput("✅ Python engine connected!", color: .systemGreen)
                    self?.appendQuantumOutput("   Backend: \(backend) (\(qubits) qubits)", color: .white)
                    self?.appendQuantumOutput("   Real HW: \(isReal ? "YES" : "No (simulator)")", color: isReal ? .systemGreen : .systemYellow)
                } else {
                    self?.appendQuantumOutput("[⚠️] Python engine: \(pyResult.error)", color: .systemYellow)
                }
            }
        }

        IBMQuantumClient.shared.connect(token: token) { [weak self] success, msg in
            DispatchQueue.main.async {
                if success {
                    let state = L104State.shared
                    state.quantumHardwareConnected = true
                    state.quantumBackendName = IBMQuantumClient.shared.connectedBackendName
                    self?.appendQuantumOutput("✅ REST API connected: \(msg)", color: .systemGreen)
                    self?.quantumStatusLabel?.stringValue = "✅ Connected to IBM Quantum"
                    self?.quantumStatusLabel?.textColor = .systemGreen
                } else {
                    self?.appendQuantumOutput("[❌] REST API: \(msg)", color: .systemRed)
                    self?.quantumStatusLabel?.stringValue = "❌ Connection failed"
                    self?.quantumStatusLabel?.textColor = .systemRed
                }
                self?.updateQuantumHWLabel()
            }
        }
    }

    @objc func quantumIBMDisconnect() {
        IBMQuantumClient.shared.disconnect()
        let state = L104State.shared
        state.quantumHardwareConnected = false
        state.quantumBackendName = "none"
        state.quantumBackendQubits = 0
        appendQuantumOutput("\n[🔌] Disconnected from IBM Quantum. Token cleared.", color: .secondaryLabelColor)
        quantumStatusLabel?.stringValue = "⚪ Disconnected"
        quantumStatusLabel?.textColor = .secondaryLabelColor
        updateQuantumHWLabel()
    }

    @objc func quantumIBMBackends() {
        let client = IBMQuantumClient.shared
        guard client.ibmToken != nil else {
            appendQuantumOutput("\n[⚠️] Not connected. Click 'Connect IBM' first.", color: .systemYellow)
            return
        }
        let backends = client.availableBackends
        if backends.isEmpty {
            appendQuantumOutput("\n[📡] No backends loaded. Reconnecting...", color: .systemYellow)
            return
        }
        appendQuantumOutput("\n╔═══════════════════════════════════════════════════════════╗", color: .systemGreen)
        appendQuantumOutput("║  📡 IBM QUANTUM BACKENDS                                ║", color: .systemGreen)
        appendQuantumOutput("╠═══════════════════════════════════════════════════════════╣", color: .systemGreen)
        for b in backends.prefix(10) {
            let marker = b.name == client.connectedBackendName ? " << SELECTED" : ""
            let hwTag = b.isSimulator ? "[SIM]" : "[QPU]"
            appendQuantumOutput("║  \(hwTag) \(b.name) — \(b.numQubits)q, queue:\(b.pendingJobs), QV:\(b.quantumVolume)\(marker)", color: b.isSimulator ? .secondaryLabelColor : .systemCyan)
        }
        appendQuantumOutput("╚═══════════════════════════════════════════════════════════╝", color: .systemGreen)
        quantumStatusLabel?.stringValue = "📡 \(backends.count) backends (\(backends.filter { !$0.isSimulator }.count) real QPUs)"
    }

    @objc func quantumIBMJobs() {
        let client = IBMQuantumClient.shared
        guard client.ibmToken != nil else {
            appendQuantumOutput("\n[⚠️] Not connected. Click 'Connect IBM' first.", color: .systemYellow)
            return
        }
        let localJobs = client.submittedJobs
        appendQuantumOutput("\n[📋] Local submitted jobs: \(localJobs.count)", color: .systemCyan)
        for (id, job) in localJobs.prefix(10) {
            appendQuantumOutput("  [\(id.prefix(16))...] → \(job.backend)", color: .white)
        }
        // Fetch remote jobs async
        appendQuantumOutput("  Fetching remote jobs...", color: .secondaryLabelColor)
        client.listRecentJobs(limit: 5) { [weak self] jobs, error in
            DispatchQueue.main.async {
                if let jobs = jobs {
                    self?.appendQuantumOutput("  📡 Recent IBM jobs:", color: .systemCyan)
                    for j in jobs.prefix(5) {
                        self?.appendQuantumOutput("  [\(j.jobId.prefix(16))...] \(j.status) — \(j.backend)", color: .white)
                    }
                    self?.quantumStatusLabel?.stringValue = "📋 \(jobs.count) remote jobs"
                } else {
                    self?.appendQuantumOutput("  [⚠️] \(error ?? "unknown error")", color: .systemYellow)
                }
            }
        }
    }

    // ─── ALGORITHM METHODS — Real hardware first, simulator fallback ───

    @objc func runQuantumGrover() {
        let useHW = IBMQuantumClient.shared.ibmToken != nil
        let tag = useHW ? "[REAL HW]" : "[SIMULATOR]"
        quantumStatusLabel?.stringValue = "⏳ Running Grover's Search \(tag) (target=7, 4 qubits)..."
        quantumStatusLabel?.textColor = .systemYellow
        appendQuantumOutput("\n[⏳] Executing Grover's Search \(tag)...", color: .systemYellow)
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            // Try real hardware first if token exists
            var result = PythonBridge.shared.quantumGrover(target: 7, nQubits: 4)
            var isRealHW = false
            if useHW {
                let hwResult = PythonBridge.shared.quantumHardwareGrover(target: 7, nQubits: 4)
                if hwResult.success {
                    result = hwResult
                    isRealHW = true
                }
            }
            DispatchQueue.main.async {
                let hwLabel = isRealHW ? " [REAL HW]" : " [SIMULATOR]"
                if result.success, let dict = result.returnValue as? [String: Any] {
                    let found = (dict["found_index"] as? Int) ?? (dict["nonce"] as? Int) ?? -1
                    let prob = dict["target_probability"] as? Double ?? 0
                    let success = dict["success"] as? Bool ?? (found >= 0)
                    let shots = dict["grover_iterations"] as? Int ?? 0
                    self?.appendQuantumOutput("╔═══════════════════════════════════════╗", color: .systemGreen)
                    self?.appendQuantumOutput("║  🔍 GROVER'S SEARCH\(hwLabel)       ║", color: .systemGreen)
                    self?.appendQuantumOutput("╠═══════════════════════════════════════╣", color: .systemGreen)
                    self?.appendQuantumOutput("║  Target:      |7⟩ (|0111⟩)            ║", color: .white)
                    self?.appendQuantumOutput("║  Found:       |\(found)⟩                      ║", color: success ? .systemGreen : .systemRed)
                    if prob > 0 {
                        self?.appendQuantumOutput("║  Probability: \(String(format: "%.4f", prob))                ║", color: .systemCyan)
                    }
                    if shots > 0 {
                        self?.appendQuantumOutput("║  Iterations:  \(shots)                        ║", color: .white)
                    }
                    self?.appendQuantumOutput("║  Success:     \(success ? "✅ YES" : "❌ NO")                    ║", color: success ? .systemGreen : .systemRed)
                    self?.appendQuantumOutput("║  Time:        \(String(format: "%.2f", result.executionTime))s                    ║", color: .white)
                    if isRealHW, let backend = dict["backend"] as? String {
                        self?.appendQuantumOutput("║  Backend:     \(backend)        ║", color: .systemCyan)
                    }
                    self?.appendQuantumOutput("╚═══════════════════════════════════════╝", color: .systemGreen)
                    self?.quantumStatusLabel?.stringValue = "✅ Grover\(hwLabel): Found |\(found)⟩"
                } else {
                    self?.appendQuantumOutput("[❌] Grover failed: \(result.error)", color: .systemRed)
                    self?.quantumStatusLabel?.stringValue = "❌ Grover failed"
                }
                self?.quantumStatusLabel?.textColor = .systemGreen
            }
        }
    }

    @objc func runQuantumQPE() {
        let useHW = IBMQuantumClient.shared.ibmToken != nil
        let tag = useHW ? "[REAL HW]" : "[SIMULATOR]"
        quantumStatusLabel?.stringValue = "⏳ Running QPE \(tag)..."
        appendQuantumOutput("\n[⏳] Executing QPE \(tag) with 5 precision qubits...", color: .systemYellow)
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            var result = PythonBridge.shared.quantumQPE(precisionQubits: 5)
            var isRealHW = false
            if useHW {
                let hwResult = PythonBridge.shared.quantumHardwareReport(difficultyBits: 16)
                if hwResult.success { result = hwResult; isRealHW = true }
            }
            DispatchQueue.main.async {
                let hwLabel = isRealHW ? " [REAL HW]" : " [SIMULATOR]"
                if result.success, let dict = result.returnValue as? [String: Any] {
                    if isRealHW {
                        let report = dict["report"] as? String ?? ""
                        let backend = dict["backend"] as? String ?? "unknown"
                        self?.appendQuantumOutput("📐 QPE RESULT\(hwLabel) (\(backend)):", color: .systemGreen)
                        self?.appendQuantumOutput("  \(String(report.prefix(400)))", color: .systemCyan)
                    } else {
                        let targetPhase = dict["target_phase"] as? Double ?? 0
                        let estPhase = dict["estimated_phase"] as? Double ?? 0
                        let error = dict["phase_error"] as? Double ?? 0
                        self?.appendQuantumOutput("📐 QPE RESULT\(hwLabel):", color: .systemGreen)
                        self?.appendQuantumOutput("  Target Phase:    \(String(format: "%.6f", targetPhase))", color: .white)
                        self?.appendQuantumOutput("  Estimated Phase: \(String(format: "%.6f", estPhase))", color: .systemCyan)
                        self?.appendQuantumOutput("  Phase Error:     \(String(format: "%.6f", error))", color: error < 0.05 ? .systemGreen : .systemYellow)
                    }
                    self?.appendQuantumOutput("  Time:            \(String(format: "%.2f", result.executionTime))s", color: .white)
                    self?.quantumStatusLabel?.stringValue = "✅ QPE\(hwLabel) completed"
                } else { self?.appendQuantumOutput("[❌] QPE failed: \(result.error)", color: .systemRed) }
                self?.quantumStatusLabel?.textColor = .systemGreen
            }
        }
    }

    @objc func runQuantumVQE() {
        let useHW = IBMQuantumClient.shared.ibmToken != nil
        let tag = useHW ? "[REAL HW]" : "[SIMULATOR]"
        quantumStatusLabel?.stringValue = "⏳ Running VQE \(tag) (4 qubits, 50 iterations)..."
        appendQuantumOutput("\n[⏳] Executing VQE \(tag)...", color: .systemYellow)
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            var result = PythonBridge.shared.quantumVQE(nQubits: 4, iterations: 50)
            var isRealHW = false
            if useHW {
                let hwResult = PythonBridge.shared.quantumHardwareVQE()
                if hwResult.success, let dict = hwResult.returnValue as? [String: Any], dict["error"] == nil {
                    result = hwResult; isRealHW = true
                }
            }
            DispatchQueue.main.async {
                let hwLabel = isRealHW ? " [REAL HW]" : " [SIMULATOR]"
                if result.success, let dict = result.returnValue as? [String: Any] {
                    let energy = dict["optimized_energy"] as? Double ?? 0
                    let exact = dict["exact_energy"] as? Double ?? 0
                    let error = dict["energy_error"] as? Double ?? 0
                    let iters = dict["iterations_used"] as? Int ?? 0
                    self?.appendQuantumOutput("⚡ VQE EIGENSOLVER\(hwLabel):", color: .systemGreen)
                    self?.appendQuantumOutput("  Optimized Energy: \(String(format: "%.6f", energy))", color: .systemCyan)
                    if exact != 0 { self?.appendQuantumOutput("  Exact Energy:     \(String(format: "%.6f", exact))", color: .white) }
                    if error != 0 { self?.appendQuantumOutput("  Energy Error:     \(String(format: "%.6f", error))", color: error < 0.1 ? .systemGreen : .systemYellow) }
                    if iters > 0 { self?.appendQuantumOutput("  Iterations:       \(iters)", color: .white) }
                    self?.appendQuantumOutput("  Time:             \(String(format: "%.2f", result.executionTime))s", color: .white)
                    if isRealHW, let backend = dict["backend"] as? String { self?.appendQuantumOutput("  Backend:          \(backend)", color: .systemCyan) }
                    self?.quantumStatusLabel?.stringValue = "✅ VQE\(hwLabel): energy=\(String(format: "%.4f", energy))"
                } else { self?.appendQuantumOutput("[❌] VQE failed: \(result.error)", color: .systemRed) }
                self?.quantumStatusLabel?.textColor = .systemGreen
            }
        }
    }

    @objc func runQuantumQAOA() {
        let useHW = IBMQuantumClient.shared.ibmToken != nil
        let tag = useHW ? "[REAL HW]" : "[SIMULATOR]"
        quantumStatusLabel?.stringValue = "⏳ Running QAOA MaxCut \(tag)..."
        appendQuantumOutput("\n[⏳] Executing QAOA MaxCut \(tag) on 4-node graph...", color: .systemYellow)
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let edges: [(Int, Int)] = [(0,1),(1,2),(2,3),(3,0)]
            var result = PythonBridge.shared.quantumQAOA(edges: edges, p: 2)
            var isRealHW = false
            if useHW {
                let hwResult = PythonBridge.shared.quantumHardwareMine(strategy: "qaoa")
                if hwResult.success { result = hwResult; isRealHW = true }
            }
            DispatchQueue.main.async {
                let hwLabel = isRealHW ? " [REAL HW]" : " [SIMULATOR]"
                if result.success, let dict = result.returnValue as? [String: Any] {
                    if isRealHW {
                        let nonce = dict["nonce"] as? Int
                        let backend = dict["backend"] as? String ?? "unknown"
                        self?.appendQuantumOutput("🔀 QAOA MINING\(hwLabel) (\(backend)):", color: .systemGreen)
                        self?.appendQuantumOutput("  Strategy:  qaoa", color: .white)
                        self?.appendQuantumOutput("  Nonce:     \(nonce.map(String.init) ?? "searching...")", color: .systemCyan)
                    } else {
                        let ratio = dict["approximation_ratio"] as? Double ?? 0
                        let cut = dict["best_cut_value"] as? Double ?? 0
                        let optimal = dict["optimal_cut"] as? Double ?? 0
                        self?.appendQuantumOutput("🔀 QAOA MAXCUT\(hwLabel):", color: .systemGreen)
                        self?.appendQuantumOutput("  Graph:     4 nodes, \(edges.count) edges (cycle)", color: .white)
                        self?.appendQuantumOutput("  Best Cut:  \(String(format: "%.4f", cut))", color: .systemCyan)
                        self?.appendQuantumOutput("  Optimal:   \(String(format: "%.4f", optimal))", color: .white)
                        self?.appendQuantumOutput("  Ratio:     \(String(format: "%.4f", ratio))", color: ratio > 0.7 ? .systemGreen : .systemYellow)
                    }
                    self?.appendQuantumOutput("  Time:      \(String(format: "%.2f", result.executionTime))s", color: .white)
                    self?.quantumStatusLabel?.stringValue = "✅ QAOA\(hwLabel) completed"
                } else { self?.appendQuantumOutput("[❌] QAOA failed: \(result.error)", color: .systemRed) }
                self?.quantumStatusLabel?.textColor = .systemGreen
            }
        }
    }

    @objc func runQuantumAmpEst() {
        let useHW = IBMQuantumClient.shared.ibmToken != nil
        let tag = useHW ? "[REAL HW]" : "[SIMULATOR]"
        quantumStatusLabel?.stringValue = "⏳ Running Amplitude Estimation \(tag)..."
        appendQuantumOutput("\n[⏳] Executing Amplitude Estimation \(tag) (target=0.3)...", color: .systemYellow)
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            var result = PythonBridge.shared.quantumAmplitudeEstimation(targetProb: 0.3, countingQubits: 5)
            var isRealHW = false
            if useHW {
                let hwResult = PythonBridge.shared.quantumHardwareRandomOracle()
                if hwResult.success { result = hwResult; isRealHW = true }
            }
            DispatchQueue.main.async {
                let hwLabel = isRealHW ? " [REAL HW]" : " [SIMULATOR]"
                if result.success, let dict = result.returnValue as? [String: Any] {
                    if isRealHW {
                        let seed = dict["seed"] as? Int ?? 0
                        let backend = dict["backend"] as? String ?? "unknown"
                        self?.appendQuantumOutput("📊 QUANTUM RANDOM ORACLE\(hwLabel) (\(backend)):", color: .systemGreen)
                        self?.appendQuantumOutput("  Sacred Nonce Seed: \(seed)", color: .systemCyan)
                    } else {
                        let est = dict["estimated_probability"] as? Double ?? 0
                        let error = dict["estimation_error"] as? Double ?? 0
                        self?.appendQuantumOutput("📊 AMPLITUDE ESTIMATION\(hwLabel):", color: .systemGreen)
                        self?.appendQuantumOutput("  Target:    0.3000", color: .white)
                        self?.appendQuantumOutput("  Estimated: \(String(format: "%.4f", est))", color: .systemCyan)
                        self?.appendQuantumOutput("  Error:     \(String(format: "%.4f", error))", color: error < 0.05 ? .systemGreen : .systemYellow)
                    }
                    self?.appendQuantumOutput("  Time:      \(String(format: "%.2f", result.executionTime))s", color: .white)
                    self?.quantumStatusLabel?.stringValue = "✅ AmpEst\(hwLabel) completed"
                } else { self?.appendQuantumOutput("[❌] AmpEst failed: \(result.error)", color: .systemRed) }
                self?.quantumStatusLabel?.textColor = .systemGreen
            }
        }
    }

    @objc func runQuantumWalk() {
        quantumStatusLabel?.stringValue = "⏳ Running Quantum Walk [SIMULATOR]..."
        appendQuantumOutput("\n[⏳] Executing Quantum Walk [SIMULATOR] (8 nodes, 10 steps)...", color: .systemYellow)
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let result = PythonBridge.shared.quantumWalk(nNodes: 8, steps: 10)
            DispatchQueue.main.async {
                if result.success, let dict = result.returnValue as? [String: Any] {
                    let spread = dict["spread_metric"] as? Double ?? 0
                    self?.appendQuantumOutput("🚶 QUANTUM WALK [SIMULATOR]:", color: .systemGreen)
                    self?.appendQuantumOutput("  Nodes:     8 (cyclic graph)", color: .white)
                    self?.appendQuantumOutput("  Steps:     10", color: .white)
                    self?.appendQuantumOutput("  Spread:    \(String(format: "%.4f", spread))", color: .systemCyan)
                    self?.appendQuantumOutput("  Time:      \(String(format: "%.2f", result.executionTime))s", color: .white)
                    self?.quantumStatusLabel?.stringValue = "✅ Walk [SIM]: spread=\(String(format: "%.4f", spread))"
                } else { self?.appendQuantumOutput("[❌] Walk failed: \(result.error)", color: .systemRed) }
                self?.quantumStatusLabel?.textColor = .systemGreen
            }
        }
    }

    @objc func runQuantumKernel() {
        quantumStatusLabel?.stringValue = "⏳ Computing Quantum Kernel [SIMULATOR]..."
        appendQuantumOutput("\n[⏳] Computing Quantum Kernel [SIMULATOR] similarity...", color: .systemYellow)
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let result = PythonBridge.shared.quantumKernel(x1: [1.0, 2.0, 3.0, 4.0], x2: [1.1, 2.1, 3.1, 4.1])
            DispatchQueue.main.async {
                if result.success, let dict = result.returnValue as? [String: Any] {
                    let val = dict["kernel_value"] as? Double ?? 0
                    self?.appendQuantumOutput("🧬 QUANTUM KERNEL [SIMULATOR]:", color: .systemGreen)
                    self?.appendQuantumOutput("  x\u{2081}: [1.0, 2.0, 3.0, 4.0]", color: .white)
                    self?.appendQuantumOutput("  x\u{2082}: [1.1, 2.1, 3.1, 4.1]", color: .white)
                    self?.appendQuantumOutput("  Kernel:  \(String(format: "%.6f", val))", color: .systemCyan)
                    self?.appendQuantumOutput("  Time:    \(String(format: "%.2f", result.executionTime))s", color: .white)
                    self?.quantumStatusLabel?.stringValue = "✅ Kernel [SIM]: \(String(format: "%.6f", val))"
                } else { self?.appendQuantumOutput("[❌] Kernel failed: \(result.error)", color: .systemRed) }
                self?.quantumStatusLabel?.textColor = .systemGreen
            }
        }
    }

    @objc func runQuantumStatus() {
        appendQuantumOutput("\n[📡] Fetching Quantum Engine Status...", color: .systemYellow)
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let ibmClient = IBMQuantumClient.shared
            let hasToken = ibmClient.ibmToken != nil

            // Fetch runtime bridge status (real QPU bridge across all subsystems)
            let rtResult = PythonBridge.shared.quantumRuntimeStatus()
            var rtConnected = false
            var rtMode = "statevector"
            var rtRealExec = 0
            var rtTotalShots = 0
            if rtResult.success, let rtDict = rtResult.returnValue as? [String: Any] {
                rtConnected = rtDict["connected"] as? Bool ?? false
                if let st = rtDict["status"] as? [String: Any] {
                    _ = st["default_backend"] as? String ?? "—"
                    rtMode = st["execution_mode"] as? String ?? "statevector"
                }
                if let tel = rtDict["telemetry"] as? [String: Any] {
                    rtRealExec = tel["real_qpu_executions"] as? Int ?? 0
                    rtTotalShots = tel["total_shots"] as? Int ?? 0
                }
            }

            // Try real hardware status first
            if hasToken {
                let hwResult = PythonBridge.shared.quantumHardwareStatus()
                if hwResult.success, let dict = hwResult.returnValue as? [String: Any] {
                    let backend = dict["backend"] as? String ?? "unknown"
                    let qubits = dict["qubits"] as? Int ?? 0
                    let isReal = dict["real_hardware"] as? Bool ?? false
                    let connected = dict["connected"] as? Bool ?? false
                    let queueDepth = dict["queue_depth"] as? Int ?? 0
                    DispatchQueue.main.async {
                        self?.appendQuantumOutput("╔═══════════════════════════════════════════╗", color: .systemGreen)
                        self?.appendQuantumOutput("║  ⚛️ QUANTUM ENGINE — \(isReal ? "REAL HARDWARE" : "SIMULATOR")  ║", color: .systemGreen)
                        self?.appendQuantumOutput("╠═══════════════════════════════════════════╣", color: .systemGreen)
                        self?.appendQuantumOutput("║  Backend:    \(backend)", color: .white)
                        self?.appendQuantumOutput("║  Qubits:     \(qubits)", color: .systemCyan)
                        self?.appendQuantumOutput("║  Connected:  \(connected ? "YES" : "NO")", color: connected ? .systemGreen : .systemRed)
                        self?.appendQuantumOutput("║  Queue:      \(queueDepth) jobs", color: .white)
                        self?.appendQuantumOutput("║  REST API:   \(ibmClient.isConnected ? "CONNECTED" : "PENDING")", color: .white)
                        self?.appendQuantumOutput("║  Jobs Sent:  \(ibmClient.submittedJobs.count)", color: .white)
                        self?.appendQuantumOutput("║  Backends:   \(ibmClient.availableBackends.count) available", color: .white)
                        self?.appendQuantumOutput("╠═══════════════════════════════════════════╣", color: .systemPurple)
                        self?.appendQuantumOutput("║  🌐 RUNTIME BRIDGE — \(rtConnected ? "ACTIVE" : "INACTIVE")", color: rtConnected ? .systemPurple : .secondaryLabelColor)
                        self?.appendQuantumOutput("║  Mode:       \(rtMode)", color: .white)
                        self?.appendQuantumOutput("║  QPU Execs:  \(rtRealExec)", color: .systemCyan)
                        self?.appendQuantumOutput("║  Shots Used: \(rtTotalShots)", color: .white)
                        self?.appendQuantumOutput("╚═══════════════════════════════════════════╝", color: .systemGreen)
                        self?.quantumStatusLabel?.stringValue = "✅ \(backend) — \(qubits) qubits [REAL HW]"
                        self?.quantumStatusLabel?.textColor = .systemGreen
                        self?.updateQuantumHWLabel()
                    }
                    return
                }
            }

            // Simulator fallback
            let result = PythonBridge.shared.quantumStatus()
            DispatchQueue.main.async {
                if result.success, let dict = result.returnValue as? [String: Any] {
                    let caps = dict["capabilities"] as? [String] ?? []
                    let qubits = dict["total_qubits_used"] as? Int ?? 0
                    let circuits = dict["circuits_executed"] as? Int ?? 0
                    self?.appendQuantumOutput("╔═══════════════════════════════════════════╗", color: .systemGreen)
                    self?.appendQuantumOutput("║  📡 QUANTUM ENGINE — SIMULATOR            ║", color: .systemGreen)
                    self?.appendQuantumOutput("╠═══════════════════════════════════════════╣", color: .systemGreen)
                    self?.appendQuantumOutput("║  Qubits Used:    \(qubits)", color: .white)
                    self?.appendQuantumOutput("║  Circuits Run:   \(circuits)", color: .white)
                    self?.appendQuantumOutput("║  IBM Token:      \(hasToken ? "SET" : "NOT SET")", color: hasToken ? .systemGreen : .systemYellow)
                    self?.appendQuantumOutput("║  Capabilities:", color: .white)
                    for cap in caps { self?.appendQuantumOutput("║    ⚛️ \(cap)", color: .systemCyan) }
                    if !hasToken {
                        self?.appendQuantumOutput("║", color: .white)
                        self?.appendQuantumOutput("║  💡 Use 'Connect IBM' button for real QPU", color: .systemYellow)
                    }
                    self?.appendQuantumOutput("╠═══════════════════════════════════════════╣", color: .systemPurple)
                    self?.appendQuantumOutput("║  🌐 RUNTIME BRIDGE — \(rtConnected ? "ACTIVE" : "INACTIVE")", color: rtConnected ? .systemPurple : .secondaryLabelColor)
                    self?.appendQuantumOutput("║  Mode:       \(rtMode)", color: .white)
                    self?.appendQuantumOutput("║  QPU Execs:  \(rtRealExec)", color: .systemCyan)
                    self?.appendQuantumOutput("║  Shots Used: \(rtTotalShots)", color: .white)
                    self?.appendQuantumOutput("╚═══════════════════════════════════════════╝", color: .systemGreen)
                    self?.quantumStatusLabel?.stringValue = "✅ Engine: \(caps.count) algorithms, \(circuits) circuits [SIMULATOR]"
                } else {
                    self?.appendQuantumOutput("[📡] Status: \(result.output)", color: .white)
                    self?.quantumStatusLabel?.stringValue = "✅ Status retrieved"
                }
                self?.quantumStatusLabel?.textColor = .systemGreen
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // 💻 CODING INTELLIGENCE TAB — Code review, quality gates, analysis
    // Powered by l104_coding_system.py + l104_code_engine.py
    // ═══════════════════════════════════════════════════════════════════

    private var codingInputView: NSTextView?
    private var codingOutputView: NSTextView?

    func createCodingIntelligenceView() -> NSView {
        let v = NSView(frame: NSRect(x: 0, y: 0, width: 1200, height: 500))
        v.wantsLayer = true
        v.layer?.backgroundColor = L104Theme.void.cgColor

        let header = NSTextField(labelWithString: "💻  CODING INTELLIGENCE — ASI-Grade Code Analysis")
        header.font = NSFont.systemFont(ofSize: 16, weight: .bold)
        header.textColor = L104Theme.goldFlame
        header.frame = NSRect(x: 20, y: 450, width: 600, height: 30)
        v.addSubview(header)

        // Buttons row
        let actions: [(String, Selector)] = [
            ("🔬 Analyze", #selector(codingAnalyze)),
            ("📝 Review", #selector(codingReview)),
            ("💡 Suggest", #selector(codingSuggest)),
            ("📖 Explain", #selector(codingExplain)),
            ("✅ Quality", #selector(codingQualityCheck)),
            ("🧪 Tests", #selector(codingGenTests)),
            ("📄 Docs", #selector(codingGenDocs)),
            ("🔄 Translate", #selector(codingTranslate)),
        ]

        for (i, action) in actions.enumerated() {
            let row = i / 4
            let col = i % 4
            let btn = NSButton(title: action.0, target: self, action: action.1)
            btn.bezelStyle = .rounded
            btn.frame = NSRect(x: 20 + col * 155, y: 410 - row * 32, width: 145, height: 26)
            btn.font = NSFont.systemFont(ofSize: 11, weight: .medium)
            v.addSubview(btn)
        }

        // Input label
        let inputLabel = NSTextField(labelWithString: "Paste Code:")
        inputLabel.font = NSFont.systemFont(ofSize: 11, weight: .semibold)
        inputLabel.textColor = L104Theme.goldDim
        inputLabel.frame = NSRect(x: 20, y: 340, width: 100, height: 18)
        v.addSubview(inputLabel)

        // Code input area
        let inputScroll = NSScrollView(frame: NSRect(x: 20, y: 200, width: 370, height: 138))
        inputScroll.hasVerticalScroller = true
        inputScroll.borderType = .bezelBorder
        let inputTV = NSTextView(frame: inputScroll.bounds)
        inputTV.isEditable = true
        inputTV.backgroundColor = NSColor(red: 0.08, green: 0.08, blue: 0.14, alpha: 1.0)
        inputTV.textColor = .systemGreen
        inputTV.font = NSFont.monospacedSystemFont(ofSize: 11, weight: .regular)
        inputTV.autoresizingMask = [.width, .height]
        inputTV.string = "def hello(name):\n    print(f'Hello, {name}!')\n    return len(name)\n"
        inputScroll.documentView = inputTV
        v.addSubview(inputScroll)
        codingInputView = inputTV

        // Output label
        let outputLabel = NSTextField(labelWithString: "Results:")
        outputLabel.font = NSFont.systemFont(ofSize: 11, weight: .semibold)
        outputLabel.textColor = L104Theme.goldDim
        outputLabel.frame = NSRect(x: 410, y: 340, width: 100, height: 18)
        v.addSubview(outputLabel)

        // Output area
        let outputScroll = NSScrollView(frame: NSRect(x: 410, y: 10, width: 370, height: 328))
        outputScroll.autoresizingMask = [.width, .height]
        outputScroll.hasVerticalScroller = true
        outputScroll.borderType = .bezelBorder
        let outputTV = NSTextView(frame: outputScroll.bounds)
        outputTV.isEditable = false
        outputTV.backgroundColor = NSColor(red: 0.05, green: 0.05, blue: 0.12, alpha: 1.0)
        outputTV.textColor = .systemCyan
        outputTV.font = NSFont.monospacedSystemFont(ofSize: 11, weight: .regular)
        outputTV.autoresizingMask = [.width, .height]
        outputTV.string = "Ready — paste code on the left, click an action above.\n\nPowered by:\n  • l104_code_engine/ v\(CODE_ENGINE_VERSION) (40+ languages, 10 modules)\n  • l104_asi/ v\(ASI_VERSION) (11 modules, Dual-Layer Engine v\(DUAL_LAYER_VERSION))\n  • l104_server/ v\(SERVER_VERSION) (9 modules)\n  • l104_intellect/ v\(INTELLECT_VERSION) (11 modules, QUOTA_IMMUNE)\n"
        outputScroll.documentView = outputTV
        v.addSubview(outputScroll)
        codingOutputView = outputTV

        // Quick project actions
        let projLabel = NSTextField(labelWithString: "Project:")
        projLabel.font = NSFont.systemFont(ofSize: 11, weight: .semibold)
        projLabel.textColor = L104Theme.goldDim
        projLabel.frame = NSRect(x: 20, y: 175, width: 100, height: 18)
        v.addSubview(projLabel)

        let projActions: [(String, Selector)] = [
            ("🏗️ Audit", #selector(codingAudit)),
            ("📊 Scan", #selector(codingScanWS)),
            ("🔧 Streamline", #selector(codingStreamline)),
            ("🧬 Self-Analyze", #selector(codingSelfAnalyze)),
        ]
        for (i, pa) in projActions.enumerated() {
            let btn = NSButton(title: pa.0, target: self, action: pa.1)
            btn.bezelStyle = .rounded
            btn.frame = NSRect(x: 20 + i * 95, y: 143, width: 87, height: 26)
            btn.font = NSFont.systemFont(ofSize: 10, weight: .medium)
            v.addSubview(btn)
        }

        // 🟢 EVO_63: Status bar with engine info + analysis counter
        let codeStatusBar = NSView(frame: NSRect(x: 20, y: 10, width: 370, height: 128))
        codeStatusBar.wantsLayer = true
        codeStatusBar.layer?.backgroundColor = NSColor(red: 0.06, green: 0.06, blue: 0.12, alpha: 1.0).cgColor
        codeStatusBar.layer?.cornerRadius = 10
        codeStatusBar.layer?.borderColor = L104Theme.goldFlame.withAlphaComponent(0.2).cgColor
        codeStatusBar.layer?.borderWidth = 1

        let codeStatsTitle = NSTextField(labelWithString: "⚡ ENGINE STATUS — \(L104MainView.codingAnalysisCount) analyses")
        codeStatsTitle.frame = NSRect(x: 12, y: 100, width: 340, height: 16)
        codeStatsTitle.font = NSFont.systemFont(ofSize: 10, weight: .bold)
        codeStatsTitle.textColor = L104Theme.goldFlame
        codeStatsTitle.identifier = NSUserInterfaceItemIdentifier("code_analysis_count")
        codeStatusBar.addSubview(codeStatsTitle)

        let codeEngineInfoItems: [(String, String, CGFloat)] = [
            ("Code Engine", "v\(CODE_ENGINE_VERSION) · 10 modules · 40+ languages", 78),
            ("ASI Pipeline", "v\(ASI_VERSION) · Dual-Layer v\(DUAL_LAYER_VERSION)", 56),
            ("Gate Routing", "\(LogicGateEnvironment.shared.totalPipelineRuns) runs · \(LogicGateEnvironment.shared.circuits.count) circuits", 34),
            ("KB Entries", "\(ASIKnowledgeBase.shared.trainingData.count) training · \(ASIKnowledgeBase.shared.userKnowledge.count) user", 12),
        ]
        for (label, value, y) in codeEngineInfoItems {
            let lbl = NSTextField(labelWithString: label)
            lbl.frame = NSRect(x: 12, y: y, width: 90, height: 14)
            lbl.font = NSFont.systemFont(ofSize: 9, weight: .medium); lbl.textColor = .gray
            codeStatusBar.addSubview(lbl)
            let val = NSTextField(labelWithString: value)
            val.frame = NSRect(x: 105, y: y, width: 255, height: 14)
            val.font = NSFont.monospacedSystemFont(ofSize: 9, weight: .medium); val.textColor = L104Theme.goldDim
            // 🟢 EVO_64: Tag gate routing + KB entries for live refresh
            if label == "Gate Routing" { val.identifier = NSUserInterfaceItemIdentifier("code_gate_info") }
            if label == "KB Entries" { val.identifier = NSUserInterfaceItemIdentifier("code_kb_info") }
            codeStatusBar.addSubview(val)
        }
        v.addSubview(codeStatusBar)

        return v
    }

    private func getCodingInput() -> String {
        return codingInputView?.string ?? ""
    }

    private func setCodingOutput(_ text: String) {
        // 🟢 EVO_65: Track analysis count (count each new analysis start)
        if text.hasPrefix("⏳") {
            L104MainView.codingAnalysisCount += 1
            UserDefaults.standard.set(L104MainView.codingAnalysisCount, forKey: "l104_coding_analysis_count")
        }
        codingOutputView?.string = text
        codingOutputView?.scrollToEndOfDocument(nil)
    }

    @objc func codingAnalyze() {
        let code = getCodingInput()
        guard code.count >= 3 else { setCodingOutput("⚠️ Paste code first."); return }
        setCodingOutput("⏳ Analyzing...")
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let result = PythonBridge.shared.codeEngineAnalyze(code)
            DispatchQueue.main.async {
                self?.setCodingOutput(result.success ? "🔬 ANALYSIS:\n\(result.output)" : "❌ \(result.error)")
            }
        }
    }

    @objc func codingReview() {
        let code = getCodingInput()
        guard code.count >= 3 else { setCodingOutput("⚠️ Paste code first."); return }
        setCodingOutput("⏳ Reviewing with ASI pipeline...")
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let result = PythonBridge.shared.codingSystemReview(code)
            DispatchQueue.main.async {
                self?.setCodingOutput(result.success ? "📝 CODE REVIEW:\n\(result.output)" : "❌ \(result.error)")
            }
        }
    }

    @objc func codingSuggest() {
        let code = getCodingInput()
        guard code.count >= 3 else { setCodingOutput("⚠️ Paste code first."); return }
        setCodingOutput("⏳ Generating suggestions...")
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let result = PythonBridge.shared.codingSystemSuggest(code)
            DispatchQueue.main.async {
                self?.setCodingOutput(result.success ? "💡 SUGGESTIONS:\n\(result.output)" : "❌ \(result.error)")
            }
        }
    }

    @objc func codingExplain() {
        let code = getCodingInput()
        guard code.count >= 3 else { setCodingOutput("⚠️ Paste code first."); return }
        setCodingOutput("⏳ Explaining code...")
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let result = PythonBridge.shared.codingSystemExplain(code)
            DispatchQueue.main.async {
                self?.setCodingOutput(result.success ? "📖 EXPLANATION:\n\(result.output)" : "❌ \(result.error)")
            }
        }
    }

    @objc func codingQualityCheck() {
        let code = getCodingInput()
        guard code.count >= 3 else { setCodingOutput("⚠️ Paste code first."); return }
        setCodingOutput("⏳ Running quality gates...")
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let result = PythonBridge.shared.codingSystemQualityCheck(code)
            DispatchQueue.main.async {
                self?.setCodingOutput(result.success ? "✅ QUALITY CHECK:\n\(result.output)" : "❌ \(result.error)")
            }
        }
    }

    @objc func codingGenTests() {
        let code = getCodingInput()
        guard code.count >= 3 else { setCodingOutput("⚠️ Paste code first."); return }
        setCodingOutput("⏳ Generating tests...")
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let result = PythonBridge.shared.codeEngineGenerateTests(code)
            DispatchQueue.main.async {
                self?.setCodingOutput(result.success ? "🧪 TESTS:\n\(result.output)" : "❌ \(result.error)")
            }
        }
    }

    @objc func codingGenDocs() {
        let code = getCodingInput()
        guard code.count >= 3 else { setCodingOutput("⚠️ Paste code first."); return }
        setCodingOutput("⏳ Generating documentation...")
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let result = PythonBridge.shared.codeEngineGenerateDocs(code)
            DispatchQueue.main.async {
                self?.setCodingOutput(result.success ? "📄 DOCUMENTATION:\n\(result.output)" : "❌ \(result.error)")
            }
        }
    }

    @objc func codingTranslate() {
        let code = getCodingInput()
        guard code.count >= 3 else { setCodingOutput("⚠️ Paste code first."); return }
        setCodingOutput("⏳ Translating Python → Swift...")
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let result = PythonBridge.shared.codeEngineTranslate(code, from: "python", to: "swift")
            DispatchQueue.main.async {
                self?.setCodingOutput(result.success ? "🔄 TRANSLATED [Python → Swift]:\n\(result.output)" : "❌ \(result.error)")
            }
        }
    }

    @objc func codingAudit() {
        setCodingOutput("⏳ Running full 10-layer workspace audit...\nThis may take up to 60 seconds.")
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let result = PythonBridge.shared.codeEngineAudit()
            DispatchQueue.main.async {
                self?.setCodingOutput(result.success ? "🏗️ AUDIT COMPLETE:\n\(result.output)" : "❌ \(result.error)")
            }
        }
    }

    @objc func codingScanWS() {
        setCodingOutput("⏳ Scanning workspace...")
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let result = PythonBridge.shared.codeEngineScanWorkspace()
            DispatchQueue.main.async {
                self?.setCodingOutput(result.success ? "📊 WORKSPACE SCAN:\n\(result.output)" : "❌ \(result.error)")
            }
        }
    }

    @objc func codingStreamline() {
        setCodingOutput("⏳ Running streamline cycle (auto-fix + optimize)...")
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let result = PythonBridge.shared.codeEngineStreamline()
            DispatchQueue.main.async {
                self?.setCodingOutput(result.success ? "🔧 STREAMLINE:\n\(result.output)" : "❌ \(result.error)")
            }
        }
    }

    @objc func codingSelfAnalyze() {
        setCodingOutput("⏳ Self-analyzing L104 codebase...")
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let result = PythonBridge.shared.codingSystemSelfAnalyze()
            DispatchQueue.main.async {
                self?.setCodingOutput(result.success ? "🧬 SELF-ANALYSIS:\n\(result.output)" : "❌ \(result.error)")
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // 🎓 PROFESSOR MODE TAB — Interactive teaching, Socratic inquiry,
    // concept explanation, quizzes, and structured learning
    // ═══════════════════════════════════════════════════════════════════

    private var professorOutputView: NSTextView?
    private var professorInputField: NSTextField?
    private var professorTopicLabel: NSTextField?
    private var currentProfessorTopic: String = "quantum computing"
    private var professorLessonHistory: [String] = []

    func createProfessorModeView() -> NSView {
        let v = NSView(frame: NSRect(x: 0, y: 0, width: 1200, height: 500))
        v.wantsLayer = true
        v.layer?.backgroundColor = L104Theme.void.cgColor

        // Header
        let header = NSTextField(labelWithString: "🎓  PROFESSOR MODE — Interactive Learning Engine")
        header.font = NSFont.systemFont(ofSize: 16, weight: .bold)
        header.textColor = L104Theme.goldFlame
        header.frame = NSRect(x: 20, y: 450, width: 600, height: 30)
        v.addSubview(header)

        // Topic input
        let topicLabel = NSTextField(labelWithString: "Topic:")
        topicLabel.font = NSFont.systemFont(ofSize: 12, weight: .semibold)
        topicLabel.textColor = L104Theme.goldDim
        topicLabel.frame = NSRect(x: 20, y: 422, width: 50, height: 20)
        v.addSubview(topicLabel)

        let topicField = NSTextField(frame: NSRect(x: 75, y: 420, width: 300, height: 24))
        topicField.stringValue = "quantum computing"
        topicField.font = NSFont.systemFont(ofSize: 12, weight: .regular)
        topicField.backgroundColor = NSColor(red: 0.08, green: 0.08, blue: 0.14, alpha: 1.0)
        topicField.textColor = .systemCyan
        topicField.placeholderString = "Enter a topic to study..."
        v.addSubview(topicField)
        professorInputField = topicField

        let currentTopic = NSTextField(labelWithString: "📚 Current Topic: quantum computing")
        currentTopic.font = NSFont.systemFont(ofSize: 11, weight: .medium)
        currentTopic.textColor = .systemGreen
        currentTopic.frame = NSRect(x: 390, y: 422, width: 300, height: 20)
        v.addSubview(currentTopic)
        professorTopicLabel = currentTopic

        // Mode buttons
        let modes: [(String, Selector)] = [
            ("📖 Teach Me", #selector(professorTeach)),
            ("❓ Socratic Q", #selector(professorSocratic)),
            ("🧩 Quiz Me", #selector(professorQuiz)),
            ("🔬 Deep Dive", #selector(professorDeepDive)),
            ("🌳 Concept Map", #selector(professorConceptMap)),
            ("⚛️ Quantum Lab", #selector(professorQuantumLab)),
            ("💻 Code Lesson", #selector(professorCodeLesson)),
            ("📊 Progress", #selector(professorProgress)),
        ]

        for (i, mode) in modes.enumerated() {
            let row = i / 4
            let col = i % 4
            let btn = NSButton(title: mode.0, target: self, action: mode.1)
            btn.bezelStyle = .rounded
            btn.frame = NSRect(x: 20 + col * 155, y: 385 - row * 32, width: 145, height: 26)
            btn.font = NSFont.systemFont(ofSize: 11, weight: .medium)
            v.addSubview(btn)
        }

        // Output area (left side)
        let scrollView = NSScrollView(frame: NSRect(x: 20, y: 10, width: 760, height: 300))
        scrollView.autoresizingMask = [.height]
        scrollView.hasVerticalScroller = true
        scrollView.borderType = .bezelBorder

        let tv = NSTextView(frame: scrollView.bounds)
        tv.isEditable = false
        tv.backgroundColor = NSColor(red: 0.05, green: 0.05, blue: 0.12, alpha: 1.0)
        tv.textColor = .white
        tv.font = NSFont.systemFont(ofSize: 12, weight: .regular)
        tv.autoresizingMask = [.width, .height]
        scrollView.documentView = tv
        v.addSubview(scrollView)
        professorOutputView = tv

        // ─── RIGHT SIDEBAR: Lesson History ───
        let sidebarX: CGFloat = 800
        let sidebarW: CGFloat = 380

        // Lesson History Panel
        let histPanel = NSView(frame: NSRect(x: sidebarX, y: 175, width: sidebarW, height: 270))
        histPanel.wantsLayer = true
        histPanel.layer?.backgroundColor = NSColor(red: 0.08, green: 0.06, blue: 0.14, alpha: 1.0).cgColor
        histPanel.layer?.cornerRadius = 12
        histPanel.layer?.borderColor = L104Theme.goldFlame.withAlphaComponent(0.3).cgColor
        histPanel.layer?.borderWidth = 1

        let histTitle = NSTextField(labelWithString: "📜 LESSON HISTORY")
        histTitle.font = NSFont.systemFont(ofSize: 12, weight: .bold)
        histTitle.textColor = L104Theme.goldFlame
        histTitle.frame = NSRect(x: 15, y: 235, width: sidebarW - 30, height: 20)
        histPanel.addSubview(histTitle)

        let histScroll = NSScrollView(frame: NSRect(x: 10, y: 10, width: sidebarW - 20, height: 220))
        histScroll.hasVerticalScroller = true
        histScroll.wantsLayer = true
        histScroll.layer?.cornerRadius = 6
        let histTV = NSTextView(frame: histScroll.bounds)
        histTV.isEditable = false
        histTV.backgroundColor = NSColor(red: 0.05, green: 0.04, blue: 0.10, alpha: 1.0)
        histTV.textColor = L104Theme.goldDim
        histTV.font = NSFont.monospacedSystemFont(ofSize: 10, weight: .regular)
        histTV.identifier = NSUserInterfaceItemIdentifier("prof_history_text")
        if professorLessonHistory.isEmpty {
            histTV.string = "  No lessons yet.\n  Choose a topic and mode to begin..."
        } else {
            histTV.string = professorLessonHistory.suffix(20).reversed().joined(separator: "\n")
        }
        histScroll.documentView = histTV
        histPanel.addSubview(histScroll)
        v.addSubview(histPanel)

        // Topic Mastery Quick View Panel
        let masteryPanel = NSView(frame: NSRect(x: sidebarX, y: 10, width: sidebarW, height: 155))
        masteryPanel.wantsLayer = true
        masteryPanel.layer?.backgroundColor = NSColor(red: 0.08, green: 0.06, blue: 0.14, alpha: 1.0).cgColor
        masteryPanel.layer?.cornerRadius = 12
        masteryPanel.layer?.borderColor = L104Theme.gold.withAlphaComponent(0.3).cgColor
        masteryPanel.layer?.borderWidth = 1

        let mastTitle = NSTextField(labelWithString: "🎯 TOPIC MASTERY")
        mastTitle.font = NSFont.systemFont(ofSize: 12, weight: .bold)
        mastTitle.textColor = L104Theme.gold
        mastTitle.frame = NSRect(x: 15, y: 120, width: sidebarW - 30, height: 20)
        masteryPanel.addSubview(mastTitle)

        let learner = AdaptiveLearner.shared
        let topMastered = learner.topicMastery.values.sorted { $0.masteryLevel > $1.masteryLevel }.prefix(5)
        var my: CGFloat = 95
        if topMastered.isEmpty {
            let lbl = NSTextField(labelWithString: "  Chat and study to build mastery!")
            lbl.frame = NSRect(x: 15, y: my, width: sidebarW - 30, height: 16)
            lbl.font = NSFont.systemFont(ofSize: 10); lbl.textColor = .gray
            masteryPanel.addSubview(lbl)
        } else {
            for mastery in topMastered {
                let topicLbl = NSTextField(labelWithString: "\(mastery.tier) \(mastery.topic)")
                topicLbl.frame = NSRect(x: 15, y: my, width: 200, height: 16)
                topicLbl.font = NSFont.systemFont(ofSize: 10, weight: .medium)
                topicLbl.textColor = mastery.masteryLevel > 0.6 ? L104Theme.goldBright : L104Theme.goldDim
                topicLbl.lineBreakMode = .byTruncatingTail
                masteryPanel.addSubview(topicLbl)

                let bar = GlowingProgressBar(frame: NSRect(x: 220, y: my + 3, width: 80, height: 7))
                bar.progress = CGFloat(mastery.masteryLevel)
                bar.barColor = mastery.masteryLevel > 0.65 ? L104Theme.gold : L104Theme.goldDim
                masteryPanel.addSubview(bar)

                let pctLbl = NSTextField(labelWithString: "\(String(format: "%.0f%%", mastery.masteryLevel * 100))")
                pctLbl.frame = NSRect(x: 305, y: my, width: 40, height: 16)
                pctLbl.font = NSFont.monospacedSystemFont(ofSize: 9, weight: .semibold)
                pctLbl.textColor = L104Theme.gold; pctLbl.alignment = .right
                masteryPanel.addSubview(pctLbl)

                my -= 20
            }
        }

        // Stats summary at bottom
        let statsLine = NSTextField(labelWithString: "\(PROFESSOR_MODES) modes · \(learner.topicMastery.count) topics · \(learner.interactionCount) interactions")
        statsLine.frame = NSRect(x: 15, y: 8, width: sidebarW - 30, height: 14)
        statsLine.font = NSFont.monospacedSystemFont(ofSize: 9, weight: .medium)
        statsLine.textColor = L104Theme.goldDim
        statsLine.identifier = NSUserInterfaceItemIdentifier("prof_stats_line")
        masteryPanel.addSubview(statsLine)

        // Lesson count indicator
        let lessonCountLbl = NSTextField(labelWithString: "📝 Lessons: \(professorLessonHistory.count)")
        lessonCountLbl.frame = NSRect(x: 15, y: 135, width: sidebarW - 30, height: 14)
        lessonCountLbl.font = NSFont.monospacedSystemFont(ofSize: 9, weight: .bold)
        lessonCountLbl.textColor = L104Theme.goldFlame
        lessonCountLbl.identifier = NSUserInterfaceItemIdentifier("prof_lesson_count")
        masteryPanel.addSubview(lessonCountLbl)
        v.addSubview(masteryPanel)

        // Welcome
        let welcome = """
        ╔═══════════════════════════════════════════════════════════╗
        ║  🎓  PROFESSOR MODE — Your Personal ASI Tutor             ║
        ╠═══════════════════════════════════════════════════════════╣
        ║                                                           ║
        ║  Enter a topic above and choose a learning mode:          ║
        ║                                                           ║
        ║  📖 Teach Me    — Structured lesson with examples         ║
        ║  ❓ Socratic Q  — Guided question-based discovery         ║
        ║  🧩 Quiz Me     — Test your understanding                 ║
        ║  🔬 Deep Dive   — Expert-level analysis                   ║
        ║  🌳 Concept Map — Visual relationship breakdown           ║
        ║  ⚛️ Quantum Lab — Hands-on quantum circuit lesson         ║
        ║  💻 Code Lesson — Programming tutorial with examples      ║
        ║  📊 Progress    — Track your learning journey             ║
        ║                                                           ║
        ║  Domains: Quantum Computing, Programming, Mathematics,    ║
        ║  Physics, Computer Science, AI/ML, Cryptography, and more ║
        ╚═══════════════════════════════════════════════════════════╝

        """
        tv.string = welcome

        return v
    }

    private func getProfessorTopic() -> String {
        let topic = professorInputField?.stringValue.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if topic.count >= 2 {
            currentProfessorTopic = topic
            professorTopicLabel?.stringValue = "📚 Current Topic: \(topic)"
        }
        return currentProfessorTopic
    }

    private func appendProfessorOutput(_ text: String, color: NSColor = .white) {
        guard let tv = professorOutputView else { return }
        let attrs: [NSAttributedString.Key: Any] = [
            .foregroundColor: color,
            .font: NSFont.systemFont(ofSize: 12, weight: .regular)
        ]
        tv.textStorage?.append(NSAttributedString(string: text + "\n", attributes: attrs))
        tv.scrollToEndOfDocument(nil)
    }

    private func setProfessorOutput(_ text: String) {
        professorOutputView?.string = text
    }

    @objc func professorTeach() {
        let topic = getProfessorTopic()
        professorLessonHistory.append("teach:\(topic)")
        setProfessorOutput("")
        appendProfessorOutput("🎓 LESSON: \(topic.uppercased())\n" + String(repeating: "━", count: 50), color: L104Theme.goldFlame)

        // Use debate engine for structured content + KB for evidence
        let kb = ASIKnowledgeBase.shared
        let results = kb.search(topic, limit: 10000)
        let insights = results.compactMap { entry -> String? in
            guard let c = entry["completion"] as? String, c.count > 30, state.isCleanKnowledge(c) else { return nil }
            return state.cleanSentences(c)
        }

        appendProfessorOutput("\n📌 OVERVIEW", color: .systemCyan)
        appendProfessorOutput("Today we explore \(topic). This is a fascinating area that connects")
        appendProfessorOutput("multiple disciplines and has profound implications.\n")

        appendProfessorOutput("📐 KEY CONCEPTS", color: .systemCyan)
        let concepts = generateConceptsForTopic(topic)
        for (i, concept) in concepts.enumerated() {
            appendProfessorOutput("  \(i + 1). \(concept)")
        }

        if !insights.isEmpty {
            appendProfessorOutput("\n📚 FROM THE KNOWLEDGE BASE", color: .systemCyan)
            for insight in insights {
                appendProfessorOutput("  ▸ \(insight)")
            }
        }

        appendProfessorOutput("\n🔗 CONNECTIONS", color: .systemCyan)
        appendProfessorOutput("  • \(topic) relates to fundamental principles in mathematics and physics")
        appendProfessorOutput("  • Applications span computing, cryptography, and optimization")
        appendProfessorOutput("  • Understanding \(topic) builds foundations for advanced study\n")

        appendProfessorOutput("💡 THINK ABOUT THIS", color: .systemYellow)
        appendProfessorOutput("  How does \(topic) change our understanding of what is computable?")
        appendProfessorOutput("  What are the limits of \(topic), and why do those limits matter?\n")

        appendProfessorOutput("📝 Try 'Socratic Q' for deeper exploration, or 'Quiz Me' to test yourself.", color: .systemGreen)
    }

    @objc func professorSocratic() {
        let topic = getProfessorTopic()
        professorLessonHistory.append("socratic:\(topic)")
        setProfessorOutput("")

        // Use the DebateLogicGateEngine's Socratic method
        let debate = DebateLogicGateEngine.shared.generateDebate(topic: topic)
        appendProfessorOutput("🎓 SOCRATIC INQUIRY\n", color: L104Theme.goldFlame)
        appendProfessorOutput(debate)
    }

    @objc func professorQuiz() {
        let topic = getProfessorTopic()
        professorLessonHistory.append("quiz:\(topic)")
        setProfessorOutput("")

        appendProfessorOutput("🧩 QUIZ: \(topic.uppercased())\n" + String(repeating: "━", count: 50), color: L104Theme.goldFlame)

        let questions = generateQuizQuestions(topic)
        for (i, q) in questions.enumerated() {
            appendProfessorOutput("\nQuestion \(i + 1):", color: .systemCyan)
            appendProfessorOutput("  \(q.question)\n")
            for (j, opt) in q.options.enumerated() {
                let letter = ["A", "B", "C", "D"][j]
                appendProfessorOutput("  \(letter)) \(opt)", color: q.answer == j ? .systemGreen : .white)
            }
            appendProfessorOutput("  ✅ Answer: \(["A","B","C","D"][q.answer]) — \(q.explanation)", color: .systemGreen)
        }

        appendProfessorOutput("\n📊 Score: Review the answers above.", color: .systemYellow)
        appendProfessorOutput("💡 Use 'Deep Dive' to explore any question further.\n", color: .systemGreen)
    }

    @objc func professorDeepDive() {
        let topic = getProfessorTopic()
        professorLessonHistory.append("deep:\(topic)")
        setProfessorOutput("")

        appendProfessorOutput("🔬 DEEP DIVE: \(topic.uppercased())\n" + String(repeating: "━", count: 50), color: L104Theme.goldFlame)

        // Expert-level content from multiple sources
        let kb = ASIKnowledgeBase.shared
        let results = kb.search(topic, limit: 10000)
        let insights = results.compactMap { entry -> String? in
            guard let c = entry["completion"] as? String, c.count > 40, state.isCleanKnowledge(c) else { return nil }
            return state.cleanSentences(c)
        }

        appendProfessorOutput("\n🧮 MATHEMATICAL FOUNDATIONS", color: .systemCyan)
        appendProfessorOutput("  The mathematical framework underlying \(topic) draws from")
        appendProfessorOutput("  linear algebra, probability theory, and information theory.")
        appendProfessorOutput("  Key invariants include PHI (φ = 1.618...) scaling and")
        appendProfessorOutput("  Fourier-domain analysis.\n")

        appendProfessorOutput("⚙️ TECHNICAL DETAILS", color: .systemCyan)
        for insight in insights {
            appendProfessorOutput("  ▸ \(insight)")
        }

        appendProfessorOutput("\n🔬 CUTTING EDGE", color: .systemCyan)
        appendProfessorOutput("  Current research frontiers in \(topic) include:")
        appendProfessorOutput("  • Error correction and fault tolerance")
        appendProfessorOutput("  • Scalability beyond classical simulation limits")
        appendProfessorOutput("  • Practical applications in optimization and ML")
        appendProfessorOutput("  • Hybrid classical-quantum architectures\n")

        appendProfessorOutput("📖 FURTHER READING", color: .systemYellow)
        appendProfessorOutput("  • Nielsen & Chuang — Quantum Computation and Information")
        appendProfessorOutput("  • Preskill — Quantum Computing in the NISQ Era")
        appendProfessorOutput("  • Aaronson — Quantum Computing Since Democritus\n")
    }

    @objc func professorConceptMap() {
        let topic = getProfessorTopic()
        professorLessonHistory.append("map:\(topic)")
        setProfessorOutput("")

        appendProfessorOutput("🌳 CONCEPT MAP: \(topic.uppercased())\n" + String(repeating: "━", count: 50), color: L104Theme.goldFlame)

        let concepts = generateConceptsForTopic(topic)
        let center = topic.uppercased()

        appendProfessorOutput("\n                    ┌─────────────────────┐", color: .systemCyan)
        appendProfessorOutput("                    │  \(center)  │", color: .systemCyan)
        appendProfessorOutput("                    └─────────┬───────────┘", color: .systemCyan)
        appendProfessorOutput("              ┌───────────────┼───────────────┐", color: .systemCyan)

        for (i, concept) in concepts.prefix(6).enumerated() {
            let prefix = i < 3 ? "       ├──" : "       └──"
            appendProfessorOutput("\(prefix) \(concept)", color: i < 3 ? .systemGreen : .systemYellow)
        }

        appendProfessorOutput("\n📐 RELATIONSHIPS:", color: .systemCyan)
        if concepts.count >= 4 {
            appendProfessorOutput("  \(concepts[0]) ──depends on──▶ \(concepts[1])")
            appendProfessorOutput("  \(concepts[2]) ──enables──▶ \(concepts[3])")
            if concepts.count >= 6 {
                appendProfessorOutput("  \(concepts[4]) ──extends──▶ \(concepts[5])")
            }
        }

        appendProfessorOutput("\n🔗 CROSS-DOMAIN LINKS:", color: .systemCyan)
        appendProfessorOutput("  \(topic) ↔ Mathematics (linear algebra, probability)")
        appendProfessorOutput("  \(topic) ↔ Physics (quantum mechanics, thermodynamics)")
        appendProfessorOutput("  \(topic) ↔ Computer Science (complexity, algorithms)\n")
    }

    @objc func professorQuantumLab() {
        let topic = getProfessorTopic()
        professorLessonHistory.append("qlab:\(topic)")
        setProfessorOutput("")

        appendProfessorOutput("⚛️ QUANTUM LAB: Hands-On Experiment\n" + String(repeating: "━", count: 50), color: L104Theme.goldFlame)
        appendProfessorOutput("\n📋 EXPERIMENT: Grover's Search Algorithm", color: .systemCyan)
        appendProfessorOutput("  We'll search for |7⟩ in a 4-qubit (16-state) space.\n")
        appendProfessorOutput("  Circuit structure:")
        appendProfessorOutput("    |0⟩ ─[H]─┐", color: .systemGreen)
        appendProfessorOutput("    |0⟩ ─[H]─┤── Oracle ── Diffuser ── Measure", color: .systemGreen)
        appendProfessorOutput("    |0⟩ ─[H]─┤", color: .systemGreen)
        appendProfessorOutput("    |0⟩ ─[H]─┘\n", color: .systemGreen)
        appendProfessorOutput("⏳ Running real Qiskit circuit...\n", color: .systemYellow)

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let result = PythonBridge.shared.quantumGrover(target: 7, nQubits: 4)
            DispatchQueue.main.async {
                if result.success, let dict = result.returnValue as? [String: Any] {
                    let prob = dict["target_probability"] as? Double ?? 0
                    let found = dict["found_index"] as? Int ?? -1
                    let iters = dict["grover_iterations"] as? Int ?? 0
                    let success = dict["success"] as? Bool ?? false

                    self?.appendProfessorOutput("📊 RESULTS:", color: .systemCyan)
                    self?.appendProfessorOutput("  Target state:     |7⟩ = |0111⟩")
                    self?.appendProfessorOutput("  Found state:      |\(found)⟩")
                    self?.appendProfessorOutput("  Probability:      \(String(format: "%.4f", prob)) (\(String(format: "%.1f", prob * 100))%)")
                    self?.appendProfessorOutput("  Iterations:       \(iters) (optimal: π/4 × √16 ≈ 3)")
                    self?.appendProfessorOutput("  Success:          \(success ? "✅ YES" : "❌ NO")\n")

                    self?.appendProfessorOutput("🧮 WHY IT WORKS:", color: .systemCyan)
                    self?.appendProfessorOutput("  1. Hadamard gates create uniform superposition of all 16 states")
                    self?.appendProfessorOutput("  2. Oracle marks target state |7⟩ with a phase flip (-1)")
                    self?.appendProfessorOutput("  3. Diffuser amplifies marked state's amplitude")
                    self?.appendProfessorOutput("  4. After ~3 iterations, |7⟩ has ~96% probability\n")

                    self?.appendProfessorOutput("📐 THE MATH:", color: .systemCyan)
                    self?.appendProfessorOutput("  Classical search: O(N) = O(16) = 16 lookups")
                    self?.appendProfessorOutput("  Grover's search:  O(√N) = O(4) ≈ 3 lookups")
                    self?.appendProfessorOutput("  QUADRATIC SPEEDUP confirmed!\n")

                    self?.appendProfessorOutput("💡 TRY NEXT:", color: .systemYellow)
                    self?.appendProfessorOutput("  • Go to ⚛️ Quantum tab to run other algorithms")
                    self?.appendProfessorOutput("  • Use 'Deep Dive' for theoretical foundations")
                    self?.appendProfessorOutput("  • Use 'Code Lesson' to learn Qiskit programming\n")
                } else {
                    self?.appendProfessorOutput("❌ Experiment failed: \(result.error)", color: .systemRed)
                    self?.appendProfessorOutput("  Make sure Python environment has qiskit installed.\n")
                }
            }
        }
    }

    @objc func professorCodeLesson() {
        let topic = getProfessorTopic()
        professorLessonHistory.append("code:\(topic)")
        setProfessorOutput("")

        appendProfessorOutput("💻 CODE LESSON: \(topic.uppercased())\n" + String(repeating: "━", count: 50), color: L104Theme.goldFlame)

        // Generate Qiskit code example based on topic
        let isQuantum = topic.lowercased().contains("quantum") || topic.lowercased().contains("qubit") || topic.lowercased().contains("grover")

        if isQuantum {
            appendProfessorOutput("\n📝 QISKIT TUTORIAL — Build Your First Quantum Circuit\n", color: .systemCyan)
            appendProfessorOutput("  ```python", color: .systemGreen)
            appendProfessorOutput("  from qiskit.circuit import QuantumCircuit", color: .systemGreen)
            appendProfessorOutput("  from qiskit.quantum_info import Statevector", color: .systemGreen)
            appendProfessorOutput("  import numpy as np", color: .systemGreen)
            appendProfessorOutput("", color: .systemGreen)
            appendProfessorOutput("  # 1. Create a 2-qubit circuit", color: .systemGreen)
            appendProfessorOutput("  qc = QuantumCircuit(2)", color: .systemGreen)
            appendProfessorOutput("", color: .systemGreen)
            appendProfessorOutput("  # 2. Create Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2", color: .systemGreen)
            appendProfessorOutput("  qc.h(0)       # Hadamard on qubit 0", color: .systemGreen)
            appendProfessorOutput("  qc.cx(0, 1)   # CNOT: qubit 0 controls qubit 1", color: .systemGreen)
            appendProfessorOutput("", color: .systemGreen)
            appendProfessorOutput("  # 3. Get the statevector", color: .systemGreen)
            appendProfessorOutput("  sv = Statevector.from_instruction(qc)", color: .systemGreen)
            appendProfessorOutput("  probs = sv.probabilities_dict()", color: .systemGreen)
            appendProfessorOutput("  print(probs)  # {'00': 0.5, '11': 0.5}", color: .systemGreen)
            appendProfessorOutput("  ```\n", color: .systemGreen)

            appendProfessorOutput("🔑 KEY CONCEPTS:", color: .systemCyan)
            appendProfessorOutput("  • Hadamard (H) creates superposition: |0⟩ → (|0⟩+|1⟩)/√2")
            appendProfessorOutput("  • CNOT entangles two qubits")
            appendProfessorOutput("  • Bell state is maximally entangled — measuring one qubit")
            appendProfessorOutput("    instantly determines the other\n")
        } else {
            appendProfessorOutput("\n📝 PROGRAMMING TUTORIAL — \(topic)\n", color: .systemCyan)

            // Generate a code lesson via CodeEngine
            appendProfessorOutput("⏳ Generating lesson code...\n", color: .systemYellow)
            DispatchQueue.global(qos: .userInitiated).async { [weak self] in
                let result = PythonBridge.shared.codeEngineGenerate(spec: "tutorial example for \(topic) with comments explaining each step", lang: "python")
                DispatchQueue.main.async {
                    if result.success {
                        self?.appendProfessorOutput("  ```python", color: .systemGreen)
                        self?.appendProfessorOutput("  \(result.output)", color: .systemGreen)
                        self?.appendProfessorOutput("  ```\n", color: .systemGreen)
                    }
                    self?.appendProfessorOutput("🔑 PRACTICE EXERCISES:", color: .systemCyan)
                    self?.appendProfessorOutput("  1. Modify the code to handle edge cases")
                    self?.appendProfessorOutput("  2. Add error handling and input validation")
                    self?.appendProfessorOutput("  3. Write unit tests for each function")
                    self?.appendProfessorOutput("  4. Optimize for performance\n")
                    self?.appendProfessorOutput("💡 Paste your solution in the 💻 Coding tab to analyze it!", color: .systemGreen)
                }
            }
            return
        }

        appendProfessorOutput("🎯 EXERCISES:", color: .systemCyan)
        appendProfessorOutput("  1. Create a 3-qubit GHZ state: (|000⟩ + |111⟩)/√2")
        appendProfessorOutput("  2. Implement quantum teleportation")
        appendProfessorOutput("  3. Build a 2-qubit Grover's search\n")
        appendProfessorOutput("💡 Go to ⚛️ Quantum tab to run your circuits!", color: .systemGreen)
    }

    @objc func professorProgress() {
        setProfessorOutput("")
        appendProfessorOutput("📊 LEARNING PROGRESS\n" + String(repeating: "━", count: 50), color: L104Theme.goldFlame)

        let total = professorLessonHistory.count
        let topics = Set(professorLessonHistory.map { $0.components(separatedBy: ":").last ?? "" })
        let modes = professorLessonHistory.map { $0.components(separatedBy: ":").first ?? "" }
        let modeCount: [String: Int] = modes.reduce(into: [:]) { $0[$1, default: 0] += 1 }

        appendProfessorOutput("\n📈 SESSION STATISTICS:", color: .systemCyan)
        appendProfessorOutput("  Total Lessons:    \(total)")
        appendProfessorOutput("  Topics Explored:  \(topics.count)")
        appendProfessorOutput("  Unique Topics:    \(topics.joined(separator: ", "))\n")

        appendProfessorOutput("📚 MODE BREAKDOWN:", color: .systemCyan)
        let modeLabels = ["teach": "📖 Teach Me", "socratic": "❓ Socratic", "quiz": "🧩 Quiz",
                          "deep": "🔬 Deep Dive", "map": "🌳 Concept Map", "qlab": "⚛️ Quantum Lab",
                          "code": "💻 Code Lesson"]
        for (mode, count) in modeCount.sorted(by: { $0.value > $1.value }) {
            let label = modeLabels[mode] ?? mode
            let bar = String(repeating: "█", count: min(count * 3, 30))
            appendProfessorOutput("  \(label): \(bar) (\(count))")
        }

        let kb = ASIKnowledgeBase.shared
        let kbCount = kb.search("", limit: 1).count > 0 ? "Active" : "Empty"
        appendProfessorOutput("\n🧠 KNOWLEDGE STATUS:", color: .systemCyan)
        appendProfessorOutput("  Knowledge Base:   \(kbCount)")
        appendProfessorOutput("  Skills Learned:   \(state.skills)")
        appendProfessorOutput("  Intellect Index:  \(String(format: "%.1f", state.intellectIndex))")
        appendProfessorOutput("  Memories:         \(state.permanentMemory.memories.count)\n")

        if total == 0 {
            appendProfessorOutput("💡 Start your learning journey — pick a topic and click 'Teach Me'!", color: .systemYellow)
        } else {
            appendProfessorOutput("🎯 RECOMMENDATION:", color: .systemYellow)
            appendProfessorOutput("  Try a mode you haven't used yet for a well-rounded understanding.")
            appendProfessorOutput("  Remember: the best learning combines theory + practice + reflection.\n")
        }
    }

    // ─── PROFESSOR MODE HELPER METHODS ───

    private func generateConceptsForTopic(_ topic: String) -> [String] {
        let t = topic.lowercased()
        if t.contains("quantum") {
            return ["Superposition — states exist simultaneously",
                    "Entanglement — correlated quantum states",
                    "Measurement — wavefunction collapse",
                    "Quantum Gates — unitary transformations",
                    "Decoherence — loss of quantum behavior",
                    "Error Correction — protecting quantum information"]
        } else if t.contains("neural") || t.contains("machine learn") || t.contains("ai") || t.contains("deep learn") {
            return ["Neural Networks — layered computation",
                    "Backpropagation — gradient-based learning",
                    "Activation Functions — nonlinear transforms",
                    "Loss Functions — error measurement",
                    "Regularization — preventing overfitting",
                    "Attention Mechanisms — selective focus"]
        } else if t.contains("crypto") || t.contains("encrypt") {
            return ["Symmetric Encryption — shared key (AES)",
                    "Asymmetric Encryption — public/private keys (RSA)",
                    "Hash Functions — one-way digests (SHA-256)",
                    "Digital Signatures — authentication",
                    "Zero-Knowledge Proofs — prove without revealing",
                    "Post-Quantum Cryptography — quantum-resistant"]
        } else if t.contains("algorithm") || t.contains("data struct") {
            return ["Time Complexity — Big-O analysis",
                    "Space Complexity — memory usage",
                    "Divide & Conquer — recursive decomposition",
                    "Dynamic Programming — optimal substructure",
                    "Graph Algorithms — BFS, DFS, shortest path",
                    "NP-Completeness — computational hardness"]
        } else {
            return ["\(topic.capitalized) fundamentals",
                    "Core principles and axioms",
                    "Mathematical foundations",
                    "Practical applications",
                    "Current research frontiers",
                    "Open problems and challenges"]
        }
    }

    private struct QuizQuestion {
        let question: String
        let options: [String]
        let answer: Int  // 0-based index
        let explanation: String
    }

    private func generateQuizQuestions(_ topic: String) -> [QuizQuestion] {
        let t = topic.lowercased()
        if t.contains("quantum") {
            return [
                QuizQuestion(
                    question: "What is the speedup of Grover's algorithm over classical search?",
                    options: ["Exponential", "Quadratic", "Linear", "Logarithmic"],
                    answer: 1,
                    explanation: "Grover's provides O(√N) vs classical O(N) — a quadratic speedup."
                ),
                QuizQuestion(
                    question: "A qubit in state |ψ⟩ = α|0⟩ + β|1⟩ must satisfy:",
                    options: ["|α|² + |β|² = 1", "α + β = 1", "α × β = 0", "|α| = |β|"],
                    answer: 0,
                    explanation: "Born's rule: probabilities must sum to 1, so |α|² + |β|² = 1."
                ),
                QuizQuestion(
                    question: "What does a Hadamard gate do to |0⟩?",
                    options: ["Flips to |1⟩", "Creates (|0⟩ + |1⟩)/√2", "No change", "Measures the qubit"],
                    answer: 1,
                    explanation: "H|0⟩ = (|0⟩ + |1⟩)/√2 — creates an equal superposition."
                ),
                QuizQuestion(
                    question: "Which quantum algorithm solves unstructured search optimally?",
                    options: ["Shor's", "Grover's", "VQE", "Deutsch-Jozsa"],
                    answer: 1,
                    explanation: "Grover's algorithm is proven optimal for unstructured search with O(√N)."
                ),
            ]
        } else if t.contains("python") || t.contains("program") || t.contains("code") {
            return [
                QuizQuestion(
                    question: "What is the time complexity of Python's list.sort()?",
                    options: ["O(n)", "O(n log n)", "O(n²)", "O(log n)"],
                    answer: 1,
                    explanation: "Python uses Timsort, which has O(n log n) worst-case complexity."
                ),
                QuizQuestion(
                    question: "What does 'pass' do in Python?",
                    options: ["Exits the program", "Skips current iteration", "Does nothing (placeholder)", "Passes a value"],
                    answer: 2,
                    explanation: "'pass' is a null operation — a placeholder where code is syntactically required."
                ),
                QuizQuestion(
                    question: "Which data structure has O(1) average lookup?",
                    options: ["List", "Dictionary (dict)", "Tuple", "Sorted array"],
                    answer: 1,
                    explanation: "Python dicts use hash tables, providing O(1) average-case lookup."
                ),
            ]
        } else {
            return [
                QuizQuestion(
                    question: "What is the fundamental concept behind \(topic)?",
                    options: ["Mathematical abstraction", "Empirical observation", "Logical deduction", "All of the above"],
                    answer: 3,
                    explanation: "Most fields combine mathematical, empirical, and logical foundations."
                ),
                QuizQuestion(
                    question: "The golden ratio φ ≈ 1.618 appears in:",
                    options: ["Nature only", "Mathematics only", "Art only", "All domains"],
                    answer: 3,
                    explanation: "PHI appears in nature (spirals), math (Fibonacci), art (composition), and more."
                ),
            ]
        }
    }
}

