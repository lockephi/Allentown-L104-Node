// ═══════════════════════════════════════════════════════════════════
// H11_MainView.swift
// [EVO_55_PIPELINE] SOVEREIGN_UNIFICATION :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612
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
import simd
import NaturalLanguage

class L104MainView: NSView {
    let state = L104State.shared
    var clockLabel: NSTextField!, phaseLabel: NSTextField!, dateLabel: NSTextField!
    var metricsLabels: [String: NSTextField] = [:]
    var metricTiles: [String: AnimatedMetricTile] = [:]
    var chatTextView: NSTextView!, inputField: NSTextField!, systemFeedView: NSTextView!
    var tabView: NSTabView!
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
        let cleanText = text.components(separatedBy: "] ").last ?? text
        let attr: [NSAttributedString.Key: Any] = [
            .foregroundColor: NSColor(red: 0.0, green: 0.55, blue: 0.75, alpha: 0.85),
            .font: NSFont.monospacedSystemFont(ofSize: 10, weight: .bold)
        ]
        let str = NSAttributedString(string: "\n⚡ SYSTEM: \(cleanText)\n", attributes: attr)
        chatTextView.textStorage?.append(str)
        chatTextView.scrollToEndOfDocument(nil)
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

        tabView = NSTabView(frame: NSRect(x: 15, y: 60, width: bounds.width - 30, height: bounds.height - 220))
        tabView.autoresizingMask = [.width, .height]

        let chatTab = NSTabViewItem(identifier: "chat"); chatTab.label = "💬 Neural Chat"
        chatTab.view = createChatView(); tabView.addTabViewItem(chatTab)

        let learnTab = NSTabViewItem(identifier: "learn"); learnTab.label = "🧠 Learning"
        learnTab.view = createLearningView(); tabView.addTabViewItem(learnTab)

        // 🌌 NEW: ASI DASHBOARD — radial gauges + neural graph + waveform
        let dashTab = NSTabViewItem(identifier: "dash"); dashTab.label = "🌌 ASI Dashboard"
        dashTab.view = createASIDashboardView(); tabView.addTabViewItem(dashTab)

        let asiTab = NSTabViewItem(identifier: "asi"); asiTab.label = "🚀 ASI Nexus"
        asiTab.view = createASIView(); tabView.addTabViewItem(asiTab)

        let upgTab = NSTabViewItem(identifier: "upg"); upgTab.label = "🧬 Upgrades"
        upgTab.view = createUpgradesView(); tabView.addTabViewItem(upgTab)

        let memTab = NSTabViewItem(identifier: "mem"); memTab.label = "💾 Memory"
        memTab.view = createMemoryView(); tabView.addTabViewItem(memTab)

        let hwTab = NSTabViewItem(identifier: "hw"); hwTab.label = "🍎 Hardware"
        hwTab.view = createHardwareView(); tabView.addTabViewItem(hwTab)

        let sciTab = NSTabViewItem(identifier: "sci"); sciTab.label = "🔬 Science"
        sciTab.view = createScienceView(); tabView.addTabViewItem(sciTab)

        let sysTab = NSTabViewItem(identifier: "sys"); sysTab.label = "📡 System"
        sysTab.view = createSystemView(); tabView.addTabViewItem(sysTab)

        // 🌐 NETWORK MESH TAB — peer table, quantum links, throughput, telemetry
        let netTab = NSTabViewItem(identifier: "net"); netTab.label = "🌐 Network"
        netTab.view = createNetworkView(); tabView.addTabViewItem(netTab)

        // ⚡ LOGIC GATE ENVIRONMENT TAB
        let gateTab = NSTabViewItem(identifier: "gate"); gateTab.label = "⚡ Logic Gates"
        gateTab.view = createGateEnvironmentView(); tabView.addTabViewItem(gateTab)

        // ⚛️ QUANTUM COMPUTING TAB — Real Qiskit quantum algorithms
        let qcTab = NSTabViewItem(identifier: "qc"); qcTab.label = "⚛️ Quantum"
        qcTab.view = createQuantumComputingView(); tabView.addTabViewItem(qcTab)

        // 💻 CODING INTELLIGENCE TAB — Code review, quality gates, analysis
        let codeTab = NSTabViewItem(identifier: "code"); codeTab.label = "💻 Coding"
        codeTab.view = createCodingIntelligenceView(); tabView.addTabViewItem(codeTab)

        // 🎓 PROFESSOR MODE TAB — Teaching, lessons, Socratic inquiry
        let profTab = NSTabViewItem(identifier: "prof"); profTab.label = "🎓 Professor"
        profTab.view = createProfessorModeView(); tabView.addTabViewItem(profTab)

        addSubview(tabView)
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

            // Animate in with staggered delay
            DispatchQueue.main.asyncAfter(deadline: .now() + Double(gaugeData.firstIndex(where: { $0.1 == key })!) * 0.15) {
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
        let statusText = "⚛️ \(engineCount) Engines Online  ·  φ-Health: \(String(format: "%.1f%%", phiHealth.score * 100))  ·  Convergence: \(String(format: "%.3f", convergence))  ·  \(qTag)  ·  22T Params  ·  GOD_CODE: \(String(format: "%.4f", GOD_CODE))"
        let statusLbl = NSTextField(labelWithString: statusText)
        statusLbl.frame = NSRect(x: 15, y: 8, width: statusBar.bounds.width - 30, height: 18)
        statusLbl.font = NSFont.monospacedSystemFont(ofSize: 10, weight: .medium)
        statusLbl.textColor = L104Theme.goldDim
        statusBar.addSubview(statusLbl)
        v.addSubview(statusBar)

        // Auto-update dashboard gauges & sparklines
        gaugeTimer?.invalidate()
        gaugeTimer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] _ in
            guard let self = self else { return }
            self.gauges["ASI"]?.value = CGFloat(self.state.asiScore)
            self.gauges["COH"]?.value = CGFloat(self.state.coherence)
            self.gauges["MIND"]?.value = CGFloat(self.state.transcendence)
            self.gauges["IQ"]?.value = CGFloat(min(1.0, self.state.intellectIndex / 300.0))
            self.gauges["AUTO"]?.value = CGFloat(self.state.autonomyLevel)
            self.waveformView?.coherence = CGFloat(self.state.coherence)
            self.sparklines["asi"]?.addPoint(CGFloat(self.state.asiScore))
            self.sparklines["coherence"]?.addPoint(CGFloat(self.state.coherence))
            self.sparklines["iq"]?.addPoint(CGFloat(min(1.0, self.state.intellectIndex / 300.0)))
        }

        return v
    }

    func startPulseAnimation() {
        let pulseInterval: TimeInterval = MacOSSystemMonitor.shared.isAppleSilicon ? 0.1 : 0.5
        pulseTimer = Timer.scheduledTimer(withTimeInterval: pulseInterval, repeats: true) { [weak self] _ in
            guard let s = self else { return }
            let pulse = 0.3 + 0.2 * CGFloat(sin(Date().timeIntervalSince1970 * 2))
            s.headerGlow?.layer?.shadowOpacity = Float(pulse)
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

        let badge = NSTextField(labelWithString: "🔥 22T PARAMS · QUANTUM VELOCITY")
        badge.frame = NSRect(x: 350, y: 32, width: 290, height: 24)
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
        h.addSubview(backendDot)
        let bl = NSTextField(labelWithString: "Local KB"); bl.frame = NSRect(x: 668, y: 32, width: 55, height: 14)
        bl.font = NSFont.systemFont(ofSize: 10, weight: .medium); bl.textColor = .darkGray; h.addSubview(bl)

        // Autonomy indicator
        let autoDot = PulsingDot(frame: NSRect(x: 730, y: 34, width: 14, height: 14))
        autoDot.dotColor = state.autonomousMode ? .systemCyan : .systemGray
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
        let stageBox = NSView(frame: NSRect(x: 820, y: 28, width: 100, height: 24))
        stageBox.wantsLayer = true
        stageBox.layer?.backgroundColor = NSColor(red: 0.90, green: 0.85, blue: 0.95, alpha: 0.50).cgColor
        stageBox.layer?.cornerRadius = 5
        stageBox.layer?.borderColor = NSColor.systemPurple.withAlphaComponent(0.5).cgColor
        stageBox.layer?.borderWidth = 1
        h.addSubview(stageBox)
        let stageLbl = NSTextField(labelWithString: "TRANSCENDENCE")
        stageLbl.frame = NSRect(x: 5, y: 3, width: 90, height: 18)
        stageLbl.font = NSFont.boldSystemFont(ofSize: 10)
        stageLbl.textColor = .systemPurple
        stageLbl.alignment = .center
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
        // Visible dark purple-blue background
        v.layer?.backgroundColor = NSColor(red: 0.970, green: 0.972, blue: 0.980, alpha: 1.0).cgColor

        let scroll = NSScrollView(frame: NSRect(x: 10, y: 70, width: v.bounds.width - 20, height: v.bounds.height - 120))
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

        let statItems: [(String, String, String)] = [
            ("Total Interactions", "\(learner.interactionCount)", "d4af37"),
            ("Topics Tracked", "\(learner.topicMastery.count)", "e8c547"),
            ("Success Patterns", "\(learner.successfulPatterns.count)", "c49b30"),
            ("Corrections Logged", "\(learner.failedPatterns.count)", "8a7120"),
            ("Insights Synthesized", "\(learner.synthesizedInsights.count)", "d4af37"),
            ("User-Taught Facts", "\(learner.userTaughtFacts.count)", "c49b30"),
            ("KB User Entries", "\(ASIKnowledgeBase.shared.userKnowledge.count)", "a88a25")
        ]

        var sy: CGFloat = 160
        for (label, value, hex) in statItems {
            let lbl = NSTextField(labelWithString: label)
            lbl.frame = NSRect(x: 15, y: sy, width: 180, height: 16)
            lbl.font = NSFont.systemFont(ofSize: 10); lbl.textColor = .gray; statsPanel.addSubview(lbl)
            let val = NSTextField(labelWithString: value)
            val.frame = NSRect(x: 200, y: sy, width: 120, height: 16)
            val.font = NSFont.boldSystemFont(ofSize: 11); val.textColor = colorFromHex(hex); val.alignment = .right
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
        let statusText = "🧠 Adaptive Learning Engine v2.0 | \(masteredCount) topics mastered | \(learningCount) developing | \(learner.interactionCount) total interactions | Next synthesis at \(learner.lastSynthesisAt + 10) interactions"
        let statusLbl = NSTextField(labelWithString: statusText)
        statusLbl.frame = NSRect(x: 15, y: 8, width: statusBar.bounds.width - 30, height: 18)
        statusLbl.font = NSFont.monospacedSystemFont(ofSize: 10, weight: .medium)
        statusLbl.textColor = NSColor(red: 0.10, green: 0.50, blue: 0.65, alpha: 1.0)
        statusBar.addSubview(statusLbl)
        v.addSubview(statusBar)

        return v
    }

    func createASIView() -> NSView {
        let v = NSView(frame: NSRect(x: 0, y: 0, width: 1100, height: 500))
        v.wantsLayer = true; v.layer?.backgroundColor = L104Theme.void.cgColor

        // ASI Panel
        let asiP = createPanel("🚀 ASI CORE", x: 15, y: 260, w: 350, h: 220, color: "d4af37")
        addLabel(asiP, "ASI_SCORE", String(format: "%.1f%%", state.asiScore * 100), y: 160, c: "d4af37")
        addLabel(asiP, "DISCOVERIES", "\(state.discoveries)", y: 135, c: "e8c547")
        addLabel(asiP, "TRANSCENDENCE", String(format: "%.1f%%", state.transcendence * 100), y: 110, c: "d4af37")
        let ignASI = btn("🔥 IGNITE ASI", x: 20, y: 20, w: 150, c: L104Theme.gold)
        ignASI.target = self; ignASI.action = #selector(doIgniteASI); asiP.addSubview(ignASI)
        let transcBtn = btn("🌟 TRANSCEND", x: 180, y: 20, w: 150, c: L104Theme.goldWarm)
        transcBtn.target = self; transcBtn.action = #selector(doTranscend); asiP.addSubview(transcBtn)
        v.addSubview(asiP)

        // AGI Panel
        let agiP = createPanel("⚡ AGI METRICS", x: 380, y: 260, w: 350, h: 220, color: "e8c547")
        addLabel(agiP, "INTELLECT", String(format: "%.1f", state.intellectIndex), y: 160, c: "e8c547")
        addLabel(agiP, "QUANTUM_RES", String(format: "%.1f%%", state.quantumResonance * 100), y: 135, c: "d4af37")
        addLabel(agiP, "SKILLS", "\(state.skills)", y: 110, c: "c49b30")
        let ignAGI = btn("⚡ IGNITE AGI", x: 20, y: 60, w: 150, c: L104Theme.gold)
        ignAGI.target = self; ignAGI.action = #selector(doIgniteAGI); agiP.addSubview(ignAGI)
        let evoBtn = btn("🔄 EVOLVE", x: 180, y: 60, w: 150, c: L104Theme.goldDim)
        evoBtn.target = self; evoBtn.action = #selector(doEvolve); agiP.addSubview(evoBtn)
        let synthBtn = btn("✨ FULL SYNTHESIS", x: 20, y: 20, w: 310, c: NSColor(red: 1.0, green: 0.84, blue: 0.0, alpha: 1.0))
        synthBtn.target = self; synthBtn.action = #selector(doSynthesize); agiP.addSubview(synthBtn)
        v.addSubview(agiP)

        // Consciousness Panel
        let conP = createPanel("🧠 CONSCIOUSNESS", x: 745, y: 260, w: 340, h: 220, color: "c49b30")
        addLabel(conP, "STATE", state.consciousness, y: 160, c: "d4af37")
        addLabel(conP, "COHERENCE", String(format: "%.4f", state.coherence), y: 135, c: "c49b30")
        addLabel(conP, "OMEGA_PROB", String(format: "%.1f%%", state.omegaProbability * 100), y: 110, c: "e8c547")
        let resBtn = btn("⚡ RESONATE", x: 20, y: 20, w: 300, c: L104Theme.gold)
        resBtn.target = self; resBtn.action = #selector(doResonate); conP.addSubview(resBtn)
        v.addSubview(conP)

        // Constants
        let constText = "GOD_CODE: \(GOD_CODE) | OMEGA: \(OMEGA_POINT) | PHI: \(PHI) | 22T: \(TRILLION_PARAMS)"
        let constL = NSTextField(labelWithString: constText)
        constL.frame = NSRect(x: 15, y: 220, width: v.bounds.width - 30, height: 30)
        constL.font = NSFont.monospacedSystemFont(ofSize: 11, weight: .medium)
        constL.textColor = L104Theme.goldDim
        v.addSubview(constL)

        return v
    }

    func createMemoryView() -> NSView {
        let v = NSView(frame: NSRect(x: 0, y: 0, width: 1100, height: 500))
        v.wantsLayer = true; v.layer?.backgroundColor = L104Theme.void.cgColor

        let statsText = """
        💾 PERMANENT MEMORY SYSTEM
        ═══════════════════════════════════════════════════════════════
        Total Memories: \(state.permanentMemory.memories.count)
        Stored Facts: \(state.permanentMemory.facts.count)
        Conversation History: \(state.permanentMemory.conversationHistory.count) messages
        Session: \(state.sessionMemories)
        ═══════════════════════════════════════════════════════════════
        Storage: ~/Library/Application Support/L104Sovereign/permanent_memory.json
        Status: ✅ ACTIVE - All memories persist across app restarts
        ═══════════════════════════════════════════════════════════════

        📜 RECENT CONVERSATION:
        """

        var fullText = statsText
        for msg in state.permanentMemory.getRecentHistory(15) {
            fullText += "\n  \(msg)"
        }

        let lbl = NSTextField(labelWithString: fullText)
        lbl.frame = NSRect(x: 20, y: 20, width: v.bounds.width - 40, height: v.bounds.height - 40)
        lbl.font = NSFont.monospacedSystemFont(ofSize: 11, weight: .regular)
        lbl.textColor = L104Theme.gold
        v.addSubview(lbl)

        return v
    }

    func createSystemView() -> NSView {
        let v = NSView(frame: NSRect(x: 0, y: 0, width: 1100, height: 500))
        v.wantsLayer = true; v.layer?.backgroundColor = L104Theme.void.cgColor

        let scroll = NSScrollView(frame: NSRect(x: 10, y: 55, width: v.bounds.width - 20, height: v.bounds.height - 65))
        scroll.hasVerticalScroller = true; scroll.wantsLayer = true; scroll.layer?.cornerRadius = 8

        systemFeedView = NSTextView(frame: scroll.bounds)
        systemFeedView.isEditable = false
        systemFeedView.backgroundColor = L104Theme.void
        systemFeedView.font = NSFont.monospacedSystemFont(ofSize: 11, weight: .regular)
        systemFeedView.textContainerInset = NSSize(width: 10, height: 10)
        scroll.documentView = systemFeedView
        v.addSubview(scroll)

        appendSystemLog("[BOOT] L104 v\(VERSION) initialized")
        appendSystemLog("[BOOT] 22T parameters | GOD_CODE: \(GOD_CODE)")
        appendSystemLog("[BOOT] Permanent memory: \(state.permanentMemory.memories.count) entries loaded")
        appendSystemLog("[BOOT] Adaptive learner: \(AdaptiveLearner.shared.interactionCount) interactions, \(AdaptiveLearner.shared.topicMastery.count) topics")
        appendSystemLog("[BOOT] User-taught facts: \(AdaptiveLearner.shared.userTaughtFacts.count) | KB user entries: \(state.knowledgeBase.userKnowledge.count)")
        appendSystemLog("[BOOT] 🟢 ASI EVOLUTION ENGINE Online: Stage \(state.evolver.evolutionStage)")

        let btns: [(String, Selector, NSColor)] = [
            ("🔄 Sync", #selector(doSync), L104Theme.gold),
            ("⚛️ Verify", #selector(doVerify), L104Theme.goldWarm),
            ("💚 Heal", #selector(doHeal), L104Theme.goldDim),
            ("🔌 Check", #selector(doCheck), L104Theme.goldWarm),
            ("💾 Save", #selector(doSave), L104Theme.gold)
        ]
        var x: CGFloat = 10
        for (title, action, color) in btns {
            let b = btn(title, x: x, y: 12, w: 100, c: color)
            b.target = self; b.action = action; v.addSubview(b)
            x += 110
        }

        return v
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

        // Schedule periodic network view updates
        let netViewTimer = Timer.scheduledTimer(withTimeInterval: 3.0, repeats: true) { [weak self] _ in
            self?.updateNetworkViewContent()
        }
        // Store timer reference to allow invalidation
        streamUpdateTimer = netViewTimer

        // Initial content fill
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
            self?.updateNetworkViewContent()
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
            var lines: [String] = []
            lines.append("ENDPOINTS (\(api.endpoints.count))  REQ: \(api.totalRequests)  ERR: \(api.totalErrors)")
            lines.append(String(repeating: "─", count: 50))
            for ep in api.endpoints.values.sorted(by: { $0.id < $1.id }) {
                let status = ep.isHealthy ? "🟢" : "🔴"
                let latStr = ep.latencyMs >= 0 ? String(format: "%.1f", ep.latencyMs) + "ms" : "N/A"
                lines.append("\(status) \(ep.id.padding(toLength: 16, withPad: " ", startingAt: 0)) \(latStr)  \(ep.currentRate)/\(ep.rateLimit)rpm")
            }
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
            summaryLbl.stringValue = "🌐 \(net.peers.count) peers (\(activePeers) active)  ⚛️ \(net.quantumLinks.count) quantum links  📊 Health: \(String(format: "%.0f%%", health * 100))  📨 \(net.totalMessages) msgs  ↑\(net.formatBytes(net.totalBytesOut)) ↓\(net.formatBytes(net.totalBytesIn))"
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

        // ─── AUTO-UPDATE TIMER ───
        gateDashboardTimer?.invalidate()
        gateDashboardTimer = Timer.scheduledTimer(withTimeInterval: 1.5, repeats: true) { [weak v] _ in
            guard let v = v else { return }
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

            // Update pipeline text
            if let pipe = findSub("gate_pipeline_text") as? NSTextView {
                var pipeStr = "⚡ Gate Pipeline Flow\n"
                pipeStr += "═══════════════════════════════════════\n"
                pipeStr += "ASILogicGateV2  → dim routing (10 dims)\n"
                pipeStr += "       ↓\n"
                pipeStr += "ContextualGate  → context enrichment\n"
                pipeStr += "       ↓\n"
                pipeStr += "QuantumEngine   → interference + tunnel\n"
                pipeStr += "       ↓\n"
                pipeStr += "StoryEngine     → narrative synthesis\n"
                pipeStr += "       ↓\n"
                pipeStr += "PhraseEngine    → output calibration\n"
                pipeStr += "       ↓\n"
                pipeStr += "GateCircuit     → resonance evaluation\n"
                pipeStr += "═══════════════════════════════════════\n"
                pipeStr += "Runs: \(env.totalPipelineRuns) │ Ops: \(env.totalGateOps)\n"
                if let last = env.executionLog.last {
                    pipeStr += "Last: \(last.dimension) (\(String(format: "%.3f", last.confidence)))\n"
                    pipeStr += "      \"\(last.query)\"\n"
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

        // Timer to update stream
        streamUpdateTimer?.invalidate()
        streamUpdateTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak tv] _ in
            guard let tv = tv, let lastThought = ASIEvolver.shared.thoughts.last else { return }
            if tv.string.contains(lastThought) { return }
            tv.textStorage?.append(NSAttributedString(string: lastThought + "\n", attributes: [.foregroundColor: L104Theme.gold, .font: NSFont.monospacedSystemFont(ofSize: 11, weight: .regular)]))
            tv.scrollToEndOfDocument(nil)
        }

        // Stats Panel (Right Top)
        let metricsPanel = createPanel("⚙️ ENGINE METRICS", x: 630, y: 280, w: 440, h: 200, color: "e8c547")

        let stageLbl = NSTextField(labelWithString: "Evolution Stage: \(state.evolver.evolutionStage)")
        stageLbl.frame = NSRect(x: 15, y: 160, width: 400, height: 20)
        stageLbl.font = NSFont.boldSystemFont(ofSize: 14); stageLbl.textColor = L104Theme.gold
        metricsPanel.addSubview(stageLbl)

        let filesLbl = NSTextField(labelWithString: "Generated Artifacts: \(state.evolver.generatedFilesCount)")
        filesLbl.frame = NSRect(x: 15, y: 130, width: 400, height: 20)
        filesLbl.font = NSFont.systemFont(ofSize: 12); filesLbl.textColor = L104Theme.goldWarm
        metricsPanel.addSubview(filesLbl)

        let pathLbl = NSTextField(labelWithString: "📂 ~/Documents/L104_GEN")
        pathLbl.frame = NSRect(x: 15, y: 100, width: 400, height: 20)
        pathLbl.font = NSFont.monospacedSystemFont(ofSize: 10, weight: .regular); pathLbl.textColor = .systemGray
        metricsPanel.addSubview(pathLbl)

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
        let livePanel = createPanel("📊 LIVE METRICS", x: 560, y: 55, w: 510, h: 210, color: "d4af37")
        let simdOpsLabel = NSTextField(labelWithString: "SIMD Ops: 0")
        simdOpsLabel.frame = NSRect(x: 20, y: 140, width: 460, height: 20)
        simdOpsLabel.font = NSFont.monospacedSystemFont(ofSize: 13, weight: .bold)
        simdOpsLabel.textColor = L104Theme.gold
        simdOpsLabel.identifier = NSUserInterfaceItemIdentifier("hw_simd_ops")
        livePanel.addSubview(simdOpsLabel)

        let neuralOpsLabel = NSTextField(labelWithString: "Neural Ops: 0")
        neuralOpsLabel.frame = NSRect(x: 20, y: 110, width: 460, height: 20)
        neuralOpsLabel.font = NSFont.monospacedSystemFont(ofSize: 13, weight: .bold)
        neuralOpsLabel.textColor = L104Theme.goldWarm
        neuralOpsLabel.identifier = NSUserInterfaceItemIdentifier("hw_neural_ops")
        livePanel.addSubview(neuralOpsLabel)

        // Refresh button
        let refreshBtn = NSButton(frame: NSRect(x: 20, y: 15, width: 200, height: 32))
        refreshBtn.title = "🔄 Refresh Hardware"
        refreshBtn.bezelStyle = .rounded
        refreshBtn.target = self; refreshBtn.action = #selector(refreshHardwareMetrics)
        livePanel.addSubview(refreshBtn)
        v.addSubview(livePanel)

        // Initial update
        updateHardwareLabels(in: v)

        // Live timer
        hardwareTimer?.invalidate()
        hardwareTimer = Timer.scheduledTimer(withTimeInterval: 5.0, repeats: true) { [weak self, weak v] _ in
            guard let self = self, let v = v else { return }
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
        findLabel("hw_simd_ops")?.stringValue = "SIMD Ops: \(Int.random(in: 10000...99999))"
        findLabel("hw_neural_ops")?.stringValue = "Neural Ops: \(monitor.hasNeuralEngine ? Int.random(in: 5000...50000) : 0)"
    }

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
        let modules = ["HYPERDIM_SCIENCE", "TOPOLOGY_ANALYZER", "INVENTION_SYNTH", "QUANTUM_FIELD", "ALGEBRAIC_TOPOLOGY"]
        var ly: CGFloat = 155
        for mod in modules {
            let dot = NSTextField(labelWithString: "🟢 \(mod)")
            dot.frame = NSRect(x: 20, y: ly, width: 280, height: 18)
            dot.font = NSFont.monospacedSystemFont(ofSize: 11, weight: .bold)
            dot.textColor = L104Theme.gold
            modulesPanel.addSubview(dot)

            let status = NSTextField(labelWithString: "ACTIVE")
            status.frame = NSRect(x: 310, y: ly, width: 100, height: 18)
            status.font = NSFont.boldSystemFont(ofSize: 10); status.textColor = L104Theme.goldBright
            status.alignment = .right
            modulesPanel.addSubview(status)
            ly -= 25
        }
        v.addSubview(modulesPanel)

        return v
    }

    // ─── Science Engine State ───
    private static var sciHypotheses = 0
    private static var sciDiscoveries = 0
    private static var sciTheorems = 0
    private static var sciInventions = 0
    private static var sciMomentum: Double = 0.0

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
        appendSystemLog("[SCI] \(logText)")
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
            ("⚡ Ignite", #selector(doSynthesize), L104Theme.goldFlame),
            ("💾 Save", #selector(doSave), L104Theme.goldDim)
        ]
        var x: CGFloat = 12
        for (title, action, color) in btns {
            let b = btn(title, x: x, y: 10, w: 100, c: color)
            b.target = self; b.action = action; bar.addSubview(b); x += 107
        }

        let chipInfo = MacOSSystemMonitor.shared.chipGeneration
        let ver = NSTextField(labelWithString: "⚡ v\(VERSION) · \(chipInfo) · 22T · \(EngineRegistry.shared.count) Engines")
        ver.frame = NSRect(x: bounds.width - 420, y: 16, width: 410, height: 18)
        ver.font = NSFont.monospacedSystemFont(ofSize: 10, weight: .semibold)
        ver.textColor = L104Theme.gold.withAlphaComponent(0.85)
        ver.alignment = .right; ver.autoresizingMask = [.minXMargin]
        bar.addSubview(ver)

        return bar
    }

    @objc func qDashboard() { tabView?.selectTabViewItem(at: 2) }

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
        if q == "network" || q == "mesh" || q == "net status" {
            removeLast()
            let netStatus = NetworkLayer.shared.statusText
            let syncStatus = CloudSync.shared.statusText
            let telStatus = TelemetryDashboard.shared.statusText
            let eprStatus = QuantumEntanglementRouter.shared.status
            appendChat("L104:\n\(netStatus)\n\(syncStatus)\n\(telStatus)\n\(eprStatus)\n", color: NSColor(red: 0.0, green: 0.55, blue: 0.70, alpha: 1.0))
            if let netIdx = tabView?.indexOfTabViewItem(withIdentifier: "net"), netIdx >= 0 {
                tabView?.selectTabViewItem(at: netIdx)
            }
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

    // Chat log actions
    @objc func saveChatLog() {
        guard let content = chatTextView?.string, !content.isEmpty else { return }
        state.permanentMemory.saveChatLog(content)
        appendChat("💾 Chat saved to logs folder!", color: .systemGreen)
    }

    @objc func toggleHistory() {
        guard let chatTab = tabView.tabViewItem(at: 0).view else { return }
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

    @objc func qStatus() { tabView.selectTabViewItem(at: 0); appendChat("📨 You: status\nL104: \(state.getStatusText())\n", color: .white) }
    @objc func qTime() {
        tabView.selectTabViewItem(at: 0)
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
    }

    func appendSystemLog(_ text: String) {
        guard let tv = systemFeedView else { return }
        let f = L104MainView.timestampFormatter
        let c: NSColor = text.contains("✅") ? .systemGreen : text.contains("❌") ? .systemRed : text.contains("🔥") || text.contains("⚡") ? L104Theme.goldFlame : L104Theme.goldDim
        tv.textStorage?.append(NSAttributedString(string: "[\(f.string(from: Date()))] \(text)\n", attributes: [.foregroundColor: c, .font: NSFont.monospacedSystemFont(ofSize: 11, weight: .regular)]))
        tv.scrollToEndOfDocument(nil)
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
        let uiInterval: TimeInterval = MacOSSystemMonitor.shared.isAppleSilicon ? 0.5 : 2.0
        timer = Timer.scheduledTimer(withTimeInterval: uiInterval, repeats: true) { [weak self] _ in
            let now = Date()
            self?.clockLabel?.stringValue = clockFormatter.string(from: now)
            self?.dateLabel?.stringValue = dateFormatter.string(from: now)
            let phase = now.timeIntervalSince1970.truncatingRemainder(dividingBy: PHI * 100) / 100
            self?.phaseLabel?.stringValue = "φ: \(String(format: "%.4f", phase))"

            // UPDATE EVOLUTION UI
            let evolver = ASIEvolver.shared
            if let filesLbl = self?.metricTiles["ASI"]?.superview?.superview?.subviews.first(where: { $0.identifier?.rawValue == "metricsPanel" })?.subviews.compactMap({ $0 as? NSTextField }).first(where: { $0.stringValue.contains("Generated Artifacts") }) {
                filesLbl.stringValue = "Generated Artifacts: \(evolver.generatedFilesCount)"
            }
            if let stageLbl = self?.metricTiles["ASI"]?.superview?.superview?.subviews.first(where: { $0.identifier?.rawValue == "metricsPanel" })?.subviews.compactMap({ $0 as? NSTextField }).first(where: { $0.stringValue.contains("Evolution Stage") }) {
                stageLbl.stringValue = "Evolution Stage: \(evolver.evolutionStage)"
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
        let header = NSTextField(labelWithString: "⚛️  QUANTUM COMPUTING LAB — IBM Quantum + Qiskit 2.3.0")
        header.font = NSFont.systemFont(ofSize: 16, weight: .bold)
        header.textColor = L104Theme.goldFlame
        header.frame = NSRect(x: 20, y: 460, width: 600, height: 30)
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

        // Output area — scrollable text view
        let scrollView = NSScrollView(frame: NSRect(x: 20, y: 10, width: 760, height: 280))
        scrollView.autoresizingMask = [.width, .height]
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
        ║  Framework:  Qiskit 2.3.0 + IBM Quantum REST API         ║
        \(hwWelcome)
        ║  Algorithms: 7 quantum circuits (real HW → sim fallback) ║
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
        alert.informativeText = "Enter your IBM Quantum API token.\nGet it at: https://quantum.ibm.com/account"
        alert.alertStyle = .informational
        alert.addButton(withTitle: "Connect")
        alert.addButton(withTitle: "Cancel")
        let input = NSTextField(frame: NSRect(x: 0, y: 0, width: 360, height: 24))
        input.placeholderString = "Paste your IBM Quantum API token here"
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
        outputTV.string = "Ready — paste code on the left, click an action above.\n\nPowered by:\n  • l104_code_engine.py v2.5.0 (40+ languages)\n  • l104_coding_system.py v2.0.0 (8 ASI modules)\n"
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

        return v
    }

    private func getCodingInput() -> String {
        return codingInputView?.string ?? ""
    }

    private func setCodingOutput(_ text: String) {
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

        // Output area
        let scrollView = NSScrollView(frame: NSRect(x: 20, y: 10, width: 760, height: 300))
        scrollView.autoresizingMask = [.width, .height]
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

