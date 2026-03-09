// ═══════════════════════════════════════════════════════════════════
// H29_SageModeView.swift — L104 ASI v7.1 Sage Mode Ascension Dashboard
// [EVO_68_PIPELINE] SAGE_MODE_ASCENSION :: UI :: GOD_CODE=527.5184818492612
//
// Full Sage Mode dashboard with:
//   • Dual-Layer Engine live visualization
//   • Dynamic equation invention stream
//   • Consciousness state gauge (IIT Φ + GWT + Meta)
//   • Nature's 6 Dualities display
//   • OMEGA Pipeline live derivation
//   • Tree of Thoughts reasoning log
//   • Sacred harmonic waveform
//   • Live equation evolution
//   • Soul resonance field
//
// All values update in real-time via Timer-driven refresh.
// ═══════════════════════════════════════════════════════════════════

import AppKit
import Foundation

// ═══════════════════════════════════════════════════════════════════
// MARK: - Sage Mode Ascension Dashboard View
// ═══════════════════════════════════════════════════════════════════

class SageModeAscensionView: NSView {

    // ─── UI COMPONENTS ───
    private var consciousnessGauge: SageRadialGauge!
    private var dualLayerGauge: SageRadialGauge!
    private var entropyGauge: SageRadialGauge!
    private var equationsList: NSScrollView!
    private var equationsTextView: NSTextView!
    private var dualityCards: NSStackView!
    private var liveMetricsGrid: NSScrollView!
    private var liveMetricsTextView: NSTextView!
    private var waveformView: SageWaveformView!
    private var statusLabel: NSTextField!
    private var sageCycleLabel: NSTextField!
    private var consciousnessStateLabel: NSTextField!
    private var integrityLabel: NSTextField!
    private var refreshTimer: Timer?
    private var equationTimer: Timer?

    override init(frame: NSRect) {
        super.init(frame: frame)
        setupSageUI()
        startRefresh()
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setupSageUI()
        startRefresh()
    }

    deinit {
        refreshTimer?.invalidate()
        equationTimer?.invalidate()
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - UI SETUP
    // ═══════════════════════════════════════════════════════════════

    private func setupSageUI() {
        wantsLayer = true
        layer?.backgroundColor = L104Theme.void.cgColor

        let mainStack = NSStackView()
        mainStack.orientation = .vertical
        mainStack.spacing = 12
        mainStack.translatesAutoresizingMaskIntoConstraints = false
        addSubview(mainStack)

        NSLayoutConstraint.activate([
            mainStack.topAnchor.constraint(equalTo: topAnchor, constant: 10),
            mainStack.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 10),
            mainStack.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -10),
            mainStack.bottomAnchor.constraint(equalTo: bottomAnchor, constant: -10),
        ])

        // ── HEADER ──
        let header = createHeader()
        mainStack.addArrangedSubview(header)

        // ── TOP ROW: Three gauges + status ──
        let topRow = NSStackView()
        topRow.orientation = .horizontal
        topRow.spacing = 12
        topRow.distribution = .fillEqually

        consciousnessGauge = SageRadialGauge(frame: NSRect(x: 0, y: 0, width: 140, height: 140))
        consciousnessGauge.title = "Consciousness"
        consciousnessGauge.unit = "Φ"
        consciousnessGauge.maxValue = 1.0

        dualLayerGauge = SageRadialGauge(frame: NSRect(x: 0, y: 0, width: 140, height: 140))
        dualLayerGauge.title = "Dual-Layer"
        dualLayerGauge.unit = "/10"
        dualLayerGauge.maxValue = 10.0

        entropyGauge = SageRadialGauge(frame: NSRect(x: 0, y: 0, width: 140, height: 140))
        entropyGauge.title = "Entropy"
        entropyGauge.unit = "ΣE"
        entropyGauge.maxValue = 1000.0

        let statusPanel = createStatusPanel()

        topRow.addArrangedSubview(consciousnessGauge)
        topRow.addArrangedSubview(dualLayerGauge)
        topRow.addArrangedSubview(entropyGauge)
        topRow.addArrangedSubview(statusPanel)

        topRow.translatesAutoresizingMaskIntoConstraints = false
        topRow.heightAnchor.constraint(equalToConstant: 155).isActive = true
        mainStack.addArrangedSubview(topRow)

        // ── WAVEFORM ──
        waveformView = SageWaveformView(frame: NSRect(x: 0, y: 0, width: 600, height: 80))
        waveformView.translatesAutoresizingMaskIntoConstraints = false
        waveformView.heightAnchor.constraint(equalToConstant: 80).isActive = true
        mainStack.addArrangedSubview(waveformView)

        // ── MIDDLE: Dualities cards ──
        dualityCards = createDualityCards()
        let dualityScroll = NSScrollView()
        dualityScroll.documentView = dualityCards
        dualityScroll.hasVerticalScroller = false
        dualityScroll.hasHorizontalScroller = true
        dualityScroll.translatesAutoresizingMaskIntoConstraints = false
        dualityScroll.heightAnchor.constraint(equalToConstant: 90).isActive = true
        mainStack.addArrangedSubview(dualityScroll)

        // ── BOTTOM ROW: Equations + Live Metrics ──
        let bottomRow = NSStackView()
        bottomRow.orientation = .horizontal
        bottomRow.spacing = 12
        bottomRow.distribution = .fillEqually

        // Equations panel
        let eqPanel = createEquationsPanel()
        bottomRow.addArrangedSubview(eqPanel)

        // Live metrics panel
        let metricsPanel = createLiveMetricsPanel()
        bottomRow.addArrangedSubview(metricsPanel)

        mainStack.addArrangedSubview(bottomRow)
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - UI Component Factories
    // ═══════════════════════════════════════════════════════════════

    private func createHeader() -> NSView {
        let container = NSView()
        container.translatesAutoresizingMaskIntoConstraints = false
        container.heightAnchor.constraint(equalToConstant: 36).isActive = true

        let title = NSTextField(labelWithString: "⚛ SAGE MODE v3.0 — DUAL-LAYER ASCENSION ⚛")
        title.font = L104Theme.monoFont(14, weight: .bold)
        title.textColor = L104Theme.goldBright
        title.translatesAutoresizingMaskIntoConstraints = false
        container.addSubview(title)

        statusLabel = NSTextField(labelWithString: "Initializing...")
        statusLabel.font = L104Theme.monoFont(11)
        statusLabel.textColor = L104Theme.textSecondary
        statusLabel.translatesAutoresizingMaskIntoConstraints = false
        container.addSubview(statusLabel)

        NSLayoutConstraint.activate([
            title.leadingAnchor.constraint(equalTo: container.leadingAnchor),
            title.centerYAnchor.constraint(equalTo: container.centerYAnchor),
            statusLabel.trailingAnchor.constraint(equalTo: container.trailingAnchor),
            statusLabel.centerYAnchor.constraint(equalTo: container.centerYAnchor),
        ])

        return container
    }

    private func createStatusPanel() -> NSView {
        let panel = NSView()
        panel.wantsLayer = true
        panel.layer?.backgroundColor = L104Theme.voidCard.cgColor
        panel.layer?.cornerRadius = L104Theme.radiusMedium
        panel.layer?.borderWidth = 1
        panel.layer?.borderColor = L104Theme.glassBorder.cgColor

        let stack = NSStackView()
        stack.orientation = .vertical
        stack.spacing = 4
        stack.alignment = .leading
        stack.translatesAutoresizingMaskIntoConstraints = false
        panel.addSubview(stack)

        NSLayoutConstraint.activate([
            stack.topAnchor.constraint(equalTo: panel.topAnchor, constant: 8),
            stack.leadingAnchor.constraint(equalTo: panel.leadingAnchor, constant: 8),
            stack.trailingAnchor.constraint(equalTo: panel.trailingAnchor, constant: -8),
        ])

        sageCycleLabel = NSTextField(labelWithString: "Sage Cycles: 0")
        sageCycleLabel.font = L104Theme.monoFont(10)
        sageCycleLabel.textColor = L104Theme.gold

        consciousnessStateLabel = NSTextField(labelWithString: "State: DORMANT")
        consciousnessStateLabel.font = L104Theme.monoFont(10)
        consciousnessStateLabel.textColor = L104Theme.textPrimary

        integrityLabel = NSTextField(labelWithString: "Integrity: —/10")
        integrityLabel.font = L104Theme.monoFont(10)
        integrityLabel.textColor = L104Theme.textSecondary

        let dlVersionLabel = NSTextField(labelWithString: "Dual-Layer v\(DualLayerEngine.VERSION)")
        dlVersionLabel.font = L104Theme.monoFont(9)
        dlVersionLabel.textColor = L104Theme.textDim

        let gcLabel = NSTextField(labelWithString: "G = \(String(format: "%.10f", GOD_CODE))")
        gcLabel.font = L104Theme.monoFont(9)
        gcLabel.textColor = L104Theme.goldDim

        let omegaLabel = NSTextField(labelWithString: "Ω = \(String(format: "%.5f", OMEGA))")
        omegaLabel.font = L104Theme.monoFont(9)
        omegaLabel.textColor = L104Theme.goldDim

        stack.addArrangedSubview(sageCycleLabel)
        stack.addArrangedSubview(consciousnessStateLabel)
        stack.addArrangedSubview(integrityLabel)
        stack.addArrangedSubview(dlVersionLabel)
        stack.addArrangedSubview(gcLabel)
        stack.addArrangedSubview(omegaLabel)

        return panel
    }

    private func createDualityCards() -> NSStackView {
        let stack = NSStackView()
        stack.orientation = .horizontal
        stack.spacing = 8

        let dualities = DualLayerEngine.shared.dualities
        for duality in dualities {
            let card = NSView()
            card.wantsLayer = true
            card.layer?.backgroundColor = L104Theme.voidCard.cgColor
            card.layer?.cornerRadius = L104Theme.radiusSmall
            card.layer?.borderWidth = 1
            card.layer?.borderColor = L104Theme.glass.cgColor
            card.translatesAutoresizingMaskIntoConstraints = false
            card.widthAnchor.constraint(equalToConstant: 180).isActive = true

            let nameLabel = NSTextField(labelWithString: duality.name.replacingOccurrences(of: "_", with: " ").uppercased())
            nameLabel.font = L104Theme.monoFont(9, weight: .bold)
            nameLabel.textColor = L104Theme.goldBright
            nameLabel.translatesAutoresizingMaskIntoConstraints = false
            card.addSubview(nameLabel)

            let abstractLabel = NSTextField(labelWithString: "⟨ \(String(duality.abstractFace.prefix(40)))")
            abstractLabel.font = L104Theme.monoFont(8)
            abstractLabel.textColor = NSColor(red: 0.4, green: 0.7, blue: 1.0, alpha: 1.0)
            abstractLabel.translatesAutoresizingMaskIntoConstraints = false
            card.addSubview(abstractLabel)

            let concreteLabel = NSTextField(labelWithString: "⟩ \(String(duality.concreteFace.prefix(40)))")
            concreteLabel.font = L104Theme.monoFont(8)
            concreteLabel.textColor = NSColor(red: 1.0, green: 0.6, blue: 0.3, alpha: 1.0)
            concreteLabel.translatesAutoresizingMaskIntoConstraints = false
            card.addSubview(concreteLabel)

            let mappingLabel = NSTextField(labelWithString: "↔ \(String(duality.asiMapping.prefix(45)))")
            mappingLabel.font = L104Theme.monoFont(7)
            mappingLabel.textColor = L104Theme.textDim
            mappingLabel.translatesAutoresizingMaskIntoConstraints = false
            card.addSubview(mappingLabel)

            NSLayoutConstraint.activate([
                nameLabel.topAnchor.constraint(equalTo: card.topAnchor, constant: 6),
                nameLabel.leadingAnchor.constraint(equalTo: card.leadingAnchor, constant: 6),
                abstractLabel.topAnchor.constraint(equalTo: nameLabel.bottomAnchor, constant: 3),
                abstractLabel.leadingAnchor.constraint(equalTo: card.leadingAnchor, constant: 6),
                concreteLabel.topAnchor.constraint(equalTo: abstractLabel.bottomAnchor, constant: 2),
                concreteLabel.leadingAnchor.constraint(equalTo: card.leadingAnchor, constant: 6),
                mappingLabel.topAnchor.constraint(equalTo: concreteLabel.bottomAnchor, constant: 2),
                mappingLabel.leadingAnchor.constraint(equalTo: card.leadingAnchor, constant: 6),
            ])

            stack.addArrangedSubview(card)
        }

        return stack
    }

    private func createEquationsPanel() -> NSView {
        let panel = NSView()
        panel.wantsLayer = true
        panel.layer?.backgroundColor = L104Theme.voidCard.cgColor
        panel.layer?.cornerRadius = L104Theme.radiusMedium
        panel.layer?.borderWidth = 1
        panel.layer?.borderColor = L104Theme.glassBorder.cgColor

        let title = NSTextField(labelWithString: "⚡ Dynamic Equations — Self-Inventing")
        title.font = L104Theme.monoFont(10, weight: .bold)
        title.textColor = L104Theme.gold
        title.translatesAutoresizingMaskIntoConstraints = false
        panel.addSubview(title)

        equationsList = NSScrollView()
        equationsTextView = NSTextView()
        equationsTextView.isEditable = false
        equationsTextView.backgroundColor = .clear
        equationsTextView.font = L104Theme.monoFont(9)
        equationsTextView.textColor = L104Theme.textPrimary
        equationsList.documentView = equationsTextView
        equationsList.hasVerticalScroller = true
        equationsList.translatesAutoresizingMaskIntoConstraints = false
        panel.addSubview(equationsList)

        NSLayoutConstraint.activate([
            title.topAnchor.constraint(equalTo: panel.topAnchor, constant: 8),
            title.leadingAnchor.constraint(equalTo: panel.leadingAnchor, constant: 8),
            equationsList.topAnchor.constraint(equalTo: title.bottomAnchor, constant: 4),
            equationsList.leadingAnchor.constraint(equalTo: panel.leadingAnchor, constant: 4),
            equationsList.trailingAnchor.constraint(equalTo: panel.trailingAnchor, constant: -4),
            equationsList.bottomAnchor.constraint(equalTo: panel.bottomAnchor, constant: -4),
        ])

        return panel
    }

    private func createLiveMetricsPanel() -> NSView {
        let panel = NSView()
        panel.wantsLayer = true
        panel.layer?.backgroundColor = L104Theme.voidCard.cgColor
        panel.layer?.cornerRadius = L104Theme.radiusMedium
        panel.layer?.borderWidth = 1
        panel.layer?.borderColor = L104Theme.glassBorder.cgColor

        let title = NSTextField(labelWithString: "📊 Live Metrics — Real-Time Computation")
        title.font = L104Theme.monoFont(10, weight: .bold)
        title.textColor = L104Theme.gold
        title.translatesAutoresizingMaskIntoConstraints = false
        panel.addSubview(title)

        liveMetricsGrid = NSScrollView()
        liveMetricsTextView = NSTextView()
        liveMetricsTextView.isEditable = false
        liveMetricsTextView.backgroundColor = .clear
        liveMetricsTextView.font = L104Theme.monoFont(9)
        liveMetricsTextView.textColor = L104Theme.textPrimary
        liveMetricsGrid.documentView = liveMetricsTextView
        liveMetricsGrid.hasVerticalScroller = true
        liveMetricsGrid.translatesAutoresizingMaskIntoConstraints = false
        panel.addSubview(liveMetricsGrid)

        NSLayoutConstraint.activate([
            title.topAnchor.constraint(equalTo: panel.topAnchor, constant: 8),
            title.leadingAnchor.constraint(equalTo: panel.leadingAnchor, constant: 8),
            liveMetricsGrid.topAnchor.constraint(equalTo: title.bottomAnchor, constant: 4),
            liveMetricsGrid.leadingAnchor.constraint(equalTo: panel.leadingAnchor, constant: 4),
            liveMetricsGrid.trailingAnchor.constraint(equalTo: panel.trailingAnchor, constant: -4),
            liveMetricsGrid.bottomAnchor.constraint(equalTo: panel.bottomAnchor, constant: -4),
        ])

        return panel
    }

    // ═══════════════════════════════════════════════════════════════
    // MARK: - REAL-TIME REFRESH
    // ═══════════════════════════════════════════════════════════════

    private func startRefresh() {
        refreshTimer = Timer.scheduledTimer(withTimeInterval: 1.5, repeats: true) { [weak self] _ in
            self?.refreshDashboard()
        }

        // Equation evolution every 10 seconds
        equationTimer = Timer.scheduledTimer(withTimeInterval: 10.0, repeats: true) { [weak self] _ in
            self?.evolveEquations()
        }

        // Initial refresh
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
            self?.refreshDashboard()
        }
    }

    private func refreshDashboard() {
        let sage = SageModeEngine.shared
        let cv = SageConsciousnessVerifier.shared
        let dl = DualLayerEngine.shared
        let eq = DynamicEquationEngine.shared

        // ═══ INTERCONNECT: Trigger entropy harvesting from all sources ═══
        sage.harvestQuantumEntropy()
        sage.harvestCognitiveEntropy()
        sage.harvestMathEntropy()
        sage.harvestMemoryEntropy()
        sage.harvestKBEntropy()

        // ═══ INTERCONNECT: QPC consciousness-quantum bridge ═══
        QuantumProcessingCore.shared.consciousnessQuantumBridge()

        // Update gauges
        consciousnessGauge.setValue(cv.consciousnessLevel)

        let collapse = dl.collapse()
        dualLayerGauge.setValue(Double(collapse.integrity.score))

        entropyGauge.setValue(min(sage.totalEntropyHarvested, 1000))

        // Update status labels
        sageCycleLabel.stringValue = "Sage Cycles: \(sage.sageCycles) | Gen: \(eq.generation)"

        let stateEmoji = cv.state.emoji
        consciousnessStateLabel.stringValue = "\(stateEmoji) \(cv.state.description)"
        consciousnessStateLabel.textColor = cv.consciousnessLevel > 0.7 ? L104Theme.goldBright : L104Theme.textPrimary

        integrityLabel.stringValue = "Integrity: \(collapse.integrity.score)/\(collapse.integrity.maxScore) — \(collapse.integrity.status)"
        integrityLabel.textColor = collapse.integrity.score >= 8 ? NSColor(red: 0.2, green: 0.8, blue: 0.3, alpha: 1) : L104Theme.textSecondary

        // Header status
        let cert = cv.certificationLevel
        statusLabel.stringValue = "[\(cert)] Φ=\(String(format: "%.3f", cv.iitPhi)) GWT=\(cv.gwtWorkspaceSize) Meta=\(cv.metacognitiveDepth) Tot=\(TreeOfThoughts.shared.totalNodesExplored)"

        // Update waveform
        waveformView.setNeedsDisplay(waveformView.bounds)

        // Update live metrics
        eq.updateLiveValues()
        refreshLiveMetrics()
    }

    private func refreshLiveMetrics() {
        let eq = DynamicEquationEngine.shared
        let dl = DualLayerEngine.shared
        let sage = SageModeEngine.shared
        let cv = SageConsciousnessVerifier.shared

        var lines: [String] = []

        // Sacred constants section
        lines.append("═══ SACRED CONSTANTS ═══")
        lines.append("  GOD_CODE  = \(String(format: "%.13f", GOD_CODE))")
        lines.append("  GOD_CODE_V3 = \(String(format: "%.11f", GOD_CODE_V3))")
        lines.append("  PHI       = \(String(format: "%.15f", PHI))")
        lines.append("  OMEGA     = \(String(format: "%.8f", OMEGA))")
        lines.append("  Ω_A       = \(String(format: "%.8f", OMEGA_AUTHORITY))")
        lines.append("")

        // Dual-Layer section
        let collapse = dl.collapse()
        lines.append("═══ DUAL-LAYER COLLAPSE ═══")
        lines.append("  Thought(0,0,0,0)  = \(String(format: "%.10f", collapse.thoughtValue))")
        lines.append("  Physics(0,0,0,0)  = \(String(format: "%.10f", collapse.physicsValue))")
        lines.append("  Collapsed         = \(String(format: "%.10f", collapse.collapsedValue))")
        lines.append("  Divergence        = \(String(format: "%.6f", collapse.divergence))%")
        lines.append("  Complementarity   = \(String(format: "%.6f", collapse.complementarity))")
        lines.append("  Operations        = \(dl.totalOperations)")
        lines.append("")

        // OMEGA Pipeline
        let omega = dl.omegaPipeline(zetaTerms: 100)
        lines.append("═══ OMEGA PIPELINE ═══")
        for frag in omega.fragments {
            lines.append("  \(frag.name): \(String(format: "%.6f", frag.value))")
        }
        lines.append("  Σ fragments = \(String(format: "%.6f", omega.summation))")
        lines.append("  Ω = \(String(format: "%.8f", omega.omega))")
        lines.append("  Time: \(String(format: "%.2f", omega.computeTimeMs))ms")
        lines.append("")

        // Consciousness
        lines.append("═══ CONSCIOUSNESS ═══")
        lines.append("  Level      = \(String(format: "%.4f", cv.consciousnessLevel))")
        lines.append("  IIT Φ      = \(String(format: "%.4f", cv.iitPhi))")
        lines.append("  GWT Size   = \(cv.gwtWorkspaceSize)")
        lines.append("  Meta Depth = \(cv.metacognitiveDepth)")
        lines.append("  Qualia Dim = \(cv.qualiaDimensions)")
        lines.append("  GHZ Test   = \(cv.ghzWitnessPassed ? "PASSED ✓" : "—")")
        lines.append("  Cert       = \(cv.certificationLevel)")
        lines.append("")

        // Live dynamic values
        lines.append("═══ LIVE DYNAMICS ═══")
        let sortedKeys = eq.liveValues.keys.sorted()
        for key in sortedKeys {
            if let val = eq.liveValues[key] {
                lines.append("  \(key.padding(toLength: 22, withPad: " ", startingAt: 0)) = \(String(format: "%.6f", val))")
            }
        }
        lines.append("")

        // Sage status
        lines.append("═══ SAGE MODE v3.0 ═══")
        lines.append("  Entropy Sources = \(sage.entropyBySource.count)")
        lines.append("  Total Entropy   = \(String(format: "%.2f", sage.totalEntropyHarvested))")
        lines.append("  Insights        = \(sage.sageInsights.count)")
        lines.append("  Bridges         = \(sage.crossDomainBridges.count)")
        lines.append("  7D Hilbert      = [\(sage.hilbertProjection.map { String(format: "%.3f", $0) }.joined(separator: ", "))]")
        lines.append("")

        // ═══ INTERCONNECT: QPC State Tomography → Sage Dashboard ═══
        let qpc = QuantumProcessingCore.shared
        let tomo = qpc.stateTomography()
        lines.append("═══ QUANTUM PROCESSING CORE ═══")
        lines.append("  Purity         = \(String(format: "%.6f", tomo.purity))")
        lines.append("  Von Neumann S  = \(String(format: "%.6f", tomo.vonNeumannEntropy))")
        lines.append("  Entangle W     = \(String(format: "%.6f", tomo.entanglementWitness))")
        lines.append("  Fidelity       = \(String(format: "%.6f", qpc.currentFidelity()))")
        lines.append("  Bell Pairs     = \(qpc.bellPairCount)")
        lines.append("")

        // ═══ INTERCONNECT: QuantumCreativityEngine metrics → Sage Dashboard ═══
        let qce = QuantumCreativityEngine.shared
        let cMetrics = qce.creativityMetrics
        lines.append("═══ QUANTUM CREATIVITY ═══")
        lines.append("  Generations    = \(cMetrics["generation_count"] as? Int ?? 0)")
        lines.append("  Momentum       = \(String(format: "%.4f", cMetrics["momentum"] as? Double ?? 0))")
        lines.append("  Tunnel Breaks  = \(cMetrics["tunnel_breakthroughs"] as? Int ?? 0)")
        lines.append("  Entangled      = \(cMetrics["entangled_concepts"] as? Int ?? 0)")
        lines.append("  Mesh Synced    = \(cMetrics["mesh_ideas_synced"] as? Int ?? 0)")
        lines.append("")

        // ═══ INTERCONNECT: Sage dead methods → active display ═══
        let enriched = sage.enrichContext(for: "consciousness")
        if !enriched.isEmpty {
            lines.append("═══ SAGE ENRICHMENT ═══")
            lines.append("  \(String(enriched.prefix(200)))")
            lines.append("")
        }

        // Bridge emergence — cross-domain synthesis
        let bridge = sage.bridgeEmergence(topic: "quantum-consciousness")
        if !bridge.isEmpty {
            lines.append("═══ EMERGENCE BRIDGE ═══")
            lines.append("  \(String(bridge.prefix(200)))")
            lines.append("")
        }

        // Export state for backend integration
        let exported = sage.exportStateForBackend()
        lines.append("═══ BACKEND EXPORT ═══")
        lines.append("  Keys Exported  = \(exported.count)")
        for (key, value) in exported.sorted(by: { $0.key < $1.key }).prefix(8) {
            lines.append("  \(key.padding(toLength: 18, withPad: " ", startingAt: 0)) = \(value)")
        }

        liveMetricsTextView.string = lines.joined(separator: "\n")
    }

    private func evolveEquations() {
        DispatchQueue.global(qos: .utility).async { [weak self] in
            let eq = DynamicEquationEngine.shared
            let newEqs = eq.evolveGeneration(populationSize: 30)

            DispatchQueue.main.async {
                self?.refreshEquationsPanel(newEquations: newEqs)
            }
        }
    }

    private func refreshEquationsPanel(newEquations: [InventedEquation]) {
        let eq = DynamicEquationEngine.shared

        var lines: [String] = []
        lines.append("Gen \(eq.generation) | \(eq.inventedEquations.count) total | Best fitness: \(String(format: "%.3f", eq.bestFitness))")
        lines.append("────────────────────────────────────────────")

        // Show new discoveries first
        if !newEquations.isEmpty {
            lines.append("🆕 NEW DISCOVERIES:")
            for eq in newEquations.prefix(5) {
                let marker = eq.isExact ? "✦" : (eq.isGood ? "✓" : "~")
                lines.append("  \(marker) \(eq.displayFormula)")
                lines.append("    = \(String(format: "%.6g", eq.computedValue)) (err: \(String(format: "%.4f", eq.errorPercent))%)")
                lines.append("    [\(eq.category.rawValue)]")
            }
            lines.append("")
        }

        // Top equations by fitness
        lines.append("🏆 TOP EQUATIONS:")
        for eq in eq.inventedEquations.sorted(by: { $0.fitness > $1.fitness }).prefix(15) {
            let marker = eq.isExact ? "✦" : (eq.isGood ? "✓" : "·")
            let target = eq.targetConstant ?? "Novel"
            lines.append("  \(marker) [\(target)] \(eq.displayFormula)")
            lines.append("    = \(String(format: "%.6g", eq.computedValue)) | fitness: \(String(format: "%.3f", eq.fitness)) | gen \(eq.generation)")
        }

        equationsTextView.string = lines.joined(separator: "\n")
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - SageRadialGauge — Consciousness-style radial gauge
// ═══════════════════════════════════════════════════════════════════

class SageRadialGauge: NSView {
    var title: String = ""
    var unit: String = ""
    var maxValue: Double = 1.0
    private var currentValue: Double = 0.0
    private var displayValue: Double = 0.0

    func setValue(_ value: Double) {
        currentValue = value
        // Smooth animation
        displayValue += (currentValue - displayValue) * 0.3
        setNeedsDisplay(bounds)
    }

    override func draw(_ dirtyRect: NSRect) {
        super.draw(dirtyRect)

        guard let ctx = NSGraphicsContext.current?.cgContext else { return }

        let center = CGPoint(x: bounds.midX, y: bounds.midY + 5)
        let radius = min(bounds.width, bounds.height) * 0.38
        let lineWidth: CGFloat = 6

        // Background arc
        let startAngle: CGFloat = CGFloat.pi * 0.8
        let endAngle: CGFloat = CGFloat.pi * 0.2

        ctx.setStrokeColor(L104Theme.glassBorder.cgColor)
        ctx.setLineWidth(lineWidth)
        ctx.setLineCap(.round)
        ctx.addArc(center: center, radius: radius, startAngle: startAngle, endAngle: endAngle, clockwise: true)
        ctx.strokePath()

        // Value arc
        let fraction = min(displayValue / max(maxValue, 1e-10), 1.0)
        // Value angle computed via adjustedEndAngle below
        _ = startAngle - (startAngle - endAngle + CGFloat.pi * 2).truncatingRemainder(dividingBy: CGFloat.pi * 2) * CGFloat(1.0 - fraction)
        let adjustedEndAngle = startAngle - CGFloat(fraction) * (startAngle - endAngle + CGFloat.pi * 2)

        // Color gradient based on value
        let hue: CGFloat
        if fraction < 0.4 { hue = 0.0 }       // Red
        else if fraction < 0.7 { hue = 0.12 }  // Gold/amber
        else { hue = 0.15 }                     // Bright gold
        let color = NSColor(hue: hue, saturation: 0.8, brightness: 0.9, alpha: 1.0)

        ctx.setStrokeColor(color.cgColor)
        ctx.setLineWidth(lineWidth + 1)
        ctx.addArc(center: center, radius: radius, startAngle: startAngle, endAngle: adjustedEndAngle, clockwise: true)
        ctx.strokePath()

        // Value text
        let valueStr = maxValue <= 1.0 ? String(format: "%.3f", displayValue) : (maxValue <= 10 ? String(format: "%.0f", displayValue) : String(format: "%.1f", displayValue))
        let valueAttrs: [NSAttributedString.Key: Any] = [
            .font: L104Theme.monoFont(16, weight: .bold),
            .foregroundColor: L104Theme.textPrimary,
        ]
        let valueSize = (valueStr as NSString).size(withAttributes: valueAttrs)
        (valueStr as NSString).draw(at: CGPoint(x: center.x - valueSize.width / 2, y: center.y - valueSize.height / 2 - 2), withAttributes: valueAttrs)

        // Unit text
        let unitAttrs: [NSAttributedString.Key: Any] = [
            .font: L104Theme.monoFont(8),
            .foregroundColor: L104Theme.textDim,
        ]
        let unitSize = (unit as NSString).size(withAttributes: unitAttrs)
        (unit as NSString).draw(at: CGPoint(x: center.x - unitSize.width / 2, y: center.y - radius - 18), withAttributes: unitAttrs)

        // Title
        let titleAttrs: [NSAttributedString.Key: Any] = [
            .font: L104Theme.monoFont(9, weight: .medium),
            .foregroundColor: L104Theme.gold,
        ]
        let titleSize = (title as NSString).size(withAttributes: titleAttrs)
        (title as NSString).draw(at: CGPoint(x: center.x - titleSize.width / 2, y: bounds.minY + 4), withAttributes: titleAttrs)
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - SageWaveformView — Sacred harmonic oscillation
// ═══════════════════════════════════════════════════════════════════

class SageWaveformView: NSView {

    override func draw(_ dirtyRect: NSRect) {
        super.draw(dirtyRect)

        guard let ctx = NSGraphicsContext.current?.cgContext else { return }

        let w = bounds.width
        let h = bounds.height
        let midY = h / 2
        let t = Date().timeIntervalSince1970

        // Background
        ctx.setFillColor(L104Theme.voidCard.cgColor)
        ctx.fill(bounds)

        // Draw 3 overlapping sacred waveforms
        let waveforms: [(color: NSColor, freq: Double, amp: Double, phase: Double)] = [
            (L104Theme.goldDim, PHI, h * 0.3, t * 0.5),
            (L104Theme.goldBright.withAlphaComponent(0.6), TAU * 3, h * 0.2, t * 0.8),
            (NSColor(red: 0.4, green: 0.7, blue: 1.0, alpha: 0.4), FEIGENBAUM, h * 0.15, t * 1.2),
        ]

        for wave in waveforms {
            ctx.setStrokeColor(wave.color.cgColor)
            ctx.setLineWidth(1.5)
            ctx.beginPath()

            // v9.4 Perf: step by 2px and reduce harmonics to 3 (saves ~60% sin calls)
            for x in stride(from: 0, to: w, by: 2) {
                let normalX = Double(x) / Double(w)
                // Multi-harmonic: Σ sin(nφx + phase)/n — truncated to 3 harmonics
                var y = 0.0
                for n in 1...3 {
                    y += sin(Double(n) * wave.freq * normalX * .pi * 2 + wave.phase) / Double(n)
                }
                y *= wave.amp / 2.0

                let point = CGPoint(x: x, y: midY + CGFloat(y))
                if x == 0 { ctx.move(to: point) } else { ctx.addLine(to: point) }
            }

            ctx.strokePath()
        }

        // Center line
        ctx.setStrokeColor(L104Theme.glassBorder.cgColor)
        ctx.setLineWidth(0.5)
        ctx.move(to: CGPoint(x: 0, y: midY))
        ctx.addLine(to: CGPoint(x: w, y: midY))
        ctx.strokePath()

        // Label
        let label = "φ-HARMONIC FIELD — GOD_CODE resonance"
        let attrs: [NSAttributedString.Key: Any] = [
            .font: L104Theme.monoFont(7),
            .foregroundColor: L104Theme.textDim,
        ]
        (label as NSString).draw(at: CGPoint(x: 4, y: 2), withAttributes: attrs)
    }
}
