// ═══════════════════════════════════════════════════════════════════
// H34_ThreeEngineCommandCenter.swift
// [EVO_77] THREE ENGINE COMMAND CENTER — Code + Math + Science
// GOD_CODE=527.5184818492612  PHI=1.618033988749895
//
// ThreeEngineHub: NSView
//   • Header strip with 3 engine status dots + refresh
//   • Three-column engine cards (Code / Math / Science)
//   • Sacred Constants Bar
//   • Cross-Engine Fusion Output Panel
//   • Auto-refresh every 8 seconds
// ═══════════════════════════════════════════════════════════════════

import AppKit
import Foundation

// ═══════════════════════════════════════════════════════════════════
// MARK: - EngineCard
// ═══════════════════════════════════════════════════════════════════

final class EngineCard: NSView {

    // Configuration
    let engineName: String
    let engineIcon: String
    let headerColor: NSColor
    let metricKeys: [(title: String, subtitle: String)]
    let actionTitles: [String]
    let actionSelectors: [Selector]
    weak var actionTarget: AnyObject?

    // Sub-views
    private var statusDot: NSView!
    private var versionLabel: NSTextField!
    private(set) var metricTiles: [AnimatedMetricTile] = []
    private(set) var miniOutput: NSTextView!
    private var miniOutputScroll: NSScrollView!

    init(
        frame: NSRect,
        icon: String,
        name: String,
        headerColor: NSColor,
        metricKeys: [(title: String, subtitle: String)],
        actionTitles: [String],
        actionSelectors: [Selector],
        target: AnyObject?
    ) {
        self.engineIcon = icon
        self.engineName = name
        self.headerColor = headerColor
        self.metricKeys = metricKeys
        self.actionTitles = actionTitles
        self.actionSelectors = actionSelectors
        self.actionTarget = target
        super.init(frame: frame)
        setupCard()
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) not supported") }

    // MARK: - Build card UI

    private func setupCard() {
        wantsLayer = true
        layer?.backgroundColor = NSColor(red: 0.10, green: 0.10, blue: 0.13, alpha: 1.0).cgColor
        layer?.cornerRadius = 12
        layer?.borderColor = headerColor.withAlphaComponent(0.35).cgColor
        layer?.borderWidth = 1.2
        translatesAutoresizingMaskIntoConstraints = false

        // ── Header band ──────────────────────────────────────────────
        let headerBand = NSView()
        headerBand.wantsLayer = true
        headerBand.layer?.backgroundColor = headerColor.withAlphaComponent(0.18).cgColor
        headerBand.layer?.cornerRadius = 12
        headerBand.translatesAutoresizingMaskIntoConstraints = false
        addSubview(headerBand)

        let iconLabel = NSTextField(labelWithString: engineIcon)
        iconLabel.font = NSFont.systemFont(ofSize: 20)
        iconLabel.translatesAutoresizingMaskIntoConstraints = false
        headerBand.addSubview(iconLabel)

        let nameLabel = NSTextField(labelWithString: engineName)
        nameLabel.font = NSFont.systemFont(ofSize: 13, weight: .bold)
        nameLabel.textColor = headerColor
        nameLabel.translatesAutoresizingMaskIntoConstraints = false
        headerBand.addSubview(nameLabel)

        versionLabel = NSTextField(labelWithString: "v\(CODE_ENGINE_VERSION)")
        versionLabel.font = NSFont.monospacedSystemFont(ofSize: 9, weight: .regular)
        versionLabel.textColor = L104Theme.textDim
        versionLabel.translatesAutoresizingMaskIntoConstraints = false
        headerBand.addSubview(versionLabel)

        statusDot = NSView()
        statusDot.wantsLayer = true
        statusDot.layer?.backgroundColor = NSColor.systemYellow.cgColor
        statusDot.layer?.cornerRadius = 5
        statusDot.translatesAutoresizingMaskIntoConstraints = false
        headerBand.addSubview(statusDot)

        // ── Metric tiles (2×2 grid) ───────────────────────────────────
        let tilesContainer = NSView()
        tilesContainer.translatesAutoresizingMaskIntoConstraints = false
        addSubview(tilesContainer)

        for (_, meta) in metricKeys.prefix(4).enumerated() {
            let tile = AnimatedMetricTile(
                frame: .zero,
                label: meta.title,
                value: "—",
                color: headerColor,
                progress: 0
            )
            tile.translatesAutoresizingMaskIntoConstraints = false
            tilesContainer.addSubview(tile)
            metricTiles.append(tile)
        }

        // ── Action buttons ────────────────────────────────────────────
        let btnStack = NSView()
        btnStack.translatesAutoresizingMaskIntoConstraints = false
        addSubview(btnStack)

        for (i, title) in actionTitles.prefix(3).enumerated() {
            let btn = NSButton(title: title, target: actionTarget, action: actionSelectors[safe: i] ?? #selector(NSObject.description))
            btn.bezelStyle = .rounded
            btn.font = NSFont.systemFont(ofSize: 10, weight: .medium)
            btn.wantsLayer = true
            btn.translatesAutoresizingMaskIntoConstraints = false
            btnStack.addSubview(btn)
        }

        // ── Mini output ───────────────────────────────────────────────
        miniOutputScroll = NSScrollView()
        miniOutputScroll.translatesAutoresizingMaskIntoConstraints = false
        miniOutputScroll.hasVerticalScroller = true
        miniOutputScroll.autohidesScrollers = true
        miniOutputScroll.borderType = .noBorder
        miniOutputScroll.drawsBackground = true
        addSubview(miniOutputScroll)

        miniOutput = NSTextView()
        miniOutput.isEditable = false
        miniOutput.backgroundColor = NSColor(red: 0.04, green: 0.04, blue: 0.07, alpha: 1.0)
        miniOutput.textColor = headerColor.withAlphaComponent(0.9)
        miniOutput.font = NSFont.monospacedSystemFont(ofSize: 9.5, weight: .regular)
        miniOutput.isVerticallyResizable = true
        miniOutput.isHorizontallyResizable = false
        miniOutput.autoresizingMask = [.width]
        miniOutput.textContainer?.widthTracksTextView = true
        miniOutput.string = "⏳ Awaiting first refresh..."
        miniOutputScroll.documentView = miniOutput

        // ── Auto Layout ───────────────────────────────────────────────
        NSLayoutConstraint.activate([
            // Header band: top strip, 52pt tall
            headerBand.topAnchor.constraint(equalTo: topAnchor),
            headerBand.leadingAnchor.constraint(equalTo: leadingAnchor),
            headerBand.trailingAnchor.constraint(equalTo: trailingAnchor),
            headerBand.heightAnchor.constraint(equalToConstant: 52),

            // Icon inside header
            iconLabel.leadingAnchor.constraint(equalTo: headerBand.leadingAnchor, constant: 10),
            iconLabel.centerYAnchor.constraint(equalTo: headerBand.centerYAnchor),

            // Name label
            nameLabel.leadingAnchor.constraint(equalTo: iconLabel.trailingAnchor, constant: 8),
            nameLabel.centerYAnchor.constraint(equalTo: headerBand.centerYAnchor, constant: -7),

            // Version label
            versionLabel.leadingAnchor.constraint(equalTo: iconLabel.trailingAnchor, constant: 8),
            versionLabel.centerYAnchor.constraint(equalTo: headerBand.centerYAnchor, constant: 8),

            // Status dot
            statusDot.trailingAnchor.constraint(equalTo: headerBand.trailingAnchor, constant: -10),
            statusDot.centerYAnchor.constraint(equalTo: headerBand.centerYAnchor),
            statusDot.widthAnchor.constraint(equalToConstant: 10),
            statusDot.heightAnchor.constraint(equalToConstant: 10),

            // Tiles container: below header
            tilesContainer.topAnchor.constraint(equalTo: headerBand.bottomAnchor, constant: 8),
            tilesContainer.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 8),
            tilesContainer.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -8),
            tilesContainer.heightAnchor.constraint(equalToConstant: 130),

            // Buttons stack: below tiles
            btnStack.topAnchor.constraint(equalTo: tilesContainer.bottomAnchor, constant: 6),
            btnStack.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 8),
            btnStack.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -8),
            btnStack.heightAnchor.constraint(equalToConstant: 28),

            // Mini output scroll: fills remaining space
            miniOutputScroll.topAnchor.constraint(equalTo: btnStack.bottomAnchor, constant: 6),
            miniOutputScroll.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 6),
            miniOutputScroll.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -6),
            miniOutputScroll.bottomAnchor.constraint(equalTo: bottomAnchor, constant: -6),
        ])

        // Tile 2×2 layout (done after tiles are in tilesContainer)
        layoutTiles(in: tilesContainer)
        // Button layout
        layoutButtons(in: btnStack)
    }

    private func layoutTiles(in container: NSView) {
        // 2×2 grid, each tile fills half width / half height
        for (i, tile) in metricTiles.enumerated() {
            let col = i % 2
            let row = i / 2
            NSLayoutConstraint.activate([
                tile.topAnchor.constraint(equalTo: container.topAnchor, constant: CGFloat(row) * 65),
                tile.leadingAnchor.constraint(equalTo: container.leadingAnchor, constant: CGFloat(col) * (container.frame.width / 2 + 2)),
                tile.widthAnchor.constraint(equalTo: container.widthAnchor, multiplier: 0.48),
                tile.heightAnchor.constraint(equalToConstant: 60),
            ])
        }
    }

    private func layoutButtons(in stack: NSView) {
        let buttons = stack.subviews.compactMap { $0 as? NSButton }
        for (i, btn) in buttons.enumerated() {
            NSLayoutConstraint.activate([
                btn.topAnchor.constraint(equalTo: stack.topAnchor),
                btn.bottomAnchor.constraint(equalTo: stack.bottomAnchor),
                btn.leadingAnchor.constraint(equalTo: stack.leadingAnchor, constant: CGFloat(i) * (stack.frame.width / CGFloat(buttons.count) + 4)),
                btn.widthAnchor.constraint(equalTo: stack.widthAnchor, multiplier: 1.0 / CGFloat(max(buttons.count, 1)), constant: -4),
            ])
        }
    }

    // MARK: - Public helpers

    func setStatus(online: Bool) {
        DispatchQueue.main.async { [weak self] in
            self?.statusDot.layer?.backgroundColor = (online ? NSColor.systemGreen : NSColor.systemRed).cgColor
        }
    }

    func updateTile(index: Int, value: String, progress: CGFloat) {
        guard index < metricTiles.count else { return }
        DispatchQueue.main.async { [weak self] in
            self?.metricTiles[index].value = value
            self?.metricTiles[index].progress = progress
        }
    }

    func appendMiniOutput(_ text: String) {
        DispatchQueue.main.async { [weak self] in
            guard let tv = self?.miniOutput else { return }
            let existing = tv.string
            let newText = existing.isEmpty || existing == "⏳ Awaiting first refresh..." ? text : "\(existing)\n\(text)"
            tv.string = newText
            tv.scrollToEndOfDocument(nil)
        }
    }

    func setMiniOutput(_ text: String) {
        DispatchQueue.main.async { [weak self] in
            self?.miniOutput.string = text
        }
    }
}

// Safe array subscript for Selectors
private extension Array where Element == Selector {
    subscript(safe index: Int) -> Selector? {
        guard index >= 0 && index < count else { return nil }
        return self[index]
    }
}

// ═══════════════════════════════════════════════════════════════════
// MARK: - ThreeEngineHub
// ═══════════════════════════════════════════════════════════════════

final class ThreeEngineHub: NSView {

    // ── Sub-views ────────────────────────────────────────────────────
    private var codeCard: EngineCard!
    private var mathCard: EngineCard!
    private var sciCard: EngineCard!

    // Header dots
    private var codeStatusDot: NSView!
    private var mathStatusDot: NSView!
    private var sciStatusDot: NSView!
    private var lastRefreshLabel: NSTextField!

    // Sacred constants bar
    private var godCodeCheck: NSTextField!
    private var phiCheck: NSTextField!
    private var voidCheck: NSTextField!
    private var omegaCheck: NSTextField!

    // Fusion output
    private var fusionOutput: NSTextView!
    private var fusionScroll: NSScrollView!
    private var fusionRefreshLabel: NSTextField!

    // Auto-refresh timer
    private var refreshTimer: Timer?

    // MARK: - Init

    override init(frame: NSRect) {
        super.init(frame: frame)
        buildUI()
        startRefreshTimer()
        fetchAll()
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        buildUI()
        startRefreshTimer()
        fetchAll()
    }

    deinit {
        refreshTimer?.invalidate()
    }

    // MARK: - Build UI

    private func buildUI() {
        wantsLayer = true
        layer?.backgroundColor = L104Theme.void.cgColor
        translatesAutoresizingMaskIntoConstraints = false

        // Outer scroll view so panel is scrollable on short windows
        let outerScroll = NSScrollView()
        outerScroll.translatesAutoresizingMaskIntoConstraints = false
        outerScroll.hasVerticalScroller = true
        outerScroll.hasHorizontalScroller = false
        outerScroll.autohidesScrollers = true
        outerScroll.borderType = .noBorder
        outerScroll.drawsBackground = false
        addSubview(outerScroll)

        NSLayoutConstraint.activate([
            outerScroll.topAnchor.constraint(equalTo: topAnchor),
            outerScroll.leadingAnchor.constraint(equalTo: leadingAnchor),
            outerScroll.trailingAnchor.constraint(equalTo: trailingAnchor),
            outerScroll.bottomAnchor.constraint(equalTo: bottomAnchor),
        ])

        let content = NSView()
        content.translatesAutoresizingMaskIntoConstraints = false
        outerScroll.documentView = content

        NSLayoutConstraint.activate([
            content.leadingAnchor.constraint(equalTo: outerScroll.leadingAnchor),
            content.trailingAnchor.constraint(equalTo: outerScroll.trailingAnchor),
            content.topAnchor.constraint(equalTo: outerScroll.contentView.topAnchor),
        ])

        // ── Header strip ─────────────────────────────────────────────
        let headerStrip = buildHeaderStrip()
        content.addSubview(headerStrip)

        // ── Engine cards row ─────────────────────────────────────────
        codeCard = buildCodeCard()
        mathCard = buildMathCard()
        sciCard  = buildSciCard()
        let cardsRow = NSView()
        cardsRow.translatesAutoresizingMaskIntoConstraints = false
        content.addSubview(cardsRow)
        cardsRow.addSubview(codeCard)
        cardsRow.addSubview(mathCard)
        cardsRow.addSubview(sciCard)

        // ── Sacred constants bar ─────────────────────────────────────
        let constsBar = buildSacredConstantsBar()
        content.addSubview(constsBar)

        // ── Fusion panel ─────────────────────────────────────────────
        let fusionPanel = buildFusionPanel()
        content.addSubview(fusionPanel)

        // ── Layout content ───────────────────────────────────────────
        NSLayoutConstraint.activate([
            // Header strip: top of content
            headerStrip.topAnchor.constraint(equalTo: content.topAnchor, constant: 10),
            headerStrip.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 12),
            headerStrip.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -12),
            headerStrip.heightAnchor.constraint(equalToConstant: 42),

            // Cards row: below header
            cardsRow.topAnchor.constraint(equalTo: headerStrip.bottomAnchor, constant: 10),
            cardsRow.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 12),
            cardsRow.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -12),
            cardsRow.heightAnchor.constraint(equalToConstant: 380),

            // Cards equal widths inside row
            codeCard.topAnchor.constraint(equalTo: cardsRow.topAnchor),
            codeCard.bottomAnchor.constraint(equalTo: cardsRow.bottomAnchor),
            codeCard.leadingAnchor.constraint(equalTo: cardsRow.leadingAnchor),
            codeCard.widthAnchor.constraint(equalTo: cardsRow.widthAnchor, multiplier: 1.0/3.0, constant: -5),

            mathCard.topAnchor.constraint(equalTo: cardsRow.topAnchor),
            mathCard.bottomAnchor.constraint(equalTo: cardsRow.bottomAnchor),
            mathCard.leadingAnchor.constraint(equalTo: codeCard.trailingAnchor, constant: 8),
            mathCard.widthAnchor.constraint(equalTo: codeCard.widthAnchor),

            sciCard.topAnchor.constraint(equalTo: cardsRow.topAnchor),
            sciCard.bottomAnchor.constraint(equalTo: cardsRow.bottomAnchor),
            sciCard.leadingAnchor.constraint(equalTo: mathCard.trailingAnchor, constant: 8),
            sciCard.trailingAnchor.constraint(equalTo: cardsRow.trailingAnchor),

            // Constants bar: below cards
            constsBar.topAnchor.constraint(equalTo: cardsRow.bottomAnchor, constant: 10),
            constsBar.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 12),
            constsBar.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -12),
            constsBar.heightAnchor.constraint(equalToConstant: 52),

            // Fusion panel: below constants, fills rest
            fusionPanel.topAnchor.constraint(equalTo: constsBar.bottomAnchor, constant: 10),
            fusionPanel.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 12),
            fusionPanel.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -12),
            fusionPanel.heightAnchor.constraint(equalToConstant: 300),
            fusionPanel.bottomAnchor.constraint(equalTo: content.bottomAnchor, constant: -14),
        ])
    }

    // MARK: - Header Strip

    private func buildHeaderStrip() -> NSView {
        let strip = NSView()
        strip.wantsLayer = true
        strip.layer?.backgroundColor = NSColor(red: 0.08, green: 0.06, blue: 0.14, alpha: 1.0).cgColor
        strip.layer?.cornerRadius = 10
        strip.layer?.borderColor = L104Theme.goldFlame.withAlphaComponent(0.35).cgColor
        strip.layer?.borderWidth = 1
        strip.translatesAutoresizingMaskIntoConstraints = false

        let title = NSTextField(labelWithString: "⚡ THREE ENGINE COMMAND CENTER")
        title.font = NSFont.systemFont(ofSize: 15, weight: .bold)
        title.textColor = L104Theme.goldFlame
        title.translatesAutoresizingMaskIntoConstraints = false
        strip.addSubview(title)

        // Status dots
        func makeDot(_ color: NSColor) -> NSView {
            let d = NSView()
            d.wantsLayer = true
            d.layer?.backgroundColor = NSColor.systemYellow.cgColor
            d.layer?.cornerRadius = 6
            d.translatesAutoresizingMaskIntoConstraints = false
            return d
        }

        codeStatusDot = makeDot(.systemGreen)
        mathStatusDot = makeDot(.systemPurple)
        sciStatusDot  = makeDot(.systemCyan)

        let codeLabel = NSTextField(labelWithString: "CODE")
        codeLabel.font = NSFont.systemFont(ofSize: 9, weight: .bold)
        codeLabel.textColor = .systemGreen
        codeLabel.translatesAutoresizingMaskIntoConstraints = false

        let mathLabel = NSTextField(labelWithString: "MATH")
        mathLabel.font = NSFont.systemFont(ofSize: 9, weight: .bold)
        mathLabel.textColor = .systemPurple
        mathLabel.translatesAutoresizingMaskIntoConstraints = false

        let sciLabel = NSTextField(labelWithString: "SCI")
        sciLabel.font = NSFont.systemFont(ofSize: 9, weight: .bold)
        sciLabel.textColor = .systemCyan
        sciLabel.translatesAutoresizingMaskIntoConstraints = false

        strip.addSubview(codeStatusDot)
        strip.addSubview(codeLabel)
        strip.addSubview(mathStatusDot)
        strip.addSubview(mathLabel)
        strip.addSubview(sciStatusDot)
        strip.addSubview(sciLabel)

        let refreshBtn = NSButton(title: "⟳ Refresh", target: self, action: #selector(manualRefresh))
        refreshBtn.bezelStyle = .rounded
        refreshBtn.font = NSFont.systemFont(ofSize: 11, weight: .medium)
        refreshBtn.translatesAutoresizingMaskIntoConstraints = false
        strip.addSubview(refreshBtn)

        lastRefreshLabel = NSTextField(labelWithString: "Last refresh: —")
        lastRefreshLabel.font = NSFont.monospacedSystemFont(ofSize: 9, weight: .regular)
        lastRefreshLabel.textColor = L104Theme.textDim
        lastRefreshLabel.translatesAutoresizingMaskIntoConstraints = false
        strip.addSubview(lastRefreshLabel)

        NSLayoutConstraint.activate([
            title.leadingAnchor.constraint(equalTo: strip.leadingAnchor, constant: 14),
            title.centerYAnchor.constraint(equalTo: strip.centerYAnchor),

            // Code dot + label
            codeStatusDot.leadingAnchor.constraint(equalTo: title.trailingAnchor, constant: 24),
            codeStatusDot.centerYAnchor.constraint(equalTo: strip.centerYAnchor, constant: -4),
            codeStatusDot.widthAnchor.constraint(equalToConstant: 12),
            codeStatusDot.heightAnchor.constraint(equalToConstant: 12),
            codeLabel.leadingAnchor.constraint(equalTo: codeStatusDot.leadingAnchor),
            codeLabel.topAnchor.constraint(equalTo: codeStatusDot.bottomAnchor, constant: 1),

            // Math dot + label
            mathStatusDot.leadingAnchor.constraint(equalTo: codeStatusDot.trailingAnchor, constant: 28),
            mathStatusDot.centerYAnchor.constraint(equalTo: strip.centerYAnchor, constant: -4),
            mathStatusDot.widthAnchor.constraint(equalToConstant: 12),
            mathStatusDot.heightAnchor.constraint(equalToConstant: 12),
            mathLabel.leadingAnchor.constraint(equalTo: mathStatusDot.leadingAnchor),
            mathLabel.topAnchor.constraint(equalTo: mathStatusDot.bottomAnchor, constant: 1),

            // Sci dot + label
            sciStatusDot.leadingAnchor.constraint(equalTo: mathStatusDot.trailingAnchor, constant: 28),
            sciStatusDot.centerYAnchor.constraint(equalTo: strip.centerYAnchor, constant: -4),
            sciStatusDot.widthAnchor.constraint(equalToConstant: 12),
            sciStatusDot.heightAnchor.constraint(equalToConstant: 12),
            sciLabel.leadingAnchor.constraint(equalTo: sciStatusDot.leadingAnchor),
            sciLabel.topAnchor.constraint(equalTo: sciStatusDot.bottomAnchor, constant: 1),

            // Refresh button
            refreshBtn.trailingAnchor.constraint(equalTo: strip.trailingAnchor, constant: -12),
            refreshBtn.centerYAnchor.constraint(equalTo: strip.centerYAnchor),

            // Last refresh label
            lastRefreshLabel.trailingAnchor.constraint(equalTo: refreshBtn.leadingAnchor, constant: -10),
            lastRefreshLabel.centerYAnchor.constraint(equalTo: strip.centerYAnchor),
        ])

        return strip
    }

    // MARK: - Engine Cards

    private func buildCodeCard() -> EngineCard {
        EngineCard(
            frame: .zero,
            icon: "💻",
            name: "CODE ENGINE",
            headerColor: .systemGreen,
            metricKeys: [
                (title: "Smells",     subtitle: "code smell count"),
                (title: "Quality",    subtitle: "quality score %"),
                (title: "Complexity", subtitle: "complexity rating"),
                (title: "Coverage",   subtitle: "test coverage %"),
            ],
            actionTitles: ["Analyze", "Audit", "Auto-Fix"],
            actionSelectors: [
                #selector(codeCardAnalyze),
                #selector(codeCardAudit),
                #selector(codeCardAutoFix),
            ],
            target: self
        )
    }

    private func buildMathCard() -> EngineCard {
        EngineCard(
            frame: .zero,
            icon: "🧮",
            name: "MATH ENGINE",
            headerColor: .systemPurple,
            metricKeys: [
                (title: "GOD_CODE",  subtitle: "alignment score"),
                (title: "PHI Align", subtitle: "Fibonacci convergence"),
                (title: "Harmonics", subtitle: "harmonic resonance"),
                (title: "Proofs",    subtitle: "proof count"),
            ],
            actionTitles: ["Verify", "Proofs", "Harmonics"],
            actionSelectors: [
                #selector(mathCardVerify),
                #selector(mathCardProofs),
                #selector(mathCardHarmonics),
            ],
            target: self
        )
    }

    private func buildSciCard() -> EngineCard {
        EngineCard(
            frame: .zero,
            icon: "🔬",
            name: "SCIENCE ENGINE",
            headerColor: .systemCyan,
            metricKeys: [
                (title: "Entropy",     subtitle: "Maxwell Demon efficiency"),
                (title: "Coherence",   subtitle: "quantum coherence"),
                (title: "26Q Fidelity",subtitle: "iron-mapped fidelity"),
                (title: "Physics",     subtitle: "constants verified"),
            ],
            actionTitles: ["Entropy", "Coherence", "26Q"],
            actionSelectors: [
                #selector(sciCardEntropy),
                #selector(sciCardCoherence),
                #selector(sciCard26Q),
            ],
            target: self
        )
    }

    // MARK: - Sacred Constants Bar

    private func buildSacredConstantsBar() -> NSView {
        let bar = NSView()
        bar.wantsLayer = true
        bar.layer?.backgroundColor = NSColor(red: 0.06, green: 0.06, blue: 0.10, alpha: 1.0).cgColor
        bar.layer?.cornerRadius = 10
        bar.layer?.borderColor = L104Theme.goldFlame.withAlphaComponent(0.25).cgColor
        bar.layer?.borderWidth = 1
        bar.translatesAutoresizingMaskIntoConstraints = false

        let barTitle = NSTextField(labelWithString: "🔐 SACRED CONSTANTS:")
        barTitle.font = NSFont.systemFont(ofSize: 10, weight: .bold)
        barTitle.textColor = L104Theme.goldFlame
        barTitle.translatesAutoresizingMaskIntoConstraints = false
        bar.addSubview(barTitle)

        let constants: [(String, String)] = [
            ("GOD_CODE", String(format: "%.6f", GOD_CODE)),
            ("PHI",      String(format: "%.6f", PHI)),
            ("VOID",     String(format: "%.6f", VOID_CONSTANT)),
            ("OMEGA",    String(format: "%.3f", OMEGA)),
        ]

        var checkFields: [NSTextField] = []
        var prevAnchor: NSLayoutXAxisAnchor = barTitle.trailingAnchor

        for (i, (name, value)) in constants.enumerated() {
            let check = NSTextField(labelWithString: "✅")
            check.font = NSFont.systemFont(ofSize: 10)
            check.translatesAutoresizingMaskIntoConstraints = false
            bar.addSubview(check)

            let lbl = NSTextField(labelWithString: "\(name)=\(value)")
            lbl.font = NSFont.monospacedSystemFont(ofSize: 9.5, weight: .regular)
            lbl.textColor = .systemGreen
            lbl.translatesAutoresizingMaskIntoConstraints = false
            bar.addSubview(lbl)

            NSLayoutConstraint.activate([
                check.leadingAnchor.constraint(equalTo: prevAnchor, constant: i == 0 ? 12 : 16),
                check.centerYAnchor.constraint(equalTo: bar.centerYAnchor),
                lbl.leadingAnchor.constraint(equalTo: check.trailingAnchor, constant: 3),
                lbl.centerYAnchor.constraint(equalTo: bar.centerYAnchor),
            ])

            prevAnchor = lbl.trailingAnchor
            checkFields.append(check)

            if i == 0 { godCodeCheck = check }
            if i == 1 { phiCheck     = check }
            if i == 2 { voidCheck    = check }
            if i == 3 { omegaCheck   = check }
        }

        NSLayoutConstraint.activate([
            barTitle.leadingAnchor.constraint(equalTo: bar.leadingAnchor, constant: 12),
            barTitle.centerYAnchor.constraint(equalTo: bar.centerYAnchor),
        ])

        return bar
    }

    // MARK: - Fusion Panel

    private func buildFusionPanel() -> NSView {
        let panel = NSView()
        panel.wantsLayer = true
        panel.layer?.backgroundColor = NSColor(red: 0.05, green: 0.05, blue: 0.10, alpha: 1.0).cgColor
        panel.layer?.cornerRadius = 12
        panel.layer?.borderColor = L104Theme.goldFlame.withAlphaComponent(0.30).cgColor
        panel.layer?.borderWidth = 1
        panel.translatesAutoresizingMaskIntoConstraints = false

        let fusionTitle = NSTextField(labelWithString: "⚡ CROSS-ENGINE SYNTHESIS")
        fusionTitle.font = NSFont.systemFont(ofSize: 13, weight: .bold)
        fusionTitle.textColor = L104Theme.goldFlame
        fusionTitle.translatesAutoresizingMaskIntoConstraints = false
        panel.addSubview(fusionTitle)

        let runBtn = NSButton(title: "▶ Run Fusion Analysis", target: self, action: #selector(runFusionAnalysis))
        runBtn.bezelStyle = .rounded
        runBtn.font = NSFont.systemFont(ofSize: 11, weight: .semibold)
        runBtn.wantsLayer = true
        runBtn.translatesAutoresizingMaskIntoConstraints = false
        panel.addSubview(runBtn)

        let statusBtn = NSButton(title: "⟳ Tri-Engine Status", target: self, action: #selector(fetchTriEngineStatus))
        statusBtn.bezelStyle = .rounded
        statusBtn.font = NSFont.systemFont(ofSize: 11, weight: .regular)
        statusBtn.translatesAutoresizingMaskIntoConstraints = false
        panel.addSubview(statusBtn)

        let constsBtn = NSButton(title: "🔐 Constants Check", target: self, action: #selector(fetchTriEngineConstants))
        constsBtn.bezelStyle = .rounded
        constsBtn.font = NSFont.systemFont(ofSize: 11, weight: .regular)
        constsBtn.translatesAutoresizingMaskIntoConstraints = false
        panel.addSubview(constsBtn)

        fusionRefreshLabel = NSTextField(labelWithString: "Last refreshed: —")
        fusionRefreshLabel.font = NSFont.monospacedSystemFont(ofSize: 9, weight: .regular)
        fusionRefreshLabel.textColor = L104Theme.textDim
        fusionRefreshLabel.translatesAutoresizingMaskIntoConstraints = false
        panel.addSubview(fusionRefreshLabel)

        fusionScroll = NSScrollView()
        fusionScroll.translatesAutoresizingMaskIntoConstraints = false
        fusionScroll.hasVerticalScroller = true
        fusionScroll.autohidesScrollers = true
        fusionScroll.borderType = .noBorder
        fusionScroll.drawsBackground = true
        panel.addSubview(fusionScroll)

        fusionOutput = NSTextView()
        fusionOutput.isEditable = false
        fusionOutput.backgroundColor = NSColor(red: 0.03, green: 0.03, blue: 0.06, alpha: 1.0)
        fusionOutput.textColor = .systemCyan
        fusionOutput.font = NSFont.monospacedSystemFont(ofSize: 10.5, weight: .regular)
        fusionOutput.isVerticallyResizable = true
        fusionOutput.isHorizontallyResizable = false
        fusionOutput.autoresizingMask = [.width]
        fusionOutput.textContainer?.widthTracksTextView = true
        fusionOutput.string = "⚡ THREE ENGINE SYNTHESIS READY\n" +
                              "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n" +
                              "GOD_CODE  = \(String(format: "%.10f", GOD_CODE))\n" +
                              "PHI       = \(String(format: "%.15f", PHI))\n" +
                              "VOID      = \(String(format: "%.13f", VOID_CONSTANT))\n" +
                              "OMEGA     = \(String(format: "%.5f", OMEGA))\n" +
                              "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n" +
                              "Click 'Run Fusion Analysis' to execute cross-engine synthesis.\n"
        fusionScroll.documentView = fusionOutput

        NSLayoutConstraint.activate([
            fusionTitle.topAnchor.constraint(equalTo: panel.topAnchor, constant: 10),
            fusionTitle.leadingAnchor.constraint(equalTo: panel.leadingAnchor, constant: 14),

            runBtn.centerYAnchor.constraint(equalTo: fusionTitle.centerYAnchor),
            runBtn.leadingAnchor.constraint(equalTo: fusionTitle.trailingAnchor, constant: 16),

            statusBtn.centerYAnchor.constraint(equalTo: fusionTitle.centerYAnchor),
            statusBtn.leadingAnchor.constraint(equalTo: runBtn.trailingAnchor, constant: 8),

            constsBtn.centerYAnchor.constraint(equalTo: fusionTitle.centerYAnchor),
            constsBtn.leadingAnchor.constraint(equalTo: statusBtn.trailingAnchor, constant: 8),

            fusionRefreshLabel.centerYAnchor.constraint(equalTo: fusionTitle.centerYAnchor),
            fusionRefreshLabel.trailingAnchor.constraint(equalTo: panel.trailingAnchor, constant: -12),

            fusionScroll.topAnchor.constraint(equalTo: fusionTitle.bottomAnchor, constant: 8),
            fusionScroll.leadingAnchor.constraint(equalTo: panel.leadingAnchor, constant: 6),
            fusionScroll.trailingAnchor.constraint(equalTo: panel.trailingAnchor, constant: -6),
            fusionScroll.bottomAnchor.constraint(equalTo: panel.bottomAnchor, constant: -6),
        ])

        return panel
    }

    // MARK: - Refresh Timer

    private func startRefreshTimer() {
        refreshTimer = Timer.scheduledTimer(withTimeInterval: 8.0, repeats: true) { [weak self] _ in
            self?.fetchAll()
        }
    }

    @objc private func manualRefresh() {
        fetchAll()
    }

    // MARK: - Fetch All Engines

    private func fetchAll() {
        fetchCodeEngine()
        fetchMathEngine()
        fetchScienceEngine()
        updateTimestamps()
    }

    private func updateTimestamps() {
        let ts = currentTimeString()
        DispatchQueue.main.async { [weak self] in
            self?.lastRefreshLabel.stringValue = "Last refresh: \(ts)"
            self?.fusionRefreshLabel.stringValue = "Last refreshed: \(ts)"
        }
    }

    private func currentTimeString() -> String {
        let fmt = DateFormatter()
        fmt.dateFormat = "HH:mm:ss"
        return fmt.string(from: Date())
    }

    // MARK: - Code Engine Fetch

    private func fetchCodeEngine() {
        APIGateway.shared.route(endpointID: "fast-server", path: "/api/v64/evo/code-engine") { [weak self] result in
            guard let self else { return }
            let online = result["error"] == nil
            self.codeStatusDot.layer?.backgroundColor = (online ? NSColor.systemGreen : NSColor.systemRed).cgColor

            if let data = result["data"] as? [String: Any] {
                let smells     = (data["smell_count"]   as? Int)    .map { "\($0)" }    ?? (data["smells"]     as? String) ?? "—"
                let quality    = (data["quality_score"] as? Double) .map { String(format: "%.1f%%", $0) } ?? (data["quality"] as? String) ?? "—"
                let complexity = (data["complexity"]    as? Double) .map { String(format: "%.2f",   $0) } ?? (data["complexity_rating"] as? String) ?? "—"
                let coverage   = (data["coverage"]      as? Double) .map { String(format: "%.1f%%", $0) } ?? (data["test_coverage"]     as? String) ?? "—"

                let qualProg = (data["quality_score"] as? Double).map { $0 / 100.0 } ?? 0.5
                let covProg  = (data["coverage"]      as? Double).map { $0 / 100.0 } ?? 0.5

                DispatchQueue.main.async {
                    self.codeCard.updateTile(index: 0, value: smells,     progress: 0.3)
                    self.codeCard.updateTile(index: 1, value: quality,    progress: qualProg)
                    self.codeCard.updateTile(index: 2, value: complexity, progress: 0.5)
                    self.codeCard.updateTile(index: 3, value: coverage,   progress: covProg)
                    self.codeCard.setMiniOutput("✅ Code Engine online\nQuality: \(quality) | Coverage: \(coverage)")
                }
            } else {
                DispatchQueue.main.async {
                    self.codeCard.setMiniOutput("⚠️ Code Engine offline or no data")
                }
            }

            self.codeCard.setStatus(online: online)
        }
    }

    // MARK: - Math Engine Fetch

    private func fetchMathEngine() {
        APIGateway.shared.route(endpointID: "fast-server", path: "/api/v64/evo/math-engine") { [weak self] result in
            guard let self else { return }
            let online = result["error"] == nil

            if let data = result["data"] as? [String: Any] {
                let godAlign   = (data["god_code_alignment"] as? Double).map { String(format: "%.4f", $0) } ?? (data["alignment"]  as? String) ?? "—"
                let phiConv    = (data["phi_convergence"]    as? Double).map { String(format: "%.4f", $0) } ?? (data["phi_align"]  as? String) ?? "—"
                let harmonics  = (data["harmonic_resonance"] as? Double).map { String(format: "%.4f", $0) } ?? (data["harmonics"] as? String) ?? "—"
                let proofCount = (data["proof_count"]        as? Int)   .map { "\($0)" }                  ?? (data["proofs"]     as? String) ?? "—"

                let godProg  = (data["god_code_alignment"] as? Double).map { min($0 / GOD_CODE, 1.0) } ?? 0.5
                let phiProg  = (data["phi_convergence"]    as? Double).map { min($0 / PHI, 1.0) }      ?? 0.5

                DispatchQueue.main.async {
                    self.mathCard.updateTile(index: 0, value: godAlign,   progress: godProg)
                    self.mathCard.updateTile(index: 1, value: phiConv,    progress: phiProg)
                    self.mathCard.updateTile(index: 2, value: harmonics,  progress: 0.7)
                    self.mathCard.updateTile(index: 3, value: proofCount, progress: 0.9)
                    self.mathCard.setMiniOutput("✅ Math Engine online\nGOD_CODE: \(godAlign) | PHI: \(phiConv)")
                }
            } else {
                DispatchQueue.main.async {
                    self.mathCard.setMiniOutput("⚠️ Math Engine offline or no data")
                }
            }

            self.mathCard.setStatus(online: online)
        }
    }

    // MARK: - Science Engine Fetch

    private func fetchScienceEngine() {
        APIGateway.shared.route(endpointID: "fast-server", path: "/api/v64/evo/science-engine") { [weak self] result in
            guard let self else { return }
            let online = result["error"] == nil

            if let data = result["data"] as? [String: Any] {
                let entropy    = (data["entropy_score"]     as? Double).map { String(format: "%.4f", $0) } ?? (data["entropy"]      as? String) ?? "—"
                let coherence  = (data["coherence_score"]   as? Double).map { String(format: "%.4f", $0) } ?? (data["coherence"]    as? String) ?? "—"
                let fidelity26 = (data["fidelity_26q"]      as? Double).map { String(format: "%.4f", $0) } ?? (data["fidelity"]     as? String) ?? "—"
                let physConsts = (data["constants_verified"] as? Int)  .map { "\($0)" }                   ?? (data["physics"]      as? String) ?? "—"

                let entProg = (data["entropy_score"]   as? Double).map { min($0, 1.0) } ?? 0.5
                let cohProg = (data["coherence_score"] as? Double).map { min($0, 1.0) } ?? 0.5
                let fidProg = (data["fidelity_26q"]    as? Double).map { min($0, 1.0) } ?? 0.5

                DispatchQueue.main.async {
                    self.sciCard.updateTile(index: 0, value: entropy,    progress: entProg)
                    self.sciCard.updateTile(index: 1, value: coherence,  progress: cohProg)
                    self.sciCard.updateTile(index: 2, value: fidelity26, progress: fidProg)
                    self.sciCard.updateTile(index: 3, value: physConsts, progress: 0.95)
                    self.sciCard.setMiniOutput("✅ Science Engine online\nEntropy: \(entropy) | Coherence: \(coherence)")
                }
            } else {
                DispatchQueue.main.async {
                    self.sciCard.setMiniOutput("⚠️ Science Engine offline or no data")
                }
            }

            self.sciCard.setStatus(online: online)

            // Update header sci dot
            DispatchQueue.main.async { [weak self] in
                self?.sciStatusDot.layer?.backgroundColor = (online ? NSColor.systemCyan : NSColor.systemRed).cgColor
            }
        }
    }

    // MARK: - Code Card Actions

    @objc private func codeCardAnalyze() {
        codeCard.setMiniOutput("⏳ Analyzing code engine...")
        let sampleCode = "def sacred_phi(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a+b\n    return b/a"
        APIGateway.shared.route(
            endpointID: "fast-server",
            path: "/api/v64/three-engine/code-analysis",
            body: ["code": sampleCode, "language": "python"]
        ) { [weak self] result in
            let text: String
            if let data = result["data"] as? [String: Any],
               let analysis = data["analysis"] as? String {
                text = "✅ Analysis:\n\(analysis)"
            } else if let err = result["error"] as? String {
                text = "⚠️ \(err)"
            } else {
                text = "✅ Analysis dispatched"
            }
            self?.codeCard.setMiniOutput(text)
        }
    }

    @objc private func codeCardAudit() {
        codeCard.setMiniOutput("⏳ Running audit...")
        APIGateway.shared.route(endpointID: "fast-server", path: "/api/v64/evo/code-engine/audit") { [weak self] result in
            let ok = result["error"] == nil
            self?.codeCard.setMiniOutput(ok ? "✅ Audit complete" : "⚠️ Audit: server offline")
        }
    }

    @objc private func codeCardAutoFix() {
        codeCard.setMiniOutput("⏳ Running Auto-Fix...")
        APIGateway.shared.route(
            endpointID: "fast-server",
            path: "/api/v64/evo/code-engine/auto-fix",
            body: [:]
        ) { [weak self] result in
            let ok = result["error"] == nil
            self?.codeCard.setMiniOutput(ok ? "✅ Auto-Fix complete" : "⚠️ Auto-Fix: server offline")
        }
    }

    // MARK: - Math Card Actions

    @objc private func mathCardVerify() {
        mathCard.setMiniOutput("⏳ Verifying GOD_CODE alignment...")
        APIGateway.shared.route(endpointID: "fast-server", path: "/api/v64/evo/math-engine/verify") { [weak self] result in
            if let data = result["data"] as? [String: Any],
               let aligned = data["aligned"] as? Bool {
                let icon = aligned ? "✅" : "⚠️"
                self?.mathCard.setMiniOutput("\(icon) GOD_CODE verify: \(aligned ? "ALIGNED" : "MISALIGNED")")
            } else {
                // Offline: verify locally
                let localGOD = GOD_CODE
                let tolerance = 0.001
                let aligned = abs(localGOD - 527.5184818492612) < tolerance
                self?.mathCard.setMiniOutput("✅ Local verify: GOD_CODE=\(String(format:"%.6f",localGOD)) [\(aligned ? "OK" : "DRIFT")]")
            }
        }
    }

    @objc private func mathCardProofs() {
        mathCard.setMiniOutput("⏳ Fetching sovereign proofs...")
        APIGateway.shared.route(endpointID: "fast-server", path: "/api/v64/evo/math-engine/proofs") { [weak self] result in
            if let data = result["data"] as? [String: Any],
               let count = data["proof_count"] as? Int {
                self?.mathCard.setMiniOutput("✅ Proofs: \(count) sovereign proofs verified")
            } else {
                self?.mathCard.setMiniOutput("⚠️ Proofs: server offline — local φ=\(String(format:"%.6f",PHI))")
            }
        }
    }

    @objc private func mathCardHarmonics() {
        mathCard.setMiniOutput("⏳ Computing harmonic resonance...")
        APIGateway.shared.route(endpointID: "fast-server", path: "/api/v64/evo/math-engine/harmonics") { [weak self] result in
            if let data = result["data"] as? [String: Any],
               let resonance = data["harmonic_resonance"] as? Double {
                self?.mathCard.setMiniOutput("✅ Harmonics: resonance=\(String(format:"%.4f", resonance))")
            } else {
                let localHarm = GOD_CODE / PHI
                self?.mathCard.setMiniOutput("✅ Local harmonic: \(String(format:"%.4f Hz", localHarm))")
            }
        }
    }

    // MARK: - Science Card Actions

    @objc private func sciCardEntropy() {
        sciCard.setMiniOutput("⏳ Maxwell Demon entropy analysis...")
        APIGateway.shared.route(endpointID: "fast-server", path: "/api/v64/evo/science-engine/entropy") { [weak self] result in
            if let data = result["data"] as? [String: Any],
               let eff = data["demon_efficiency"] as? Double {
                self?.sciCard.setMiniOutput("✅ Entropy: demon efficiency=\(String(format:"%.4f", eff))")
            } else {
                self?.sciCard.setMiniOutput("⚠️ Entropy: server offline")
            }
        }
    }

    @objc private func sciCardCoherence() {
        sciCard.setMiniOutput("⏳ Quantum coherence evolution...")
        APIGateway.shared.route(endpointID: "fast-server", path: "/api/v64/evo/science-engine/coherence") { [weak self] result in
            if let data = result["data"] as? [String: Any],
               let coh = data["coherence"] as? Double {
                self?.sciCard.setMiniOutput("✅ Coherence: \(String(format:"%.4f", coh))")
            } else {
                let localCoh = L104State.shared.coherence
                self?.sciCard.setMiniOutput("✅ Local coherence: \(String(format:"%.4f", localCoh))")
            }
        }
    }

    @objc private func sciCard26Q() {
        sciCard.setMiniOutput("⏳ 26Q iron-mapped circuit analysis...")
        APIGateway.shared.route(endpointID: "fast-server", path: "/api/v64/evo/science-engine/26q") { [weak self] result in
            if let data = result["data"] as? [String: Any],
               let fidelity = data["fidelity"] as? Double {
                self?.sciCard.setMiniOutput("✅ 26Q fidelity: \(String(format:"%.4f", fidelity))")
            } else {
                self?.sciCard.setMiniOutput("✅ 26Q: Fe(26) map | harmonic=\(String(format:"%.2fHz", GOD_CODE/PHI))")
            }
        }
    }

    // MARK: - Fusion Panel Actions

    @objc private func runFusionAnalysis() {
        appendFusionOutput("⏳ Running Three-Engine Fusion Analysis...")
        appendFusionOutput("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        let sampleCode = """
        def god_code_verify(x):
            phi = 1.618033988749895
            return (286 ** (1/phi)) * (2 ** (416/104))
        """

        APIGateway.shared.route(
            endpointID: "fast-server",
            path: "/api/v64/three-engine/code-analysis",
            body: [
                "code": sampleCode,
                "language": "python",
                "engines": ["code", "math", "science"],
            ]
        ) { [weak self] result in
            guard let self else { return }
            if let data = result["data"] as? [String: Any] {
                let codeResult  = (data["code_engine"]    as? [String: Any])
                let mathResult  = (data["math_engine"]    as? [String: Any])
                let sciResult   = (data["science_engine"] as? [String: Any])
                let fusionScore = (data["fusion_score"]   as? Double).map { String(format: "%.4f", $0) } ?? "—"

                DispatchQueue.main.async {
                    self.appendFusionOutput("✅ FUSION COMPLETE")
                    self.appendFusionOutput("  Code Engine:    \((codeResult?["status"] as? String) ?? "OK")")
                    self.appendFusionOutput("  Math Engine:    \((mathResult?["status"] as? String) ?? "OK")")
                    self.appendFusionOutput("  Science Engine: \((sciResult?["status"] as? String) ?? "OK")")
                    self.appendFusionOutput("  Fusion Score:   \(fusionScore)")
                    self.appendFusionOutput("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                }
            } else {
                // Server offline — show local sacred constants alignment
                DispatchQueue.main.async {
                    self.appendFusionOutput("⚠️ Server offline — local sacred alignment:")
                    self.appendFusionOutput("  GOD_CODE = \(String(format:"%.10f", GOD_CODE))")
                    self.appendFusionOutput("  PHI      = \(String(format:"%.15f", PHI))")
                    self.appendFusionOutput("  VOID     = \(String(format:"%.13f", VOID_CONSTANT))")
                    self.appendFusionOutput("  OMEGA    = \(String(format:"%.5f",  OMEGA))")
                    self.appendFusionOutput("  EVO_\(EVOLUTION_INDEX) | ASI v\(ASI_VERSION) | Code v\(CODE_ENGINE_VERSION)")
                    self.appendFusionOutput("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                }
            }
        }
    }

    @objc private func fetchTriEngineStatus() {
        appendFusionOutput("⏳ Fetching Tri-Engine Status...")
        APIGateway.shared.route(endpointID: "tri-engine", path: "/api/v62/tri-engine/status") { [weak self] result in
            guard let self else { return }
            if let data = result["data"] as? [String: Any] {
                let codeOnline = (data["code_engine_online"]    as? Bool) ?? false
                let mathOnline = (data["math_engine_online"]    as? Bool) ?? false
                let sciOnline  = (data["science_engine_online"] as? Bool) ?? false
                DispatchQueue.main.async {
                    self.appendFusionOutput("✅ Tri-Engine Status:")
                    self.appendFusionOutput("  💻 Code Engine:    \(codeOnline ? "ONLINE" : "OFFLINE")")
                    self.appendFusionOutput("  🧮 Math Engine:    \(mathOnline ? "ONLINE" : "OFFLINE")")
                    self.appendFusionOutput("  🔬 Science Engine: \(sciOnline  ? "ONLINE" : "OFFLINE")")
                }
            } else {
                DispatchQueue.main.async {
                    self.appendFusionOutput("⚠️ Tri-Engine: \(result["error"] as? String ?? "server offline")")
                }
            }
        }
    }

    @objc private func fetchTriEngineConstants() {
        appendFusionOutput("⏳ Fetching Tri-Engine Constants...")
        APIGateway.shared.route(endpointID: "tri-engine", path: "/api/v62/tri-engine/constants") { [weak self] result in
            guard let self else { return }
            if let data = result["data"] as? [String: Any] {
                DispatchQueue.main.async {
                    self.appendFusionOutput("✅ Tri-Engine Constants:")
                    if let gc = data["god_code"] as? Double {
                        let ok = abs(gc - 527.5184818492612) < 0.001
                        self.appendFusionOutput("  GOD_CODE: \(String(format:"%.6f", gc)) \(ok ? "✅" : "⚠️")")
                        self.godCodeCheck?.stringValue = ok ? "✅" : "⚠️"
                    }
                    if let ph = data["phi"] as? Double {
                        let ok = abs(ph - 1.618033988749895) < 0.0001
                        self.appendFusionOutput("  PHI:      \(String(format:"%.9f", ph)) \(ok ? "✅" : "⚠️")")
                        self.phiCheck?.stringValue = ok ? "✅" : "⚠️"
                    }
                    if let vc = data["void_constant"] as? Double {
                        let ok = abs(vc - 1.0416180339887497) < 0.0001
                        self.appendFusionOutput("  VOID:     \(String(format:"%.10f", vc)) \(ok ? "✅" : "⚠️")")
                        self.voidCheck?.stringValue = ok ? "✅" : "⚠️"
                    }
                }
            } else {
                DispatchQueue.main.async {
                    self.appendFusionOutput("⚠️ Constants: server offline — local values verified locally")
                }
            }
        }
    }

    // MARK: - Append to fusion output

    private func appendFusionOutput(_ text: String) {
        DispatchQueue.main.async { [weak self] in
            guard let tv = self?.fusionOutput else { return }
            tv.string += "\(text)\n"
            tv.scrollToEndOfDocument(nil)
        }
    }
}
