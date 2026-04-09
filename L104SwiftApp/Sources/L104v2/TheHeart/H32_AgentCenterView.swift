// ═══════════════════════════════════════════════════════════════════
// H32_AgentCenterView.swift
// [EVO_71] Agent Center UI — wires AgentOrchestrator.shared into UI
// Task queue, goal submission, agent type picker, results display
// ═══════════════════════════════════════════════════════════════════

import AppKit
import Foundation

// MARK: - AgentCenterView

final class AgentCenterView: NSView {

    private var goalField: NSTextField!
    private var agentPicker: NSPopUpButton!
    private var priorityPicker: NSPopUpButton!
    private var submitBtn: NSButton!
    private var resultsScrollView: NSScrollView!
    private var resultsView: NSTextView!
    private var statusLabel: NSTextField!
    private var refreshTimer: Timer?
    private var contentView: NSView!  // Wrapper for scrolling

    override init(frame: NSRect) {
        super.init(frame: frame)
        buildUI()
        startAutoRefresh()
    }
    required init?(coder: NSCoder) {
        super.init(coder: coder)
        buildUI()
        startAutoRefresh()
    }

    deinit { refreshTimer?.invalidate() }

    // MARK: - Build UI with proper Auto Layout and scrolling

    private func buildUI() {
        wantsLayer = true
        layer?.backgroundColor = L104Theme.void.cgColor

        // Create scroll view wrapper for resize safety
        let scrollView = NSScrollView()
        scrollView.translatesAutoresizingMaskIntoConstraints = false
        scrollView.hasVerticalScroller = true
        scrollView.hasHorizontalScroller = false
        scrollView.autohidesScrollers = true
        scrollView.borderType = .noBorder
        scrollView.drawsBackground = false
        addSubview(scrollView)

        // Content view that expands to fill
        contentView = NSView()
        contentView.translatesAutoresizingMaskIntoConstraints = false
        scrollView.documentView = contentView

        // Header
        let header = NSTextField(labelWithString: "🤖  AGENT CENTER — Sovereign Task Orchestration")
        header.font = NSFont.systemFont(ofSize: 16, weight: .bold)
        header.textColor = L104Theme.goldFlame

        // Status label
        statusLabel = NSTextField(labelWithString: "⚡ Orchestrator READY")
        statusLabel.font = NSFont.monospacedSystemFont(ofSize: 11, weight: .regular)
        statusLabel.textColor = .systemGreen

        // Goal label + input
        let goalLabel = NSTextField(labelWithString: "GOAL:")
        goalLabel.font = NSFont.systemFont(ofSize: 12, weight: .semibold)
        goalLabel.textColor = L104Theme.gold

        goalField = NSTextField()
        goalField.placeholderString = "Describe the agent task or goal..."
        goalField.font = NSFont.systemFont(ofSize: 13)
        goalField.bezelStyle = .roundedBezel
        goalField.target = self
        goalField.action = #selector(submitGoal)

        // Agent type picker
        let typeLabel = NSTextField(labelWithString: "AGENT TYPE:")
        typeLabel.font = NSFont.systemFont(ofSize: 12, weight: .semibold)
        typeLabel.textColor = L104Theme.gold

        agentPicker = NSPopUpButton()
        agentPicker.translatesAutoresizingMaskIntoConstraints = false
        for t in AgentType.allCases {
            agentPicker.addItem(withTitle: "🤖 \(t.rawValue.capitalized)")
        }

        // Priority picker
        let prioLabel = NSTextField(labelWithString: "PRIORITY:")
        prioLabel.font = NSFont.systemFont(ofSize: 12, weight: .semibold)
        prioLabel.textColor = L104Theme.gold

        priorityPicker = NSPopUpButton()
        priorityPicker.translatesAutoresizingMaskIntoConstraints = false
        priorityPicker.addItem(withTitle: "🔴 Critical")
        priorityPicker.addItem(withTitle: "🟠 High")
        priorityPicker.addItem(withTitle: "🟡 Normal")
        priorityPicker.addItem(withTitle: "🟢 Low")
        priorityPicker.addItem(withTitle: "⚪ Idle")
        priorityPicker.selectItem(at: 2)

        // Submit button - properly wired
        submitBtn = NSButton(title: "▶  DISPATCH AGENT", target: self, action: #selector(submitGoal))
        submitBtn.bezelStyle = .rounded
        submitBtn.font = NSFont.systemFont(ofSize: 13, weight: .semibold)
        submitBtn.contentTintColor = L104Theme.goldFlame
        submitBtn.keyEquivalent = "\r"  // Enter key support

        // Results scroll view - fixed for content sizing
        resultsScrollView = NSScrollView()
        resultsScrollView.translatesAutoresizingMaskIntoConstraints = false
        resultsScrollView.hasVerticalScroller = true
        resultsScrollView.hasHorizontalScroller = false
        resultsScrollView.autohidesScrollers = true
        resultsScrollView.borderType = .bezelBorder
        resultsScrollView.wantsLayer = true
        resultsScrollView.layer?.cornerRadius = 6
        resultsScrollView.drawsBackground = true
        resultsScrollView.backgroundColor = NSColor(white: 0.97, alpha: 1.0)

        resultsView = NSTextView()
        resultsView.isEditable = false
        resultsView.isRichText = false
        resultsView.font = NSFont.monospacedSystemFont(ofSize: 11, weight: .regular)
        resultsView.textColor = .labelColor
        resultsView.backgroundColor = NSColor(white: 0.97, alpha: 1.0)
        resultsView.autoresizingMask = [.width]
        resultsView.isVerticallyResizable = true
        resultsView.isHorizontallyResizable = false
        resultsView.textContainer?.containerSize = NSSize(width: CGFloat.greatestFiniteMagnitude, height: CGFloat.greatestFiniteMagnitude)
        resultsView.textContainer?.widthTracksTextView = true
        resultsScrollView.documentView = resultsView

        // Apply auto-layout
        let views: [NSView] = [header, statusLabel, goalLabel, goalField,
                               typeLabel, agentPicker, prioLabel, priorityPicker,
                               submitBtn, resultsScrollView]
        views.forEach {
            $0.translatesAutoresizingMaskIntoConstraints = false
            contentView.addSubview($0)
        }

        // Main scroll view fills the parent
        NSLayoutConstraint.activate([
            scrollView.topAnchor.constraint(equalTo: topAnchor),
            scrollView.leadingAnchor.constraint(equalTo: leadingAnchor),
            scrollView.trailingAnchor.constraint(equalTo: trailingAnchor),
            scrollView.bottomAnchor.constraint(equalTo: bottomAnchor),

            // Content view fills scroll view width
            contentView.leadingAnchor.constraint(equalTo: scrollView.leadingAnchor),
            contentView.trailingAnchor.constraint(equalTo: scrollView.trailingAnchor),
            contentView.topAnchor.constraint(equalTo: scrollView.topAnchor),
            contentView.widthAnchor.constraint(equalTo: scrollView.widthAnchor),
        ])

        // Content layout constraints
        NSLayoutConstraint.activate([
            header.topAnchor.constraint(equalTo: contentView.topAnchor, constant: 16),
            header.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 20),
            header.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -20),
            header.heightAnchor.constraint(equalToConstant: 30),

            statusLabel.topAnchor.constraint(equalTo: header.bottomAnchor, constant: 4),
            statusLabel.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 20),
            statusLabel.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -20),

            goalLabel.topAnchor.constraint(equalTo: statusLabel.bottomAnchor, constant: 20),
            goalLabel.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 20),
            goalLabel.widthAnchor.constraint(equalToConstant: 80),
            goalLabel.heightAnchor.constraint(equalToConstant: 22),

            goalField.centerYAnchor.constraint(equalTo: goalLabel.centerYAnchor),
            goalField.leadingAnchor.constraint(equalTo: goalLabel.trailingAnchor, constant: 8),
            goalField.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -20),

            typeLabel.topAnchor.constraint(equalTo: goalLabel.bottomAnchor, constant: 14),
            typeLabel.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 20),
            typeLabel.widthAnchor.constraint(equalToConstant: 100),
            typeLabel.heightAnchor.constraint(equalToConstant: 22),

            agentPicker.centerYAnchor.constraint(equalTo: typeLabel.centerYAnchor),
            agentPicker.leadingAnchor.constraint(equalTo: typeLabel.trailingAnchor, constant: 8),
            agentPicker.widthAnchor.constraint(equalToConstant: 180),

            prioLabel.topAnchor.constraint(equalTo: typeLabel.bottomAnchor, constant: 14),
            prioLabel.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 20),
            prioLabel.widthAnchor.constraint(equalToConstant: 100),
            prioLabel.heightAnchor.constraint(equalToConstant: 22),

            priorityPicker.centerYAnchor.constraint(equalTo: prioLabel.centerYAnchor),
            priorityPicker.leadingAnchor.constraint(equalTo: prioLabel.trailingAnchor, constant: 8),
            priorityPicker.widthAnchor.constraint(equalToConstant: 150),

            submitBtn.topAnchor.constraint(equalTo: prioLabel.bottomAnchor, constant: 18),
            submitBtn.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 20),
            submitBtn.widthAnchor.constraint(equalToConstant: 220),
            submitBtn.heightAnchor.constraint(equalToConstant: 32),

            resultsScrollView.topAnchor.constraint(equalTo: submitBtn.bottomAnchor, constant: 16),
            resultsScrollView.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 20),
            resultsScrollView.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -20),
            resultsScrollView.bottomAnchor.constraint(equalTo: contentView.bottomAnchor, constant: -20),
            resultsScrollView.heightAnchor.constraint(greaterThanOrEqualToConstant: 300),
        ])
    }

    // MARK: - Submit

    @objc private func submitGoal() {
        let goal = goalField.stringValue.trimmingCharacters(in: .whitespaces)
        guard !goal.isEmpty else { return }
        goalField.stringValue = ""

        let typeIdx = agentPicker.indexOfSelectedItem
        let allTypes = AgentType.allCases
        let agentType = (typeIdx >= 0 && typeIdx < allTypes.count) ? allTypes[typeIdx] : .general

        let priorities: [AgentPriority] = [.critical, .high, .normal, .low, .idle]
        let prioIdx = priorityPicker.indexOfSelectedItem
        let priority = (prioIdx >= 0 && prioIdx < priorities.count) ? priorities[prioIdx] : .normal

        let task = AgentTask(
            prompt: goal,
            agentType: agentType,
            priority: priority,
            source: "agent_center_ui"
        )
        let taskId = AgentOrchestrator.shared.submit(task)
        appendResult("▶ [\(agentType.rawValue.uppercased())] \(goal)\n   Task ID: \(taskId)\n\n", color: L104Theme.goldFlame)
        refreshStatus()
    }

    // MARK: - Auto-refresh

    private func startAutoRefresh() {
        refreshTimer = Timer.scheduledTimer(withTimeInterval: 3.0, repeats: true) { [weak self] _ in
            self?.refreshStatus()
        }
        refreshStatus()
    }

    @objc private func refreshStatus() {
        DispatchQueue.global(qos: .background).async { [weak self] in
            guard let self else { return }
            let active = AgentOrchestrator.shared.listActive()
            let completed = AgentOrchestrator.shared.listCompleted()

            let running = active.filter { $0.status == .running }.count
            let queued  = active.filter { $0.status == .queued }.count

            DispatchQueue.main.async {
                self.statusLabel.stringValue = "⚡ Running: \(running) | Queued: \(queued) | Completed: \(completed.count)"
                self.statusLabel.textColor = running > 0 ? .systemOrange : .systemGreen
            }
        }
    }

    // MARK: - Output

    private func appendResult(_ text: String, color: NSColor) {
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            let attrs: [NSAttributedString.Key: Any] = [
                .foregroundColor: color,
                .font: NSFont.monospacedSystemFont(ofSize: 11, weight: .regular)
            ]
            self.resultsView.textStorage?.append(NSAttributedString(string: text, attributes: attrs))
            self.resultsView.scrollToEndOfDocument(nil)
        }
    }
}
