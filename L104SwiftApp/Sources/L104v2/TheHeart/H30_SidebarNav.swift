// ═══════════════════════════════════════════════════════════════════
// H30_SidebarNav.swift
// [EVO_68_PIPELINE] SOVEREIGN_CONVERGENCE :: SIDEBAR_NAV
// L104 ASI — Sidebar Navigation (replaces 16-tab bar)
//
// Groups 16 views into categorized sidebar sections with icons.
// Uses NSOutlineView source list style for native macOS look.
// ═══════════════════════════════════════════════════════════════════

import AppKit
import Foundation

// ─── DATA MODEL ───

/// A section header in the sidebar (e.g. "Core", "Intelligence")
class SidebarSection {
    let title: String
    let items: [SidebarItem]
    var isExpanded: Bool = true

    init(_ title: String, items: [SidebarItem]) {
        self.title = title
        self.items = items
    }
}

/// A clickable row in the sidebar that maps to a tab identifier
class SidebarItem {
    let icon: String       // emoji / SF symbol name
    let label: String      // display label
    let tabID: String      // maps to NSTabViewItem identifier

    init(_ icon: String, _ label: String, tab: String) {
        self.icon = icon
        self.label = label
        self.tabID = tab
    }
}

// ─── SIDEBAR VIEW ───

class L104SidebarView: NSView, NSOutlineViewDelegate, NSOutlineViewDataSource {
    private var outlineView: NSOutlineView!
    private var scrollView: NSScrollView!
    /// Called when the user clicks a sidebar row — payload is the tab identifier string
    var onSelect: ((String) -> Void)?

    /// The grouped navigation structure (4 groups, 16 items)
    let sections: [SidebarSection] = [
        SidebarSection("CORE", items: [
            SidebarItem("💬", "Chat", tab: "chat"),
            SidebarItem("🌌", "Dashboard", tab: "dash"),
            SidebarItem("📡", "System", tab: "sys"),
        ]),
        SidebarSection("INTELLIGENCE", items: [
            SidebarItem("🚀", "ASI Nexus", tab: "asi"),
            SidebarItem("🧠", "Learning", tab: "learn"),
            SidebarItem("🎓", "Professor", tab: "prof"),
            SidebarItem("🔮", "Sage Mode", tab: "sage"),
        ]),
        SidebarSection("ENGINES", items: [
            SidebarItem("⚡", "Logic Gates", tab: "gate"),
            SidebarItem("⚛️", "Quantum", tab: "qc"),
            SidebarItem("💻", "Coding", tab: "code"),
            SidebarItem("🔬", "Science", tab: "sci"),
            SidebarItem("🌌", "Unified Field", tab: "ufield"),
        ]),
        SidebarSection("SYSTEM", items: [
            SidebarItem("🧬", "Upgrades", tab: "upg"),
            SidebarItem("💾", "Memory", tab: "mem"),
            SidebarItem("🍎", "Hardware", tab: "hw"),
            SidebarItem("🌐", "Network", tab: "net"),
            SidebarItem("🛠", "Debug Console", tab: "debug"),
        ]),
    ]

    override init(frame: NSRect) {
        super.init(frame: frame)
        wantsLayer = true
        layer?.backgroundColor = NSColor(red: 0.955, green: 0.955, blue: 0.965, alpha: 1.0).cgColor
        buildOutlineView()
    }

    required init?(coder: NSCoder) { super.init(coder: coder) }

    private func buildOutlineView() {
        // Column
        let col = NSTableColumn(identifier: NSUserInterfaceItemIdentifier("nav"))
        col.title = ""

        // Outline view
        outlineView = NSOutlineView()
        outlineView.addTableColumn(col)
        outlineView.outlineTableColumn = col
        outlineView.headerView = nil
        outlineView.style = .sourceList
        outlineView.selectionHighlightStyle = .regular
        outlineView.indentationPerLevel = 14
        outlineView.rowSizeStyle = .medium
        outlineView.floatsGroupRows = false
        outlineView.delegate = self
        outlineView.dataSource = self

        // Scroll view
        scrollView = NSScrollView(frame: bounds)
        scrollView.documentView = outlineView
        scrollView.hasVerticalScroller = true
        scrollView.autohidesScrollers = true
        scrollView.autoresizingMask = [.width, .height]
        scrollView.drawsBackground = false
        addSubview(scrollView)

        // Expand all sections
        outlineView.reloadData()
        for section in sections {
            outlineView.expandItem(section)
        }

        // Select "Chat" by default
        if let chatItem = sections.first?.items.first {
            let row = outlineView.row(forItem: chatItem)
            if row >= 0 {
                outlineView.selectRowIndexes(IndexSet(integer: row), byExtendingSelection: false)
            }
        }
    }

    // ─── NSOutlineViewDataSource ───

    func outlineView(_ outlineView: NSOutlineView, numberOfChildrenOfItem item: Any?) -> Int {
        if item == nil { return sections.count }
        if let section = item as? SidebarSection { return section.items.count }
        return 0
    }

    func outlineView(_ outlineView: NSOutlineView, child index: Int, ofItem item: Any?) -> Any {
        if item == nil { return sections[index] }
        if let section = item as? SidebarSection { return section.items[index] }
        return ""
    }

    func outlineView(_ outlineView: NSOutlineView, isItemExpandable item: Any) -> Bool {
        return item is SidebarSection
    }

    // ─── NSOutlineViewDelegate ───

    func outlineView(_ outlineView: NSOutlineView, isGroupItem item: Any) -> Bool {
        return item is SidebarSection
    }

    func outlineView(_ outlineView: NSOutlineView, shouldSelectItem item: Any) -> Bool {
        return item is SidebarItem
    }

    func outlineView(_ outlineView: NSOutlineView, viewFor tableColumn: NSTableColumn?, item: Any) -> NSView? {
        if let section = item as? SidebarSection {
            let cell = NSTableCellView()
            let tf = NSTextField(labelWithString: section.title)
            tf.font = NSFont.systemFont(ofSize: 11, weight: .bold)
            tf.textColor = NSColor.secondaryLabelColor
            tf.translatesAutoresizingMaskIntoConstraints = false
            cell.addSubview(tf)
            cell.textField = tf
            NSLayoutConstraint.activate([
                tf.leadingAnchor.constraint(equalTo: cell.leadingAnchor, constant: 2),
                tf.centerYAnchor.constraint(equalTo: cell.centerYAnchor),
            ])
            return cell
        }

        if let item = item as? SidebarItem {
            let cell = NSTableCellView()
            cell.identifier = NSUserInterfaceItemIdentifier("navItem")

            let tf = NSTextField(labelWithString: "\(item.icon)  \(item.label)")
            tf.font = NSFont.systemFont(ofSize: 13, weight: .regular)
            tf.textColor = NSColor.labelColor
            tf.lineBreakMode = .byTruncatingTail
            tf.translatesAutoresizingMaskIntoConstraints = false
            cell.addSubview(tf)
            cell.textField = tf
            NSLayoutConstraint.activate([
                tf.leadingAnchor.constraint(equalTo: cell.leadingAnchor, constant: 4),
                tf.trailingAnchor.constraint(equalTo: cell.trailingAnchor, constant: -4),
                tf.centerYAnchor.constraint(equalTo: cell.centerYAnchor),
            ])
            return cell
        }

        return nil
    }

    func outlineView(_ outlineView: NSOutlineView, heightOfRowByItem item: Any) -> CGFloat {
        if item is SidebarSection { return 24 }
        return 28
    }

    func outlineViewSelectionDidChange(_ notification: Notification) {
        let row = outlineView.selectedRow
        guard row >= 0, let item = outlineView.item(atRow: row) as? SidebarItem else { return }
        onSelect?(item.tabID)
    }

    /// Programmatically select a sidebar row by tab identifier (used when code navigates via tabView)
    func selectItem(withTabID tabID: String) {
        for section in sections {
            for item in section.items {
                if item.tabID == tabID {
                    let row = outlineView.row(forItem: item)
                    if row >= 0 {
                        outlineView.selectRowIndexes(IndexSet(integer: row), byExtendingSelection: false)
                    }
                    return
                }
            }
        }
    }
}
