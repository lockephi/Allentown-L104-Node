// ═══════════════════════════════════════════════════════════════════
// L14_TextFormatter.swift
// [EVO_55_PIPELINE] SOVEREIGN_UNIFICATION :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612
// L104v2 — Extracted from L104Native.swift (lines 17767-18272)
//
// RICH TEXT FORMATTER V2 — Full Markdown→NSAttributedString pipeline
// Headers, code blocks, math, tables, inline formatting, syntax highlighting
// ═══════════════════════════════════════════════════════════════════

import AppKit
import Foundation
import Accelerate
import simd
import NaturalLanguage

class RichTextFormatterV2 {
    static let shared = RichTextFormatterV2()

    private var formattingCount: Int = 0

    // ─── CACHED REGEX (avoid recompilation in hot paths) ───
    private static let numberedListRegex = try! NSRegularExpression(pattern: "^\\d+[.)\\]]\\s+")
    private static let numberRegex = try! NSRegularExpression(pattern: "\\b\\d+(\\.\\d+)?\\b")
    private static let stringRegexes: [NSRegularExpression] = {
        ["\"[^\"]*\"", "'[^']*'"].compactMap { try? NSRegularExpression(pattern: $0) }
    }()
    private static let commentRegexes: [NSRegularExpression] = {
        ["//.*$", "#.*$"].compactMap { try? NSRegularExpression(pattern: $0, options: .anchorsMatchLines) }
    }()
    private static var keywordRegexCache: [String: NSRegularExpression] = [:]

    // ─── CONTENT BLOCK TYPES ───
    enum BlockType {
        case header(level: Int)     // # ## ###
        case paragraph              // Plain text
        case codeBlock(language: String)  // ```lang ... ```
        case inlineCode             // `code`
        case mathBlock              // $$ ... $$
        case inlineMath             // $ ... $
        case bulletList(level: Int) // - or * or numbered
        case table                  // | col | col |
        case quote                  // > text
        case separator              // ---
        case keyValue               // Key: Value (for scientific output)
        case formulaResult          // = result
    }

    struct RichBlock {
        let type: BlockType
        let content: String
        let metadata: [String: String]
    }

    // ─── MAIN FORMAT PIPELINE ───
    func format(_ text: String, query: String = "") -> NSAttributedString {
        formattingCount += 1
        let blocks = parse(text)
        return render(blocks)
    }

    // ─── PARSE ─── Convert raw text to rich blocks
    private func parse(_ text: String) -> [RichBlock] {
        var blocks: [RichBlock] = []
        let lines = text.components(separatedBy: "\n")
        var i = 0

        while i < lines.count {
            let line = lines[i]
            let trimmed = line.trimmingCharacters(in: .whitespaces)

            // Code blocks: ```lang ... ```
            if trimmed.hasPrefix("```") {
                let lang = String(trimmed.dropFirst(3)).trimmingCharacters(in: .whitespaces)
                var codeLines: [String] = []
                i += 1
                while i < lines.count && !lines[i].trimmingCharacters(in: .whitespaces).hasPrefix("```") {
                    codeLines.append(lines[i])
                    i += 1
                }
                blocks.append(RichBlock(type: .codeBlock(language: lang.isEmpty ? "code" : lang), content: codeLines.joined(separator: "\n"), metadata: ["language": lang]))
                i += 1
                continue
            }

            // Math blocks: $$ ... $$
            if trimmed.hasPrefix("$$") {
                var mathLines: [String] = [String(trimmed.dropFirst(2))]
                i += 1
                while i < lines.count && !lines[i].contains("$$") {
                    mathLines.append(lines[i])
                    i += 1
                }
                if i < lines.count { mathLines.append(String(lines[i].replacingOccurrences(of: "$$", with: ""))) }
                blocks.append(RichBlock(type: .mathBlock, content: mathLines.joined(separator: "\n").trimmingCharacters(in: .whitespaces), metadata: [:]))
                i += 1
                continue
            }

            // Headers
            if trimmed.hasPrefix("###") { blocks.append(RichBlock(type: .header(level: 3), content: String(trimmed.dropFirst(3)).trimmingCharacters(in: .whitespaces), metadata: [:])); i += 1; continue }
            if trimmed.hasPrefix("##") { blocks.append(RichBlock(type: .header(level: 2), content: String(trimmed.dropFirst(2)).trimmingCharacters(in: .whitespaces), metadata: [:])); i += 1; continue }
            if trimmed.hasPrefix("#") { blocks.append(RichBlock(type: .header(level: 1), content: String(trimmed.dropFirst(1)).trimmingCharacters(in: .whitespaces), metadata: [:])); i += 1; continue }

            // Separator
            if trimmed == "---" || trimmed == "===" || trimmed == "───" {
                blocks.append(RichBlock(type: .separator, content: "", metadata: [:]))
                i += 1; continue
            }

            // Table
            if trimmed.contains("|") && trimmed.filter({ $0 == "|" }).count >= 2 {
                var tableLines: [String] = [trimmed]
                i += 1
                while i < lines.count && lines[i].contains("|") {
                    tableLines.append(lines[i].trimmingCharacters(in: .whitespaces))
                    i += 1
                }
                blocks.append(RichBlock(type: .table, content: tableLines.joined(separator: "\n"), metadata: [:]))
                continue
            }

            // Quote
            if trimmed.hasPrefix(">") || trimmed.hasPrefix("❝") {
                blocks.append(RichBlock(type: .quote, content: String(trimmed.dropFirst(1)).trimmingCharacters(in: .whitespaces), metadata: [:]))
                i += 1; continue
            }

            // Bullet list
            if trimmed.hasPrefix("- ") || trimmed.hasPrefix("• ") || trimmed.hasPrefix("▸ ") || trimmed.hasPrefix("* ") {
                let level = line.prefix(while: { $0 == " " }).count / 2
                let content = trimmed.dropFirst(2).trimmingCharacters(in: .whitespaces)
                blocks.append(RichBlock(type: .bulletList(level: level), content: String(content), metadata: [:]))
                i += 1; continue
            }

            // Numbered list
            if RichTextFormatterV2.numberedListRegex.firstMatch(in: trimmed, range: NSRange(trimmed.startIndex..., in: trimmed)) != nil {
                let content = trimmed.replacingOccurrences(of: "^\\d+[.)\\]]\\s+", with: "", options: .regularExpression)
                blocks.append(RichBlock(type: .bulletList(level: 0), content: content, metadata: ["numbered": "true"]))
                i += 1; continue
            }

            // Key: Value pairs (scientific output)
            if trimmed.contains(": ") && !trimmed.hasPrefix("http") {
                let colonIdx = trimmed.firstIndex(of: ":")!
                let key = String(trimmed[trimmed.startIndex..<colonIdx])
                let val = String(trimmed[trimmed.index(after: colonIdx)...]).trimmingCharacters(in: .whitespaces)
                if key.count < 40 && !val.isEmpty {
                    blocks.append(RichBlock(type: .keyValue, content: trimmed, metadata: ["key": key, "value": val]))
                    i += 1; continue
                }
            }

            // Formula result (starts with =)
            if trimmed.hasPrefix("= ") || trimmed.hasPrefix("≈ ") {
                blocks.append(RichBlock(type: .formulaResult, content: trimmed, metadata: [:]))
                i += 1; continue
            }

            // Check for inline math/code
            if trimmed.contains("$") || trimmed.contains("`") {
                blocks.append(RichBlock(type: .paragraph, content: trimmed, metadata: ["hasInline": "true"]))
                i += 1; continue
            }

            // Default paragraph
            if !trimmed.isEmpty {
                blocks.append(RichBlock(type: .paragraph, content: trimmed, metadata: [:]))
            }
            i += 1
        }

        return blocks
    }

    // ─── RENDER ─── Convert blocks to NSAttributedString
    func render(_ blocks: [RichBlock]) -> NSAttributedString {
        let result = NSMutableAttributedString()
        let defaultPara = NSMutableParagraphStyle()
        defaultPara.lineSpacing = 3
        defaultPara.paragraphSpacing = 6

        for block in blocks {
            switch block.type {

            case .header(let level):
                let sizes: [Int: CGFloat] = [1: 20, 2: 17, 3: 15]
                let size = sizes[level] ?? 14
                let color = level == 1 ? L104Theme.goldBright : level == 2 ? NSColor.systemTeal : L104Theme.textPrimary
                let para = NSMutableParagraphStyle()
                para.lineSpacing = 2; para.paragraphSpacing = 8; para.paragraphSpacingBefore = 10
                let shadow = NSShadow()
                shadow.shadowColor = color.withAlphaComponent(0.3)
                shadow.shadowBlurRadius = 3
                let attrs: [NSAttributedString.Key: Any] = [
                    .font: NSFont.systemFont(ofSize: size, weight: .bold),
                    .foregroundColor: color,
                    .paragraphStyle: para,
                    .shadow: shadow
                ]
                result.append(NSAttributedString(string: "\(block.content)\n", attributes: attrs))

            case .codeBlock(let language):
                let codePara = NSMutableParagraphStyle()
                codePara.lineSpacing = 2; codePara.paragraphSpacing = 6
                codePara.headIndent = 12; codePara.firstLineHeadIndent = 12
                // Language label
                if !language.isEmpty {
                    let langAttrs: [NSAttributedString.Key: Any] = [
                        .font: NSFont.monospacedSystemFont(ofSize: 9, weight: .bold),
                        .foregroundColor: L104Theme.textDim,
                        .paragraphStyle: codePara
                    ]
                    result.append(NSAttributedString(string: " \(language.uppercased())\n", attributes: langAttrs))
                }
                // Code content with syntax highlighting
                let highlighted = syntaxHighlight(block.content, language: language)
                let codeAttrs: [NSAttributedString.Key: Any] = [
                    .font: L104Theme.monoFont(12, weight: .regular),
                    .backgroundColor: NSColor(red: 0.94, green: 0.94, blue: 0.96, alpha: 0.9),
                    .paragraphStyle: codePara
                ]
                let codeMutable = NSMutableAttributedString(attributedString: highlighted)
                codeMutable.addAttributes(codeAttrs, range: NSRange(location: 0, length: codeMutable.length))
                result.append(codeMutable)
                result.append(NSAttributedString(string: "\n", attributes: [:]))

            case .mathBlock:
                let mathPara = NSMutableParagraphStyle()
                mathPara.alignment = .center
                mathPara.paragraphSpacing = 8; mathPara.paragraphSpacingBefore = 8
                let mathShadow = NSShadow()
                mathShadow.shadowColor = NSColor.systemTeal.withAlphaComponent(0.4)
                mathShadow.shadowBlurRadius = 4
                let attrs: [NSAttributedString.Key: Any] = [
                    .font: NSFont(name: "Menlo", size: 15) ?? NSFont.monospacedSystemFont(ofSize: 15, weight: .medium),
                    .foregroundColor: NSColor.systemTeal,
                    .paragraphStyle: mathPara,
                    .shadow: mathShadow,
                    .backgroundColor: NSColor(red: 0.92, green: 0.96, blue: 0.98, alpha: 0.5)
                ]
                result.append(NSAttributedString(string: "  \(block.content)  \n", attributes: attrs))

            case .bulletList(let level):
                let indent = CGFloat(level * 16 + 12)
                let bulletPara = NSMutableParagraphStyle()
                bulletPara.lineSpacing = 2; bulletPara.paragraphSpacing = 3
                bulletPara.headIndent = indent + 12; bulletPara.firstLineHeadIndent = indent
                let bullet = level == 0 ? "▸" : level == 1 ? "◦" : "·"
                let attrs: [NSAttributedString.Key: Any] = [
                    .font: NSFont.systemFont(ofSize: 13, weight: .regular),
                    .foregroundColor: L104Theme.textPrimary,
                    .paragraphStyle: bulletPara
                ]
                result.append(NSAttributedString(string: "\(bullet) \(block.content)\n", attributes: attrs))

            case .table:
                renderTable(block.content, into: result)

            case .quote:
                let quotePara = NSMutableParagraphStyle()
                quotePara.lineSpacing = 2; quotePara.headIndent = 20; quotePara.firstLineHeadIndent = 12
                quotePara.paragraphSpacing = 4
                let attrs: [NSAttributedString.Key: Any] = [
                    .font: NSFont.systemFont(ofSize: 13, weight: .regular),
                    .foregroundColor: L104Theme.goldDim,
                    .paragraphStyle: quotePara,
                    .backgroundColor: NSColor(red: 0.98, green: 0.96, blue: 0.90, alpha: 0.4)
                ]
                result.append(NSAttributedString(string: "  ❝ \(block.content) ❞\n", attributes: attrs))

            case .separator:
                let sepAttrs: [NSAttributedString.Key: Any] = [
                    .font: NSFont.systemFont(ofSize: 8),
                    .foregroundColor: L104Theme.textDim
                ]
                result.append(NSAttributedString(string: "  ─────────────────────────────\n", attributes: sepAttrs))

            case .keyValue:
                let key = block.metadata["key"] ?? ""
                let value = block.metadata["value"] ?? ""
                let keyAttrs: [NSAttributedString.Key: Any] = [
                    .font: NSFont.systemFont(ofSize: 13, weight: .semibold),
                    .foregroundColor: L104Theme.goldBright
                ]
                let valAttrs: [NSAttributedString.Key: Any] = [
                    .font: NSFont.systemFont(ofSize: 13, weight: .regular),
                    .foregroundColor: L104Theme.textPrimary
                ]
                result.append(NSAttributedString(string: "\(key): ", attributes: keyAttrs))
                result.append(NSAttributedString(string: "\(value)\n", attributes: valAttrs))

            case .formulaResult:
                let mathShadow = NSShadow()
                mathShadow.shadowColor = NSColor.systemTeal.withAlphaComponent(0.5)
                mathShadow.shadowBlurRadius = 5
                let attrs: [NSAttributedString.Key: Any] = [
                    .font: NSFont.monospacedSystemFont(ofSize: 16, weight: .bold),
                    .foregroundColor: NSColor.systemTeal,
                    .shadow: mathShadow
                ]
                result.append(NSAttributedString(string: "\(block.content)\n", attributes: attrs))

            case .paragraph:
                let hasInline = block.metadata["hasInline"] == "true"
                if hasInline {
                    result.append(renderInlineFormatting(block.content))
                    result.append(NSAttributedString(string: "\n", attributes: [:]))
                } else {
                    // Apply bold markers **text**
                    let formatted = applyBoldMarkers(block.content)
                    let attrs: [NSAttributedString.Key: Any] = [
                        .font: NSFont.systemFont(ofSize: 13, weight: .regular),
                        .foregroundColor: L104Theme.textPrimary,
                        .paragraphStyle: defaultPara
                    ]
                    let mutStr = NSMutableAttributedString(string: formatted, attributes: attrs)
                    applyBoldRanges(mutStr)
                    result.append(mutStr)
                    result.append(NSAttributedString(string: "\n", attributes: [:]))
                }

            case .inlineCode, .inlineMath:
                break // Handled within paragraph
            }
        }

        return result
    }

    // ─── SYNTAX HIGHLIGHTING ─── Basic keyword highlighting for code
    private func syntaxHighlight(_ code: String, language: String) -> NSAttributedString {
        let result = NSMutableAttributedString(string: code)
        let fullRange = NSRange(location: 0, length: result.length)

        // Base style
        result.addAttribute(.foregroundColor, value: L104Theme.textPrimary, range: fullRange)

        // Keywords
        let keywords: [String]
        switch language.lowercased() {
        case "swift":
            keywords = ["func", "var", "let", "class", "struct", "enum", "protocol", "import",
                        "return", "if", "else", "guard", "for", "while", "switch", "case",
                        "self", "Self", "true", "false", "nil", "public", "private", "static",
                        "override", "init", "deinit", "throws", "try", "catch", "async", "await",
                        "typealias", "where", "extension", "some", "any", "inout"]
        case "python", "py":
            keywords = ["def", "class", "return", "if", "elif", "else", "for", "while",
                        "import", "from", "as", "try", "except", "finally", "with",
                        "True", "False", "None", "self", "lambda", "yield", "async", "await",
                        "and", "or", "not", "in", "is", "pass", "break", "continue", "raise"]
        case "javascript", "js", "typescript", "ts":
            keywords = ["function", "const", "let", "var", "class", "return", "if", "else",
                        "for", "while", "switch", "case", "import", "export", "from",
                        "true", "false", "null", "undefined", "this", "new", "async", "await",
                        "try", "catch", "throw", "typeof", "instanceof", "of", "in"]
        default:
            keywords = ["func", "def", "class", "return", "if", "else", "for", "while",
                        "true", "false", "nil", "null", "import", "var", "let", "const"]
        }

        // Highlight keywords — combined regex for all keywords at once
        let combined = keywords.map { NSRegularExpression.escapedPattern(for: $0) }.joined(separator: "|")
        let kwPattern = "\\b(\(combined))\\b"
        if let kwRegex = try? NSRegularExpression(pattern: kwPattern) {
            let matches = kwRegex.matches(in: code, range: fullRange)
            for match in matches {
                result.addAttribute(.foregroundColor, value: NSColor(red: 0.8, green: 0.3, blue: 0.9, alpha: 1.0), range: match.range)
                result.addAttribute(.font, value: L104Theme.monoFont(12, weight: .bold), range: match.range)
            }
        }

        // Strings
        for regex in RichTextFormatterV2.stringRegexes {
            let matches = regex.matches(in: code, range: fullRange)
            for match in matches {
                result.addAttribute(.foregroundColor, value: NSColor(red: 0.3, green: 0.8, blue: 0.3, alpha: 1.0), range: match.range)
            }
        }

        // Numbers
        let numMatches = RichTextFormatterV2.numberRegex.matches(in: code, range: fullRange)
        for match in numMatches {
            result.addAttribute(.foregroundColor, value: NSColor(red: 0.9, green: 0.7, blue: 0.2, alpha: 1.0), range: match.range)
        }

        // Comments
        for regex in RichTextFormatterV2.commentRegexes {
            let matches = regex.matches(in: code, range: fullRange)
            for match in matches {
                result.addAttribute(.foregroundColor, value: NSColor(red: 0.4, green: 0.5, blue: 0.4, alpha: 1.0), range: match.range)
            }
        }

        return result
    }

    // ─── INLINE FORMATTING ─── Handle `code` and $math$ within text
    private func renderInlineFormatting(_ text: String) -> NSAttributedString {
        let result = NSMutableAttributedString()
        var remaining = text

        while !remaining.isEmpty {
            // Look for inline code
            if let codeStart = remaining.range(of: "`") {
                let before = String(remaining[remaining.startIndex..<codeStart.lowerBound])
                if !before.isEmpty {
                    result.append(NSAttributedString(string: before, attributes: [
                        .font: NSFont.systemFont(ofSize: 13), .foregroundColor: L104Theme.textPrimary
                    ]))
                }
                let afterStart = remaining[codeStart.upperBound...]
                if let codeEnd = afterStart.range(of: "`") {
                    let code = String(afterStart[afterStart.startIndex..<codeEnd.lowerBound])
                    result.append(NSAttributedString(string: code, attributes: [
                        .font: L104Theme.monoFont(12, weight: .medium),
                        .foregroundColor: L104Theme.goldBright,
                        .backgroundColor: NSColor(red: 0.94, green: 0.93, blue: 0.96, alpha: 0.8)
                    ]))
                    remaining = String(afterStart[codeEnd.upperBound...])
                    continue
                }
            }

            // Look for inline math
            if let mathStart = remaining.range(of: "$") {
                let before = String(remaining[remaining.startIndex..<mathStart.lowerBound])
                if !before.isEmpty {
                    result.append(NSAttributedString(string: before, attributes: [
                        .font: NSFont.systemFont(ofSize: 13), .foregroundColor: L104Theme.textPrimary
                    ]))
                }
                let afterStart = remaining[mathStart.upperBound...]
                if let mathEnd = afterStart.range(of: "$") {
                    let math = String(afterStart[afterStart.startIndex..<mathEnd.lowerBound])
                    result.append(NSAttributedString(string: math, attributes: [
                        .font: NSFont(name: "Menlo", size: 13) ?? NSFont.monospacedSystemFont(ofSize: 13, weight: .medium),
                        .foregroundColor: NSColor.systemTeal
                    ]))
                    remaining = String(afterStart[mathEnd.upperBound...])
                    continue
                }
            }

            // No more special chars
            result.append(NSAttributedString(string: remaining, attributes: [
                .font: NSFont.systemFont(ofSize: 13), .foregroundColor: L104Theme.textPrimary
            ]))
            break
        }

        return result
    }

    // ─── TABLE RENDERER ───
    private func renderTable(_ content: String, into result: NSMutableAttributedString) {
        let rows = content.components(separatedBy: "\n")
            .map { row in
                row.components(separatedBy: "|")
                    .map { $0.trimmingCharacters(in: .whitespaces) }
                    .filter { !$0.isEmpty }
            }
            .filter { !$0.isEmpty && !$0.allSatisfy({ $0.allSatisfy({ $0 == "-" || $0 == ":" }) }) }

        let tablePara = NSMutableParagraphStyle()
        tablePara.lineSpacing = 1; tablePara.paragraphSpacing = 2

        for (rowIdx, row) in rows.enumerated() {
            let isHeader = rowIdx == 0
            let attrs: [NSAttributedString.Key: Any] = [
                .font: isHeader ? NSFont.monospacedSystemFont(ofSize: 11, weight: .bold) : NSFont.monospacedSystemFont(ofSize: 11, weight: .regular),
                .foregroundColor: isHeader ? L104Theme.goldBright : L104Theme.textPrimary,
                .paragraphStyle: tablePara
            ]
            let line = "  " + row.map { $0.padding(toLength: 16, withPad: " ", startingAt: 0) }.joined(separator: " │ ")
            result.append(NSAttributedString(string: "\(line)\n", attributes: attrs))
            if isHeader {
                let sep = "  " + row.map { _ in String(repeating: "─", count: 16) }.joined(separator: "─┼─")
                result.append(NSAttributedString(string: "\(sep)\n", attributes: [
                    .font: NSFont.monospacedSystemFont(ofSize: 11, weight: .regular),
                    .foregroundColor: L104Theme.textDim
                ]))
            }
        }
    }

    // ─── BOLD MARKERS ─── Apply **bold** formatting
    private func applyBoldMarkers(_ text: String) -> String {
        text.replacingOccurrences(of: "**", with: "")
    }

    private func applyBoldRanges(_ attrStr: NSMutableAttributedString) {
        // Already stripped ** markers in the text, so skip for now
        // In a full implementation, would track positions and apply .bold font weight
    }

    var status: String {
        """
        ╔═══════════════════════════════════════════════════════════╗
        ║  ✨ RICH TEXT FORMATTER v29.0                              ║
        ╠═══════════════════════════════════════════════════════════╣
        ║  Formats Rendered: \(formattingCount)
        ║  Capabilities:
        ║    • Headers (H1/H2/H3) with glow effects
        ║    • Code blocks with syntax highlighting
        ║    • Math blocks with centered cyan rendering
        ║    • Inline code and inline math
        ║    • Bullet lists (multi-level)
        ║    • Tables with header formatting
        ║    • Block quotes with gold styling
        ║    • Key-Value pairs for scientific output
        ║    • Formula results with neon glow
        ╚═══════════════════════════════════════════════════════════╝
        """
    }
}
