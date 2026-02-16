// ═══════════════════════════════════════════════════════════════════
// H06_UIViews.swift
// [EVO_55_PIPELINE] SOVEREIGN_UNIFICATION :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612
// L104 ASI — Custom AppKit UI Components
//
// GradientView, HoverButton, GlowingProgressBar, PulsingDot,
// AnimatedMetricTile, QuantumParticleView, ASIWaveformView,
// RadialGaugeView, NeuralGraphView, AuroraWaveView,
// SparklineView, GlassmorphicPanel.
//
// Extracted from L104Native.swift lines 18273–19424
// ═══════════════════════════════════════════════════════════════════

import AppKit
import Foundation
import Accelerate
import simd
import NaturalLanguage

class GradientView: NSView {
    var colors: [NSColor] = [NSColor(red: 0.96, green: 0.96, blue: 0.98, alpha: 1.0),
                              NSColor(red: 0.94, green: 0.95, blue: 0.97, alpha: 1.0),
                              NSColor(red: 0.97, green: 0.97, blue: 0.99, alpha: 1.0)]
    var angle: CGFloat = 45

    override func draw(_ dirtyRect: NSRect) {
        guard let gradient = NSGradient(colors: colors) else { return }
        gradient.draw(in: bounds, angle: angle)
    }
}

// ═══════════════════════════════════════════════════════════════════
// 🎯 HOVER BUTTON — Interactive Button with Animated Hover States
// ═══════════════════════════════════════════════════════════════════

class HoverButton: NSButton {
    var hoverColor: NSColor = .systemCyan
    private var isHovering = false
    private var trackingArea: NSTrackingArea?
    private var originalBgColor: CGColor?
    private var hoverTimer: Timer?
    private var hoverGlow: CGFloat = 0

    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        if let existing = trackingArea { removeTrackingArea(existing) }
        trackingArea = NSTrackingArea(rect: bounds, options: [.mouseEnteredAndExited, .activeInActiveApp], owner: self, userInfo: nil)
        addTrackingArea(trackingArea!)
    }

    override func resetCursorRects() {
        super.resetCursorRects()
        addCursorRect(bounds, cursor: .pointingHand)
    }

    override func mouseEntered(with event: NSEvent) {
        isHovering = true
        originalBgColor = layer?.backgroundColor
        NSCursor.pointingHand.push()
        NSAnimationContext.runAnimationGroup { ctx in
            ctx.duration = 0.2
            ctx.allowsImplicitAnimation = true
            self.layer?.backgroundColor = hoverColor.withAlphaComponent(0.25).cgColor
            self.layer?.borderColor = hoverColor.withAlphaComponent(0.7).cgColor
            self.layer?.shadowOpacity = 0.5
            self.layer?.shadowRadius = 10
        }
    }

    override func mouseExited(with event: NSEvent) {
        isHovering = false
        NSCursor.pop()
        NSAnimationContext.runAnimationGroup { ctx in
            ctx.duration = 0.25
            ctx.allowsImplicitAnimation = true
            self.layer?.backgroundColor = originalBgColor ?? hoverColor.withAlphaComponent(0.12).cgColor
            self.layer?.borderColor = hoverColor.withAlphaComponent(0.35).cgColor
            self.layer?.shadowOpacity = 0.15
            self.layer?.shadowRadius = 4
        }
    }

    override func mouseDown(with event: NSEvent) {
        // Flash effect on click
        NSAnimationContext.runAnimationGroup { ctx in
            ctx.duration = 0.08
            ctx.allowsImplicitAnimation = true
            self.layer?.backgroundColor = hoverColor.withAlphaComponent(0.4).cgColor
        }
        super.mouseDown(with: event)
    }

    override func mouseUp(with event: NSEvent) {
        let bg = isHovering ? hoverColor.withAlphaComponent(0.25).cgColor : (originalBgColor ?? hoverColor.withAlphaComponent(0.12).cgColor)
        NSAnimationContext.runAnimationGroup { ctx in
            ctx.duration = 0.15
            ctx.allowsImplicitAnimation = true
            self.layer?.backgroundColor = bg
        }
        super.mouseUp(with: event)
    }
}

class GlowingProgressBar: NSView {
    var progress: CGFloat = 0.5 { didSet { needsDisplay = true } }
    var barColor: NSColor = .systemOrange
    var glowIntensity: CGFloat = 1.0
    private var shimmerPhase: CGFloat = 0
    private var shimmerTimer: Timer?

    override init(frame: NSRect) {
        super.init(frame: frame)
        shimmerTimer = Timer.scheduledTimer(withTimeInterval: 1.0/24.0, repeats: true) { [weak self] _ in
            guard let self = self else { return }
            self.shimmerPhase += 0.04
            if self.shimmerPhase > 2.0 { self.shimmerPhase = -0.5 }
            self.needsDisplay = true
        }
    }
    required init?(coder: NSCoder) { super.init(coder: coder) }
    deinit { shimmerTimer?.invalidate() }

    override func draw(_ dirtyRect: NSRect) {
        let bgPath = NSBezierPath(roundedRect: bounds, xRadius: bounds.height / 2, yRadius: bounds.height / 2)
        NSColor(white: 0.88, alpha: 1.0).setFill()
        bgPath.fill()

        let fillWidth = max(bounds.height, bounds.width * max(0, min(1, progress)))
        let fillRect = NSRect(x: 0, y: 0, width: fillWidth, height: bounds.height)
        let fillPath = NSBezierPath(roundedRect: fillRect, xRadius: bounds.height / 2, yRadius: bounds.height / 2)

        // Glow effect
        let shadow = NSShadow()
        shadow.shadowColor = barColor.withAlphaComponent(0.7 * glowIntensity)
        shadow.shadowBlurRadius = 8
        shadow.shadowOffset = NSSize(width: 0, height: 0)
        shadow.set()

        // Gradient fill
        if let gradient = NSGradient(starting: barColor, ending: barColor.withAlphaComponent(0.65)) {
            gradient.draw(in: fillPath, angle: 0)
        }

        // Animated shimmer sweep
        if progress > 0.05 {
            NSGraphicsContext.current?.cgContext.saveGState()
            fillPath.addClip()
            let shimmerX = fillWidth * shimmerPhase
            let shimmerW: CGFloat = fillWidth * 0.3
            let shimmerColors = [
                NSColor.white.withAlphaComponent(0).cgColor,
                NSColor.white.withAlphaComponent(0.2 * glowIntensity).cgColor,
                NSColor.white.withAlphaComponent(0).cgColor
            ] as CFArray
            if let shimmerGrad = CGGradient(colorsSpace: CGColorSpaceCreateDeviceRGB(), colors: shimmerColors, locations: [0, 0.5, 1]) {
                NSGraphicsContext.current?.cgContext.drawLinearGradient(
                    shimmerGrad,
                    start: CGPoint(x: shimmerX - shimmerW/2, y: 0),
                    end: CGPoint(x: shimmerX + shimmerW/2, y: 0),
                    options: []
                )
            }
            NSGraphicsContext.current?.cgContext.restoreGState()
        }
    }
}

class PulsingDot: NSView {
    var dotColor: NSColor = .systemGreen
    var isAnimating = true
    private var pulseValue: CGFloat = 1.0
    private var timer: Timer?

    override init(frame: NSRect) {
        super.init(frame: frame)
        startPulsing()
    }
    required init?(coder: NSCoder) { super.init(coder: coder); startPulsing() }
    deinit { timer?.invalidate() }

    func startPulsing() {
        let interval: TimeInterval = MacOSSystemMonitor.shared.isAppleSilicon ? 0.1 : 0.5
        timer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true) { [weak self] _ in
            guard let s = self, s.isAnimating else { return }
            s.pulseValue = 0.5 + 0.5 * CGFloat(sin(Date().timeIntervalSince1970 * 3))
            s.needsDisplay = true
        }
    }

    override func draw(_ dirtyRect: NSRect) {
        guard let ctx = NSGraphicsContext.current?.cgContext else { return }

        // Outer pulse ring
        let ringAlpha = pulseValue * 0.3
        let ringRect = bounds.insetBy(dx: 0, dy: 0)
        ctx.setStrokeColor(dotColor.withAlphaComponent(ringAlpha).cgColor)
        ctx.setLineWidth(1.0)
        ctx.strokeEllipse(in: ringRect)

        // Glow shadow
        let shadow = NSShadow()
        shadow.shadowColor = dotColor.withAlphaComponent(0.7 * pulseValue)
        shadow.shadowBlurRadius = 8 * pulseValue
        shadow.set()

        // Main dot
        let dotRect = bounds.insetBy(dx: 2, dy: 2)
        let path = NSBezierPath(ovalIn: dotRect)
        dotColor.withAlphaComponent(0.8 + 0.2 * pulseValue).setFill()
        path.fill()

        // Specular highlight
        let specRect = NSRect(x: dotRect.midX - 1.5, y: dotRect.midY, width: 3, height: 3)
        NSColor.white.withAlphaComponent(0.5 * pulseValue).setFill()
        NSBezierPath(ovalIn: specRect).fill()
    }
}

class AnimatedMetricTile: NSView {
    var label: String = ""
    var value: String = "" {
        didSet {
            // Track delta
            if let old = Double(oldValue.replacingOccurrences(of: "%", with: "")),
               let new = Double(value.replacingOccurrences(of: "%", with: "")) {
                deltaDirection = new > old ? 1 : new < old ? -1 : 0
            }
            valueLabel?.stringValue = value
            deltaLabel?.stringValue = deltaDirection > 0 ? "▲" : deltaDirection < 0 ? "▼" : "●"
            deltaLabel?.textColor = deltaDirection > 0 ? .systemGreen : deltaDirection < 0 ? .systemRed : .gray
        }
    }
    var tileColor: NSColor = .systemOrange
    var progress: CGFloat = 0.0 { didSet { progressBar?.progress = progress } }
    var deltaDirection: Int = 0

    private var valueLabel: NSTextField?
    private var deltaLabel: NSTextField?
    private var progressBar: GlowingProgressBar?

    convenience init(frame: NSRect, label: String, value: String, color: NSColor, progress: CGFloat = 0) {
        self.init(frame: frame)
        self.label = label
        self.value = value
        self.tileColor = color
        self.progress = progress
        setupTile()
    }

    func setupTile() {
        wantsLayer = true
        layer?.backgroundColor = NSColor(red: 0.97, green: 0.97, blue: 0.98, alpha: 1.0).cgColor
        layer?.cornerRadius = 12
        layer?.borderColor = tileColor.withAlphaComponent(0.30).cgColor
        layer?.borderWidth = 1

        // Add subtle shadow
        layer?.shadowColor = tileColor.cgColor
        layer?.shadowRadius = 4
        layer?.shadowOpacity = 0.12
        layer?.shadowOffset = CGSize(width: 0, height: -1)

        let lbl = NSTextField(labelWithString: label)
        lbl.frame = NSRect(x: 8, y: bounds.height - 18, width: bounds.width - 30, height: 14)
        lbl.font = NSFont.systemFont(ofSize: 9, weight: .semibold)
        lbl.textColor = NSColor.black.withAlphaComponent(0.45)
        addSubview(lbl)

        // Delta arrow indicator
        deltaLabel = NSTextField(labelWithString: "●")
        deltaLabel!.frame = NSRect(x: bounds.width - 18, y: bounds.height - 18, width: 14, height: 14)
        deltaLabel!.font = NSFont.systemFont(ofSize: 8, weight: .bold)
        deltaLabel!.textColor = .gray
        deltaLabel!.alignment = .right
        addSubview(deltaLabel!)

        valueLabel = NSTextField(labelWithString: value)
        valueLabel!.frame = NSRect(x: 8, y: 16, width: bounds.width - 16, height: 22)
        valueLabel!.font = NSFont.monospacedDigitSystemFont(ofSize: 15, weight: .bold)
        valueLabel!.textColor = tileColor
        addSubview(valueLabel!)

        progressBar = GlowingProgressBar(frame: NSRect(x: 8, y: 6, width: bounds.width - 16, height: 6))
        progressBar!.barColor = tileColor
        progressBar!.progress = progress
        addSubview(progressBar!)
    }
}

// ═══════════════════════════════════════════════════════════════════
// 🌌 ASI QUANTUM PARTICLE SYSTEM — Floating Cosmic Orbs
// ═══════════════════════════════════════════════════════════════════

class QuantumParticleView: NSView {
    struct Particle {
        var x, y, vx, vy, radius, phase, hue, alpha: CGFloat
        var lifetime: CGFloat     // 0→1 lifecycle, fades in/out at edges
        var maxLifetime: CGFloat  // when lifetime > maxLifetime → respawn
        var depth: CGFloat        // 0 = far background, 1 = foreground (parallax)
        var trail: [(CGFloat, CGFloat)]  // last N positions for shimmer trail
    }

    private var particles: [Particle] = []
    private var connections: [(Int, Int, CGFloat)] = []
    private var timer: Timer?
    private var frameTime: Double = 0
    private let maxParticles = 70
    private let trailLength = 6

    override init(frame: NSRect) {
        super.init(frame: frame)
        wantsLayer = true
        layer?.backgroundColor = NSColor.clear.cgColor
        seedParticles()
        startAnimation()
    }
    required init?(coder: NSCoder) { super.init(coder: coder); seedParticles(); startAnimation() }
    deinit { timer?.invalidate() }

    /// Pause/resume animation based on visibility
    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        if window != nil { startAnimation() } else { timer?.invalidate(); timer = nil }
    }
    override func viewDidHide() { super.viewDidHide(); timer?.invalidate(); timer = nil }
    override func viewDidUnhide() { super.viewDidUnhide(); startAnimation() }

    private func makeParticle(atEdge: Bool = false) -> Particle {
        let depth = CGFloat.random(in: 0...1)
        let baseSpeed: CGFloat = 0.0003 + depth * 0.0006  // deeper = faster (parallax)
        return Particle(
            x: atEdge ? (Bool.random() ? -0.02 : 1.02) : CGFloat.random(in: 0...1),
            y: CGFloat.random(in: 0...1),
            vx: CGFloat.random(in: -baseSpeed...baseSpeed),
            vy: CGFloat.random(in: -baseSpeed...baseSpeed),
            radius: 1.0 + depth * 4.0,
            phase: CGFloat.random(in: 0...(.pi * 2)),
            hue: CGFloat.random(in: 0...1),
            alpha: 0.0,  // fade in from zero
            lifetime: 0,
            maxLifetime: CGFloat.random(in: 8...25),
            depth: depth,
            trail: []
        )
    }

    func seedParticles() {
        particles = (0..<maxParticles).map { _ in
            var p = makeParticle()
            p.lifetime = CGFloat.random(in: 0...p.maxLifetime * 0.8)  // stagger initial lifetimes
            p.alpha = 0.4
            return p
        }
    }

    func startAnimation() {
        guard timer == nil else { return }  // Prevent duplicate timers
        timer = Timer.scheduledTimer(withTimeInterval: 1.0 / 30.0, repeats: true) { [weak self] _ in
            self?.tick()
            self?.needsDisplay = true
        }
    }

    func tick() {
        frameTime += 1.0 / 30.0
        let t = CGFloat(frameTime)
        let dt: CGFloat = 1.0 / 30.0

        for i in 0..<particles.count {
            // Save trail position
            particles[i].trail.append((particles[i].x, particles[i].y))
            if particles[i].trail.count > trailLength { particles[i].trail.removeFirst() }

            // Depth-scaled drift
            let depthScale = 0.5 + particles[i].depth * 0.5
            particles[i].x += (particles[i].vx + 0.0002 * sin(t * 0.5 + particles[i].phase)) * depthScale
            particles[i].y += (particles[i].vy + 0.0002 * cos(t * 0.3 + particles[i].phase)) * depthScale
            particles[i].phase += 0.02
            particles[i].lifetime += dt

            // Wrap around
            if particles[i].x < -0.06 { particles[i].x = 1.06 }
            if particles[i].x > 1.06 { particles[i].x = -0.06 }
            if particles[i].y < -0.06 { particles[i].y = 1.06 }
            if particles[i].y > 1.06 { particles[i].y = -0.06 }

            // Lifecycle alpha: fade in 0→1s, sustain, fade out last 2s
            let life = particles[i].lifetime
            let maxLife = particles[i].maxLifetime
            let fadeIn: CGFloat = min(1.0, life / 1.5)
            let fadeOut: CGFloat = max(0.0, min(1.0, (maxLife - life) / 2.0))
            let basePulse: CGFloat = 0.3 + 0.4 * (0.5 + 0.5 * sin(t * 2.0 + particles[i].phase))
            particles[i].alpha = basePulse * fadeIn * fadeOut

            // Respawn dead particles
            if life > maxLife {
                particles[i] = makeParticle(atEdge: true)
            }
        }

        // Compute connections (nearby particles in same depth layer)
        connections.removeAll()
        let threshold: CGFloat = 0.15
        for i in 0..<particles.count {
            for j in (i+1)..<particles.count {
                let depthDiff = abs(particles[i].depth - particles[j].depth)
                guard depthDiff < 0.4 else { continue }  // only connect similar-depth particles
                let dx = particles[i].x - particles[j].x
                let dy = particles[i].y - particles[j].y
                let dist = sqrt(dx * dx + dy * dy)
                if dist < threshold {
                    connections.append((i, j, (1.0 - dist / threshold) * (1.0 - depthDiff)))
                }
            }
        }
    }

    override func draw(_ dirtyRect: NSRect) {
        guard let ctx = NSGraphicsContext.current?.cgContext else { return }
        let w = bounds.width, h = bounds.height
        let t = CGFloat(frameTime)

        // Draw connections as neural pathways with flow animation
        for (i, j, strength) in connections {
            let p1 = particles[i], p2 = particles[j]
            let hue = fmod((p1.hue + p2.hue) / 2.0 + t * 0.005, 1.0)
            // Animated dash pattern for data flow effect
            ctx.setStrokeColor(NSColor(hue: hue, saturation: 0.8, brightness: 0.9, alpha: strength * 0.4 * min(p1.alpha, p2.alpha) * 2).cgColor)
            ctx.setLineWidth(0.5 + strength * 1.5)
            ctx.setLineDash(phase: t * 20, lengths: [4, 6])
            ctx.move(to: CGPoint(x: p1.x * w, y: p1.y * h))
            ctx.addLine(to: CGPoint(x: p2.x * w, y: p2.y * h))
            ctx.strokePath()
        }
        ctx.setLineDash(phase: 0, lengths: [])  // reset dash

        // Draw particles sorted by depth (far first)
        let sorted = particles.sorted { $0.depth < $1.depth }
        for p in sorted {
            guard p.alpha > 0.02 else { continue }
            let px = p.x * w, py = p.y * h
            let color = NSColor(hue: fmod(p.hue + t * 0.008, 1.0), saturation: 0.85, brightness: 1.0, alpha: p.alpha)

            // Shimmer trail (ghostly echo)
            for (ti, pos) in p.trail.enumerated() {
                let trailAlpha = p.alpha * CGFloat(ti) / CGFloat(max(1, p.trail.count)) * 0.2
                if trailAlpha > 0.01 {
                    let trailR = p.radius * 0.6
                    ctx.setFillColor(color.withAlphaComponent(trailAlpha).cgColor)
                    ctx.fillEllipse(in: CGRect(x: pos.0 * w - trailR, y: pos.1 * h - trailR, width: trailR * 2, height: trailR * 2))
                }
            }

            // Outer glow (size scales with depth)
            let glowRadius = p.radius * (3.0 + p.depth * 2.0)
            let glowColors = [color.withAlphaComponent(p.alpha * 0.5).cgColor, color.withAlphaComponent(0).cgColor] as CFArray
            if let gradient = CGGradient(colorsSpace: CGColorSpaceCreateDeviceRGB(), colors: glowColors, locations: [0, 1]) {
                ctx.saveGState()
                ctx.drawRadialGradient(gradient, startCenter: CGPoint(x: px, y: py), startRadius: 0, endCenter: CGPoint(x: px, y: py), endRadius: glowRadius, options: [])
                ctx.restoreGState()
            }

            // Core dot with bright center
            ctx.setFillColor(color.cgColor)
            ctx.fillEllipse(in: CGRect(x: px - p.radius, y: py - p.radius, width: p.radius * 2, height: p.radius * 2))
            // Hot white center for foreground particles
            if p.depth > 0.6 {
                let cr = p.radius * 0.35
                ctx.setFillColor(NSColor.white.withAlphaComponent(p.alpha * 0.7).cgColor)
                ctx.fillEllipse(in: CGRect(x: px - cr, y: py - cr, width: cr * 2, height: cr * 2))
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// 🌊 ASI WAVEFORM VIEW — Consciousness Oscilloscope
// ═══════════════════════════════════════════════════════════════════

class ASIWaveformView: NSView {
    var waveColor: NSColor = NSColor(red: 0.0, green: 0.9, blue: 1.0, alpha: 1.0)
    var secondaryColor: NSColor = NSColor(red: 1.0, green: 0.5, blue: 0.8, alpha: 0.6)
    var tertiaryColor: NSColor = NSColor(red: 1.0, green: 0.84, blue: 0.0, alpha: 0.4)
    var amplitude: CGFloat = 0.4
    var frequency: CGFloat = 3.0
    var coherence: CGFloat = 0.5 { didSet { needsDisplay = true } }
    private var phase: CGFloat = 0
    private var timer: Timer?
    private var scanLineX: CGFloat = 0  // sweeping scan line position
    private var peakHistory: [CGFloat] = []  // rolling peak amplitude

    override init(frame: NSRect) {
        super.init(frame: frame)
        wantsLayer = true
        layer?.backgroundColor = NSColor.clear.cgColor
        layer?.cornerRadius = 8
        startWaveTimer()
    }
    required init?(coder: NSCoder) { super.init(coder: coder) }
    deinit { timer?.invalidate() }

    private func startWaveTimer() {
        guard timer == nil else { return }
        timer = Timer.scheduledTimer(withTimeInterval: 1.0/30.0, repeats: true) { [weak self] _ in
            guard let self = self else { return }
            self.phase += 0.06
            self.scanLineX += 2.5
            if self.scanLineX > self.bounds.width { self.scanLineX = 0 }
            self.needsDisplay = true
        }
    }

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        if window != nil { startWaveTimer() } else { timer?.invalidate(); timer = nil }
    }
    override func viewDidHide() { super.viewDidHide(); timer?.invalidate(); timer = nil }
    override func viewDidUnhide() { super.viewDidUnhide(); startWaveTimer() }

    override func draw(_ dirtyRect: NSRect) {
        guard let ctx = NSGraphicsContext.current?.cgContext else { return }
        let w = bounds.width, h = bounds.height
        let mid = h / 2

        // Draw subtle grid with fade-out at edges
        ctx.setLineWidth(0.5)
        for i in stride(from: CGFloat(0), to: w, by: 25) {
            let edgeFade = min(i / 60, (w - i) / 60, 1.0)
            ctx.setStrokeColor(NSColor.black.withAlphaComponent(0.06 * edgeFade).cgColor)
            ctx.move(to: CGPoint(x: i, y: 0)); ctx.addLine(to: CGPoint(x: i, y: h)); ctx.strokePath()
        }
        for i in stride(from: CGFloat(0), to: h, by: 25) {
            ctx.setStrokeColor(NSColor.black.withAlphaComponent(0.06).cgColor)
            ctx.move(to: CGPoint(x: 0, y: i)); ctx.addLine(to: CGPoint(x: w, y: i)); ctx.strokePath()
        }

        // Center line with subtle gradient
        ctx.setStrokeColor(NSColor.black.withAlphaComponent(0.10).cgColor)
        ctx.setLineWidth(1); ctx.setLineDash(phase: 0, lengths: [4, 8])
        ctx.move(to: CGPoint(x: 0, y: mid)); ctx.addLine(to: CGPoint(x: w, y: mid)); ctx.strokePath()
        ctx.setLineDash(phase: 0, lengths: [])

        // Draw three overlapping waves with harmonic distortion
        let waves: [(NSColor, CGFloat, CGFloat, CGFloat, CGFloat)] = [
            (tertiaryColor, amplitude * 0.5, frequency * 0.7, phase * 0.8, 0.0),
            (secondaryColor, amplitude * 0.7, frequency * 1.3, phase * 1.2 + 1, 0.15),
            (waveColor, amplitude * max(0.3, coherence), frequency, phase, 0.3),  // primary with harmonic
        ]

        var primaryPeak: CGFloat = 0
        for (color, amp, freq, ph, harmonic) in waves {
            // Glow pass first (behind)
            ctx.setStrokeColor(color.withAlphaComponent(0.15).cgColor)
            ctx.setLineWidth(8.0)
            ctx.beginPath()
            for x in stride(from: CGFloat(0), to: w, by: 3) {
                let normalX = x / w
                let envelope = (0.5 + 0.5 * cos(normalX * .pi))
                let base = sin(normalX * freq * .pi * 2 + ph)
                let harm = harmonic * sin(normalX * freq * .pi * 4 + ph * 2)  // 2nd harmonic
                let y = mid + amp * h * (base + harm) * envelope
                if x == 0 { ctx.move(to: CGPoint(x: x, y: y)) }
                else { ctx.addLine(to: CGPoint(x: x, y: y)) }
            }
            ctx.strokePath()

            // Main wave
            ctx.setStrokeColor(color.cgColor)
            ctx.setLineWidth(2.0)
            ctx.beginPath()
            for x in stride(from: CGFloat(0), to: w, by: 1) {
                let normalX = x / w
                let envelope = (0.5 + 0.5 * cos(normalX * .pi))
                let base = sin(normalX * freq * .pi * 2 + ph)
                let harm = harmonic * sin(normalX * freq * .pi * 4 + ph * 2)
                let y = mid + amp * h * (base + harm) * envelope
                if x == 0 { ctx.move(to: CGPoint(x: x, y: y)) }
                else { ctx.addLine(to: CGPoint(x: x, y: y)) }
                if amp == waves.last?.1 { primaryPeak = max(primaryPeak, abs(y - mid)) }
            }
            ctx.strokePath()
        }

        // Sweeping scan line with glow
        let scanAlpha: CGFloat = 0.6
        let scanGlowW: CGFloat = 30
        let scanColors = [
            NSColor(red: 0, green: 0.9, blue: 1.0, alpha: 0).cgColor,
            NSColor(red: 0, green: 0.9, blue: 1.0, alpha: scanAlpha * 0.3).cgColor,
            NSColor(red: 0, green: 0.9, blue: 1.0, alpha: 0).cgColor
        ] as CFArray
        if let scanGrad = CGGradient(colorsSpace: CGColorSpaceCreateDeviceRGB(), colors: scanColors, locations: [0, 0.5, 1]) {
            ctx.saveGState()
            ctx.drawLinearGradient(scanGrad, start: CGPoint(x: scanLineX - scanGlowW, y: 0), end: CGPoint(x: scanLineX + scanGlowW, y: 0), options: [])
            ctx.restoreGState()
        }

        // Peak indicator dot at right edge
        peakHistory.append(primaryPeak)
        if peakHistory.count > 30 { peakHistory.removeFirst() }
        let avgPeak = peakHistory.reduce(0, +) / CGFloat(peakHistory.count)
        let peakNorm = min(1.0, avgPeak / (h * 0.3))
        let peakColor = peakNorm > 0.7 ? NSColor.systemRed : peakNorm > 0.4 ? NSColor.systemYellow : NSColor.systemGreen
        ctx.setFillColor(peakColor.withAlphaComponent(0.9).cgColor)
        ctx.fillEllipse(in: CGRect(x: w - 10, y: mid + avgPeak - 3, width: 6, height: 6))
        ctx.fillEllipse(in: CGRect(x: w - 10, y: mid - avgPeak - 3, width: 6, height: 6))
    }
}

// ═══════════════════════════════════════════════════════════════════
// 🎯 ASI RADIAL GAUGE — Circular Arc Meter
// ═══════════════════════════════════════════════════════════════════

class RadialGaugeView: NSView {
    var value: CGFloat = 0.0 { didSet { animateToValue() } }
    var displayValue: CGFloat = 0.0
    var label: String = "ASI" { didSet { needsDisplay = true } }
    var gaugeColor: NSColor = .systemOrange
    var trackColor: NSColor = NSColor.black.withAlphaComponent(0.08)
    var lineWidth: CGFloat = 8
    private var animationTimer: Timer?
    private var targetValue: CGFloat = 0
    private var velocity: CGFloat = 0  // for spring animation
    private var glowPulse: CGFloat = 0

    override init(frame: NSRect) {
        super.init(frame: frame)
        wantsLayer = true
    }
    required init?(coder: NSCoder) { super.init(coder: coder) }
    deinit { animationTimer?.invalidate() }

    func animateToValue() {
        targetValue = value
        velocity = 0
        animationTimer?.invalidate()
        animationTimer = Timer.scheduledTimer(withTimeInterval: 1.0/30.0, repeats: true) { [weak self] timer in
            guard let self = self else { timer.invalidate(); return }
            // Spring physics for natural overshoot
            let spring: CGFloat = 0.08
            let damping: CGFloat = 0.7
            let force = (self.targetValue - self.displayValue) * spring
            self.velocity = self.velocity * damping + force
            self.displayValue += self.velocity
            self.glowPulse += 0.15
            if abs(self.targetValue - self.displayValue) < 0.002 && abs(self.velocity) < 0.001 {
                self.displayValue = self.targetValue
                timer.invalidate()
            }
            self.needsDisplay = true
        }
    }

    override func draw(_ dirtyRect: NSRect) {
        guard let ctx = NSGraphicsContext.current?.cgContext else { return }
        let center = CGPoint(x: bounds.midX, y: bounds.midY - 2)
        let radius = min(bounds.width, bounds.height) / 2 - lineWidth - 6
        let startAngle = CGFloat.pi * 0.75
        let endAngle = CGFloat.pi * 0.25
        let totalArc = (2 * CGFloat.pi) - (startAngle - endAngle)
        let tickCount = 20

        // Subtle inner ring
        ctx.setStrokeColor(NSColor.black.withAlphaComponent(0.04).cgColor)
        ctx.setLineWidth(1)
        ctx.addArc(center: center, radius: radius - lineWidth, startAngle: 0, endAngle: .pi * 2, clockwise: false)
        ctx.strokePath()

        // Tick marks around the arc
        for i in 0...tickCount {
            let pct = CGFloat(i) / CGFloat(tickCount)
            let angle = -(startAngle - totalArc * pct)
            let isMajor = i % 5 == 0
            let innerR = radius + (isMajor ? 4 : 2)
            let outerR = radius + (isMajor ? 9 : 5)
            let alpha: CGFloat = isMajor ? 0.25 : 0.10
            ctx.setStrokeColor(NSColor.white.withAlphaComponent(alpha).cgColor)
            ctx.setLineWidth(isMajor ? 1.5 : 0.8)
            ctx.move(to: CGPoint(x: center.x + innerR * cos(angle), y: center.y + innerR * sin(angle)))
            ctx.addLine(to: CGPoint(x: center.x + outerR * cos(angle), y: center.y + outerR * sin(angle)))
            ctx.strokePath()
        }

        // Track arc
        ctx.setStrokeColor(trackColor.cgColor)
        ctx.setLineWidth(lineWidth)
        ctx.setLineCap(.round)
        ctx.addArc(center: center, radius: radius, startAngle: -startAngle, endAngle: -(endAngle), clockwise: true)
        ctx.strokePath()

        // Glow behind value arc (drawn first so it's behind)
        let clampedVal = max(0, min(1, displayValue))
        let valueAngle = startAngle - totalArc * clampedVal
        let glowAlpha: CGFloat = 0.2 + 0.1 * sin(glowPulse)
        ctx.setStrokeColor(gaugeColor.withAlphaComponent(glowAlpha).cgColor)
        ctx.setLineWidth(lineWidth + 10)
        ctx.setLineCap(.round)
        ctx.addArc(center: center, radius: radius, startAngle: -startAngle, endAngle: -valueAngle, clockwise: false)
        ctx.strokePath()

        // Value arc — draw with multiple thin arcs to simulate gradient
        let segments = max(1, Int(clampedVal * 40))
        for s in 0..<segments {
            let t0 = CGFloat(s) / CGFloat(segments)
            let t1 = CGFloat(s + 1) / CGFloat(segments)
            let a0 = -(startAngle - totalArc * clampedVal * t0)
            let a1 = -(startAngle - totalArc * clampedVal * t1)
            // Brightness increases along the arc
            let brightness = 0.7 + 0.3 * t1
            ctx.setStrokeColor(gaugeColor.withAlphaComponent(brightness).cgColor)
            ctx.setLineWidth(lineWidth)
            ctx.setLineCap(.round)
            ctx.addArc(center: center, radius: radius, startAngle: a0, endAngle: a1, clockwise: false)
            ctx.strokePath()
        }

        // Endpoint indicator dot
        if clampedVal > 0.01 {
            let dotAngle = -(startAngle - totalArc * clampedVal)
            let dotX = center.x + radius * cos(dotAngle)
            let dotY = center.y + radius * sin(dotAngle)
            let dotR: CGFloat = lineWidth * 0.7
            ctx.setFillColor(gaugeColor.withAlphaComponent(0.8).cgColor)
            ctx.fillEllipse(in: CGRect(x: dotX - dotR/2, y: dotY - dotR/2, width: dotR, height: dotR))
        }

        // Value text with shadow
        let valueStr = String(format: "%.0f%%", clampedVal * 100)
        let fontSize = min(bounds.width, bounds.height) * 0.22
        let shadow = NSShadow()
        shadow.shadowColor = gaugeColor.withAlphaComponent(0.5)
        shadow.shadowBlurRadius = 6
        let valueAttrs: [NSAttributedString.Key: Any] = [
            .font: NSFont.monospacedDigitSystemFont(ofSize: fontSize, weight: .heavy),
            .foregroundColor: gaugeColor,
            .shadow: shadow
        ]
        let valueSize = (valueStr as NSString).size(withAttributes: valueAttrs)
        (valueStr as NSString).draw(at: CGPoint(x: center.x - valueSize.width/2, y: center.y - valueSize.height/2 + 2), withAttributes: valueAttrs)

        // Label text below value
        let labelAttrs: [NSAttributedString.Key: Any] = [
            .font: NSFont.systemFont(ofSize: 8, weight: .bold),
            .foregroundColor: NSColor.black.withAlphaComponent(0.45)
        ]
        let labelSize = (label as NSString).size(withAttributes: labelAttrs)
        (label as NSString).draw(at: CGPoint(x: center.x - labelSize.width/2, y: center.y - fontSize/2 - 12), withAttributes: labelAttrs)
    }
}

// ═══════════════════════════════════════════════════════════════════
// 🧠 ASI NEURAL GRAPH VIEW — Live Engine Connection Map
// ═══════════════════════════════════════════════════════════════════

class NeuralGraphView: NSView {
    struct Node {
        var name: String
        var x, y: CGFloat
        var health: CGFloat
        var color: NSColor
        var pulsePhase: CGFloat
    }

    private var nodes: [Node] = []
    private var edges: [(Int, Int)] = []
    private var timer: Timer?
    private var time: CGFloat = 0

    override init(frame: NSRect) {
        super.init(frame: frame)
        wantsLayer = true
        layer?.backgroundColor = NSColor.clear.cgColor
        layer?.cornerRadius = 12
        buildGraph()
        startGraphTimer()
    }
    required init?(coder: NSCoder) { super.init(coder: coder); buildGraph() }
    deinit { timer?.invalidate() }

    private func startGraphTimer() {
        guard timer == nil else { return }
        timer = Timer.scheduledTimer(withTimeInterval: 1.0/20.0, repeats: true) { [weak self] _ in
            self?.time += 0.05
            self?.updateNodes()
            self?.needsDisplay = true
        }
    }
    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        if window != nil { startGraphTimer() } else { timer?.invalidate(); timer = nil }
    }
    override func viewDidHide() { super.viewDidHide(); timer?.invalidate(); timer = nil }
    override func viewDidUnhide() { super.viewDidUnhide(); startGraphTimer() }

    func buildGraph() {
        let engines: [(String, NSColor)] = [
            ("HyperBrain", NSColor.systemOrange),
            ("Nexus", NSColor.systemCyan),
            ("SQC", NSColor.systemPurple),
            ("Steering", NSColor.systemYellow),
            ("Evolution", NSColor.systemGreen),
            ("Consciousness", NSColor.systemPink),
            ("Resonance", NSColor.systemTeal),
            ("Entanglement", NSColor.systemBlue),
            ("Invention", NSColor(red: 1.0, green: 0.5, blue: 0.0, alpha: 1.0)),
            ("Superfluid", NSColor(red: 0.4, green: 0.8, blue: 1.0, alpha: 1.0)),
            ("FeOrbital", NSColor(red: 0.8, green: 0.4, blue: 0.2, alpha: 1.0)),
            ("QShellMemory", NSColor(red: 0.6, green: 0.3, blue: 0.9, alpha: 1.0)),
        ]

        let count = engines.count
        nodes = engines.enumerated().map { (i: Int, engine: (String, NSColor)) -> Node in
            let angle: CGFloat = CGFloat(i) / CGFloat(count) * CGFloat.pi * 2.0 - CGFloat.pi / 2.0
            let radius: CGFloat = 0.35
            let xPos: CGFloat = 0.5 + radius * cos(angle)
            let yPos: CGFloat = 0.5 + radius * sin(angle)
            let initHealth: CGFloat = CGFloat.random(in: 0.6...1.0)
            let initPhase: CGFloat = CGFloat.random(in: 0.0...(CGFloat.pi * 2.0))
            return Node(
                name: engine.0,
                x: xPos,
                y: yPos,
                health: initHealth,
                color: engine.1,
                pulsePhase: initPhase
            )
        }

        // Connect every engine to HyperBrain (index 0) and Nexus (index 1)
        for i in 2..<count {
            edges.append((0, i)) // HyperBrain hub
            edges.append((1, i)) // Nexus hub
        }
        edges.append((0, 1)) // HyperBrain ↔ Nexus
        // Cross-connections for visual density
        edges.append((4, 5))  // Evolution ↔ Consciousness
        edges.append((6, 7))  // Resonance ↔ Entanglement
        edges.append((2, 3))  // SQC ↔ Steering
        edges.append((9, 10)) // Superfluid ↔ FeOrbital
    }

    func updateNodes() {
        let sweep = EngineRegistry.shared.healthSweep()
        for i in 0..<nodes.count {
            if let found = sweep.first(where: { $0.name == nodes[i].name }) {
                nodes[i].health = CGFloat(found.health)
            }
            nodes[i].pulsePhase += 0.05
        }
    }

    override func draw(_ dirtyRect: NSRect) {
        guard let ctx = NSGraphicsContext.current?.cgContext else { return }
        let w = bounds.width, h = bounds.height

        // Draw edges with animated flow particles
        for (ei, (i, j)) in edges.enumerated() {
            let n1 = nodes[i], n2 = nodes[j]
            let strength = (n1.health + n2.health) / 2
            let flow = 0.3 + 0.4 * (0.5 + 0.5 * sin(time * 2 + CGFloat(i + j)))
            let blended = n1.color.blended(withFraction: 0.5, of: n2.color) ?? n1.color

            // Edge line
            ctx.setStrokeColor(blended.withAlphaComponent(strength * flow * 0.4).cgColor)
            ctx.setLineWidth(1.0 + strength * 1.5)
            ctx.move(to: CGPoint(x: n1.x * w, y: n1.y * h))
            ctx.addLine(to: CGPoint(x: n2.x * w, y: n2.y * h))
            ctx.strokePath()

            // Data flow particles (2-3 packets per edge, traveling along the line)
            let packetCount = strength > 0.7 ? 3 : 2
            for p in 0..<packetCount {
                let baseT = fmod(time * (0.4 + strength * 0.3) + CGFloat(p) / CGFloat(packetCount) + CGFloat(ei) * 0.1, 1.0)
                let px = n1.x + (n2.x - n1.x) * baseT
                let py = n1.y + (n2.y - n1.y) * baseT
                let packetR: CGFloat = 2.0 + strength * 1.5
                let packetAlpha = strength * flow * 0.8 * (1.0 - abs(baseT - 0.5) * 2) // fade at endpoints
                ctx.setFillColor(blended.withAlphaComponent(packetAlpha).cgColor)
                ctx.fillEllipse(in: CGRect(x: px * w - packetR, y: py * h - packetR, width: packetR * 2, height: packetR * 2))
            }
        }

        // Draw nodes
        for node in nodes {
            let px = node.x * w, py = node.y * h
            let pulse = 0.7 + 0.3 * sin(time * 3 + node.pulsePhase)
            let r = 6 + node.health * 8

            // Outer pulse ring (expands periodically)
            let ringPhase = fmod(time * 1.5 + node.pulsePhase, 3.0)
            if ringPhase < 2.0 {
                let ringR = r + ringPhase * 8
                let ringAlpha = (1.0 - ringPhase / 2.0) * 0.3 * node.health
                ctx.setStrokeColor(node.color.withAlphaComponent(ringAlpha).cgColor)
                ctx.setLineWidth(1.0)
                ctx.addArc(center: CGPoint(x: px, y: py), radius: ringR, startAngle: 0, endAngle: .pi * 2, clockwise: false)
                ctx.strokePath()
            }

            // Glow
            let glowColors = [node.color.withAlphaComponent(0.5 * pulse).cgColor, node.color.withAlphaComponent(0).cgColor] as CFArray
            if let gradient = CGGradient(colorsSpace: CGColorSpaceCreateDeviceRGB(), colors: glowColors, locations: [0, 1]) {
                ctx.saveGState()
                ctx.drawRadialGradient(gradient, startCenter: CGPoint(x: px, y: py), startRadius: 0, endCenter: CGPoint(x: px, y: py), endRadius: r * 3, options: [])
                ctx.restoreGState()
            }

            // Node circle with border
            ctx.setFillColor(node.color.withAlphaComponent(0.9).cgColor)
            ctx.fillEllipse(in: CGRect(x: px - r, y: py - r, width: r * 2, height: r * 2))
            ctx.setStrokeColor(node.color.withAlphaComponent(0.5).cgColor)
            ctx.setLineWidth(1.5)
            ctx.strokeEllipse(in: CGRect(x: px - r, y: py - r, width: r * 2, height: r * 2))

            // Inner highlight (specular)
            ctx.setFillColor(NSColor.white.withAlphaComponent(0.35).cgColor)
            let hr = r * 0.35
            ctx.fillEllipse(in: CGRect(x: px - hr + 1, y: py - hr + 2, width: hr * 2, height: hr * 2))

            // Health percentage tiny text inside node
            let healthStr = String(format: "%.0f", node.health * 100)
            let healthAttrs: [NSAttributedString.Key: Any] = [
                .font: NSFont.monospacedDigitSystemFont(ofSize: max(6, r * 0.55), weight: .bold),
                .foregroundColor: NSColor.white.withAlphaComponent(0.8)
            ]
            let hs = (healthStr as NSString).size(withAttributes: healthAttrs)
            (healthStr as NSString).draw(at: CGPoint(x: px - hs.width/2, y: py - hs.height/2), withAttributes: healthAttrs)

            // Label below node
            let labelAttrs: [NSAttributedString.Key: Any] = [
                .font: NSFont.systemFont(ofSize: 8, weight: .bold),
                .foregroundColor: node.color.withAlphaComponent(0.9)
            ]
            let size = (node.name as NSString).size(withAttributes: labelAttrs)
            (node.name as NSString).draw(at: CGPoint(x: px - size.width/2, y: py - r - size.height - 4), withAttributes: labelAttrs)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// ✨ ASI AURORA WAVE VIEW — Animated Header Aurora
// ═══════════════════════════════════════════════════════════════════

class AuroraWaveView: NSView {
    private var phase: CGFloat = 0
    private var timer: Timer?

    override init(frame: NSRect) {
        super.init(frame: frame)
        wantsLayer = true
        layer?.backgroundColor = NSColor.clear.cgColor
        layer?.compositingFilter = "screenBlendMode"  // Additive blending
        startAuroraTimer()
    }
    required init?(coder: NSCoder) { super.init(coder: coder) }
    deinit { timer?.invalidate() }

    private func startAuroraTimer() {
        guard timer == nil else { return }
        timer = Timer.scheduledTimer(withTimeInterval: 1.0/30.0, repeats: true) { [weak self] _ in
            self?.phase += 0.04
            self?.needsDisplay = true
        }
    }
    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        if window != nil { startAuroraTimer() } else { timer?.invalidate(); timer = nil }
    }
    override func viewDidHide() { super.viewDidHide(); timer?.invalidate(); timer = nil }
    override func viewDidUnhide() { super.viewDidUnhide(); startAuroraTimer() }

    override func draw(_ dirtyRect: NSRect) {
        guard let ctx = NSGraphicsContext.current?.cgContext else { return }
        let w = bounds.width, h = bounds.height

        // 5 aurora bands with varied motion
        let colors: [(CGFloat, CGFloat, CGFloat, CGFloat, CGFloat)] = [
            (1.0, 0.84, 0.0, 0.4, 1.0),   // Gold — fast
            (0.0, 0.9, 1.0, 0.25, 0.7),   // Cyan — medium
            (1.0, 0.3, 0.6, 0.2, 0.5),    // Pink — slow
            (0.5, 0.3, 1.0, 0.15, 1.3),   // Violet — fast
            (0.2, 1.0, 0.6, 0.12, 0.4),   // Emerald — slow
        ]

        for (ci, (r, g, b, a, speed)) in colors.enumerated() {
            let freq = 1.8 + CGFloat(ci) * 0.5
            let amp = h * (0.25 + CGFloat(ci) * 0.04)
            let phaseOff = CGFloat(ci) * 1.2
            let alphaWave = a * (0.4 + 0.35 * sin(phase * speed * 0.6 + phaseOff))

            ctx.beginPath()
            ctx.move(to: CGPoint(x: 0, y: 0))
            for x in stride(from: CGFloat(0), to: w, by: 2) {
                let normalX = x / w
                let wave1 = sin(normalX * freq * .pi + phase * speed + phaseOff)
                let wave2 = 0.3 * sin(normalX * freq * .pi * 2.5 + phase * speed * 0.7)
                let envelope = 0.5 + 0.3 * cos(normalX * .pi * 3 + phase * speed * 0.5)
                let y = h * 0.5 + amp * (wave1 + wave2) * envelope
                ctx.addLine(to: CGPoint(x: x, y: y))
            }
            ctx.addLine(to: CGPoint(x: w, y: 0))
            ctx.closePath()
            ctx.setFillColor(NSColor(red: r, green: g, blue: b, alpha: alphaWave).cgColor)
            ctx.fillPath()
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// 🔮 ASI SPARKLINE VIEW — Mini Trend Chart
// ═══════════════════════════════════════════════════════════════════

class SparklineView: NSView {
    var dataPoints: [CGFloat] = [] { didSet { if dataPoints.count > maxPoints { dataPoints.removeFirst(dataPoints.count - maxPoints) }; needsDisplay = true } }
    var lineColor: NSColor = .systemCyan
    var fillColor: NSColor = NSColor.systemCyan.withAlphaComponent(0.15)
    var maxPoints: Int = 40
    var showValueLabel: Bool = true

    override init(frame: NSRect) {
        super.init(frame: frame)
        wantsLayer = true
    }
    required init?(coder: NSCoder) { super.init(coder: coder) }

    func addPoint(_ value: CGFloat) {
        dataPoints.append(value)
        if dataPoints.count > maxPoints { dataPoints.removeFirst() }
        needsDisplay = true
    }

    // Catmull-Rom spline interpolation for smooth curves
    private func catmullRomPoint(_ p0: CGPoint, _ p1: CGPoint, _ p2: CGPoint, _ p3: CGPoint, t: CGFloat) -> CGPoint {
        let t2 = t * t, t3 = t2 * t
        let x = 0.5 * ((2 * p1.x) + (-p0.x + p2.x) * t + (2 * p0.x - 5 * p1.x + 4 * p2.x - p3.x) * t2 + (-p0.x + 3 * p1.x - 3 * p2.x + p3.x) * t3)
        let y = 0.5 * ((2 * p1.y) + (-p0.y + p2.y) * t + (2 * p0.y - 5 * p1.y + 4 * p2.y - p3.y) * t2 + (-p0.y + 3 * p1.y - 3 * p2.y + p3.y) * t3)
        return CGPoint(x: x, y: y)
    }

    override func draw(_ dirtyRect: NSRect) {
        guard let ctx = NSGraphicsContext.current?.cgContext, dataPoints.count > 1 else { return }
        let w = bounds.width, h = bounds.height
        let padding: CGFloat = showValueLabel ? 22 : 2
        let drawH = h - padding
        let minVal = dataPoints.min() ?? 0
        let maxVal = max(dataPoints.max() ?? 1, minVal + 0.01)
        let range = maxVal - minVal

        let rawPoints: [CGPoint] = dataPoints.enumerated().map { (i, val) in
            let x = CGFloat(i) / CGFloat(dataPoints.count - 1) * w
            let y = ((val - minVal) / range) * (drawH - 4) + 2
            return CGPoint(x: x, y: y)
        }

        // Build smooth Catmull-Rom path
        let smoothPath = CGMutablePath()
        var smoothPoints: [CGPoint] = [rawPoints[0]]  // for fill
        smoothPath.move(to: rawPoints[0])
        for i in 0..<rawPoints.count - 1 {
            let p0 = i > 0 ? rawPoints[i - 1] : rawPoints[i]
            let p1 = rawPoints[i]
            let p2 = rawPoints[i + 1]
            let p3 = i + 2 < rawPoints.count ? rawPoints[i + 2] : rawPoints[i + 1]
            let steps = 6
            for s in 1...steps {
                let t = CGFloat(s) / CGFloat(steps)
                let pt = catmullRomPoint(p0, p1, p2, p3, t: t)
                smoothPath.addLine(to: pt)
                smoothPoints.append(pt)
            }
        }

        // Gradient fill under curve
        ctx.saveGState()
        let fillPath = CGMutablePath()
        fillPath.move(to: CGPoint(x: smoothPoints[0].x, y: 0))
        for p in smoothPoints { fillPath.addLine(to: p) }
        fillPath.addLine(to: CGPoint(x: smoothPoints.last!.x, y: 0))
        fillPath.closeSubpath()
        ctx.addPath(fillPath)
        ctx.clip()
        let gradColors = [lineColor.withAlphaComponent(0.25).cgColor, lineColor.withAlphaComponent(0.02).cgColor] as CFArray
        if let grad = CGGradient(colorsSpace: CGColorSpaceCreateDeviceRGB(), colors: gradColors, locations: [0, 1]) {
            ctx.drawLinearGradient(grad, start: CGPoint(x: 0, y: drawH), end: CGPoint(x: 0, y: 0), options: [])
        }
        ctx.restoreGState()

        // Glow line (behind main line)
        ctx.setStrokeColor(lineColor.withAlphaComponent(0.15).cgColor)
        ctx.setLineWidth(5.0)
        ctx.setLineCap(.round); ctx.setLineJoin(.round)
        ctx.addPath(smoothPath); ctx.strokePath()

        // Main smooth line
        ctx.setStrokeColor(lineColor.cgColor)
        ctx.setLineWidth(1.8)
        ctx.setLineCap(.round); ctx.setLineJoin(.round)
        ctx.addPath(smoothPath); ctx.strokePath()

        // Last point with glow
        if let last = smoothPoints.last {
            // Glow ring
            let glowR: CGFloat = 6
            let glowColors = [lineColor.withAlphaComponent(0.5).cgColor, lineColor.withAlphaComponent(0).cgColor] as CFArray
            if let gGrad = CGGradient(colorsSpace: CGColorSpaceCreateDeviceRGB(), colors: glowColors, locations: [0, 1]) {
                ctx.saveGState()
                ctx.drawRadialGradient(gGrad, startCenter: last, startRadius: 0, endCenter: last, endRadius: glowR, options: [])
                ctx.restoreGState()
            }
            // Solid dot
            ctx.setFillColor(lineColor.cgColor)
            ctx.fillEllipse(in: CGRect(x: last.x - 2.5, y: last.y - 2.5, width: 5, height: 5))
            ctx.setFillColor(NSColor.white.withAlphaComponent(0.7).cgColor)
            ctx.fillEllipse(in: CGRect(x: last.x - 1, y: last.y - 1, width: 2, height: 2))
        }

        // Value annotation
        if showValueLabel, let lastVal = dataPoints.last {
            let valStr = String(format: "%.1f%%", lastVal * 100)
            let valAttrs: [NSAttributedString.Key: Any] = [
                .font: NSFont.monospacedDigitSystemFont(ofSize: 9, weight: .semibold),
                .foregroundColor: lineColor
            ]
            let valSize = (valStr as NSString).size(withAttributes: valAttrs)
            (valStr as NSString).draw(at: CGPoint(x: w - valSize.width - 2, y: drawH + 4), withAttributes: valAttrs)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// 💎 ASI GLASSMORPHIC PANEL — Frosted Glass Container
// ═══════════════════════════════════════════════════════════════════

class GlassmorphicPanel: NSView {
    var borderColor: NSColor = NSColor.black.withAlphaComponent(0.08) { didSet { layer?.borderColor = borderColor.cgColor } }
    var accentColor: NSColor = .systemCyan {
        didSet {
            layer?.shadowColor = accentColor.withAlphaComponent(0.3).cgColor
            titleLabel?.textColor = accentColor
            needsDisplay = true
        }
    }
    var panelTitle: String = "" { didSet { titleLabel?.stringValue = panelTitle; titleLabel?.isHidden = panelTitle.isEmpty } }
    private var titleLabel: NSTextField?
    private var blurView: NSVisualEffectView?

    override init(frame: NSRect) {
        super.init(frame: frame)
        setupGlass()
    }
    required init?(coder: NSCoder) { super.init(coder: coder); setupGlass() }

    func setupGlass() {
        wantsLayer = true

        // Real backdrop blur via NSVisualEffectView
        let blur = NSVisualEffectView(frame: bounds)
        blur.autoresizingMask = [.width, .height]
        blur.blendingMode = .behindWindow
        blur.material = .hudWindow
        blur.state = .active
        blur.wantsLayer = true
        blur.layer?.cornerRadius = 16
        blur.layer?.masksToBounds = true
        addSubview(blur, positioned: .below, relativeTo: nil)
        blurView = blur

        // Semi-transparent overlay for depth
        layer?.backgroundColor = NSColor(red: 1.0, green: 1.0, blue: 1.0, alpha: 0.85).cgColor
        layer?.cornerRadius = 16
        layer?.borderColor = borderColor.cgColor
        layer?.borderWidth = 1
        layer?.shadowColor = accentColor.withAlphaComponent(0.3).cgColor
        layer?.shadowRadius = 14
        layer?.shadowOpacity = 0.25
        layer?.shadowOffset = CGSize(width: 0, height: -2)

        // Title label (properly initialized)
        let lbl = NSTextField(labelWithString: panelTitle)
        lbl.frame = NSRect(x: 16, y: bounds.height - 28, width: bounds.width - 32, height: 20)
        lbl.font = NSFont.systemFont(ofSize: 12, weight: .bold)
        lbl.textColor = accentColor
        lbl.isHidden = panelTitle.isEmpty
        lbl.autoresizingMask = [.width, .minYMargin]
        addSubview(lbl)
        titleLabel = lbl
    }

    override func draw(_ dirtyRect: NSRect) {
        guard let ctx = NSGraphicsContext.current?.cgContext else { return }

        // Top accent line with glow
        let accentRect = NSRect(x: 16, y: bounds.height - 2, width: bounds.width - 32, height: 2)
        ctx.setFillColor(accentColor.withAlphaComponent(0.6).cgColor)
        let path = CGPath(roundedRect: accentRect, cornerWidth: 1, cornerHeight: 1, transform: nil)
        ctx.addPath(path)
        ctx.fillPath()

        // Subtle inner highlight at top
        let highlightRect = NSRect(x: 1, y: bounds.height - 40, width: bounds.width - 2, height: 38)
        let highlightColors = [NSColor.black.withAlphaComponent(0.02).cgColor, NSColor.black.withAlphaComponent(0).cgColor] as CFArray
        if let grad = CGGradient(colorsSpace: CGColorSpaceCreateDeviceRGB(), colors: highlightColors, locations: [0, 1]) {
            ctx.saveGState()
            ctx.clip(to: highlightRect)
            ctx.drawLinearGradient(grad, start: CGPoint(x: 0, y: bounds.height), end: CGPoint(x: 0, y: bounds.height - 40), options: [])
            ctx.restoreGState()
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// 🌐 MESH TOPOLOGY VIEW — Visual Network Graph
// Real-time visualization of peers, quantum links, and data flow
// ═══════════════════════════════════════════════════════════════════

class MeshTopologyView: NSView {
    private var timer: Timer?
    private var phase: CGFloat = 0
    private var flowPhase: CGFloat = 0

    struct NodePosition {
        let id: String
        let label: String
        var x: CGFloat
        var y: CGFloat
        let isLocal: Bool
        let isAlive: Bool
        let role: String
    }

    override init(frame: NSRect) {
        super.init(frame: frame)
        wantsLayer = true
        layer?.backgroundColor = NSColor(red: 0.97, green: 0.97, blue: 0.99, alpha: 1.0).cgColor
        layer?.cornerRadius = 12
        startAnimation()
    }
    required init?(coder: NSCoder) { super.init(coder: coder); startAnimation() }
    deinit { timer?.invalidate() }

    func startAnimation() {
        guard timer == nil else { return }
        timer = Timer.scheduledTimer(withTimeInterval: 1.0/15.0, repeats: true) { [weak self] _ in
            guard let self = self else { return }
            self.phase += 0.03
            self.flowPhase += 0.06
            self.needsDisplay = true
        }
    }

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        if window != nil { startAnimation() } else { timer?.invalidate(); timer = nil }
    }
    override func viewDidHide() { super.viewDidHide(); timer?.invalidate(); timer = nil }
    override func viewDidUnhide() { super.viewDidUnhide(); startAnimation() }

    override func draw(_ dirtyRect: NSRect) {
        guard let ctx = NSGraphicsContext.current?.cgContext else { return }
        let w = bounds.width
        let h = bounds.height
        let centerX = w * 0.5
        let centerY = h * 0.5

        // Gather network state
        let net = NetworkLayer.shared
        let peers: [String: NetworkLayer.Peer] = net.peers
        let qLinks: [String: NetworkLayer.QuantumLink] = net.quantumLinks

        // Position nodes in a circle
        var nodes: [NodePosition] = []
        // Local node at center
        nodes.append(NodePosition(
            id: "local", label: "L104",
            x: centerX, y: centerY,
            isLocal: true, isAlive: true, role: "SOVEREIGN"
        ))
        // Remote peers around the circle
        let radius: CGFloat = min(w, h) * 0.32
        let peerValues: [NetworkLayer.Peer] = Array(peers.values)
        let totalPeers: CGFloat = CGFloat(peerValues.count)
        for i in 0..<peerValues.count {
            let peer: NetworkLayer.Peer = peerValues[i]
            let angle: CGFloat = (CGFloat(i) / max(1, totalPeers)) * .pi * 2 - .pi / 2
            let shortId: String = peer.id.count > 8 ? String(peer.id.prefix(8)) : peer.id
            let px: CGFloat = centerX + radius * cos(angle)
            let py: CGFloat = centerY + radius * sin(angle)
            let peerIsLocal: Bool = peer.role == .sovereign
            let peerIsAlive: Bool = peer.fidelity > 0.1
            nodes.append(NodePosition(
                id: peer.id, label: shortId,
                x: px, y: py,
                isLocal: peerIsLocal, isAlive: peerIsAlive,
                role: peer.role.rawValue
            ))
        }

        // Draw quantum links as animated arcs
        let qLinkValues: [NetworkLayer.QuantumLink] = Array(qLinks.values)
        for qLink in qLinkValues {
            guard let n1 = nodes.first(where: { $0.id == qLink.peerA || $0.isLocal }),
                  let n2 = nodes.first(where: { $0.id == qLink.peerB }) else { continue }
            let fidelityAlpha: CGFloat = CGFloat(qLink.eprFidelity) * 0.8
            ctx.setStrokeColor(NSColor.systemPurple.withAlphaComponent(max(0.15, fidelityAlpha)).cgColor)
            ctx.setLineWidth(2.0)
            ctx.setLineDash(phase: flowPhase * 20, lengths: [6, 4])
            ctx.move(to: CGPoint(x: n1.x, y: n1.y))
            ctx.addLine(to: CGPoint(x: n2.x, y: n2.y))
            ctx.strokePath()
        }
        ctx.setLineDash(phase: 0, lengths: [])

        // Draw peer connections as lines
        for i in 1..<nodes.count {
            let n = nodes[i]
            let local = nodes[0]
            let alpha: CGFloat = n.isAlive ? 0.3 : 0.08
            ctx.setStrokeColor(NSColor.systemTeal.withAlphaComponent(alpha).cgColor)
            ctx.setLineWidth(1.2)
            ctx.move(to: CGPoint(x: local.x, y: local.y))
            ctx.addLine(to: CGPoint(x: n.x, y: n.y))
            ctx.strokePath()

            // Animated data flow dot along the line
            if n.isAlive {
                let t = (flowPhase + CGFloat(i) * 0.7).truncatingRemainder(dividingBy: 1.0)
                let dotX = local.x + (n.x - local.x) * t
                let dotY = local.y + (n.y - local.y) * t
                ctx.setFillColor(NSColor.systemCyan.withAlphaComponent(0.7).cgColor)
                ctx.fillEllipse(in: CGRect(x: dotX - 2, y: dotY - 2, width: 4, height: 4))
            }
        }

        // Draw nodes
        for (i, node) in nodes.enumerated() {
            let nodeRadius: CGFloat = i == 0 ? 18 : 12
            let pulse: CGFloat = i == 0 ? (0.5 + 0.5 * sin(phase * 2)) : 1.0

            // Glow
            let glowColor = i == 0 ? NSColor.systemOrange : (node.isAlive ? NSColor.systemTeal : NSColor.gray)
            let glowR = nodeRadius * 2.5
            let glowColors = [glowColor.withAlphaComponent(0.3 * pulse).cgColor, glowColor.withAlphaComponent(0).cgColor] as CFArray
            if let grad = CGGradient(colorsSpace: CGColorSpaceCreateDeviceRGB(), colors: glowColors, locations: [0, 1]) {
                ctx.saveGState()
                ctx.drawRadialGradient(grad, startCenter: CGPoint(x: node.x, y: node.y), startRadius: 0,
                                       endCenter: CGPoint(x: node.x, y: node.y), endRadius: glowR, options: [])
                ctx.restoreGState()
            }

            // Node circle
            let nodeColor = i == 0 ? NSColor.systemOrange : (node.isAlive ? NSColor.systemTeal : NSColor.gray)
            ctx.setFillColor(nodeColor.withAlphaComponent(0.85).cgColor)
            ctx.fillEllipse(in: CGRect(x: node.x - nodeRadius, y: node.y - nodeRadius,
                                       width: nodeRadius * 2, height: nodeRadius * 2))

            // White center
            let innerR = nodeRadius * 0.4
            ctx.setFillColor(NSColor.white.withAlphaComponent(0.6).cgColor)
            ctx.fillEllipse(in: CGRect(x: node.x - innerR, y: node.y - innerR, width: innerR * 2, height: innerR * 2))

            // Label
            let labelAttrs: [NSAttributedString.Key: Any] = [
                .font: NSFont.systemFont(ofSize: i == 0 ? 10 : 8, weight: .bold),
                .foregroundColor: NSColor.black.withAlphaComponent(0.7)
            ]
            let labelSize = (node.label as NSString).size(withAttributes: labelAttrs)
            (node.label as NSString).draw(at: CGPoint(x: node.x - labelSize.width / 2,
                                                       y: node.y - nodeRadius - labelSize.height - 3),
                                          withAttributes: labelAttrs)
        }

        // Summary in corner
        let summary = "\(peers.count) peers · \(qLinks.count) Q-links"
        let sumAttrs: [NSAttributedString.Key: Any] = [
            .font: NSFont.monospacedDigitSystemFont(ofSize: 9, weight: .medium),
            .foregroundColor: NSColor.black.withAlphaComponent(0.4)
        ]
        (summary as NSString).draw(at: CGPoint(x: 8, y: 6), withAttributes: sumAttrs)
    }
}


// ═══════════════════════════════════════════════════════════════════
// 📊 NETWORK THROUGHPUT BAR — Animated horizontal bar gauge
// ═══════════════════════════════════════════════════════════════════

class NetworkHealthBar: NSView {
    var health: CGFloat = 0.0 { didSet { needsDisplay = true } }
    var meshStatus: String = "OFFLINE" { didSet { needsDisplay = true } }
    var peerCount: Int = 0 { didSet { needsDisplay = true } }
    var linkCount: Int = 0 { didSet { needsDisplay = true } }

    override init(frame: NSRect) {
        super.init(frame: frame)
        wantsLayer = true
        layer?.cornerRadius = 8
        layer?.backgroundColor = NSColor(red: 0.96, green: 0.96, blue: 0.98, alpha: 1.0).cgColor
    }
    required init?(coder: NSCoder) { super.init(coder: coder) }

    override func draw(_ dirtyRect: NSRect) {
        guard let ctx = NSGraphicsContext.current?.cgContext else { return }
        let w = bounds.width
        let h = bounds.height

        // Background bar track
        let barY: CGFloat = h * 0.35
        let barH: CGFloat = 8
        let barRect = CGRect(x: 12, y: barY, width: w - 24, height: barH)
        ctx.setFillColor(NSColor.black.withAlphaComponent(0.06).cgColor)
        let trackPath = CGPath(roundedRect: barRect, cornerWidth: barH / 2, cornerHeight: barH / 2, transform: nil)
        ctx.addPath(trackPath)
        ctx.fillPath()

        // Health fill bar
        let fillW = (w - 24) * max(0, min(1, health))
        let fillColor: NSColor = health > 0.7 ? .systemGreen : health > 0.4 ? .systemYellow : .systemRed
        let fillRect = CGRect(x: 12, y: barY, width: fillW, height: barH)
        ctx.setFillColor(fillColor.withAlphaComponent(0.8).cgColor)
        let fillPath = CGPath(roundedRect: fillRect, cornerWidth: barH / 2, cornerHeight: barH / 2, transform: nil)
        ctx.addPath(fillPath)
        ctx.fillPath()

        // Status label
        let statusIcon = meshStatus == "ONLINE" ? "🟢" : meshStatus == "DEGRADED" ? "🟡" : "🔴"
        let statusStr = "\(statusIcon) \(meshStatus) · \(peerCount) peers · \(linkCount) Q-links · \(String(format: "%.0f%%", health * 100))"
        let attrs: [NSAttributedString.Key: Any] = [
            .font: NSFont.monospacedDigitSystemFont(ofSize: 10, weight: .semibold),
            .foregroundColor: NSColor.black.withAlphaComponent(0.6)
        ]
        let textSize = (statusStr as NSString).size(withAttributes: attrs)
        _ = textSize  // suppress warning — retained for future layout use
        (statusStr as NSString).draw(at: CGPoint(x: 12, y: barY + barH + 6), withAttributes: attrs)

        // Health percentage at right
        let pctStr = String(format: "%.1f%%", health * 100)
        let pctAttrs: [NSAttributedString.Key: Any] = [
            .font: NSFont.monospacedDigitSystemFont(ofSize: 11, weight: .bold),
            .foregroundColor: fillColor
        ]
        let pctSize = (pctStr as NSString).size(withAttributes: pctAttrs)
        (pctStr as NSString).draw(at: CGPoint(x: w - pctSize.width - 12, y: barY + barH + 6), withAttributes: pctAttrs)
    }
}


// ═══════════════════════════════════════════════════════════════════
// 🔮 QUANTUM LINK ARC VIEW — Animated entanglement fidelity arcs
// ═══════════════════════════════════════════════════════════════════

class QuantumLinkArcView: NSView {
    private var timer: Timer?
    private var phase: CGFloat = 0

    override init(frame: NSRect) {
        super.init(frame: frame)
        wantsLayer = true
        layer?.cornerRadius = 10
        layer?.backgroundColor = NSColor(red: 0.97, green: 0.96, blue: 0.99, alpha: 1.0).cgColor
        startAnimation()
    }
    required init?(coder: NSCoder) { super.init(coder: coder); startAnimation() }
    deinit { timer?.invalidate() }

    func startAnimation() {
        guard timer == nil else { return }
        timer = Timer.scheduledTimer(withTimeInterval: 1.0/20.0, repeats: true) { [weak self] _ in
            self?.phase += 0.04
            self?.needsDisplay = true
        }
    }

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        if window != nil { startAnimation() } else { timer?.invalidate(); timer = nil }
    }
    override func viewDidHide() { super.viewDidHide(); timer?.invalidate(); timer = nil }
    override func viewDidUnhide() { super.viewDidUnhide(); startAnimation() }

    override func draw(_ dirtyRect: NSRect) {
        guard let ctx = NSGraphicsContext.current?.cgContext else { return }
        let w = bounds.width
        let h = bounds.height
        let router = QuantumEntanglementRouter.shared
        let fidelity = CGFloat(router.overallFidelity)

        // Draw concentric arcs representing entanglement strength
        let center = CGPoint(x: w / 2, y: h * 0.4)
        let maxR = min(w, h) * 0.35
        let arcCount = 5

        for i in 0..<arcCount {
            let r = maxR * CGFloat(i + 1) / CGFloat(arcCount)
            let animAngle = phase + CGFloat(i) * 0.3
            let arcStart = -CGFloat.pi * 0.8 + sin(animAngle) * 0.1
            let arcEnd = CGFloat.pi * 0.8 * fidelity
            let hue = 0.75 + CGFloat(i) * 0.04  // purple spectrum
            let alpha: CGFloat = (1.0 - CGFloat(i) / CGFloat(arcCount)) * 0.6

            ctx.setStrokeColor(NSColor(hue: hue, saturation: 0.7, brightness: 0.9, alpha: alpha).cgColor)
            ctx.setLineWidth(3.0 - CGFloat(i) * 0.3)
            ctx.setLineCap(.round)
            ctx.addArc(center: center, radius: r, startAngle: arcStart, endAngle: arcEnd, clockwise: false)
            ctx.strokePath()
        }

        // Fidelity text
        let fidStr = String(format: "F = %.4f", fidelity)
        let attrs: [NSAttributedString.Key: Any] = [
            .font: NSFont.monospacedDigitSystemFont(ofSize: 14, weight: .bold),
            .foregroundColor: NSColor.systemPurple.withAlphaComponent(0.8)
        ]
        let textSize = (fidStr as NSString).size(withAttributes: attrs)
        (fidStr as NSString).draw(at: CGPoint(x: w / 2 - textSize.width / 2, y: h - 24), withAttributes: attrs)

        // EPR label
        let eprStr = "\(router.remoteLinkCount) EPR · \(router.routeCount) routes"
        let eprAttrs: [NSAttributedString.Key: Any] = [
            .font: NSFont.systemFont(ofSize: 9, weight: .medium),
            .foregroundColor: NSColor.black.withAlphaComponent(0.4)
        ]
        let eprSize = (eprStr as NSString).size(withAttributes: eprAttrs)
        (eprStr as NSString).draw(at: CGPoint(x: w / 2 - eprSize.width / 2, y: 6), withAttributes: eprAttrs)
    }
}


// ═══════════════════════════════════════════════════════════════════
// ASI EVOLUTION ENGINE - Continuous Upgrade Cycle
// ═══════════════════════════════════════════════════════════════════

