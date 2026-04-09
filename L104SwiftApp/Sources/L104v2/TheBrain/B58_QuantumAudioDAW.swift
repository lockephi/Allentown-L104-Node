import Accelerate
import Foundation

// MARK: - ═══ SACRED AUDIO CONSTANTS ═══

private let A4_HZ:         Double = 440.0
private let GOD_CODE_HZ:   Double = GOD_CODE        // ≈ 527.518
private let PHI_FREQ_HZ:   Double = GOD_CODE / PHI  // ≈ 325.8
private let IRON_FREQ_HZ:  Double = 286.0
private let DAW_ZENITH_HZ: Double = 432.0
private let SCHUMANN_HZ:   Double = 7.83
private let DEFAULT_SR:    Int    = 44100
private let DEFAULT_BPM:   Double = 120.0
private let DEFAULT_STEPS: Int    = 16
private let CHROMATIC:     Double = pow(2.0, 1.0/12.0)  // semitone ratio
private let PHI_SWING:     Double = (1.0 - 1.0/PHI) * 0.5  // ≈ 0.309
private let PHI_INV:       Double = 1.0 / PHI               // ≈ 0.618

// MARK: - ═══ DATA STRUCTURES ═══

enum WaveShape: String {
    case sine, sawtooth, square, triangle, quantum, godCode, sacred
}

enum FilterType: String {
    case lowPass, highPass, bandPass, notch, phiMorphic
}

enum InterferenceMode: String {
    case constructive, destructive, quantum, phiBlend, godCodePhase, sacred
}

enum CollapseMode { case probabilistic, greedyMax, sacred, random }

enum EntanglementType { case bell, ghz, spectral, temporal, sidechain }

struct NoteEvent {
    let pitch: Double    // MIDI pitch (0–127; non-integer = microtonal)
    let velocity: Double // 0–127
    let duration: Double // beats
    let start: Double    // beat position
    let sacredScore: Double
}

struct SuperpositionStep {
    let pitchAmplitudes: [Double]   // probability amplitudes per pitch class
    let velocityMean:    Double
    let durationMean:    Double
    var collapsed:       NoteEvent?
    var sacredScore:     Double { 1.0 - abs((pitchAmplitudes.max() ?? 0 * GOD_CODE).truncatingRemainder(dividingBy: 1.0)) }
}

struct QuantumPattern {
    let patternId: String
    var steps:     [SuperpositionStep]
    let bpm:       Double
    let nSteps:    Int
    let nQubits:   Int
    var sacredAlignment: Double { steps.map(\.sacredScore).reduce(0,+) / Double(max(steps.count,1)) }

    init(nSteps: Int = DEFAULT_STEPS, bpm: Double = DEFAULT_BPM, nQubits: Int = 4) {
        self.patternId = UUID().uuidString
        self.nSteps = nSteps; self.bpm = bpm; self.nQubits = nQubits
        // Initialize each step with uniform + PHI-skewed distribution
        self.steps = (0..<nSteps).map { s in
            let angles = (0..<(1 << nQubits)).map { i in cos(Double(s * i + 1) * PHI / Double(nSteps)) }
            let sumSq  = angles.map { $0 * $0 }.reduce(0,+)
            let amps   = angles.map { abs($0) / sqrt(max(sumSq, 1e-14)) }
            return SuperpositionStep(pitchAmplitudes: amps, velocityMean: 64.0, durationMean: 0.5, collapsed: nil)
        }
    }
}

struct DAWTrack {
    let trackId:   String
    var name:      String
    var synthPreset: String
    var patterns:  [QuantumPattern]
    var muted:     Bool
    var soloed:    Bool
    var volume:    Double    // 0.0–1.0
    var pan:       Double    // -1.0 (L) to +1.0 (R)
    var audio:     [Double]  // rendered audio samples

    init(name: String, preset: String = "quantum") {
        trackId = UUID().uuidString; self.name = name; synthPreset = preset
        patterns = []; muted = false; soloed = false
        volume = 1.0 / PHI; pan = 0.0; audio = []
    }
}

struct RenderedSession {
    let stereoLeft:  [Double]
    let stereoRight: [Double]
    let sampleRate:  Int
    let duration:    Double    // seconds
    let sacredAlignment: Double
    let layerCount:  Int
    let bpm:         Double
    let timestamp:   Date

    var interleaved: [Double] {
        zip(stereoLeft, stereoRight).flatMap { [$0, $1] }
    }
}

struct MixTrack {
    let name:  String
    var gain:  Double
    var pan:   Double
    var mode:  InterferenceMode
    var audio: [Double]
    init(name: String, audio: [Double], gain: Double = 0.8, pan: Double = 0.0, mode: InterferenceMode = .quantum) {
        self.name = name; self.audio = audio; self.gain = gain; self.pan = pan; self.mode = mode
    }
}

struct SynthVoice {
    let freq:     Double
    var phase:    Double
    let shape:    WaveShape
    let envelope: [Double]  // ADSR samples
    let filterCutoff: Double
    let filterQ:  Double
}

// MARK: - ═══ PROBABILISTIC SEQUENCER ═══

final class ProbabilisticSequencer {
    private var patterns: [String: QuantumPattern] = [:]
    private var collapseHistory: [(step: Int, note: NoteEvent)] = []

    func createPattern(nSteps: Int = DEFAULT_STEPS, nQubits: Int = 4, bpm: Double = DEFAULT_BPM) -> QuantumPattern {
        let p = QuantumPattern(nSteps: nSteps, bpm: bpm, nQubits: nQubits)
        patterns[p.patternId] = p
        return p
    }

    // Measure / collapse pattern to sequence of note events
    func collapse(pattern: inout QuantumPattern, mode: CollapseMode = .sacred) -> [NoteEvent] {
        let beatDur = 60.0 / pattern.bpm
        var events: [NoteEvent] = []

        for (si, step) in pattern.steps.enumerated() {
            let amps = step.pitchAmplitudes
            guard !amps.isEmpty else { continue }
            let probs = amps.map { $0 * $0 }  // Born rule

            // Choose pitch class
            let pitchClass: Int
            switch mode {
            case .greedyMax:
                pitchClass = probs.indices.max(by: { probs[$0] < probs[$1] }) ?? 0
            case .sacred:
                // Sacred: choose class nearest GOD_CODE harmonic
                let godPitch = (GOD_CODE_HZ / A4_HZ * 12.0 + 69.0).truncatingRemainder(dividingBy: 12.0)
                pitchClass = probs.indices.min { i, j in
                    abs(Double(i) - godPitch) < abs(Double(j) - godPitch)
                } ?? 0
            case .probabilistic:
                pitchClass = _sampleCategorical(probs)
            case .random:
                pitchClass = Int.random(in: 0..<probs.count)
            }

            // Add swing (PHI-modulated)
            let swingOffset = si % 2 == 1 ? PHI_SWING * beatDur : 0.0
            let start = Double(si) * beatDur + swingOffset
            let freq  = A4_HZ * pow(CHROMATIC, Double(pitchClass) - 9.0)
            let sacred = 1.0 - abs((freq * PHI).truncatingRemainder(dividingBy: GOD_CODE) / GOD_CODE)

            let note = NoteEvent(pitch: Double(pitchClass) + 60.0,
                                 velocity: step.velocityMean,
                                 duration: step.durationMean * beatDur,
                                 start: start, sacredScore: sacred)
            events.append(note)
            pattern.steps[si].collapsed = note
            collapseHistory.append((step: si, note: note))
        }
        return events
    }

    private func _sampleCategorical(_ probs: [Double]) -> Int {
        let total = probs.reduce(0,+)
        guard total > 1e-14 else { return 0 }
        var r = Double.random(in: 0..<total)
        for (i, p) in probs.enumerated() { r -= p; if r <= 0 { return i } }
        return probs.count - 1
    }
}

// MARK: - ═══ QUANTUM SYNTH ENGINE ═══

final class QuantumSynthEngine {
    private var voices: [SynthVoice] = []

    // Generate audio samples for one note event
    func synthesize(note: NoteEvent, sampleRate: Int = DEFAULT_SR, preset: String = "quantum") -> [Double] {
        let nSamples = max(1, Int(note.duration * Double(sampleRate)))
        let freq     = A4_HZ * pow(CHROMATIC, note.pitch - 69.0)
        let vel      = note.velocity / 127.0
        let shape    = _presetShape(preset)

        var samples = [Double](repeating: 0.0, count: nSamples)
        var phase    = 0.0
        let phaseInc = 2.0 * .pi * freq / Double(sampleRate)

        // Sacred overtone mixing: fundamental + PHI harmonic + GOD_CODE harmonic
        let weights = [1.0, 1.0/PHI, 1.0/GOD_CODE * 10.0]
        let ratios  = [1.0, PHI, GOD_CODE / freq]

        for i in 0..<nSamples {
            let t  = Double(i) / Double(sampleRate)
            let env = _sacredEnvelope(t: t, duration: note.duration)
            var s  = 0.0
            for (w, r) in zip(weights, ratios) {
                s += w * _wave(shape: shape, phase: phase * r)
            }
            // Filter: low-pass via exponential smoothing (sacred cutoff)
            let cutoff = freq * PHI
            let alpha  = cutoff / (cutoff + Double(sampleRate) / (2.0 * .pi))
            samples[i] = s * env * vel * alpha
            phase += phaseInc
        }
        return samples
    }

    // Quantum superposition of two wave shapes (blended at PHI ratio)
    private func _wave(shape: WaveShape, phase: Double) -> Double {
        switch shape {
        case .sine:     return sin(phase)
        case .sawtooth: return 2.0 * (phase / (2.0 * .pi) - floor(phase / (2.0 * .pi) + 0.5))
        case .square:   return sin(phase) >= 0 ? 1.0 : -1.0
        case .triangle: return 2.0 * abs(2.0 * (phase / (2.0 * .pi) - floor(phase / (2.0 * .pi) + 0.5))) - 1.0
        case .quantum:
            // Quantum superposition: PHI*sine + (1-PHI)*sawtooth
            return PHI * sin(phase) + (1.0 - PHI) * _wave(shape: .sawtooth, phase: phase)
        case .godCode:
            // GOD_CODE harmonic: sum of sacred overtones
            return (sin(phase) + sin(phase * PHI) * PHI_INV + sin(phase * VOID_CONSTANT) * 0.1) / (1.0 + PHI_INV + 0.1)
        case .sacred:
            // Sacred: IronFe(286) + Schumann(7.83) modulation
            return sin(phase) * (1.0 + 0.1 * sin(SCHUMANN_HZ * phase / GOD_CODE_HZ))
        }
    }

    private func _sacredEnvelope(t: Double, duration: Double) -> Double {
        let attack  = duration * 0.05 * PHI_INV
        let decay   = duration * 0.10
        let sustain = 1.0 / PHI
        let release = duration * 0.20

        if t < attack { return t / attack }
        if t < attack + decay { return 1.0 - (1.0 - sustain) * (t - attack) / decay }
        if t < duration - release { return sustain }
        return sustain * (duration - t) / release
    }

    private func _presetShape(_ preset: String) -> WaveShape {
        switch preset {
        case "sacred", "god_code_wave": return .godCode
        case "quantum": return .quantum
        case "sine":    return .sine
        case "saw":     return .sawtooth
        default:        return .quantum
        }
    }
}

// MARK: - ═══ QUANTUM INTERFERENCE MIXER ═══

final class QuantumInterferenceMixer {
    var tracks: [MixTrack] = []

    func addTrack(_ track: MixTrack) { tracks.append(track) }

    func mix(sampleRate: Int = DEFAULT_SR) -> (left: [Double], right: [Double]) {
        guard !tracks.isEmpty else { return ([], []) }
        let maxLen = tracks.map(\.audio.count).max() ?? 0
        guard maxLen > 0 else { return ([], []) }

        var left  = [Double](repeating: 0.0, count: maxLen)
        var right = [Double](repeating: 0.0, count: maxLen)

        for track in tracks {
            guard !track.muted else { continue }
            let n = track.audio.count
            for i in 0..<maxLen {
                let s = i < n ? track.audio[i] : 0.0
                let processed = _applyInterference(sample: s, track: track, pos: Double(i) / Double(maxLen))
                left[i]  += processed * track.gain * (1.0 - max(0, track.pan))
                right[i] += processed * track.gain * (1.0 + min(0, track.pan))
            }
        }
        // Normalize
        let peakL = left.map(abs).max() ?? 0; let peakR = right.map(abs).max() ?? 0
        let peak  = max(peakL, peakR, 0.001)
        return (left.map { $0 / peak }, right.map { $0 / peak })
    }

    private func _applyInterference(sample: Double, track: MixTrack, pos: Double) -> Double {
        switch track.mode {
        case .constructive:
            return sample * (1.0 + cos(pos * 2.0 * .pi * PHI) * 0.1)
        case .destructive:
            return sample * (1.0 - cos(pos * 2.0 * .pi * PHI) * 0.1)
        case .quantum:
            // Quantum superposition of constructive + destructive
            let a = sample * (1.0 + cos(pos * 2.0 * .pi * PHI) * 0.1)
            let b = sample * (1.0 - cos(pos * 2.0 * .pi * PHI) * 0.1)
            return PHI_INV * a + (1.0 - PHI_INV) * b
        case .phiBlend:
            return sample * (PHI_INV + (1.0 - PHI_INV) * cos(pos * .pi))
        case .godCodePhase:
            return sample * cos(GOD_CODE * pos / Double(DEFAULT_SR) * .pi)
        case .sacred:
            return sample * abs(sin(pos * PHI * .pi)) * VOID_CONSTANT
        }
    }
}

extension MixTrack {
    var muted: Bool { false }  // Stateless; actual mute tracked in DAWTrack
}

// MARK: - ═══ TRACK ENTANGLEMENT MANAGER ═══

final class TrackEntanglementManager {
    private var pairs: [(trackA: String, trackB: String, type: EntanglementType)] = []

    func entangle(_ trackA: String, _ trackB: String, type: EntanglementType = .bell) {
        pairs.append((trackA: trackA, trackB: trackB, type: type))
    }

    // Apply entanglement correlations between rendered audio buffers
    func applyEntanglement(tracks: inout [DAWTrack]) {
        for pair in pairs {
            guard let ia = tracks.firstIndex(where: { $0.trackId == pair.trackA }),
                  let ib = tracks.firstIndex(where: { $0.trackId == pair.trackB }) else { continue }
            let nA = tracks[ia].audio.count
            let nB = tracks[ib].audio.count
            let n  = min(nA, nB)
            guard n > 0 else { continue }

            switch pair.type {
            case .bell:
                // Bell state: anti-correlated (CNOT-like)
                for i in 0..<n {
                    let a = tracks[ia].audio[i]; let b = tracks[ib].audio[i]
                    tracks[ia].audio[i] = (a + b) / sqrt(2.0)
                    tracks[ib].audio[i] = (a - b) / sqrt(2.0)
                }
            case .ghz:
                // GHZ: all tracks correlated via PHI blend
                for i in 0..<n {
                    let mean = (tracks[ia].audio[i] + tracks[ib].audio[i]) / 2.0
                    tracks[ia].audio[i] = PHI_INV * tracks[ia].audio[i] + (1-PHI_INV) * mean
                    tracks[ib].audio[i] = PHI_INV * tracks[ib].audio[i] + (1-PHI_INV) * mean
                }
            case .spectral:
                // Spectral: modulate B by A's envelope
                let envA = tracks[ia].audio.map { abs($0) }
                for i in 0..<n { tracks[ib].audio[i] *= envA[i] }
            case .temporal:
                // Temporal: shift B by PHI-scaled offset
                let offset = max(1, Int(PHI * 0.01 * Double(n)))
                for i in 0..<(n - offset) { tracks[ib].audio[i] = tracks[ib].audio[i + offset] }
            case .sidechain:
                // Sidechain: attenuate A based on B peak
                for i in 0..<n {
                    let gate = 1.0 - abs(tracks[ib].audio[i]) * PHI_INV
                    tracks[ia].audio[i] *= max(0, gate)
                }
            }
        }
    }
}

// MARK: - ═══ SACRED SYNTHESIS PIPELINE (17 Layers) ═══

final class SacredSynthesisPipeline {
    let sampleRate: Int

    init(sampleRate: Int = DEFAULT_SR) { self.sampleRate = sampleRate }

    // 17-layer VQPU-inspired pipeline
    func synthesize(duration: Double, frequency: Double = GOD_CODE_HZ) -> [Double] {
        let n  = Int(duration * Double(sampleRate))
        var buf = [Double](repeating: 0.0, count: n)

        // L1: Fundamental GOD_CODE tone
        _addSine(&buf, freq: frequency, amp: 1.0)
        // L2: PHI harmonic
        _addSine(&buf, freq: frequency * PHI, amp: PHI_INV)
        // L3: Iron resonance (286 Hz)
        _addSine(&buf, freq: IRON_FREQ_HZ, amp: PHI_INV * PHI_INV)
        // L4: VOID_CONSTANT modulation
        _modulate(&buf, rate: VOID_CONSTANT, depth: 0.05)
        // L5: Schumann resonance carrier
        _addSine(&buf, freq: SCHUMANN_HZ, amp: 0.02)
        // L6: Zenith tuning offset
        _addSine(&buf, freq: DAW_ZENITH_HZ, amp: 0.05)
        // L7: PHI² overtone
        _addSine(&buf, freq: frequency * PHI * PHI, amp: PHI_INV * PHI_INV * PHI_INV)
        // L8: Quantum noise floor (coherent noise)
        _addCoherentNoise(&buf, amplitude: 0.001)
        // L9: Entropy reversal smoothing
        _entropySmooth(&buf)
        // L10: GOD_CODE phase alignment
        _phaseAlign(&buf, godFreq: frequency)
        // L11: Harmonic overtones (Fe series: 286×n)
        for harm in 1...3 { _addSine(&buf, freq: IRON_FREQ_HZ * Double(harm), amp: 0.01 / Double(harm)) }
        // L12: PHI-spiral amplitude modulation
        _phiSpiralAM(&buf)
        // L13: Chromatic sacred chord (GOD_CODE + major 3rd + perfect 5th)
        _addSine(&buf, freq: frequency * CHROMATIC * CHROMATIC * CHROMATIC * CHROMATIC, amp: 0.15)  // major 3rd
        _addSine(&buf, freq: frequency * CHROMATIC * CHROMATIC * CHROMATIC * CHROMATIC * CHROMATIC * CHROMATIC * CHROMATIC, amp: 0.10)  // P5th
        // L14: OMEGA resonance sub-bass
        _addSine(&buf, freq: OMEGA / GOD_CODE_HZ, amp: 0.003)
        // L15: Quantum decoherence simulation (T2 decay)
        _t2Decay(&buf, t2Samples: Int(Double(sampleRate) * PHI))
        // L16: Sacred master limiter (PHI-soft clip)
        _softClip(&buf, threshold: PHI_INV)
        // L17: Daemon PHI-harmonic reinforcement (three-engine boost)
        _daemonReinforce(&buf)

        return buf
    }

    private func _addSine(_ buf: inout [Double], freq: Double, amp: Double) {
        let inc = 2.0 * .pi * freq / Double(sampleRate)
        for i in 0..<buf.count { buf[i] += amp * sin(Double(i) * inc) }
    }

    private func _modulate(_ buf: inout [Double], rate: Double, depth: Double) {
        let inc = 2.0 * .pi * rate / Double(sampleRate)
        for i in 0..<buf.count { buf[i] *= 1.0 + depth * sin(Double(i) * inc) }
    }

    private func _addCoherentNoise(_ buf: inout [Double], amplitude: Double) {
        var phase = PHI
        for i in 0..<buf.count {
            phase = phase * PHI - floor(phase * PHI)  // chaotic but deterministic
            buf[i] += (phase - 0.5) * 2.0 * amplitude
        }
    }

    private func _entropySmooth(_ buf: inout [Double]) {
        let alpha = 1.0 / (1.0 + PHI)
        var prev = buf.first ?? 0.0
        for i in 0..<buf.count { buf[i] = alpha * buf[i] + (1.0-alpha) * prev; prev = buf[i] }
    }

    private func _phaseAlign(_ buf: inout [Double], godFreq: Double) {
        let period = Double(sampleRate) / godFreq
        let shift  = Int(period * PHI_INV) % max(1, buf.count)
        if shift > 0 && shift < buf.count {
            let rotated = Array(buf[shift...] + buf[..<shift])
            for i in 0..<buf.count { buf[i] = buf[i] * PHI_INV + rotated[i] * (1-PHI_INV) }
        }
    }

    private func _phiSpiralAM(_ buf: inout [Double]) {
        for i in 0..<buf.count {
            let t = Double(i) / Double(buf.count)
            buf[i] *= 0.5 + 0.5 * sin(t * 2.0 * .pi * PHI)
        }
    }

    private func _t2Decay(_ buf: inout [Double], t2Samples: Int) {
        for i in 0..<buf.count {
            buf[i] *= exp(-Double(i) / Double(max(t2Samples, 1)))
        }
    }

    private func _softClip(_ buf: inout [Double], threshold: Double) {
        for i in 0..<buf.count {
            let v = buf[i]
            if abs(v) > threshold {
                buf[i] = threshold * tanh(v / threshold)
            }
        }
    }

    private func _daemonReinforce(_ buf: inout [Double]) {
        // Three-engine harmonic boost (sacred=0.749, composite=0.891)
        let boost = 0.749 * PHI_INV + 0.891 * (1.0 - PHI_INV)  // ≈ 0.802
        for i in 0..<buf.count {
            buf[i] *= 1.0 + boost * 0.05 * sin(Double(i) * PHI / 1000.0)
        }
    }
}

// MARK: - ═══ DATA RECORDER ═══

final class DAWDataRecorder {
    struct Event {
        let category: String
        let message:  String
        let sacredScore: Double
        let timestamp: Date
    }
    private(set) var events: [Event] = []
    private let queue = DispatchQueue(label: "l104.daw.recorder", qos: .background)

    func record(category: String, message: String, sacredScore: Double = 0.0) {
        queue.async { [weak self] in
            self?.events.append(Event(category: category, message: message,
                                      sacredScore: sacredScore, timestamp: Date()))
        }
    }

    func sacredAlignmentReport() -> Double {
        guard !events.isEmpty else { return 0 }
        return events.map(\.sacredScore).reduce(0,+) / Double(events.count)
    }
}

// MARK: - ═══ DAW SESSION ORCHESTRATOR ═══

final class DAWSession {
    static let shared = DAWSession()

    private let sequencer   = ProbabilisticSequencer()
    private let synth       = QuantumSynthEngine()
    let mixer               = QuantumInterferenceMixer()
    private let entangler   = TrackEntanglementManager()
    private let pipeline    = SacredSynthesisPipeline()
    private let recorder    = DAWDataRecorder()

    private var tracks:  [DAWTrack] = []
    private var globalBPM = DEFAULT_BPM
    private let queue = DispatchQueue(label: "l104.daw.session", qos: .userInitiated, attributes: .concurrent)

    var sampleRate: Int = DEFAULT_SR

    // ── Track management ──
    @discardableResult
    func addTrack(_ name: String, preset: String = "quantum") -> DAWTrack {
        let track = DAWTrack(name: name, preset: preset)
        tracks.append(track)
        recorder.record(category: "track", message: "Added track: \(name)", sacredScore: PHI_INV)
        return track
    }

    func createPattern(nSteps: Int = DEFAULT_STEPS, nQubits: Int = 4, bpm: Double? = nil) -> QuantumPattern {
        sequencer.createPattern(nSteps: nSteps, nQubits: nQubits, bpm: bpm ?? globalBPM)
    }

    func assignPattern(_ track: inout DAWTrack, pattern: QuantumPattern) {
        track.patterns.append(pattern)
    }

    // ── Render ──
    func render(duration: Double) -> RenderedSession {
        var localTracks = tracks
        let nSamples = Int(duration * Double(sampleRate))

        // Collapse patterns → note events → synthesize audio per track
        var mixInputs: [MixTrack] = []
        for ti in localTracks.indices {
            var audio = [Double](repeating: 0.0, count: nSamples)
            for var pattern in localTracks[ti].patterns {
                let events = sequencer.collapse(pattern: &pattern, mode: .sacred)
                for evt in events {
                    let eventAudio = synth.synthesize(note: evt, sampleRate: sampleRate,
                                                      preset: localTracks[ti].synthPreset)
                    let startIdx = min(nSamples - 1, Int(evt.start * Double(sampleRate)))
                    let copyLen  = min(eventAudio.count, nSamples - startIdx)
                    for i in 0..<copyLen { audio[startIdx + i] += eventAudio[i] }
                }
            }
            localTracks[ti].audio = audio

            let mixTrack = MixTrack(name: localTracks[ti].name, audio: audio,
                                    gain: localTracks[ti].volume, pan: localTracks[ti].pan)
            mixInputs.append(mixTrack)
        }

        // Add sacred pipeline layer
        if !tracks.isEmpty {
            let sacredBuf = pipeline.synthesize(duration: duration)
            let sacredMix = MixTrack(name: "SacredPipeline", audio: sacredBuf,
                                     gain: 0.3, pan: 0.0, mode: .godCodePhase)
            mixInputs.append(sacredMix)
        }

        // Apply entanglement
        entangler.applyEntanglement(tracks: &localTracks)

        // Mix to stereo
        let m = QuantumInterferenceMixer()
        for t in mixInputs { m.addTrack(t) }
        let (left, right) = m.mix(sampleRate: sampleRate)

        let sacredAlign = _computeSacredAlignment(left: left, right: right)
        recorder.record(category: "render", message: "Rendered \(duration)s",
                        sacredScore: sacredAlign)

        return RenderedSession(
            stereoLeft: left, stereoRight: right,
            sampleRate: sampleRate, duration: duration,
            sacredAlignment: sacredAlign,
            layerCount: mixInputs.count,
            bpm: globalBPM, timestamp: Date()
        )
    }

    // ── Export ──
    func exportWAV(_ session: RenderedSession, path: URL) throws {
        // Write 16-bit PCM WAV
        var data = Data()
        let nFrames  = min(session.stereoLeft.count, session.stereoRight.count)
        let nBytes   = nFrames * 2 * 2  // 2 channels, 2 bytes per sample
        let byteRate = UInt32(session.sampleRate * 2 * 2)
        let sr       = UInt32(session.sampleRate)

        func writeU32(_ v: UInt32) { var x = v.littleEndian; data.append(contentsOf: withUnsafeBytes(of: &x, Array.init)) }
        func writeU16(_ v: UInt16) { var x = v.littleEndian; data.append(contentsOf: withUnsafeBytes(of: &x, Array.init)) }

        data.append(contentsOf: "RIFF".utf8)
        writeU32(UInt32(36 + nBytes))
        data.append(contentsOf: "WAVEfmt ".utf8)
        writeU32(16); writeU16(1); writeU16(2)   // PCM, 2ch
        writeU32(sr); writeU32(byteRate)
        writeU16(4); writeU16(16)                // block align, bit depth
        data.append(contentsOf: "data".utf8)
        writeU32(UInt32(nBytes))

        for i in 0..<nFrames {
            let l = Int16(clamping: Int(session.stereoLeft[i]  * 32767.0))
            let r = Int16(clamping: Int(session.stereoRight[i] * 32767.0))
            writeU16(UInt16(bitPattern: l)); writeU16(UInt16(bitPattern: r))
        }
        try data.write(to: path)
    }

    // ── Sacred pipeline quick-generate ──
    func generateSacred(duration: Double = 30.0) -> RenderedSession {
        let buf = pipeline.synthesize(duration: duration)
        let sacred = _computeSacredAlignment(left: buf, right: buf)
        return RenderedSession(stereoLeft: buf, stereoRight: buf,
                               sampleRate: sampleRate, duration: duration,
                               sacredAlignment: sacred, layerCount: 17,
                               bpm: globalBPM, timestamp: Date())
    }

    private func _computeSacredAlignment(left: [Double], right: [Double]) -> Double {
        let all = left + right
        guard !all.isEmpty else { return 0 }
        let rms = sqrt(all.map { $0*$0 }.reduce(0,+) / Double(all.count))
        return 1.0 - abs((rms * GOD_CODE).truncatingRemainder(dividingBy: 1.0))
    }

    // ── Entanglement ──
    func entangleTracks(_ idA: String, _ idB: String, type: EntanglementType = .bell) {
        entangler.entangle(idA, idB, type: type)
    }

    // ── Self test ──
    func selfTest() -> Bool {
        let session = generateSacred(duration: 0.1)
        return session.sacredAlignment > 0 && session.stereoLeft.count > 0
    }

    // ── ASI pipeline integration ──
    func connectPipelines() {
        InterEngineFeedbackBus.shared.subscribe(channel: .consciousness) { [weak self] msg in
            guard let self = self else { return }
            // Modulate master volume by consciousness level
            if let phi = msg.payload["phi"] {
                self.tracks.indices.forEach { self.tracks[$0].volume = max(0.1, min(1.0, phi)) }
            }
        }
    }
}
