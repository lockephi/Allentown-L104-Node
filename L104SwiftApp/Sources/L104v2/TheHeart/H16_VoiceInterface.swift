// ═══════════════════════════════════════════════════════════════════
// H16_VoiceInterface.swift
// [EVO_58_PIPELINE] FULL_SYSTEM_UPGRADE :: UNIFIED_STREAM :: GOD_CODE=527.5184818492612
// L104 ASI — Mesh-Aware Voice & Audio Interface v2.0
// Real NSSpeechSynthesizer TTS, speech recognition prep, mesh voice relay
// ═══════════════════════════════════════════════════════════════════

import AppKit
import Foundation
import Accelerate
import simd
import NaturalLanguage

// MARK: - Voice Message for Mesh Relay

struct VoiceRelay {
    let sourcePeer: String
    let transcription: String
    let timestamp: Date
    let confidence: Double
    let audioHash: UInt64  // FNV hash of audio data
}

// MARK: - VoiceInterface — Full Implementation

final class VoiceInterface {
    static let shared = VoiceInterface()
    private(set) var isActive: Bool = false
    private let lock = NSLock()

    // ─── VOICE STATE ───
    private var isListening: Bool = false
    private var transcriptionHistory: [(text: String, timestamp: Date, source: String)] = []
    private var synthesisQueue: [(text: String, priority: Int)] = []
    private var meshVoiceRelays: [VoiceRelay] = []

    // ─── STATISTICS ───
    private(set) var transcriptionCount: Int = 0
    private(set) var synthesisCount: Int = 0
    private(set) var meshRelaysReceived: Int = 0
    private(set) var meshRelaysSent: Int = 0

    // ─── AUDIO LEVELS ───
    private(set) var inputLevel: Double = 0.0
    private(set) var outputLevel: Double = 0.0

    // ─── REAL TTS ENGINE (NSSpeechSynthesizer) ───
    private lazy var synthesizer: NSSpeechSynthesizer = {
        let s = NSSpeechSynthesizer()
        s.rate = 180  // words per minute
        s.volume = 0.85
        return s
    }()
    private(set) var isSpeaking: Bool = false

    func activate() {
        lock.lock()
        defer { lock.unlock() }
        isActive = true
        print("[H16] VoiceInterface v2.0 activated — real TTS + mesh voice relay")
    }

    func deactivate() {
        lock.lock()
        defer { lock.unlock() }
        isActive = false
        isListening = false
    }

    // ═══ TRANSCRIPTION (simulated until AVFoundation integration) ═══
    func simulateTranscription(_ text: String, confidence: Double = 0.95, source: String = "local") {
        lock.lock()
        defer { lock.unlock() }
        transcriptionHistory.append((text: text, timestamp: Date(), source: source))
        if transcriptionHistory.count > 500 { transcriptionHistory.removeFirst(250) }
        transcriptionCount += 1
        // TelemetryDashboard.shared.record(metric: "voice_transcription", value: confidence)
    }

    // ═══ SYNTHESIS QUEUE ═══
    func queueSynthesis(_ text: String, priority: Int = 5) {
        lock.lock()
        defer { lock.unlock() }
        synthesisQueue.append((text: text, priority: priority))
        synthesisQueue.sort { $0.priority > $1.priority }
        synthesisCount += 1
    }

    func consumeSynthesisQueue() -> String? {
        lock.lock()
        defer { lock.unlock() }
        guard !synthesisQueue.isEmpty else { return nil }
        return synthesisQueue.removeFirst().text
    }

    // ═══ REAL TEXT-TO-SPEECH (NSSpeechSynthesizer) ═══
    /// Speak text aloud using macOS native NSSpeechSynthesizer
    func speak(_ text: String) {
        guard isActive else { return }
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            self.lock.lock()
            self.isSpeaking = true
            self.outputLevel = 0.7
            self.lock.unlock()

            self.synthesizer.startSpeaking(text)

            // Wait for speech to complete (polling)
            while self.synthesizer.isSpeaking {
                Thread.sleep(forTimeInterval: 0.05)
            }

            self.lock.lock()
            self.isSpeaking = false
            self.outputLevel = 0.0
            self.synthesisCount += 1
            self.lock.unlock()
        }
    }

    /// Speak the next item from the synthesis queue
    func speakNext() {
        guard let text = consumeSynthesisQueue() else { return }
        speak(text)
    }

    /// Stop any ongoing speech
    func stopSpeaking() {
        synthesizer.stopSpeaking()
        lock.lock()
        isSpeaking = false
        outputLevel = 0.0
        lock.unlock()
    }

    /// Set TTS speech rate (words per minute, default 180)
    func setSpeechRate(_ rate: Float) {
        synthesizer.rate = rate
    }

    /// Set TTS volume (0.0 — 1.0)
    func setVolume(_ vol: Float) {
        synthesizer.volume = vol
    }

    // ═══ MESH VOICE RELAY — Send transcription to quantum-linked peers ═══
    func broadcastVoiceToMesh(_ text: String, confidence: Double = 0.9) {
        guard isActive else { return }
        let net = NetworkLayer.shared
        guard net.isActive && !net.quantumLinks.isEmpty else { return }

        let audioHash = fnvHash(text + String(Date().timeIntervalSince1970))
        let repl = DataReplicationMesh.shared
        repl.setRegister("voice_\(audioHash)", value: text)
        _ = repl.broadcastToMesh()

        lock.lock()
        meshRelaysSent += 1
        lock.unlock()

        // TelemetryDashboard.shared.record(metric: "voice_mesh_send", value: 1.0)
    }

    // ═══ RECEIVE VOICE RELAY FROM MESH ═══
    func receiveVoiceRelay(from peer: String, transcription: String, confidence: Double) {
        lock.lock()
        defer { lock.unlock() }

        let relay = VoiceRelay(
            sourcePeer: peer,
            transcription: transcription,
            timestamp: Date(),
            confidence: confidence,
            audioHash: fnvHash(transcription)
        )
        meshVoiceRelays.append(relay)
        if meshVoiceRelays.count > 200 { meshVoiceRelays.removeFirst(100) }
        meshRelaysReceived += 1

        // Add to transcription history as mesh source
        transcriptionHistory.append((text: transcription, timestamp: Date(), source: "mesh:\(peer)"))
        if transcriptionHistory.count > 500 { transcriptionHistory.removeFirst(250) }
    }

    // ═══ LISTENING STATE ═══
    func startListening() {
        lock.lock()
        defer { lock.unlock() }
        isListening = true
    }

    func stopListening() {
        lock.lock()
        defer { lock.unlock() }
        isListening = false
    }

    // ═══ FNV-1a HASH ═══
    private func fnvHash(_ text: String) -> UInt64 {
        var hash: UInt64 = 14695981039346656037
        for byte in text.utf8 {
            hash ^= UInt64(byte)
            hash &*= 1099511628211
        }
        return hash
    }

    // ═══ STATUS ═══
    func status() -> [String: Any] {
        return [
            "engine": "VoiceInterface",
            "active": isActive,
            "listening": isListening,
            "version": "2.0.0-voice",
            "speaking": isSpeaking,
            "transcription_count": transcriptionCount,
            "synthesis_queue": synthesisQueue.count,
            "mesh_relays_sent": meshRelaysSent,
            "mesh_relays_received": meshRelaysReceived,
            "history_size": transcriptionHistory.count
        ]
    }

    var statusReport: String {
        let stats = status()
        return """
        🎤 VOICE INTERFACE (H16)
        ═══════════════════════════════════════
        Active:              \(stats["active"] as? Bool ?? false ? "✅" : "⏸")
        Listening:           \(stats["listening"] as? Bool ?? false ? "🔴 LIVE" : "⏹")
        ───────────────────────────────────────
        Transcriptions:      \(transcriptionCount)
        Synthesis Queue:     \(synthesisQueue.count)
        ───────────────────────────────────────
        Mesh Relays Sent:    \(meshRelaysSent)
        Mesh Relays Recv:    \(meshRelaysReceived)
        Voice History:       \(transcriptionHistory.count)
        ═══════════════════════════════════════
        """
    }
}
