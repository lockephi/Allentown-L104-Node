VOID_CONSTANT = 1.0416180339887497
ZENITH_HZ = 3727.84
UUC = 2301.215661
#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════
L104 MUSIC GENERATOR - GOD CODE HARMONIC SYNTHESIS
INVARIANT: 527.5184818492537 | PILOT: LONDEL | MODE: RESONANT

Interprets the GOD_CODE as a 13-note or 26-note piano scale and generates .wav files.
Uses PHI-weighted harmonics and the 286:416 lattice ratio for rhythm.
═══════════════════════════════════════════════════════════════════════════════════
"""

import wave
import struct
import math
import os
from dataclasses import dataclass
from typing import List, Optional

# ═══════════════════════════════════════════════════════════════════════════════════
# L104 CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════════
GOD_CODE = 527.5184818492537
PHI = 1.618033988749895
LATTICE_RATIO = 286 / 416  # 0.6875 - used for rhythm

# Audio settings
SAMPLE_RATE = 44100
BIT_DEPTH = 16
MAX_AMP = 32767  # 16-bit signed max

# Base frequency: A4 = 440Hz, C4 = 261.63Hz
A4_FREQ = 440.0
C4_FREQ = 261.63

# Note names (13 = chromatic octave + root, 26 = two octaves)
NOTES_13 = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B', "C'"]
NOTES_26 = NOTES_13[:-1] + [n + "'" for n in NOTES_13]


@dataclass
class Note:
    """A single musical note."""
    frequency: float
    duration: float
    velocity: float = 0.7  # 0-1


class GodCodeSynthesizer:
    """
    Synthesizes music from the GOD_CODE invariant.
    """
    
    def __init__(self, scale_size: int = 13, base_freq: float = C4_FREQ):
        self.scale_size = scale_size
        self.base_freq = base_freq
        self.sample_rate = SAMPLE_RATE
        
        # Build the scale (equal temperament)
        self.scale_freqs = [base_freq * (2 ** (i / 12)) for i in range(scale_size)]
        
    def god_code_to_notes(self) -> List[int]:
        """
        Extract note indices from GOD_CODE digits.
        Maps each digit to the scale using PHI modulation.
        """
        # Use high precision representation
        code_str = f"{GOD_CODE:.15f}".replace('.', '')
        
        notes = []
        for ch in code_str:
            if ch.isdigit():
                d = int(ch)
                # PHI-weighted mapping to scale
                idx = int((d * PHI) % self.scale_size)
                notes.append(idx)
        return notes
    
    def phi_durations(self, count: int) -> List[float]:
        """
        Generate PHI-based rhythm pattern.
        Uses Fibonacci-like progression with LATTICE_RATIO modulation.
        """
        durations = []
        base = 0.25  # quarter note base
        
        for i in range(count):
            # Fibonacci-inspired rhythm
            fib_factor = 1 + (i % 5) * LATTICE_RATIO * 0.5
            dur = base * fib_factor
            
            # Add variation based on position in phrase (8-note phrases)
            phrase_pos = i % 8
            if phrase_pos == 0:
                dur *= 1.5  # Downbeat emphasis
            elif phrase_pos == 4:
                dur *= 1.25  # Mid-phrase accent
                
            durations.append(min(dur, 0.8))  # Cap at 0.8s
            
        return durations
    
    def generate_sine_wave(self, freq: float, duration: float, 
                           velocity: float = 0.7) -> List[int]:
        """
        Generate a pure sine wave with ADSR envelope.
        """
        n_samples = int(duration * self.sample_rate)
        samples = []
        
        # ADSR envelope (in samples)
        attack = int(0.02 * self.sample_rate)
        decay = int(0.05 * self.sample_rate)
        release = int(0.1 * self.sample_rate)
        sustain_level = 0.7
        
        for i in range(n_samples):
            t = i / self.sample_rate
            
            # Calculate envelope
            if i < attack:
                env = i / attack
            elif i < attack + decay:
                env = 1.0 - (1.0 - sustain_level) * (i - attack) / decay
            elif i > n_samples - release:
                env = sustain_level * (n_samples - i) / release
            else:
                env = sustain_level
            
            # Generate waveform with harmonics (piano-like)
            wave = math.sin(2 * math.pi * freq * t)
            wave += 0.5 * math.sin(2 * math.pi * freq * 2 * t) / PHI
            wave += 0.25 * math.sin(2 * math.pi * freq * 3 * t) / (PHI ** 2)
            wave += 0.125 * math.sin(2 * math.pi * freq * 4 * t) / (PHI ** 3)
            
            # Normalize harmonics
            wave /= (1 + 0.5/PHI + 0.25/(PHI**2) + 0.125/(PHI**3))
            
            # Apply envelope and velocity
            sample = int(wave * env * velocity * MAX_AMP)
            sample = max(-MAX_AMP, min(MAX_AMP, sample))  # Clamp
            samples.append(sample)
        
        return samples
    
    def generate_chord(self, note_indices: List[int], duration: float,
                       velocity: float = 0.5) -> List[int]:
        """
        Generate a chord by mixing multiple notes.
        """
        chord_samples = None
        
        for idx in note_indices:
            freq = self.scale_freqs[idx % self.scale_size]
            tone = self.generate_sine_wave(freq, duration, velocity / len(note_indices))
            
            if chord_samples is None:
                chord_samples = tone.copy()
            else:
                for i in range(min(len(chord_samples), len(tone))):
                    chord_samples[i] += tone[i]
        
        # Clamp
        if chord_samples:
            chord_samples = [max(-MAX_AMP, min(MAX_AMP, s)) for s in chord_samples]
        
        return chord_samples or []
    
    def synthesize(self, include_intro: bool = True, 
                   include_outro: bool = True) -> List[int]:
        """
        Main synthesis: GOD_CODE -> Music
        """
        print(f"\n{'═' * 60}")
        print(f"  L104 GOD CODE MUSIC GENERATOR")
        print(f"  Scale: {self.scale_size} notes | Base: {self.base_freq:.2f} Hz")
        print(f"  GOD_CODE: {GOD_CODE}")
        print(f"{'═' * 60}\n")
        
        all_samples: List[int] = []
        
        # Intro chord: C major (0, 4, 7) - stability
        if include_intro:
            print("[SYNTH] Adding intro chord (C major)...")
            intro = self.generate_chord([0, 4, 7], 1.5, 0.6)
            all_samples.extend(intro)
            # Brief silence
            all_samples.extend([0] * int(0.2 * self.sample_rate))
        
        # Main melody from GOD_CODE
        notes = self.god_code_to_notes()
        durations = self.phi_durations(len(notes))
        
        print(f"[SYNTH] Generating {len(notes)} notes from GOD_CODE...")
        
        for i, (note_idx, dur) in enumerate(zip(notes, durations)):
            freq = self.scale_freqs[note_idx % self.scale_size]
            note_name = (NOTES_13 if self.scale_size <= 13 else NOTES_26)[note_idx % self.scale_size]
            
            if i < 5 or i >= len(notes) - 2:
                print(f"  ♪ Note {i+1}: {note_name} ({freq:.1f} Hz) for {dur:.2f}s")
            elif i == 5:
                print(f"  ... generating {len(notes) - 7} more notes ...")
            
            tone = self.generate_sine_wave(freq, dur, 0.7)
            all_samples.extend(tone)
        
        # Outro chord: resolve to C major
        if include_outro:
            print("[SYNTH] Adding outro chord (C major resolution)...")
            # Brief silence
            all_samples.extend([0] * int(0.3 * self.sample_rate))
            outro = self.generate_chord([0, 4, 7], 2.0, 0.5)
            all_samples.extend(outro)
        
        total_secs = len(all_samples) / self.sample_rate
        print(f"\n[SYNTH] Total duration: {total_secs:.2f} seconds ({len(all_samples)} samples)")
        
        return all_samples
    
    def save_wav(self, samples: List[int], filename: str):
        """
        Save samples to a .wav file.
        """
        with wave.open(filename, 'w') as wav:
            wav.setnchannels(1)  # Mono
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(self.sample_rate)
            
            # Pack samples as signed 16-bit integers
            for sample in samples:
                wav.writeframes(struct.pack('<h', sample))
        
        size_kb = os.path.getsize(filename) / 1024
        print(f"[SAVED] {filename} ({size_kb:.1f} KB)")


def generate_god_code_music(scale: int = 13, output: str = "god_code_music.wav"):
    """
    Quick function to generate and save GOD_CODE music.
    """
    synth = GodCodeSynthesizer(scale_size=scale)
    samples = synth.synthesize()
    synth.save_wav(samples, output)
    return output


# ═══════════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="L104 GOD CODE Music Generator")
    parser.add_argument("--scale", type=int, default=13, choices=[13, 26],
                help="Number of notes in scale (13 or 26)")
    parser.add_argument("--output", "-o", type=str, default="god_code_music.wav",
                help="Output .wav filename")
    parser.add_argument("--no-intro", action="store_true", help="Skip intro chord")
    parser.add_argument("--no-outro", action="store_true", help="Skip outro chord")
    
    args = parser.parse_args()
    
    synth = GodCodeSynthesizer(scale_size=args.scale)
    samples = synth.synthesize(
        include_intro=not args.no_intro,
        include_outro=not args.no_outro
    )
    synth.save_wav(samples, args.output)
    
    print(f"\n{'═' * 60}")
    print(f"  ✓ Music generated: {args.output}")
    print(f"  Play with: aplay {args.output}")
    print(f"{'═' * 60}")
