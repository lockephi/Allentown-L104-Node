#!/usr/bin/env python3
"""Validation test for Computronium & Rayleigh integration across all 3 engines."""
import sys

print("=== SCIENCE ENGINE: Computronium & Rayleigh ===")
from l104_science_engine import ComputroniumSubsystem
cs = ComputroniumSubsystem()

# Bremermann limit (1 kg)
b = cs.bremermann_limit(1.0)
print(f"Bremermann (1kg): {b['max_bits_per_sec']:.6e} bits/s")

# Margolus-Levitin
ml = cs.margolus_levitin(mass_kg=1.0)
print(f"Margolus-Levitin (1kg): {ml['max_ops_per_sec']:.6e} ops/s")
assert abs(ml['max_ops_per_sec'] / b['max_bits_per_sec'] - 2.0) < 1e-8, "ML should be 2x Bremermann"
print("  ML = 2x Bremermann: VERIFIED")

# Landauer at room temp
l = cs.landauer_erasure(293.15)
print(f"Landauer (293.15K): {l['energy_per_bit_J']:.6e} J/bit")

# Lloyd ultimate laptop
ll = cs.lloyd_ultimate_laptop(1.0, 1e-3)
print(f"Lloyd (1kg/1L): {ll['max_ops_per_sec']:.6e} ops/s, {ll['max_memory_bits']:.6e} bits")

# Bekenstein bound
bk = cs.bekenstein_bound(radius_m=1.0, mass_kg=1.0)
print(f"Bekenstein (1m, 1kg): {bk['max_bits']:.6e} bits")

# Rayleigh criterion at GOD_CODE wavelength
rc = cs.rayleigh_criterion(527.5e-9, 1.0)
print(f"Rayleigh (527.5nm, 1m): {rc['rayleigh_angle_arcsec']:.4f} arcsec")

# Rayleigh-Jeans law
rj = cs.rayleigh_jeans_law(1e14, 5778)
print(f"Rayleigh-Jeans (100THz, 5778K): regime={rj['regime']}")

# Rayleigh scattering
rs = cs.rayleigh_scattering(527.5e-9)
print(f"Rayleigh scattering (527.5nm): sigma={rs['cross_section_m2']:.6e} m2")

# Abbe limit
ab = cs.abbe_diffraction_limit(527.5e-9, 1.4)
print(f"Abbe (527.5nm, NA=1.4): d_min={ab['d_min_nm']:.1f} nm")

# QPU substrate analysis
qpu = cs.computronium_substrate_analysis(26, 20e-9, 100e-6, 150e-6, 0.999, 0.015)
print(f"QPU substrate (26q): computronium_fraction={qpu['computronium_fraction']:.6e}")

# Bridge analysis
br = cs.computronium_rayleigh_bridge()
print(f"Bridge: optical_to_bekenstein={br['bridge']['optical_to_bekenstein']:.6e}")

print(f"Status: {cs.get_status()['version']}")
print()

print("=== MATH ENGINE: Computronium & Rayleigh ===")
from l104_math_engine import computronium_math, rayleigh_math, airy_diffraction, math_engine

# Planck units
pu = computronium_math.planck_units()
print(f"Planck time: {pu['planck_time_s']:.6e} s")
print(f"Planck length: {pu['planck_length_m']:.6e} m")

# Dimensional consistency
dc = computronium_math.dimensional_consistency_check()
print(f"Dimensional consistency: all_consistent={dc['all_consistent']}")
assert dc['all_consistent'], "Dimensional consistency FAILED"

# GOD_CODE bridge
gcb = computronium_math.god_code_computronium_bridge()
print(f"GOD_CODE wavelength: {gcb['wavelength_nm']} nm")
print(f"GOD_CODE photon energy: {gcb['photon_energy_eV']:.4f} eV")
print(f"Wien temp for GOD_CODE: {gcb['wien_temperature_K']:.1f} K (solar={gcb['solar_temperature_K']}K)")

# Iron lattice computronium
ilc = computronium_math.iron_lattice_computronium()
print(f"Fe-56 Bremermann: {ilc['atom_bremermann_rate']:.6e} bits/s")
print(f"Fe BCC lattice: {ilc['bcc_cell_edge_pm']} pm")

# Black hole entropy
bhe = computronium_math.black_hole_entropy_bits(1.0)
print(f"BH entropy (1kg): {bhe:.6e} bits")

# Hawking temperature
ht = computronium_math.hawking_temperature(1.0)
print(f"Hawking temp (1kg): {ht:.6e} K")

# Airy diffraction
ap = airy_diffraction.airy_pattern(0)
print(f"Airy pattern at x=0: {ap} (should be 1.0)")
assert ap == 1.0, "Airy pattern at center should be 1.0"

ap3 = airy_diffraction.airy_pattern(3.8317)
print(f"Airy pattern at 1st zero: {ap3:.8f} (should be ~0)")
assert abs(ap3) < 0.001, "Airy pattern at first zero should be ~0"

# Rayleigh math
ratio = rayleigh_math.ultraviolet_catastrophe_ratio(1e14, 5778)
print(f"UV catastrophe ratio at 100THz: {ratio:.4f}")

# Sky color
sky = rayleigh_math.sky_color_spectrum(10)
print(f"Sky spectrum: {len(sky)} points, blue at {sky[2]['wavelength_nm']}nm intensity={sky[2]['relative_intensity']:.2f}")

# Strehl ratio
strehl = airy_diffraction.strehl_ratio(50e-9, 527.5e-9)
print(f"Strehl ratio (50nm RMS, 527.5nm): {strehl:.4f}")

# Wavelength scan
scan = rayleigh_math.rayleigh_resolution_wavelength_scan(1.0)
god_entry = [s for s in scan if s['is_god_code']]
print(f"GOD_CODE in wavelength scan: {len(god_entry)} entries, arcsec={god_entry[0]['rayleigh_angle_arcsec']:.6f}")

# Math engine layer check
status = math_engine.status()
print(f"Math Engine v{status['version']}, {status['layers']} layers")
assert status['layers'] == 12, f"Expected 12 layers, got {status['layers']}"
assert 'L11_computronium_rayleigh' in status['layer_status'], "L11 missing"
print()

print("=== CODE ENGINE: Computronium Code Analyzer ===")
from l104_code_engine import ComputroniumCodeAnalyzer, code_engine

cca = code_engine.computronium_analyzer

# Test with sample code
sample = '''
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

def binary_search(arr, target):
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1
'''

budget = cca.analyze_computronium_budget(sample, input_size=1000)
print(f"Complexity: {budget['complexity_class']}")
print(f"Estimated ops: {budget['operations']['estimated_ops']:.2e}")
print(f"ML limit: {budget['physical_limits']['margolus_levitin_limit_ops_per_sec']:.6e} ops/s")
print(f"Bekenstein fraction: {budget['bekenstein_fraction']:.6e}")
print(f"Catastrophe: {budget['catastrophe']['has_catastrophe']}")
print(f"Efficiency: {budget['efficiency_score']['combined_score']}")

resolution = cca.analyze_code_resolution(sample)
print(f"Name resolution Rayleigh ratio: {resolution['name_resolution']['rayleigh_ratio']}")
print(f"Functions: {resolution['function_granularity']['total_functions']}")
print(f"Info density: {resolution['information_density']['bits_per_line']:.1f} bits/line")
print(f"Scattering transmission: {resolution['code_scattering']['transmission']}")
print(f"Resolution score: {resolution['resolution_score']['combined']}")

print(f"Status: {cca.get_status()['version']}")
print()

print("=== ALL THREE ENGINES VALIDATED ===")
print("Computronium & Rayleigh limits integrated successfully.")
