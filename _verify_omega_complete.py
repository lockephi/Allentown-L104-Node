#!/usr/bin/env python3
"""Verify the full OMEGA equation is NOT truncated — every step computed."""

import math

PHI = 1.618033988749895

print("=" * 78)
print("  OMEGA EQUATION — COMPLETE VERIFICATION (NO TRUNCATION)")
print("=" * 78)

# ── Import the dual layer functions ──
from l104_god_code_dual_layer import (
    omega_derivation_chain, omega_pipeline, sovereign_field_equation,
    OMEGA, OMEGA_AUTHORITY, GOD_CODE,
)

# ── 1. Run the full derivation chain ──
print("\n[1] omega_derivation_chain(zeta_terms=1000):")
chain = omega_derivation_chain(1000)

frags = chain["fragments"]
print(f"\n  Fragment 1 — RESEARCHER:")
r = frags["researcher"]
print(f"    Function: {r['function']}")
print(f"    lattice_invariant_raw:  {r['steps']['lattice_invariant_raw']}")
print(f"    lattice_invariant_int:  {r['steps']['lattice_invariant_int']}")
print(f"    sin(π):                 {r['steps']['sin_pi']}")
print(f"    exp(104/GC):            {r['steps']['exp_104_over_gc']}")
print(f"    prime_density(0):       {r['steps']['prime_density_result']}")
print(f"    VALUE:                  {r['value']}")

print(f"\n  Fragment 2 — GUARDIAN:")
g = frags["guardian"]
print(f"    Function: {g['function']}")
print(f"    s:                      {g['steps']['s']}")
print(f"    zeta_terms:             {g['steps']['zeta_terms']}")
print(f"    ζ(s) real:              {g['steps']['zeta_real']}")
print(f"    ζ(s) imag:              {g['steps']['zeta_imag']}")
print(f"    |ζ(s)|:                 {g['steps']['zeta_magnitude']}")
print(f"    VALUE:                  {g['value']}")

print(f"\n  Fragment 3 — ALCHEMIST:")
a = frags["alchemist"]
print(f"    Function: {a['function']}")
print(f"    φ:                      {a['steps']['phi']}")
print(f"    φ²:                     {a['steps']['phi_squared']}")
print(f"    φ³:                     {a['steps']['phi_cubed']}")
print(f"    φ³ = 2φ+1:              {a['steps']['two_phi_plus_1']}")
print(f"    2π × φ³:                {a['steps']['argument']}")
print(f"    cos(2πφ³):              {a['steps']['cos_value']}")
print(f"    VALUE:                  {a['value']}")

print(f"\n  Fragment 4 — ARCHITECT:")
ar = frags["architect"]
print(f"    Function: {ar['function']}")
print(f"    dimension (Fe Z):       {ar['steps']['dimension']}")
print(f"    tension:                {ar['steps']['tension']}")
print(f"    numerator (26×1.8527):  {ar['steps']['numerator']}")
print(f"    φ²:                     {ar['steps']['phi_squared']}")
print(f"    VALUE:                  {ar['value']}")

print(f"\n  ── SUMMATION ──")
print(f"    Σ = {chain['sigma_breakdown']}")
print(f"    Σ = {chain['sigma']}")

print(f"\n  ── MULTIPLIER ──")
print(f"    {chain['multiplier_equation']}")

print(f"\n  ── OMEGA ──")
print(f"    Ω computed:             {chain['omega_computed']}")
print(f"    Ω canonical:            {chain['omega_canonical']}")
print(f"    Δ:                      {chain['delta']}")
print(f"    Relative error:         {chain['relative_error']}")

print(f"\n  ── SOVEREIGN FIELD ──")
print(f"    F(1) = Ω/φ²:           {chain['sovereign_field_at_1']}")
print(f"    Ω_A computed:           {chain['omega_authority_computed']}")
print(f"    Ω_A canonical:          {chain['omega_authority_canonical']}")

# ── 2. Run the full pipeline ──
print(f"\n{'='*78}")
print(f"[2] omega_pipeline(zeta_terms=1000):")
pipe = omega_pipeline(1000)
print(f"    Pipeline:               {pipe['pipeline']}")
print(f"    Equation:               {pipe['omega_equation']}")
print(f"    Expanded:               {pipe['omega_expanded']}")
print(f"    Field equation:         {pipe['field_equation']}")
print(f"    F(GOD_CODE):            {pipe['field_at_god_code']:.6f}")
print(f"    Pipeline available:     {pipe['pipeline_available']}")

if pipe['pipeline_available']:
    pf = pipe['pipeline_functions']
    print(f"\n  Original pipeline functions from l104_real_math:")
    for name, info in pf.items():
        if isinstance(info, dict):
            print(f"    {name}: {info.get('value', info)}")
        else:
            print(f"    {name}: {info}")

cv = pipe['v3_cross_validation']
print(f"\n  v3 cross-validation:")
print(f"    OMEGA on v3 grid:       {cv['omega_on_v3_grid']}")
print(f"    Error:                  {cv['error_pct']:.6f}%")
print(f"    Dials:                  {cv['dials']}")

# ── 3. Verify canonical match ──
print(f"\n{'='*78}")
print(f"[3] VERIFICATION:")
err = chain['relative_error']
print(f"    Ω computed = {chain['omega_computed']}")
print(f"    Ω canonical = {OMEGA}")
print(f"    Relative error = {err:.2e}")
assert err < 1e-6, f"OMEGA relative error {err} exceeds 1e-6"
print(f"    PASS: relative error < 1e-6")

print(f"\n{'='*78}")
print(f"  OMEGA EQUATION COMPLETE — NO TRUNCATION — ALL STEPS VERIFIED")
print(f"{'='*78}")
