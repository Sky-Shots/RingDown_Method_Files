# ==============================================================
# RD5 : Estimate Resonant Frequency from Zero-Crossings
#
# Purpose of RD5:
# ----------------
# After RD4 extracts the decay constant τ and Q, RD5 measures the
# *actual oscillation frequency* f0 during the free decay.
#
# The idea:
#   A decaying sine crosses zero every half-period.
#
#       Zero-cross →   π shift →   half-period
#
# So, if the time between zero-crossings is Δt_half:
#
#        T = 2 * mean(Δt_half)       (full period)
#        f0 = 1 / T
#
# RD5 performs:
#   1) Find all sign changes → zero-crossings
#   2) Interpolate the exact crossing time (linear interpolation)
#   3) Compute average period → f0
#   4) Print warnings if:
#        • too few crossings
#        • irregular spacing (noisy)
#        • extremely low amplitude (bad signal)
#
# Equations used:
# ----------------
# Zero-cross interpolation:
#       x(k) + frac * ( x(k+1) – x(k) ) = 0
#       frac = x(k) / ( x(k) – x(k+1) )
#
# Estimated full period:
#       T = 2 * mean( zt[i+1] – zt[i] )
#
# Resonant frequency:
#       f0 = 1 / T
#
# Inputs:
#   x1_relax : normalized clean signal from RD2
#   fs       : sampling rate (Hz)
#
# Outputs:
#   f0_relax_hz   : estimated resonant frequency (Hz)
#   n_cycles_used : number of cycles used
# ==============================================================

import numpy as np

def rd5_freq_from_relax(x1_relax, fs, min_crossings=6):
    """
    Estimate f0 from zero-crossings.
    Includes warnings + suggested fixes.
    """

    # -----------------------------
    # RD5-A : Find zero-crossings
    # -----------------------------
    x = x1_relax
    s = np.signbit(x)          # True = negative, False = positive
    idx = np.where(s[:-1] != s[1:])[0]   # sign changes

    if idx.size < min_crossings:
        print("WARNING (RD5): Not enough zero-crossings detected.")
        print("FIX: Check RD2 normalization or RD1 relax slicing — decay may be too short.")
        return np.nan, 0

    print(f"[RD5-A] crossings found: {idx.size}")

    # -----------------------------
    # RD5-B : Sub-sample crossing times
    # -----------------------------
    zt = []
    for k in idx:
        x0, x1 = x[k], x[k+1]
        frac = x0 / (x0 - x1 + 1e-12)   # linear interpolation
        zt.append((k + frac) / fs)

    zt = np.array(zt)
    print(f"[RD5-B] first 3 crossings (s): {zt[:3]}")

    # warning: irregular crossing spacing → noisy or distorted signal
    dt_test = np.diff(zt)
    if np.std(dt_test) > 0.1 * np.mean(dt_test):
        print("WARNING (RD5): Zero-crossing intervals vary a lot.")
        print("FIX: Check RD2 plot for noise or clipping. Verify f0 guess in RD3/RD4.")

    # -----------------------------
    # RD5-C : Compute frequency
    # -----------------------------
    dt = np.diff(zt)
    if dt.size < 2:
        print("WARNING (RD5): Too few intervals for reliable frequency estimate.")
        print("FIX: Ensure decay contains multiple clean oscillations.")
        return np.nan, 0

    T_est = 2.0 * np.mean(dt)     # full period (seconds)
    f0_relax = 1.0 / T_est        # Hz
    n_cycles_used = dt.size + 1

    print(f"[RD5-C] f0_relax = {f0_relax/1e6:.6f} MHz | cycles used = {n_cycles_used}")

    # final warning: unrealistic f0
    if f0_relax < 1e5 or f0_relax > 50e6:
        print("WARNING (RD5): Estimated f0 out of expected QCM range.")
        print("FIX: Likely incorrect zero-cross detection — inspect RD2 zoom plot.")

    return f0_relax, n_cycles_used

