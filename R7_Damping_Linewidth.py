# ==============================================================
# RD7 : Damping Factor & Linewidth Estimation
#
# Purpose:
# --------
# After RD4 (τ extraction) and RD5 (f₀ extraction), this file
# computes additional physical quantities related to QCM damping:
#
#   1) Damping factor:
#          α = 1 / τ
#
#   2) Linewidth derived from τ:
#          Δf_τ = 1 / (2π τ)
#
#   3) Linewidth derived from Q:
#          Δf_Q = f₀ / Q
#
# These parameters quantify *how quickly* vibration energy is lost
# in the medium. Larger linewidth → stronger damping → lower Q.
#
# Why this is useful:
# -------------------
# • Helps compare MLA vs Ring-Down linewidth results
# • Detects excessive damping in electrolyte (battery SOH tracking)
# • Warns when τ or Q disagree with expected physics
#
# What RD7 also does:
# -------------------
# • Prints warnings when α, Δf_τ, or Δf_Q look unrealistic
# • Gives FIX suggestions (student-style)
# • Creates an overlay plot:
#       rd_linewidth_compare.png
#   showing τ-based vs Q-based linewidth difference
#
# Inputs:
#   tau_s : decay constant from RD4              (seconds)
#   f0_hz : resonant frequency from RD5          (Hz)
#   Q     : quality factor                       (unitless)
#
# Outputs (dict):
#   {
#     "alpha"  : α
#     "df_tau" : linewidth from τ
#     "df_Q"   : linewidth from Q
#     "warn"   : list of warnings
#     "fix"    : list of suggested fixes
#   }
#
# ==============================================================

import numpy as np
import matplotlib.pyplot as plt


def rd7_damping_linewidth(tau_s, f0_hz, Q):
    """
    Compute α, Δf_τ, and Δf_Q.
    Print warnings + fixes.
    Return a structured dictionary.
    """

    warnings = []
    fixes = []

    # ----------------------------------------------------------
    # RD7-A : Basic validation
    # ----------------------------------------------------------
    if tau_s is None or tau_s <= 0:
        w = "tau_s is non-positive; cannot compute damping or linewidth."
        f = "Check RD4 exponential fit; envelope may be corrupted."
        print("WARNING (RD7):", w)
        print("FIX:", f)
        return {"alpha": np.nan, "df_tau": np.nan, "df_Q": np.nan,
                "warn": [w], "fix": [f]}

    if f0_hz is None or f0_hz <= 0:
        w = "f0_hz is invalid; cannot compute linewidth."
        f = "Check zero-crossings in RD5; signal may be too noisy."
        print("WARNING (RD7):", w)
        print("FIX:", f)
        return {"alpha": np.nan, "df_tau": np.nan, "df_Q": np.nan,
                "warn": [w], "fix": [f]}

    if Q is None or Q <= 0:
        w = "Q is invalid; cannot compute Q-based linewidth."
        f = "Check RD4 and RD5, or inspect the RD2 and RD3 plots."
        print("WARNING (RD7):", w)
        print("FIX:", f)
        return {"alpha": np.nan, "df_tau": np.nan, "df_Q": np.nan,
                "warn": [w], "fix": [f]}

    # ----------------------------------------------------------
    # RD7-B : Compute damping factor + linewidths
    # ----------------------------------------------------------
    alpha = 1.0 / tau_s                # damping factor
    df_tau = 1.0 / (2 * np.pi * tau_s) # linewidth from τ
    df_Q = f0_hz / Q                   # linewidth from Q

    print(f"[RD7] alpha = {alpha:.3f}")
    print(f"[RD7] Δf_tau = {df_tau:.3f} Hz   (from tau)")
    print(f"[RD7] Δf_Q   = {df_Q:.3f} Hz   (from Q)")

    # ----------------------------------------------------------
    # RD7-C : Warnings + fixes
    # ----------------------------------------------------------
    # very small linewidth
    if df_tau < 0.1:
        warnings.append("df_tau extremely small (<0.1 Hz).")
        fixes.append("Decay too slow or envelope oversmoothed; check RD3 window.")
        print("WARNING (RD7): df_tau extremely small (<0.1 Hz).")
        print("FIX: Decay too slow or smoothing too strong — inspect RD3 & RD4.")

    # very large linewidth
    if df_tau > 1e5:
        warnings.append("df_tau very large (>100 kHz).")
        fixes.append("Decay too fast; signal too short or noisy. Check RD1 slicing.")
        print("WARNING (RD7): df_tau very large (>100 kHz).")
        print("FIX: Relax segment too short or noise too high — inspect RD1.")

    # mismatch between tau-based and Q-based linewidths
    if abs(df_tau - df_Q) / max(df_tau, df_Q) > 0.3:
        warnings.append("Δfτ and ΔfQ mismatch >30%.")
        fixes.append("Check Q calculation in RD4 and f0 from RD5 for consistency.")
        print("WARNING (RD7): Δf_tau and Δf_Q differ by more than 30%.")
        print("FIX: Check Q from RD4 and f0 from RD5; noise or wrong envelope region.")


    # ----------------------------------------------------------
    # RD7-D : Linewidth comparison plot
    # ----------------------------------------------------------
    plt.figure(figsize=(5,3))
    plt.bar(["df_tau", "df_Q"], [df_tau, df_Q], color=["#4a90e2", "#e24a4a"])
    plt.ylabel("Hz")
    plt.title("RD7: Linewidth Comparison (τ-based vs Q-based)")
    plt.tight_layout()
    plt.savefig("rd_linewidth_compare.png", dpi=150)
    plt.close()
    print("[RD7] saved rd_linewidth_compare.png")

    # ----------------------------------------------------------
    # Return results
    # ----------------------------------------------------------
    return {
        "alpha": alpha,
        "df_tau": df_tau,
        "df_Q": df_Q,
        "warn": warnings,
        "fix": fixes
    }

