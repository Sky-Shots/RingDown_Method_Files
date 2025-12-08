# ==============================================================
# RD6 : Final Summary + CSV Export
#
# Purpose of RD6:
# ----------------
# RD6 collects ALL the final results from the ring-down pipeline:
#
#    • Resonant frequency f0            (from RD5)
#    • Decay constant τ                 (from RD4)
#    • Quality factor Q = π f0 τ        (from RD4)
#    • Fit RMSE                          (from RD4)
#    • Damping factor α = 1/τ           (from RD7)
#    • Linewidth (τ-based) Δf_τ = 1/(2πτ)
#    • Linewidth (Q-based) Δf_Q = f0/Q
#
# Why we use RD6:
# ----------------
#  1) Saves each run into rd_results.csv
#  2) Lets us compare SIM vs FILE runs
#  3) Lets us detect drift in real QCM experiments
#  4) Ensures all values are validated before logging
#
# Equations:
# ----------
#   τ      : time constant of exponential decay
#   α      : damping factor
#               α = 1 / τ
#
#   Q      : quality factor
#               Q = π * f0 * τ
#
#   Δf_τ   : linewidth from τ
#               Δf_τ = 1 / (2π τ)
#
#   Δf_Q   : linewidth from Q
#               Δf_Q = f0 / Q
#
# Warnings + Fixes:
# -----------------
# RD6 prints useful checks for:
#   • unrealistic f0 values
#   • very low Q
#   • too small or too large linewidth
#   • high RMSE → poor fit in RD4
#   • NaN or invalid results
#
# ==============================================================

import csv
from pathlib import Path
from datetime import datetime


def rd6_save_results(f0_hz, tau_s, Q, rmse,
                     alpha=None, df_tau=None, df_Q=None,
                     csv_path="rd_results.csv", extras=None):
    """
    Save ring-down results to CSV and print full summary.
    """

    # ----------------------------------------------------------
    # RD6-A : Basic validation
    # ----------------------------------------------------------
    if f0_hz is None or f0_hz != f0_hz:
        print("WARNING (RD6): f0 is NaN. Results not saved.")
        print("FIX: RD5 zero-crossing detection failed (check RD2/RD5).")
        return

    if tau_s <= 0:
        print("WARNING (RD6): tau_s ≤ 0 is not physically valid.")
        print("FIX: RD4 exponential fit window may be incorrect.")
        return

    if Q <= 0:
        print("WARNING (RD6): Q ≤ 0 — invalid physical value.")
        print("FIX: Check RD4 and RD5 results for noise/clipping.")
        return

    # Additional warnings
    if rmse > 0.05:
        print("WARNING (RD6): High RMSE indicates a poor exponential fit.")
        print("FIX: Inspect rd_envelope_fit.png and adjust smoothing/fitting window.")

    if f0_hz < 1e5 or f0_hz > 50e6:
        print("WARNING (RD6): f0 outside expected QCM range (0.1–50 MHz).")
        print("FIX: RD5 zero-crossings may be corrupted by noise.")

    if Q < 50:
        print("WARNING (RD6): Q extremely low — strong damping or bad signal.")
        print("FIX: Check RD4 fit and RD3 envelope for irregularities.")

    # Linewidth warnings
    if df_tau is not None:
        if df_tau < 0.1:
            print("WARNING (RD6): Δf_τ is extremely small (<0.1 Hz).")
            print("FIX: Envelope may be too smooth or tau too large.")
        if df_tau > 1e5:
            print("WARNING (RD6): Δf_τ unusually large (>100 kHz).")
            print("FIX: Decay too short or noisy — check RD1 slicing.")

    if df_Q is not None:
        if df_Q < 0.1:
            print("WARNING (RD6): Δf_Q is extremely small (<0.1 Hz).")
            print("FIX: Q may be overestimated — check RD4 fitting.")
        if df_Q > 1e5:
            print("WARNING (RD6): Δf_Q is very large (>100 kHz).")
            print("FIX: Check if Q is incorrectly low or data corrupted.")


    # ----------------------------------------------------------
    # RD6-B : Build CSV row
    # ----------------------------------------------------------
    timestamp = datetime.now().isoformat(timespec="seconds")

    base_header = [
        "timestamp", "f0_hz", "tau_s", "Q", "fit_rmse",
        "alpha", "df_tau", "df_Q"
    ]
    base_row = [
        timestamp, f0_hz, tau_s, Q, rmse,
        alpha, df_tau, df_Q
    ]

    extras = extras or {}
    extra_keys = sorted(extras.keys())
    extra_vals = [extras[k] for k in extra_keys]

    header = base_header + extra_keys
    row    = base_row    + extra_vals

    # Append to CSV
    p = Path(csv_path)
    write_header = not p.exists()

    with p.open("a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)

    # ----------------------------------------------------------
    # RD6-C : Full printed summary
    # ----------------------------------------------------------
    print("--------------------------------------------------------------")
    print(f"[RD6] Results saved → {csv_path}")
    print(f"[RD6] f0          = {f0_hz/1e6:.6f} MHz")
    print(f"[RD6] tau         = {tau_s*1e6:.2f} µs")
    print(f"[RD6] Q           = {Q:.1f}")
    print(f"[RD6] RMSE        = {rmse:.3e}")

    if alpha is not None:
        print(f"[RD6] alpha       = {alpha:.3f}  (damping factor = 1/τ)")

    if df_tau is not None:
        print(f"[RD6] Δf_tau      = {df_tau:.3f} Hz  (linewidth from τ)")

    if df_Q is not None:
        print(f"[RD6] Δf_Q        = {df_Q:.3f} Hz  (linewidth from Q)")

    print("--------------------------------------------------------------")
    print("[RD6] All values logged successfully.")

