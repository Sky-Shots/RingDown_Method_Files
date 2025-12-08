# ==============================================================
# RD4 : Fit Exponential Decay to Envelope
#
# Purpose of RD4:
# ----------------
# RD3 gives a smoothed envelope E(t).  RD4 fits this envelope to
# the physical exponential model:
#
#     E(t) = A0 * exp(-t / τ)
#
# where:
#     τ  = decay time constant (seconds)
#     Q  = π f0 τ     (quality factor of the resonator)
#
# This file:
#   1) Chooses a clean fitting region (skip filter edges + stop
#      when envelope has fallen to a fraction of its initial value)
#
#   2) Converts envelope to log-scale to linearize the exponential:
#
#         ln(E(t)) = ln(A0) - t / τ
#
#      Thus a straight-line fit gives the slope b:
#
#         b ≈ -1/τ      →      τ = -1/b
#
#   3) Computes:
#         τ (time constant)
#         Q_env = π f0 τ
#         RMSE = error between measured envelope and fitted curve
#
#   4) Prints warnings and FIX suggestions if:
#        • the fit region is too small
#        • the envelope does not decay properly
#        • RMSE is too large (bad fit)
#        • f0 or fs values look suspicious
#
# Inputs:
#   env          : Envelope array from RD3
#   fs (Hz)      : Sample rate
#   f0 (Hz)      : Resonance frequency estimate
#   guard_periods: Number of cycles skipped at start
#   drop_frac    : Fit stops when envelope falls to this fraction
#
# Outputs:
#   tau_s        : time constant (seconds)
#   Q_env        : quality factor from decay
#   fit_rmse     : fitting error
#   fit_slice    : (i0, i1) index window used for fitting
# ==============================================================

import numpy as np
import matplotlib.pyplot as plt


def rd4_fit_decay(env, fs, f0, guard_periods=3, drop_frac=0.1):
    """
    Fit the exponential decay to the envelope curve.
    Includes detailed warnings + fixes.
    """

    # -----------------------------
    # RD4-A : choose clean fit window
    # -----------------------------
    samples_per_period = int(fs / f0)

    # start fitting after several periods
    i0 = guard_periods * samples_per_period

    if i0 >= len(env):
        print("WARNING (RD4): guard_periods too large, no envelope left to fit.")
        print("FIX: reduce guard_periods (1–3) or verify f0 value.")
        i0 = 0  # fallback

    env0 = env[i0] if i0 < len(env) else env[0]

    # find where envelope falls to drop_frac
    below_idx = np.where(env[i0:] <= env0 * drop_frac)[0]
    if below_idx.size > 0:
        i1 = i0 + below_idx[0]
    else:
        i1 = len(env)

    # time axis
    t = np.arange(len(env)) / fs
    t_fit = t[i0:i1]
    y_fit = env[i0:i1]

    # warnings
    if len(y_fit) < 20:
        print("WARNING (RD4): very short fitting region, fit may be inaccurate.")
        print("FIX: check relax length in RD1 or reduce drop_frac.")

    print(f"[RD4-A] fit window: i0={i0} i1={i1} | "
          f"t0={t_fit[0]*1e3:.3f} ms → t1={t_fit[-1]*1e3:.3f} ms")


    # -----------------------------
    # RD4-B : log–linear fit
    # -----------------------------
    # log transform (avoid log(0))
    ln_y = np.log(y_fit + 1e-12)

    # linear regression: ln(E) ≈ a + b t
    b, a = np.polyfit(t_fit, ln_y, 1)

    # exponential parameters
    tau_s = -1.0 / b
    Q_env = np.pi * f0 * tau_s

    print(f"[RD4-B] τ = {tau_s*1e6:.2f} µs | Q_env = {Q_env:.1f}")


    # -----------------------------
    # RD4-C : error check + overlay plot
    # -----------------------------
    y_pred = np.exp(a + b * t_fit)
    fit_rmse = np.sqrt(np.mean((y_fit - y_pred) ** 2))

    # warnings about fit quality
    if fit_rmse > 0.05:
        print("WARNING (RD4): RMSE indicates a poor exponential fit.")
        print("FIX: check RD3 envelope smoothing or verify f0 used here.")
    if tau_s < 0:
        print("WARNING (RD4): negative tau detected — incorrect fit.")
        print("FIX: inspect envelope, check for noise or inverted signal.")

    # Plot
    plt.figure(figsize=(6,3))
    plt.plot(t * 1e3, env, label="envelope")
    plt.plot(t_fit * 1e3, y_pred, label="exp fit", linewidth=2)
    plt.axvline(t[i0] * 1e3, ls="--", color="gray", label="fit start")
    plt.xlabel("time (ms)")
    plt.ylabel("envelope (a.u.)")
    plt.title("RD4: Exponential Fit → τ, Q")
    plt.legend()
    plt.tight_layout()
    plt.savefig("rd_envelope_fit.png", dpi=150)
    plt.close()
    print(f"[RD4-C] rmse={fit_rmse:.3e} | saved rd_envelope_fit.png")

    return tau_s, Q_env, fit_rmse, (i0, i1)

