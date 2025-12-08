# ==============================================================
# RD3 : Envelope Estimation (Time-Domain Amplitude Decay)
#
# Purpose of RD3:
# ----------------
# After RD2 cleans the relax waveform, RD3 extracts a smooth
# amplitude envelope from the oscillating signal. The envelope
# shows how the vibration decays over time, which is needed for
# RD4 (exponential fitting).
#
# Steps performed:
#
# 1) Magnitude envelope:
#        env_abs[n] = | x1_relax[n] |
#
# 2) Moving-average smoothing over ~ half a vibration period:
#        M = floor( (fs / f0) / 2 )
#        env[n] = (1/M) * Σ env_abs[n-k], for k = 0..M-1
#
# 3) Save envelope plot:
#        rd_envelope.png
#
# Why envelope is needed:
# ------------------------
# • RD4 fits an exponential function:
#        A(t) = A0 * exp(-t / tau)
# • The envelope provides A(t) for that fit.
#
# Warnings printed:
# -----------------
# • If smoothing window M is too small or too large
# • If envelope is almost flat (no visible decay)
# • If f0 or fs values cause unstable envelope
#
# Fixes suggested:
# ----------------
# • Adjust f0 guess used for envelope smoothing
# • Check RD2 normalization and cleaning
# • Check RD1 relax slicing if decay does not start at the beginning
#
# Inputs:
#   x1_relax : cleaned + normalized relax segment
#   fs       : sample rate in Hz
#   f0       : estimated or test resonance frequency
#
# Output:
#   env     : smoothed amplitude envelope for RD4
# ==============================================================

import numpy as np
import matplotlib.pyplot as plt


def rd3_envelope(x1_relax, fs, f0):
    """
    Create amplitude envelope using magnitude + moving average.
    Also prints warnings and suggested fixes.
    """

    # -----------------------------
    # RD3-A : Magnitude envelope
    # -----------------------------
    env_abs = np.abs(x1_relax)
    print(f"[RD3-A] env_abs computed | length={env_abs.size}")

    # Warning: signal too flat
    if np.std(env_abs) < 1e-3:
        print("WARNING (RD3): envelope magnitude is almost flat.")
        print("FIX: check RD2 cleaning; relax segment may not be correct.")


    # -----------------------------
    # RD3-B : Moving-average smooth
    # -----------------------------
    samples_per_period = fs / f0

    # window size = half-period (minimum 5 samples)
    M = int(max(5, samples_per_period // 2))

    # warning if M too small
    if M < 5:
        print("WARNING (RD3): smoothing window M is very small.")
        print("FIX: check f0 value; it may be too large or incorrect.")

    # warning if M too large (over-smoothed envelope)
    if M > len(env_abs) // 4:
        print("WARNING (RD3): smoothing window too large, envelope may lose detail.")
        print("FIX: check f0 estimate or relax length in RD1.")

    kernel = np.ones(M) / M
    env = np.convolve(env_abs, kernel, mode="same")

    print(f"[RD3-B] envelope smoothed | M={M} samples")


    # -----------------------------
    # RD3-C : Envelope plot
    # -----------------------------
    t_ms = (np.arange(len(env)) / fs) * 1e3

    plt.figure(figsize=(6,3))
    plt.plot(t_ms, env, label="envelope")
    plt.xlabel("time (ms)")
    plt.ylabel("amplitude (a.u.)")
    plt.title("RD3: Envelope (moving average)")
    plt.tight_layout()

    # mark start of decay
    plt.plot(t_ms[0], env[0], "ro")
    plt.annotate("start of decay",
                 xy=(t_ms[0], env[0]),
                 xytext=(t_ms[0] + 0.05, env[0] * 0.9),
                 arrowprops=dict(arrowstyle="->", color="red"),
                 color="red", fontsize=9)

    plt.savefig("rd_envelope.png", dpi=150)
    plt.close()
    print("[RD3-C] saved rd_envelope.png")

    return env

