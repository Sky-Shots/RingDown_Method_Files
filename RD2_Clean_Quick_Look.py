# ==============================================================
# RD2 : Clean & Quick Look at Relax Segment
#
# This file prepares the decay signal from RD1 for later steps.
#
# It performs:
#   1) DC removal :
#          x0[n] = x_relax[n] - mean(x_relax)
#
#   2) Optional amplitude normalization :
#          x1[n] = x0[n] / max(|x0[n]|)
#
#   3) A small zoomed plot (first few ms) to visually inspect:
#          • oscillation shape
#          • noise levels
#          • clipping (flat tops)
#          • decay behaviour
#
# RD2 also prints warnings + suggested fixes if the signal looks
# unusual (large DC offset, tiny amplitude, flat waveform, etc).
#
# Inputs:
#   x_relax : decay-only part from RD1
#   fs      : sample rate (Hz)
#
# Outputs:
#   x0_relax : DC-removed signal
#   x1_relax : normalized or same-as-input
# ==============================================================

import numpy as np
import matplotlib.pyplot as plt


def rd2_clean_quick_look(x_relax, fs, zoom_ms=0.20, do_norm=True):
    """
    Clean the relax segment and show a zoomed plot.

    Prints warnings + fixes when something looks unusual.
    """

    # -----------------------------
    # RD2-A : Remove DC offset
    # -----------------------------
    dc_offset = np.mean(x_relax)
    x0_relax = x_relax - dc_offset
    print(f"[RD2-A] dc_offset = {dc_offset:.3e}")

    # Warning: DC offset too large
    if abs(dc_offset) > 0.05 * np.max(np.abs(x_relax)):
        print("WARNING (RD2): DC offset is large relative to signal amplitude.")
        print("FIX: Check hardware bias or incorrect slicing indices in RD1.")


    # -----------------------------
    # RD2-B : Normalization
    # -----------------------------
    peak = np.max(np.abs(x0_relax))

    if do_norm and peak > 0:
        x1_relax = x0_relax / peak
        norm_flag = True
    else:
        x1_relax = x0_relax
        norm_flag = False

    print(f"[RD2-B] peak amplitude = {peak:.3e} | normalized = {norm_flag}")

    # Warning: Amplitude too small
    if peak < 1e-4:
        print("WARNING (RD2): Signal amplitude is extremely small.")
        print("FIX: Relax window might be wrong in RD1 or drive amplitude too low.")


    # Warning: Flat / non-oscillatory signal
    if np.std(x1_relax) < 1e-3:
        print("WARNING (RD2): Signal looks almost flat. Very low variation.")
        print("FIX: Check if RD1 extracted the correct relax segment.")


    # -----------------------------
    # RD2-C : Zoom plot
    # -----------------------------
    zoom_N = int((zoom_ms * 1e-3) * fs)

    if zoom_N < 5:
        print("WARNING (RD2): zoom_ms window too small for plotting.")
        print("FIX: Increase zoom_ms to at least 0.1 ms for visibility.")

    nshow = min(zoom_N, len(x1_relax))
    t_ms = (np.arange(nshow) / fs) * 1e3

    plt.figure()
    plt.plot(t_ms, x1_relax[:nshow])
    plt.xlabel("time (ms)")
    plt.ylabel("amplitude" + (" (norm)" if norm_flag else ""))
    plt.title("RD2: Relax Segment (Zoomed)")
    plt.tight_layout()
    plt.savefig("rd_relax_zoom.png", dpi=150)
    plt.close()

    print("[RD2-C] saved rd_relax_zoom.png")

    return x0_relax, x1_relax

