# ============================================================
# RD0 + RD1 : Mode selection, basic knobs, and data loading
#
# This file performs two important jobs:
#
# (1) RD0 : sets up all the user-controlled parameters
#           such as SIM/FILE mode, file names, sample rate,
#           burst length and relax length.
#
# (2) RD1 : depending on the mode:
#             • SIM mode → generates a synthetic burst + decay
#                   x_relax(t) = A0 * exp(-t / tau_sim) * sin(2π f0 t)
#
#             • NEW FILE mode → loads ringdown_data.raw and
#               ringdown_data.json using NEW metadata:
#                   sample_rate_hz
#                   capture_start_offset_rel   (bytes)
#                   capture_bytes              (bytes)
#
#             • OLD FILE mode → loads older raw-only or
#               writer_offset_bytes-based files.
#               Auto-detection:
#                    - If JSON with OLD metadata exists → slice
#                    - Otherwise → use full raw file
#
# After RD0 + RD1, THIS FILE RUNS THE FULL PIPELINE:
#       RD2 → clean
#       RD3 → envelope
#       RD4 → exponential fit (tau)
#       RD5 → zero-crossing frequency
#       RD6 → save results (CSV)
#       RD7 → damping + linewidth checks
#
# Note:
# • RD2–RD7 are permanent pipeline stages.
# • Only the placement of this main block is temporary.
# • This file does not create .raw or .json — the FPGA does.
# ============================================================

import numpy as np
import json
import os
import math
import matplotlib.pyplot as plt

from RD2_Clean_Quick_Look import rd2_clean_quick_look
from RD3_Envelope import rd3_envelope
from RD4_Fit_Decay import rd4_fit_decay
from RD5_freq_est import rd5_freq_from_relax
from RD6_summary import rd6_save_results
from RD_summary_tools import rd_stats
from RD7_Damping_Linewidth import rd7_damping_linewidth


# ============================================================
# RD0 : USER KNOBS
# ============================================================

SIM_MODE       = False          # Synthetic mode
NEW_FILE_MODE  = True           # New RD acquisition metadata
OLD_FILE_MODE  = False          # Old Lucas metadata format

RAW_FILE    = "ringdown_data.raw"
META_FILE   = "ringdown_data.json"

BURST_LEN_S = 200e-6
RELAX_LEN_S = 400e-6

fs          = 125e6
f0          = 4_995_000

# --- SAFETY CHECK: Exactly one mode must be chosen ---
mode_count = sum([SIM_MODE, NEW_FILE_MODE, OLD_FILE_MODE])
if mode_count != 1:
    raise SystemExit("[RD0 ERROR] Exactly ONE of SIM_MODE / NEW_FILE_MODE / OLD_FILE_MODE must be True.")

if SIM_MODE:
    print("[RD0] mode = SIM (synthetic test)")
elif NEW_FILE_MODE:
    print("[RD0] mode = NEW FILE MODE (new metadata)")
else:
    print("[RD0] mode = OLD FILE MODE (Lucas metadata)")


# ============================================================
# RD1 : LOAD OR SYNTHESIZE RELAX SEGMENT
# ============================================================

def rd0_rd1_mode_knobs_load_slice():
    """Return (fs, x_relax). Handles SIM, NEW FILE, and OLD FILE modes."""

    # --------------------------------------------------------
    # SIM MODE
    # --------------------------------------------------------
    if SIM_MODE:
        burst_len_N = int(BURST_LEN_S * fs)
        relax_len_N = int(RELAX_LEN_S * fs)
        N_total     = burst_len_N + relax_len_N

        t = np.arange(N_total) / fs
        x = np.zeros(N_total, dtype=float)

        # driven burst
        x[0:burst_len_N] = np.sin(2*np.pi*f0*t[0:burst_len_N])

        # exponential decay
        A0      = 1.0
        tau_sim = 100e-6
        t_relax = t[burst_len_N:] - t[burst_len_N]
        x[burst_len_N:] = A0 * np.exp(-t_relax/tau_sim) * np.sin(2*np.pi*f0*t_relax)

        x_relax = x[burst_len_N:]
        ms = (x_relax.size / fs) * 1e3
        print(f"[RD1/SIM] fs={fs} | relax={x_relax.size} samples (~{ms:.2f} ms)")
        return fs, x_relax


    # --------------------------------------------------------
    # NEW FILE MODE
    # --------------------------------------------------------
    if NEW_FILE_MODE:

        if not os.path.exists(META_FILE):
            raise SystemExit(f"[RD1/NEW][ERROR] Missing metadata: {META_FILE}")

        if not os.path.exists(RAW_FILE):
            raise SystemExit(f"[RD1/NEW][ERROR] Missing raw file: {RAW_FILE}")

        with open(META_FILE, 'r') as f:
            meta = json.load(f)

        fs_file   = float(meta["sample_rate_hz"])
        start_rel = int(meta["capture_start_offset_rel"])
        nbytes    = int(meta["capture_bytes"])

        print(f"[RD1/NEW] fs={fs_file} | start_rel={start_rel} bytes | size={nbytes} bytes")

        raw = np.fromfile(RAW_FILE, dtype=np.int16).astype(np.float64)

        start_samp = start_rel // 2
        nsamp      = nbytes // 2
        end_samp   = start_samp + nsamp

        print(f"[RD1/NEW] slicing samples [{start_samp}:{end_samp}] | total={raw.size}")

        x_relax = raw[start_samp:end_samp]
        ms = (x_relax.size / fs_file) * 1e3

        print(f"[RD1/NEW] relax = {x_relax.size} samples (~{ms:.2f} ms)")
        return fs_file, x_relax


    # --------------------------------------------------------
    # OLD FILE MODE — auto-detect JSON or raw-only
    # --------------------------------------------------------
    if OLD_FILE_MODE:

        if not os.path.exists(RAW_FILE):
            raise SystemExit(f"[RD1/OLD][ERROR] Missing raw file: {RAW_FILE}")

        raw = np.fromfile(RAW_FILE, dtype=np.int16).astype(np.float64)

        # Case 1: JSON exists → use OLD slicing
        if os.path.exists(META_FILE):
            print(f"[RD1/OLD] Found metadata JSON: {META_FILE}")

            with open(META_FILE, "r") as f:
                meta = json.load(f)

            if ("writer_offset_bytes" in meta) and ("capture_bytes" in meta):
                start_old  = int(meta["writer_offset_bytes"])
                nbytes_old = int(meta["capture_bytes"])
                fs_file    = float(meta.get("sample_rate_hz", fs))

                print(f"[RD1/OLD] fs={fs_file} | start={start_old} bytes | size={nbytes_old} bytes")

                start_samp = start_old // 2
                nsamp      = nbytes_old // 2
                end_samp   = start_samp + nsamp

                print(f"[RD1/OLD] slicing samples [{start_samp}:{end_samp}] | total={raw.size}")

                x_relax = raw[start_samp:end_samp]
                ms = (x_relax.size / fs_file) * 1e3
                print(f"[RD1/OLD] relax = {x_relax.size} samples (~{ms:.2f} ms)")
                return fs_file, x_relax

            print("[RD1/OLD][WARNING] JSON exists but does not contain OLD metadata keys.")
            print("[RD1/OLD] Falling back to full raw file.")

        else:
            print("[RD1/OLD] No metadata JSON found → using raw-only mode.")

        # Case 2: full-raw fallback
        fs_file = fs
        x_relax = raw
        ms = (x_relax.size / fs_file) * 1e3

        print(f"[RD1/OLD] Loaded raw-only file '{RAW_FILE}' with {raw.size} samples.")
        print(f"[RD1/OLD] relax = {raw.size} samples (~{ms:.2f} ms)")
        return fs_file, x_relax


# ============================================================
# TEMPORARY MAIN DRIVER — executes RD2 → RD7
# ============================================================

if __name__ == "__main__":

    # RD1
    fs_out, x_relax = rd0_rd1_mode_knobs_load_slice()
    print("[MAIN] RD0–RD1 complete")

    # RD2
    x0_relax, x1_relax = rd2_clean_quick_look(x_relax, fs_out)
    print("[DRIVER] RD2 done")

    # RD3
    env = rd3_envelope(x1_relax, fs_out)
    print("[DRIVER] RD3 done")

    # RD4
    tau_s, Q_env, fit_rmse, fit_slice = rd4_fit_decay(env, fs_out, f0)
    print(f"[DRIVER] RD4 done | tau={tau_s*1e6:.2f} µs | Q_env={Q_env:.1f}")

    # RD5
    f0_relax_hz, ncyc = rd5_freq_from_relax(x1_relax, fs_out)
    print(f"[DRIVER] RD5 done | f0_relax={f0_relax_hz/1e6:.6f} MHz")

    # final Q
    Q_final = math.pi * f0_relax_hz * tau_s

    # RD7
    results7 = rd7_damping_linewidth(tau_s, f0_relax_hz, Q_final)
    print(f"[DRIVER] RD7 done | alpha={results7['alpha']:.3f}")

    # RD6
    rd6_save_results(
        f0_hz=f0_relax_hz,
        tau_s=tau_s,
        Q=Q_final,
        rmse=fit_rmse,
        csv_path="rd_results.csv",
        extras={
            "mode": "SIM" if SIM_MODE else ("NEW" if NEW_FILE_MODE else "OLD"),
            "fs_hz": fs_out,
            "f0_guess_hz": f0,
            "relax_samples": int(x1_relax.size),
            "fit_slice": str(fit_slice)
        }
    )
    print("[DRIVER] RD6 done | results saved")

    rd_stats("rd_results.csv", tail=3)

