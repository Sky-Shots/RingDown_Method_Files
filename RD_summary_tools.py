# ==============================================================
# RD_summary_tools.py
#
# Purpose:
# --------
# This helper file reads the CSV produced by RD6 and prints a
# simple statistical summary of many ring-down runs.
#
# It helps you quickly see:
#     • how f0 is drifting over time
#     • how τ (decay constant) is changing
#     • whether Q factor is getting lower (more damping)
#     • how RMSE behaves across runs (fit quality)
#
# This is extremely useful for:
#     • comparing SIM vs FILE runs
#     • checking stability over many experiments
#     • detecting noise or hardware issues
#
# What it prints:
# ---------------
# Example:
#     f0_hz : n=10 | min=4.9951e6  mean=4.9953e6  max=4.9956e6
#     tau_s : n=10 | min=8.2e-5    mean=9.0e-5    max=9.6e-5
#     Q     : n=10 | ...
#     rmse  : n=10 | ...
#
# Then it prints the last few runs so you can check the raw
# values without opening Excel.
#
# Notes:
# ------
# • This tool gracefully handles missing values or misnamed
#   columns (for example fft_rmse instead of fit_rmse).
# • This file does *not* modify the CSV — it only reads it.
#
# ==============================================================

import csv, math, statistics as stats
from pathlib import Path


def _get_float(row, *keys):
    """
    Return the first column that exists and can be converted to float.
    If nothing matches, return NaN.
    """
    for k in keys:
        if k in row and row[k] != "":
            try:
                return float(row[k])
            except ValueError:
                return math.nan
    return math.nan


def _col(values, name):
    """
    Format min/mean/max for a column (ignores NaNs).
    """
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return f"{name}: (no usable data)"
    return (f"{name}:  n={len(vals)} | "
            f"min={min(vals):.6g}  mean={stats.mean(vals):.6g}  max={max(vals):.6g}")


def rd_stats(csv_path="rd_results.csv", tail=5):
    """
    Print statistics for ring-down results stored in rd_results.csv.

    tail : number of most recent runs to print.
    """
    p = Path(csv_path)
    if not p.exists():
        print(f"[RD_stats] No CSV found: {csv_path}")
        return

    # -----------------------------
    # Load all rows from CSV
    # -----------------------------
    with p.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("[RD_stats] CSV has header but no rows.")
        return

    # -----------------------------
    # Extract columns
    # -----------------------------
    f0s   = [_get_float(r, "f0_hz")                for r in rows]
    taus  = [_get_float(r, "tau_s")                for r in rows]
    Qs    = [_get_float(r, "Q")                    for r in rows]
    rmses = [_get_float(r, "fit_rmse", "fft_rmse") for r in rows]

    # -----------------------------
    # Print summary statistics
    # -----------------------------
    print(f"=== RD Summary ({csv_path}) ===")
    print(_col(f0s,   "f0_hz"))
    print(_col(taus,  "tau_s"))
    print(_col(Qs,    "Q"))
    print(_col(rmses, "rmse"))
    print()

    # -----------------------------
    # Print last few runs
    # -----------------------------
    print(f"Last {min(tail, len(rows))} run(s):")
    fields = [
        "timestamp", "f0_hz", "tau_s", "Q",
        "fit_rmse", "alpha", "df_tau", "df_Q",
        "mode", "fs_hz", "guard_periods", "drop_frac"
    ]

    for r in rows[-tail:]:
        out = []
        for k in fields:
            if k in r and r[k] != "":
                out.append(f"{k}={r[k]}")
        print("  - " + " | ".join(out))


if __name__ == "__main__":
    rd_stats()

