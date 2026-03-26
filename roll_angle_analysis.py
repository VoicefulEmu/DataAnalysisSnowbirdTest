"""
Roll angle analysis for SnowbirdPrim HR flight data.

Integrates Gyro_X (roll rate, deg/s) over flight time to produce cumulative
roll angle (deg), starting from 0 at launch (Flight_Time = 0).

Two plots are produced:
  1. Roll angle during ascent  (0 s  → apogee)
  2. Roll angle during descent (apogee → end of flight data)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

# ── File paths ─────────────────────────────────────────────────────────────────
BASE = Path(r"c:\Users\lamor\OneDrive\src\DataAnalysisSnowbirdTest")
CSV  = BASE / "SnowbirdPrim HR_03-21-2026_10_52_17.csv"

# ── Known flight events (seconds, from hr_metrics.csv / summary CSV) ───────────
APO_TIME_S  = 38.6    # apogee drogue fire
MAIN_TIME_S = 217.9   # main chute fire

# ── Load & clean data ──────────────────────────────────────────────────────────
hr = pd.read_csv(CSV, low_memory=False)
hr.columns = [c.strip() for c in hr.columns]

for col in ["Flight_Time_(s)", "Gyro_X", "Gyro_Y", "Gyro_Z"]:
    hr[col] = pd.to_numeric(hr[col], errors="coerce")

hr = hr.dropna(subset=["Flight_Time_(s)", "Gyro_X"]).sort_values("Flight_Time_(s)")

# ── Restrict to powered + ballistic flight (from launch onward) ────────────────
flight = hr[hr["Flight_Time_(s)"] >= 0.0].copy().reset_index(drop=True)

t        = flight["Flight_Time_(s)"].to_numpy(dtype=float)
roll_rate = flight["Gyro_X"].to_numpy(dtype=float)   # deg/s

# ── Integrate roll rate → cumulative roll angle (deg) ─────────────────────────
# cumulative_trapezoid returns N-1 values; prepend 0 so length matches t.
roll_angle = np.concatenate(([0.0], cumulative_trapezoid(roll_rate, t)))

# ── Split into ascent / descent ────────────────────────────────────────────────
asc_mask  = t <= APO_TIME_S
desc_mask = t >= APO_TIME_S

t_asc    = t[asc_mask]
ang_asc  = roll_angle[asc_mask]

t_desc   = t[desc_mask]
ang_desc = roll_angle[desc_mask]

# ── Print summary ──────────────────────────────────────────────────────────────
print(f"Data points in flight  : {len(t):,}")
print(f"Ascent  samples (0→{APO_TIME_S}s): {asc_mask.sum():,}")
print(f"Descent samples ({APO_TIME_S}s→end): {desc_mask.sum():,}")
print(f"Roll angle at apogee   : {ang_asc[-1]:.1f} °")
print(f"Roll angle at end      : {ang_desc[-1]:.1f} °")
print(f"Total roll (flight)    : {roll_angle[-1]:.1f} °")

# ── Plot helper ────────────────────────────────────────────────────────────────
def make_roll_plot(t_seg, ang_seg, phase_label, event_time=None, event_label=None,
                   angle_at_phase_start=None):
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle(f"Roll Angle — {phase_label}\nSnowbirdPrim  |  2026-03-21",
                 fontsize=13, fontweight="bold")

    # Top panel: cumulative roll angle
    ax = axes[0]
    ax.plot(t_seg, ang_seg, color="tab:blue", lw=1.5, label="Cumulative roll angle")
    ax.axhline(0, color="grey", lw=0.6, ls="--")
    if event_time is not None and t_seg[0] <= event_time <= t_seg[-1]:
        ax.axvline(event_time, color="red", ls="--", lw=1.2, label=event_label)
    if angle_at_phase_start is not None:
        ax.axhline(angle_at_phase_start, color="grey", lw=0.8, ls=":",
                   label=f"Angle at phase start ({angle_at_phase_start:.0f}°)")
    ax.set_ylabel("Roll Angle (°)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Bottom panel: roll rate (Gyro_X slice)
    mask = (flight["Flight_Time_(s)"] >= t_seg[0]) & (flight["Flight_Time_(s)"] <= t_seg[-1])
    t_rr  = flight.loc[mask, "Flight_Time_(s)"].to_numpy(dtype=float)
    rr    = flight.loc[mask, "Gyro_X"].to_numpy(dtype=float)
    ax2   = axes[1]
    ax2.plot(t_rr, rr, color="tab:orange", lw=1, alpha=0.85, label="Roll rate (Gyro_X)")
    ax2.axhline(0, color="grey", lw=0.6, ls="--")
    if event_time is not None and t_seg[0] <= event_time <= t_seg[-1]:
        ax2.axvline(event_time, color="red", ls="--", lw=1.2)
    ax2.set_xlabel("Flight Time (s)", fontsize=11)
    ax2.set_ylabel("Roll Rate (°/s)", fontsize=11)
    ax2.legend(fontsize=9, loc = "upper right")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    return fig

# ── Figure 1: Ascent ───────────────────────────────────────────────────────────
fig_asc = make_roll_plot(
    t_asc, ang_asc,
    phase_label="Ascent (launch → apogee)",
    event_time=APO_TIME_S,
    event_label=f"Apogee / drogue fire ({APO_TIME_S} s)",
)

# ── Figure 2: Descent ─────────────────────────────────────────────────────────
fig_desc = make_roll_plot(
    t_desc, ang_desc,
    phase_label="Descent (apogee → landing)",
    event_time=MAIN_TIME_S,
    event_label=f"Main chute fire ({MAIN_TIME_S} s)",
    angle_at_phase_start=ang_asc[-1],
)

OUT_DIR = BASE / "SnowbirdPrim HR_03-21-2026_10_52_17_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

fig_asc.savefig(OUT_DIR / "roll_angle_ascent.png", dpi=150, bbox_inches="tight")
fig_desc.savefig(OUT_DIR / "roll_angle_descent.png", dpi=150, bbox_inches="tight")
print(f"Saved plots to {OUT_DIR}")

plt.show()
