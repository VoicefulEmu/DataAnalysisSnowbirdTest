from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from path_config import default_analysis_dir, default_hr_csv, default_lr_csv, ensure_standard_dirs


# ---------------------------------------------------------------------------
# Signal-processing helpers
# ---------------------------------------------------------------------------

def find_rise_onset_time(time_s, signal, reference_time, search_pre_s, search_post_s):
    """Find the time where *signal* first departs from baseline toward its peak."""
    t = np.asarray(time_s, dtype=float)
    x = np.asarray(signal, dtype=float)
    valid = np.isfinite(t) & np.isfinite(x)
    if valid.sum() < 7:
        return float(reference_time)

    t, x = t[valid], x[valid]
    mask = (t >= reference_time - search_pre_s) & (t <= reference_time + search_post_s)
    if mask.sum() < 7:
        return float(reference_time)

    tw, xw = t[mask], x[mask]

    # Light smoothing suppresses IMU noise before slope estimation.
    window = min(9, len(xw) if len(xw) % 2 == 1 else len(xw) - 1)
    if window < 3:
        return float(reference_time)
    xw_smooth = np.convolve(xw, np.ones(window) / window, mode="same")

    slope = np.gradient(xw_smooth, tw)
    i_peak = int(np.nanargmax(xw_smooth))
    if i_peak < 4:
        return float(reference_time)

    baseline = float(np.nanmedian(xw_smooth[: max(4, int(0.25 * i_peak))]))
    rise = float(xw_smooth[i_peak]) - baseline
    if not np.isfinite(rise) or rise <= 0:
        return float(reference_time)

    # Pick an early but stable departure from baseline so Event 2 shows
    # the near-zero gyro_x region before the rapid spin-up.
    trigger_level = baseline + max(0.10 * rise, 25.0)
    sustain_pts = min(5, max(3, len(xw_smooth) // 80))

    for i in range(0, max(1, i_peak - sustain_pts)):
        if xw_smooth[i] < trigger_level:
            continue
        if not np.all(np.isfinite(xw_smooth[i : i + sustain_pts])):
            continue
        if np.nanmean(slope[i : i + sustain_pts]) <= 0:
            continue
        return float(tw[i])

    return float(tw[int(np.nanargmax(slope[: i_peak + 1]))])


def find_post_min_time(time_s, signal, start_time, search_post_s):
    """Return the time of the minimum in *signal* within a window after *start_time*."""
    t = np.asarray(time_s, dtype=float)
    x = np.asarray(signal, dtype=float)
    valid = np.isfinite(t) & np.isfinite(x)
    if valid.sum() < 3:
        return float(start_time)

    t, x = t[valid], x[valid]
    mask = (t >= start_time) & (t <= start_time + search_post_s)
    if mask.sum() < 3:
        return float(start_time)

    return float(t[mask][int(np.nanargmin(x[mask]))])


def detrend_linear(time_s, values, t_start, t_end):
    """Remove linear drift in *values* between *t_start* and *t_end*."""
    t_arr = np.asarray(time_s, dtype=float)
    v_arr = np.asarray(values, dtype=float).copy()
    mask = (
        np.isfinite(t_arr) & np.isfinite(v_arr)
        & np.isfinite(t_start) & np.isfinite(t_end)
        & (t_arr >= t_start) & (t_arr <= t_end)
    )
    if mask.sum() < 2:
        return v_arr, False

    t_fit, v_fit = t_arr[mask], v_arr[mask]
    coeff = np.polyfit(t_fit, v_fit, 1)
    trend = np.polyval(coeff, t_fit)
    # Preserve level at the start of the detrend window while removing slope drift.
    v_arr[mask] = v_fit - (trend - np.polyval(coeff, t_fit[0]))
    return v_arr, True


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _add_event_markers(ax, t0, t1, markers, labels=False):
    """Draw vertical reference lines for known event times that fall within [t0, t1]."""
    style = [
        ("drogue_time", "purple", ":", "Drogue deployment"),
        ("main_time", "brown", ":", "Main deployment"),
        ("event_75_time", "black", ":", "70 s event"),
    ]
    for key, color, ls, label_text in style:
        t = markers.get(key)
        if t is not None and t0 <= t <= t1:
            ax.axvline(t, color=color, ls=ls, label=label_text if labels else None)


def plot_event_window(win, event_id, event_name, event_center_line_t, t0, t1,
                      markers, outdir):
    """Create a 2-panel accel/gyro plot for one event window and save to disk."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    for c in ["accel_x_g", "accel_y_g", "accel_z_g"]:
        axes[0].plot(win["time_s"], win[c], label=c)
    if event_name == "main_deployment":
        axes[0].axvline(event_center_line_t, color="r", ls="--", label="Event center")
    _add_event_markers(axes[0], t0, t1, markers, labels=True)
    axes[0].set_ylabel("Acceleration (g)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(ncol=2)

    for c in ["gyro_x_deg_s", "gyro_y_deg_s", "gyro_z_deg_s"]:
        axes[1].plot(win["time_s"], win[c], label=c)
    if event_name == "main_deployment":
        axes[1].axvline(event_center_line_t, color="r", ls="--")
    _add_event_markers(axes[1], t0, t1, markers)
    axes[1].set_ylabel("Gyro (deg/s)")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(ncol=2)

    fig.suptitle(
        f"Event {event_id}: {event_name.replace('_', ' ')} near t={event_center_line_t:.3f}s"
    )
    fig.tight_layout()
    fig.savefig(outdir / f"event_{event_id:02d}_{event_name}_window.png", dpi=180)
    plt.close(fig)


def plot_overview(df, markers, outdir):
    """Full-flight 3-panel overview: accel, velocity, gyro."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    for c in ["accel_x_g", "accel_y_g", "accel_z_g"]:
        axes[0].plot(df["time_s"], df[c], label=c, lw=1)
    axes[0].set_ylabel("Acceleration (g)")

    for c in ["vel_x_m_s", "vel_y_m_s", "vel_z_m_s"]:
        axes[1].plot(df["time_s"], df[c], label=c, lw=1)
    axes[1].set_ylabel("Velocity (m/s)")

    for c in ["gyro_x_deg_s", "gyro_y_deg_s", "gyro_z_deg_s"]:
        axes[2].plot(df["time_s"], df[c], label=c, lw=1)
    axes[2].set_ylabel("Gyro (deg/s)")
    axes[2].set_xlabel("Time (s)")

    t_min = float(df["time_s"].min())
    t_max = float(df["time_s"].max())
    for ax in axes:
        _add_event_markers(ax, t_min, t_max, markers, labels=True)
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2)

    fig.suptitle(
        "SnowbirdPrim Blue Raven HR Analysis "
        "(Raw Accel + Quaternion Gravity-Corrected Velocity + Gyro)"
    )
    fig.tight_layout()
    fig.savefig(outdir / "hr_dynamics_overview.png", dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Analyze Blue Raven high-rate CSV for shocks and dynamic events."
    )
    parser.add_argument(
        "csv_path", nargs="?", default=None,
        help="Path to SnowbirdPrim HR CSV. Defaults to latest HR CSV in exports/.",
    )
    parser.add_argument("--outdir", default=None, help="Optional output directory")
    parser.add_argument(
        "--lr-csv-path", default=None,
        help="Optional LR CSV path used as primary velocity source",
    )
    parser.add_argument(
        "--lr-velocity-units", choices=["ft/s", "m/s"], default="ft/s",
        help="Units used by LR velocity columns",
    )
    parser.add_argument(
        "--detrend-velocity", action="store_true", default=True,
        help="Apply linear detrend to LR velocity traces in a selected time window",
    )
    parser.add_argument(
        "--no-detrend-velocity", action="store_false", dest="detrend_velocity",
        help="Disable LR velocity detrending",
    )
    parser.add_argument(
        "--detrend-start-time", type=float, default=None,
        help="Start time for velocity detrend window (defaults to drogue time)",
    )
    parser.add_argument(
        "--detrend-end-time", type=float, default=None,
        help="End time for velocity detrend window (defaults to end of HR data)",
    )
    parser.add_argument(
        "--event-window", type=float, default=0.75,
        help="Default symmetric seconds before/after event center when custom windows are not set",
    )
    # Event 1 – drogue deployment
    parser.add_argument("--event1-pre-window", type=float, default=0.5,
                        help="Seconds before drogue event center")
    parser.add_argument("--event1-post-window", type=float, default=1.75,
                        help="Seconds after drogue event center")
    # Event 2 – 70 s event
    parser.add_argument("--event2-pre-window", type=float, default=20.5,
                        help="Seconds before 70 s event center")
    parser.add_argument("--event2-post-window", type=float, default=20.5,
                        help="Seconds after 70 s event center")
    # Event 3 – main deployment
    parser.add_argument("--event3-pre-window", type=float, default=0.5,
                        help="Seconds before main deployment center")
    parser.add_argument("--event3-post-window", type=float, default=2.5,
                        help="Seconds after main deployment center")
    # Known event times
    parser.add_argument(
        "--event-75-time", type=float, default=72.0,
        help="Known event time near 72 s from flight timeline",
    )
    parser.add_argument(
        "--event2-center-mode", choices=["gyro_x_onset", "fixed"],
        default="fixed",
        help="How to center Event 2 window: fixed time or gyro_x rise onset",
    )
    parser.add_argument("--event2-gyro-onset-search-pre", type=float, default=8.0,
                        help="Seconds before fixed Event 2 time to search for gyro_x rise onset")
    parser.add_argument("--event2-gyro-onset-search-post", type=float, default=4.0,
                        help="Seconds after fixed Event 2 time to search for gyro_x rise onset")
    parser.add_argument("--event2-center-shift-s", type=float, default=0.0,
                        help="Additional time shift applied to Event 2 center (positive moves right)")
    parser.add_argument(
        "--no-event2-include-negative-trough", action="store_false",
        dest="event2_include_negative_trough", default=True,
        help="Disable auto-extension of Event 2 post-window for the post-rise negative gyro_x trough",
    )
    parser.add_argument("--event2-trough-search-post", type=float, default=12.0,
                        help="Seconds after Event 2 center to search for negative gyro_x trough")
    parser.add_argument("--event2-trough-pad", type=float, default=0.75,
                        help="Extra seconds after trough to keep in Event 2 window")
    parser.add_argument("--main-time", type=float, default=217.9,
                        help="Known main fire time from summary CSV")
    parser.add_argument("--drogue-time", type=float, default=38.6,
                        help="Known drogue deployment time from summary CSV")
    return parser


# ---------------------------------------------------------------------------
# LR velocity interpolation
# ---------------------------------------------------------------------------

def load_and_interpolate_lr_velocity(lr_csv_path, t_hr, velocity_units):
    """Read the LR CSV and interpolate velocity components onto HR timestamps."""
    lr = pd.read_csv(lr_csv_path, low_memory=False)
    lr.columns = [str(c).strip() for c in lr.columns]

    required_lr = ["Flight_Time_(s)", "Velocity_Up"]
    missing_lr = [c for c in required_lr if c not in lr.columns]
    if missing_lr:
        raise ValueError(f"Missing required LR velocity columns: {missing_lr}")

    t_lr = pd.to_numeric(lr["Flight_Time_(s)"], errors="coerce").to_numpy(dtype=float)
    v_up = pd.to_numeric(lr["Velocity_Up"], errors="coerce").to_numpy(dtype=float)
    v_dr = (pd.to_numeric(lr["Velocity_DR"], errors="coerce").to_numpy(dtype=float)
            if "Velocity_DR" in lr.columns else np.full(len(lr), np.nan))
    v_cr = (pd.to_numeric(lr["Velocity_CR"], errors="coerce").to_numpy(dtype=float)
            if "Velocity_CR" in lr.columns else np.full(len(lr), np.nan))

    valid_lr = np.isfinite(t_lr) & np.isfinite(v_up)
    if valid_lr.sum() < 2:
        raise ValueError("LR velocity data is invalid or too sparse to interpolate")

    sort_idx = np.argsort(t_lr[valid_lr])
    t_sorted = t_lr[valid_lr][sort_idx]
    up_sorted, dr_sorted, cr_sorted = (
        v_up[valid_lr][sort_idx],
        v_dr[valid_lr][sort_idx],
        v_cr[valid_lr][sort_idx],
    )

    unique_t, unique_idx = np.unique(t_sorted, return_index=True)
    t_sorted = unique_t
    up_sorted = up_sorted[unique_idx]
    dr_sorted = dr_sorted[unique_idx]
    cr_sorted = cr_sorted[unique_idx]

    valid_hr = np.isfinite(t_hr)
    vel_x = np.full(len(t_hr), np.nan)
    vel_y = np.full(len(t_hr), np.nan)
    vel_z = np.full(len(t_hr), np.nan)
    vel_x[valid_hr] = np.interp(t_hr[valid_hr], t_sorted, up_sorted)

    finite_dr = np.isfinite(dr_sorted)
    if finite_dr.sum() >= 2:
        vel_y[valid_hr] = np.interp(t_hr[valid_hr], t_sorted[finite_dr], dr_sorted[finite_dr])

    finite_cr = np.isfinite(cr_sorted)
    if finite_cr.sum() >= 2:
        vel_z[valid_hr] = np.interp(t_hr[valid_hr], t_sorted[finite_cr], cr_sorted[finite_cr])

    if velocity_units == "ft/s":
        vel_x *= 0.3048
        vel_y *= 0.3048
        vel_z *= 0.3048

    return vel_x, vel_y, vel_z


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = build_arg_parser().parse_args()

    ensure_standard_dirs()
    csv_path = Path(args.csv_path) if args.csv_path else default_hr_csv()
    outdir = Path(args.outdir) if args.outdir else default_analysis_dir(csv_path)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Read HR CSV ──────────────────────────────────────────────────────
    hr = pd.read_csv(csv_path, low_memory=False)
    hr.columns = [str(c).strip() for c in hr.columns]

    required = [
        "Flight_Time_(s)", "Gyro_X", "Gyro_Y", "Gyro_Z",
        "Accel_X", "Accel_Y", "Accel_Z",
    ]
    missing = [c for c in required if c not in hr.columns]
    if missing:
        raise ValueError(f"Missing required HR columns: {missing}")

    df = pd.DataFrame({
        "time_s":       pd.to_numeric(hr["Flight_Time_(s)"], errors="coerce"),
        "gyro_x_deg_s": pd.to_numeric(hr["Gyro_X"], errors="coerce"),
        "gyro_y_deg_s": pd.to_numeric(hr["Gyro_Y"], errors="coerce"),
        "gyro_z_deg_s": pd.to_numeric(hr["Gyro_Z"], errors="coerce"),
        "accel_x_g":    pd.to_numeric(hr["Accel_X"], errors="coerce"),
        "accel_y_g":    pd.to_numeric(hr["Accel_Y"], errors="coerce"),
        "accel_z_g":    pd.to_numeric(hr["Accel_Z"], errors="coerce"),
    })
    if "Aux_Volts" in hr.columns:
        df["aux_volts"] = pd.to_numeric(hr["Aux_Volts"], errors="coerce")
    if "Current" in hr.columns:
        df["current_a"] = pd.to_numeric(hr["Current"], errors="coerce")

    df["accel_mag_g"] = np.sqrt(
        df["accel_x_g"] ** 2 + df["accel_y_g"] ** 2 + df["accel_z_g"] ** 2
    )
    df["accel_excess_g"] = (df["accel_mag_g"] - 1.0).abs()
    df["gyro_mag_deg_s"] = np.sqrt(
        df["gyro_x_deg_s"] ** 2 + df["gyro_y_deg_s"] ** 2 + df["gyro_z_deg_s"] ** 2
    )

    # ── Resolve LR CSV path ─────────────────────────────────────────────
    lr_csv_path = None
    if args.lr_csv_path:
        candidate = Path(args.lr_csv_path)
        if candidate.exists():
            lr_csv_path = candidate
    else:
        candidate = csv_path.with_name(csv_path.name.replace(" HR_", " LR_"))
        if candidate.exists():
            lr_csv_path = candidate
        else:
            try:
                lr_csv_path = default_lr_csv()
            except FileNotFoundError:
                lr_csv_path = None

    if lr_csv_path is None:
        raise ValueError(
            "LR CSV with onboard velocity is required. "
            "Pass --lr-csv-path or keep matching HR_/LR_ filenames."
        )

    # ── Interpolate LR velocity onto HR timestamps ──────────────────────
    t_hr = df["time_s"].to_numpy(dtype=float)
    vel_x, vel_y, vel_z = load_and_interpolate_lr_velocity(
        lr_csv_path, t_hr, args.lr_velocity_units
    )

    # ── Optional linear detrend ──────────────────────────────────────────
    detrend_start_s = args.detrend_start_time if args.detrend_start_time is not None else args.drogue_time
    if args.detrend_end_time is not None:
        detrend_end_s = args.detrend_end_time
    else:
        finite_hr_time = t_hr[np.isfinite(t_hr)]
        detrend_end_s = float(np.max(finite_hr_time)) if len(finite_hr_time) > 0 else np.nan

    detrend_applied_x = detrend_applied_y = detrend_applied_z = False
    if args.detrend_velocity:
        vel_x, detrend_applied_x = detrend_linear(t_hr, vel_x, detrend_start_s, detrend_end_s)
        vel_y, detrend_applied_y = detrend_linear(t_hr, vel_y, detrend_start_s, detrend_end_s)
        vel_z, detrend_applied_z = detrend_linear(t_hr, vel_z, detrend_start_s, detrend_end_s)

    df["vel_x_m_s"] = vel_x
    df["vel_y_m_s"] = vel_y
    df["vel_z_m_s"] = vel_z
    df["vel_mag_m_s"] = np.sqrt(vel_x ** 2 + vel_y ** 2 + vel_z ** 2)

    # ── Jerk ─────────────────────────────────────────────────────────────
    t = t_hr
    y = df["accel_excess_g"].to_numpy(dtype=float)
    valid = np.isfinite(t) & np.isfinite(y)
    jerk = np.full(len(df), np.nan)
    if valid.sum() > 3:
        jerk[valid] = np.gradient(y[valid], t[valid])
    df["jerk_excess_g_per_s"] = jerk

    # ── Global metrics ───────────────────────────────────────────────────
    i_acc = df["accel_mag_g"].idxmax()
    i_gyro = df["gyro_mag_deg_s"].idxmax()
    i_jerk = df["jerk_excess_g_per_s"].abs().idxmax()
    i_vel = df["vel_mag_m_s"].idxmax()

    metrics = {
        "max_accel_time_s":            float(df.loc[i_acc, "time_s"]),
        "max_accel_mag_g":             float(df.loc[i_acc, "accel_mag_g"]),
        "max_gyro_time_s":             float(df.loc[i_gyro, "time_s"]),
        "max_gyro_mag_deg_s":          float(df.loc[i_gyro, "gyro_mag_deg_s"]),
        "max_abs_jerk_time_s":         float(df.loc[i_jerk, "time_s"]),
        "max_abs_jerk_excess_g_per_s": float(df.loc[i_jerk, "jerk_excess_g_per_s"]),
        "max_velocity_time_s":         float(df.loc[i_vel, "time_s"]),
        "max_velocity_mag_m_s":        float(df.loc[i_vel, "vel_mag_m_s"]),
        "velocity_source":             "lr_onboard",
        "lr_csv_path":                 str(lr_csv_path),
        "lr_velocity_units":           args.lr_velocity_units,
        "detrend_velocity":            bool(args.detrend_velocity),
        "detrend_start_time_s":        detrend_start_s,
        "detrend_end_time_s":          detrend_end_s,
        "detrend_applied_x":           bool(detrend_applied_x),
        "detrend_applied_y":           bool(detrend_applied_y),
        "detrend_applied_z":           bool(detrend_applied_z),
    }

    # ── Event windows ────────────────────────────────────────────────────
    markers = {
        "drogue_time":   args.drogue_time,
        "main_time":     args.main_time,
        "event_75_time": args.event_75_time,
    }

    focus_events = [
        ("drogue_deployment", args.drogue_time,   args.event1_pre_window, args.event1_post_window),
        ("event_70s",         args.event_75_time,  args.event2_pre_window, args.event2_post_window),
        ("main_deployment",   args.main_time,      args.event3_pre_window, args.event3_post_window),
    ]

    event_rows = []
    finite_time = t_hr
    time_arr = t_hr
    gyro_x_arr = df["gyro_x_deg_s"].to_numpy(dtype=float)

    for event_id, (event_name, center_t, pre_window, post_window) in enumerate(
        focus_events, start=1
    ):
        if center_t is None or not np.isfinite(center_t):
            continue

        requested_center_t = float(center_t)

        # Event 2 center adjustment: auto-detect gyro_x rise onset
        if event_name == "event_70s" and args.event2_center_mode == "gyro_x_onset":
            center_t = find_rise_onset_time(
                time_arr, gyro_x_arr, requested_center_t,
                args.event2_gyro_onset_search_pre,
                args.event2_gyro_onset_search_post,
            )
        if event_name == "event_70s":
            center_t = float(center_t) + args.event2_center_shift_s

        if pre_window is None or post_window is None:
            pre_window = post_window = args.event_window

        # Event 2: auto-extend post-window to include negative gyro_x trough
        if event_name == "event_70s" and args.event2_include_negative_trough:
            trough_t = find_post_min_time(
                time_arr, gyro_x_arr, float(center_t),
                max(args.event2_trough_search_post, 0.0),
            )
            needed_post = (trough_t - float(center_t)) + max(args.event2_trough_pad, 0.0)
            if np.isfinite(needed_post):
                post_window = max(post_window, needed_post)

        t0 = center_t - pre_window
        t1 = center_t + post_window
        win = df[(df["time_s"] >= t0) & (df["time_s"] <= t1)].copy()
        if win.empty:
            continue

        nearest_idx = int(np.nanargmin(np.abs(finite_time - center_t)))
        peak_t = float(df.loc[nearest_idx, "time_s"])
        event_center_line_t = peak_t

        if event_name == "main_deployment":
            accel_x_valid = win["accel_x_g"].dropna()
            if len(accel_x_valid) > 0:
                event_center_line_t = float(df.loc[int(accel_x_valid.idxmax()), "time_s"])

        win.to_csv(outdir / f"event_{event_id:02d}_{event_name}_window.csv", index=False)
        plot_event_window(win, event_id, event_name, event_center_line_t,
                          t0, t1, markers, outdir)

        row = {
            "event_id": event_id,
            "event_name": event_name,
            "start_time_s": float(win["time_s"].min()),
            "end_time_s": float(win["time_s"].max()),
            "pre_window_s": float(pre_window),
            "post_window_s": float(post_window),
            "event_time_requested_s": requested_center_t,
            "event_time_used_s": float(center_t),
            "event_center_line_time_s": float(event_center_line_t),
            "nearest_sample_time_s": peak_t,
            "accel_x_g": float(df.loc[nearest_idx, "accel_x_g"]),
            "accel_y_g": float(df.loc[nearest_idx, "accel_y_g"]),
            "accel_z_g": float(df.loc[nearest_idx, "accel_z_g"]),
            "gyro_x_deg_s": float(df.loc[nearest_idx, "gyro_x_deg_s"]),
            "gyro_y_deg_s": float(df.loc[nearest_idx, "gyro_y_deg_s"]),
            "gyro_z_deg_s": float(df.loc[nearest_idx, "gyro_z_deg_s"]),
        }
        if "current_a" in df.columns:
            row["current_a"] = float(df.loc[nearest_idx, "current_a"])
        if "aux_volts" in df.columns:
            row["aux_volts"] = float(df.loc[nearest_idx, "aux_volts"])
        row["dt_from_drogue_deploy_s"] = (
            peak_t - args.drogue_time if args.drogue_time is not None else np.nan
        )
        row["dt_from_main_fire_s"] = (
            peak_t - args.main_time if args.main_time is not None else np.nan
        )
        row["dt_from_70s_event_s"] = (
            peak_t - args.event_75_time if args.event_75_time is not None else np.nan
        )
        event_rows.append(row)

    # ── Save CSVs ────────────────────────────────────────────────────────
    df.to_csv(outdir / "hr_derived.csv", index=False)
    pd.DataFrame(event_rows).to_csv(outdir / "hr_event_summary.csv", index=False)
    pd.DataFrame([
        metrics | {
            "event_count": len(event_rows),
            "drogue_time_s": args.drogue_time,
            "event_70_time_s": args.event_75_time,
            "main_time_s": args.main_time,
            "event1_pre_window_s": args.event1_pre_window,
            "event1_post_window_s": args.event1_post_window,
            "event2_pre_window_s": args.event2_pre_window,
            "event2_post_window_s": args.event2_post_window,
            "event2_center_mode": args.event2_center_mode,
            "event2_center_shift_s": args.event2_center_shift_s,
            "event2_gyro_onset_search_pre_s": args.event2_gyro_onset_search_pre,
            "event2_gyro_onset_search_post_s": args.event2_gyro_onset_search_post,
            "event2_include_negative_trough": bool(args.event2_include_negative_trough),
            "event2_trough_search_post_s": args.event2_trough_search_post,
            "event2_trough_pad_s": args.event2_trough_pad,
            "event3_pre_window_s": args.event3_pre_window,
            "event3_post_window_s": args.event3_post_window,
        }
    ]).to_csv(outdir / "hr_metrics.csv", index=False)

    # ── Overview plot ────────────────────────────────────────────────────
    plot_overview(df, markers, outdir)

    with open(outdir / "hr_detected_columns.txt", "w", encoding="utf-8") as f:
        for name in required:
            f.write(name + "\n")
        f.write(f"drogue_time_s={args.drogue_time}\n")
        f.write(f"event_70_time_s={args.event_75_time}\n")
        f.write(f"main_time_s={args.main_time}\n")

    print(f"Saved HR outputs to {outdir}")


if __name__ == "__main__":
    main()
