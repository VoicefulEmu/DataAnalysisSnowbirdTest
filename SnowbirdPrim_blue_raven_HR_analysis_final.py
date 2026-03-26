from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def contiguous_regions(mask):
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    edges = np.diff(mask.astype(int))
    starts = list(np.where(edges == 1)[0] + 1)
    ends = list(np.where(edges == -1)[0] + 1)
    if mask[0]:
        starts = [0] + starts
    if mask[-1]:
        ends = ends + [len(mask)]
    return list(zip(starts, ends))


def event_windows(df, signal_col, threshold, min_samples=5):
    signal = pd.to_numeric(df[signal_col], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(signal) & (signal >= threshold)
    events = []
    for start, end in contiguous_regions(mask):
        if end - start < min_samples:
            continue
        block = signal[start:end]
        peak_local = int(np.nanargmax(block))
        peak_idx = start + peak_local
        events.append((start, end, peak_idx))
    return events


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Blue Raven high-rate CSV for shocks and dynamic events."
    )
    parser.add_argument("csv_path", help="Path to SnowbirdPrim HR CSV")
    parser.add_argument("--outdir", default=None, help="Optional output directory")
    parser.add_argument(
        "--event-window",
        type=float,
        default=0.75,
        help="Seconds before/after event peak to save",
    )
    parser.add_argument(
        "--event-threshold-g",
        type=float,
        default=None,
        help="Override threshold on excess g for event detection",
    )
    parser.add_argument(
        "--main-time",
        type=float,
        default=217.9,
        help="Known main fire time from summary CSV",
    )
    parser.add_argument(
        "--apo-time",
        type=float,
        default=38.6,
        help="Known apogee fire time from summary CSV",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    outdir = (
        Path(args.outdir)
        if args.outdir
        else csv_path.parent / (csv_path.stem + "_analysis")
    )
    outdir.mkdir(parents=True, exist_ok=True)

    hr = pd.read_csv(csv_path, low_memory=False)
    hr.columns = [str(c).strip() for c in hr.columns]

    required = [
        "Flight_Time_(s)",
        "Gyro_X",
        "Gyro_Y",
        "Gyro_Z",
        "Accel_X",
        "Accel_Y",
        "Accel_Z",
    ]
    missing = [c for c in required if c not in hr.columns]
    if missing:
        raise ValueError(f"Missing required HR columns: {missing}")

    df = pd.DataFrame()
    df["time_s"] = pd.to_numeric(hr["Flight_Time_(s)"], errors="coerce")
    df["gyro_x_deg_s"] = pd.to_numeric(hr["Gyro_X"], errors="coerce")
    df["gyro_y_deg_s"] = pd.to_numeric(hr["Gyro_Y"], errors="coerce")
    df["gyro_z_deg_s"] = pd.to_numeric(hr["Gyro_Z"], errors="coerce")
    df["accel_x_g"] = pd.to_numeric(hr["Accel_X"], errors="coerce")
    df["accel_y_g"] = pd.to_numeric(hr["Accel_Y"], errors="coerce")
    df["accel_z_g"] = pd.to_numeric(hr["Accel_Z"], errors="coerce")

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

    t = df["time_s"].to_numpy(dtype=float)
    y = df["accel_excess_g"].to_numpy(dtype=float)
    valid = np.isfinite(t) & np.isfinite(y)
    jerk = np.full(len(df), np.nan)
    if valid.sum() > 3:
        jerk[valid] = np.gradient(y[valid], t[valid])
    df["jerk_excess_g_per_s"] = jerk

    metrics = {}
    i_acc = df["accel_mag_g"].idxmax()
    metrics["max_accel_time_s"] = float(df.loc[i_acc, "time_s"])
    metrics["max_accel_mag_g"] = float(df.loc[i_acc, "accel_mag_g"])

    i_gyro = df["gyro_mag_deg_s"].idxmax()
    metrics["max_gyro_time_s"] = float(df.loc[i_gyro, "time_s"])
    metrics["max_gyro_mag_deg_s"] = float(df.loc[i_gyro, "gyro_mag_deg_s"])

    i_jerk = df["jerk_excess_g_per_s"].abs().idxmax()
    metrics["max_abs_jerk_time_s"] = float(df.loc[i_jerk, "time_s"])
    metrics["max_abs_jerk_excess_g_per_s"] = float(
        df.loc[i_jerk, "jerk_excess_g_per_s"]
    )

    threshold = (
        args.event_threshold_g
        if args.event_threshold_g is not None
        else max(float(df["accel_excess_g"].quantile(0.999)), 5.0)
    )
    events = event_windows(df, "accel_excess_g", threshold=threshold, min_samples=5)

    event_rows = []
    for event_id, (start, end, peak_idx) in enumerate(events, start=1):
        peak_t = float(df.loc[peak_idx, "time_s"])
        t0 = peak_t - args.event_window
        t1 = peak_t + args.event_window
        win = df[(df["time_s"] >= t0) & (df["time_s"] <= t1)].copy()
        win.to_csv(outdir / f"event_{event_id:02d}_window.csv", index=False)

        fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
        for c in ["accel_x_g", "accel_y_g", "accel_z_g", "accel_mag_g"]:
            axes[0].plot(win["time_s"], win[c], label=c)
        axes[0].axvline(peak_t, color="r", ls="--")
        if args.apo_time is not None and t0 <= args.apo_time <= t1:
            axes[0].axvline(args.apo_time, color="purple", ls=":", label="Apo fire")
        if args.main_time is not None and t0 <= args.main_time <= t1:
            axes[0].axvline(args.main_time, color="brown", ls=":", label="Main fire")
        axes[0].set_ylabel("Acceleration (g)")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(ncol=2)

        for c in ["gyro_x_deg_s", "gyro_y_deg_s", "gyro_z_deg_s", "gyro_mag_deg_s"]:
            axes[1].plot(win["time_s"], win[c], label=c)
        axes[1].axvline(peak_t, color="r", ls="--")
        if args.apo_time is not None and t0 <= args.apo_time <= t1:
            axes[1].axvline(args.apo_time, color="purple", ls=":")
        if args.main_time is not None and t0 <= args.main_time <= t1:
            axes[1].axvline(args.main_time, color="brown", ls=":")
        axes[1].set_ylabel("Gyro (deg/s)")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(ncol=2)

        plotted = False
        for c in ["jerk_excess_g_per_s", "aux_volts", "current_a"]:
            if c in win.columns:
                axes[2].plot(win["time_s"], win[c], label=c)
                plotted = True
        axes[2].axvline(peak_t, color="r", ls="--")
        if args.apo_time is not None and t0 <= args.apo_time <= t1:
            axes[2].axvline(args.apo_time, color="purple", ls=":")
        if args.main_time is not None and t0 <= args.main_time <= t1:
            axes[2].axvline(args.main_time, color="brown", ls=":")
        axes[2].set_ylabel("Shock indicators")
        axes[2].set_xlabel("Time (s)")
        axes[2].grid(True, alpha=0.3)
        if plotted:
            axes[2].legend(ncol=2)

        fig.suptitle(f"Event {event_id}: high-acceleration window near t={peak_t:.3f}s")
        fig.tight_layout()
        fig.savefig(outdir / f"event_{event_id:02d}_window.png", dpi=180)
        plt.close(fig)

        row = {
            "event_id": event_id,
            "start_time_s": float(df.loc[start, "time_s"]),
            "end_time_s": float(df.loc[end - 1, "time_s"]),
            "peak_time_s": peak_t,
            "peak_accel_mag_g": float(df.loc[peak_idx, "accel_mag_g"]),
            "peak_accel_excess_g": float(df.loc[peak_idx, "accel_excess_g"]),
            "peak_gyro_mag_deg_s": float(df.loc[peak_idx, "gyro_mag_deg_s"]),
        }
        if "current_a" in df.columns:
            row["peak_current_a"] = float(df.loc[peak_idx, "current_a"])
        if "aux_volts" in df.columns:
            row["peak_aux_volts"] = float(df.loc[peak_idx, "aux_volts"])
        row["dt_from_apo_fire_s"] = (
            peak_t - args.apo_time if args.apo_time is not None else np.nan
        )
        row["dt_from_main_fire_s"] = (
            peak_t - args.main_time if args.main_time is not None else np.nan
        )
        event_rows.append(row)

    df.to_csv(outdir / "hr_derived.csv", index=False)
    pd.DataFrame(event_rows).to_csv(outdir / "hr_event_summary.csv", index=False)
    pd.DataFrame(
        [
            metrics
            | {
                "event_threshold_excess_g": threshold,
                "event_count": len(event_rows),
                "apo_time_s": args.apo_time,
                "main_time_s": args.main_time,
            }
        ]
    ).to_csv(outdir / "hr_metrics.csv", index=False)

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    for c in ["accel_x_g", "accel_y_g", "accel_z_g", "accel_mag_g"]:
        axes[0].plot(df["time_s"], df[c], label=c, lw=1)
    axes[0].axvline(args.apo_time, color="purple", ls=":", label="Apo fire")
    axes[0].axvline(args.main_time, color="brown", ls=":", label="Main fire")
    axes[0].set_ylabel("Acceleration (g)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(ncol=2)

    for c in ["gyro_x_deg_s", "gyro_y_deg_s", "gyro_z_deg_s", "gyro_mag_deg_s"]:
        axes[1].plot(df["time_s"], df[c], label=c, lw=1)
    axes[1].axvline(args.apo_time, color="purple", ls=":")
    axes[1].axvline(args.main_time, color="brown", ls=":")
    axes[1].set_ylabel("Gyro (deg/s)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(ncol=2)

    for c in ["accel_excess_g", "jerk_excess_g_per_s"]:
        axes[2].plot(df["time_s"], df[c], label=c, lw=1)
    if "current_a" in df.columns:
        axes[2].plot(df["time_s"], df["current_a"], label="current_a", lw=1)
    axes[2].axhline(threshold, color="r", ls="--", label="Event threshold")
    axes[2].axvline(args.apo_time, color="purple", ls=":")
    axes[2].axvline(args.main_time, color="brown", ls=":")
    for row in event_rows:
        axes[2].axvline(row["peak_time_s"], color="gray", ls=":", alpha=0.5)
    axes[2].set_ylabel("Shock indicators")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(ncol=2)

    fig.suptitle("SnowbirdPrim Blue Raven HR Shock and Dynamics Analysis")
    fig.tight_layout()
    fig.savefig(outdir / "hr_dynamics_overview.png", dpi=180)
    plt.close(fig)

    with open(outdir / "hr_detected_columns.txt", "w", encoding="utf-8") as f:
        for name in required:
            f.write(name + "\n")
        f.write("event_threshold_excess_g=" + str(threshold) + "\n")

    print(f"Saved HR outputs to {outdir}")


if __name__ == "__main__":
    main()
