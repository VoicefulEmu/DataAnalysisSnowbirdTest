from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from path_config import default_analysis_dir, default_lr_csv, ensure_standard_dirs


def moving_average(values, window):
    return (
        pd.Series(values)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


def robust_smooth(values, median_window, mean_window):
    return (
        pd.Series(values)
        .rolling(window=median_window, center=True, min_periods=1)
        .median()
        .rolling(window=mean_window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


def main():
    parser = argparse.ArgumentParser(
        description="Plot descent rate vs altitude from a Blue Raven low-rate CSV."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=None,
        help="Path to SnowbirdPrim LR CSV. Defaults to latest LR CSV in exports/.",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Optional output directory. Defaults to <csv stem>_analysis",
    )
    parser.add_argument(
        "--altitude-window",
        type=int,
        default=101,
        help="Smoothing window for altitude before differentiating",
    )
    parser.add_argument(
        "--descent-window",
        type=int,
        default=61,
        help="Smoothing window for the derived descent rate",
    )
    parser.add_argument(
        "--min-altitude-ft",
        type=float,
        default=50.0,
        help="Minimum altitude to include in the plot",
    )
    parser.add_argument(
        "--altitude-bin-ft",
        type=float,
        default=250.0,
        help="Altitude bin size for the final descent profile",
    )
    args = parser.parse_args()

    ensure_standard_dirs()
    csv_path = Path(args.csv_path) if args.csv_path else default_lr_csv()
    outdir = (
        Path(args.outdir)
        if args.outdir
        else default_analysis_dir(csv_path)
    )
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = [str(column).strip() for column in df.columns]

    required = ["Flight_Time_(s)", "Baro_Altitude_AGL_(feet)"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    time_s = pd.to_numeric(df["Flight_Time_(s)"], errors="coerce")
    altitude_agl_ft = pd.to_numeric(df["Baro_Altitude_AGL_(feet)"], errors="coerce")
    velocity_up_ft_s = None
    if "Velocity_Up" in df.columns:
        velocity_up_ft_s = pd.to_numeric(df["Velocity_Up"], errors="coerce")

    altitude_smooth_ft = pd.Series(
        moving_average(altitude_agl_ft, window=max(3, args.altitude_window))
    )

    time_values = time_s.to_numpy(dtype=float)
    altitude_values = altitude_smooth_ft.to_numpy(dtype=float)
    valid = np.isfinite(time_values) & np.isfinite(altitude_values)
    if valid.sum() < 2:
        raise ValueError("Not enough valid altitude/time samples to compute descent rate")

    altitude_rate_ft_s = np.full(len(df), np.nan)
    altitude_rate_ft_s[valid] = np.gradient(altitude_values[valid], time_values[valid])

    descent_rate_ft_s = np.where(altitude_rate_ft_s < 0, -altitude_rate_ft_s, np.nan)

    apogee_index = int(altitude_smooth_ft.idxmax())
    descent_series = pd.Series(descent_rate_ft_s)
    descent_smoothed_ft_s = pd.Series(np.nan, index=df.index, dtype=float)
    descent_smoothed_ft_s.iloc[apogee_index:] = (
        descent_series.iloc[apogee_index:]
        .rolling(window=max(3, args.descent_window), center=True, min_periods=1)
        .mean()
        .to_numpy()
    )
    descent_time_smoothed_ft_s = pd.Series(np.nan, index=df.index, dtype=float)
    descent_time_smoothed_ft_s.iloc[apogee_index:] = robust_smooth(
        descent_series.iloc[apogee_index:],
        median_window=max(5, args.descent_window),
        mean_window=max(9, args.descent_window * 3),
    )

    time_plot_source_label = "Altitude-derived descent rate"
    time_plot_raw_series = descent_series.copy()
    time_plot_smoothed_series = descent_time_smoothed_ft_s.copy()
    if velocity_up_ft_s is not None:
        velocity_descent_series = pd.Series(
            np.where(velocity_up_ft_s < 0, -velocity_up_ft_s, np.nan),
            index=df.index,
            dtype=float,
        )
        sane_velocity = velocity_descent_series.iloc[apogee_index:].dropna()
        if not sane_velocity.empty and sane_velocity.quantile(0.99) <= 500:
            velocity_descent_smoothed = pd.Series(np.nan, index=df.index, dtype=float)
            velocity_descent_smoothed.iloc[apogee_index:] = robust_smooth(
                velocity_descent_series.iloc[apogee_index:],
                median_window=max(5, args.descent_window),
                mean_window=max(9, args.descent_window * 3),
            )
            time_plot_source_label = "Velocity_Up descent rate"
            time_plot_raw_series = velocity_descent_series
            time_plot_smoothed_series = velocity_descent_smoothed

    plot_mask = (
        descent_smoothed_ft_s.notna()
        & altitude_smooth_ft.notna()
        & (altitude_smooth_ft >= args.min_altitude_ft)
    )
    if not plot_mask.any():
        raise ValueError("No descent samples remained after filtering")

    plot_df = pd.DataFrame(
        {
            "time_s": time_s.loc[plot_mask],
            "altitude_agl_smooth_ft": altitude_smooth_ft.loc[plot_mask],
            "descent_rate_smoothed_ft_s": descent_smoothed_ft_s.loc[plot_mask],
        }
    ).dropna()

    bin_size_ft = max(10.0, float(args.altitude_bin_ft))
    plot_df["altitude_bin_ft"] = (
        np.round(plot_df["altitude_agl_smooth_ft"] / bin_size_ft) * bin_size_ft
    )

    profile_df = (
        plot_df.groupby("altitude_bin_ft", as_index=False)
        .agg(
            altitude_agl_ft=("altitude_agl_smooth_ft", "median"),
            descent_rate_ft_s=("descent_rate_smoothed_ft_s", "median"),
            sample_count=("descent_rate_smoothed_ft_s", "size"),
        )
        .sort_values("altitude_agl_ft", ascending=False)
        .reset_index(drop=True)
    )
    profile_df["descent_rate_profile_ft_s"] = (
        profile_df["descent_rate_ft_s"]
        .rolling(window=7, center=True, min_periods=1)
        .median()
        .rolling(window=5, center=True, min_periods=1)
        .mean()
    )

    output_csv = outdir / "lr_descent_rate_vs_altitude.csv"
    pd.DataFrame(
        {
            "time_s": time_s,
            "altitude_agl_ft": altitude_agl_ft,
            "altitude_agl_smooth_ft": altitude_smooth_ft,
            "descent_rate_from_alt_ft_s": descent_rate_ft_s,
            "descent_rate_smoothed_ft_s": descent_smoothed_ft_s,
            "descent_rate_time_smoothed_ft_s": descent_time_smoothed_ft_s,
            "velocity_up_ft_s": velocity_up_ft_s,
            "time_plot_descent_rate_ft_s": time_plot_raw_series,
            "time_plot_descent_rate_smoothed_ft_s": time_plot_smoothed_series,
        }
    ).to_csv(output_csv, index=False)

    profile_csv = outdir / "lr_descent_rate_vs_altitude_profile.csv"
    profile_df.to_csv(profile_csv, index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        plot_df["altitude_agl_smooth_ft"],
        plot_df["descent_rate_smoothed_ft_s"],
        s=6,
        alpha=0.08,
        color="tab:orange",
        edgecolors="none",
        label="Raw descent samples",
    )
    ax.plot(
        profile_df["altitude_agl_ft"],
        profile_df["descent_rate_profile_ft_s"],
        color="tab:red",
        lw=2,
        label="Binned median profile",
    )
    ax.set_xlabel("Altitude AGL (ft)")
    ax.set_ylabel("Descent rate (ft/s)")
    ax.set_title("Descent Rate vs Altitude")
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_png = outdir / "lr_descent_rate_vs_altitude.png"
    fig.savefig(output_png, dpi=180)
    plt.close(fig)

    time_plot_mask = (
        (time_s >= time_s.iloc[apogee_index])
        & time_plot_raw_series.notna()
        & (altitude_smooth_ft >= args.min_altitude_ft)
    )

    fig_time, ax_time = plt.subplots(figsize=(10, 6))
    ax_time.scatter(
        time_s.loc[time_plot_mask],
        time_plot_raw_series.loc[time_plot_mask],
        color="tab:orange",
        alpha=0.08,
        s=8,
        edgecolors="none",
        label=f"Raw {time_plot_source_label}",
    )
    ax_time.plot(
        time_s.loc[time_plot_mask],
        time_plot_smoothed_series.loc[time_plot_mask],
        color="tab:red",
        lw=2,
        label=f"Smoothed {time_plot_source_label}",
    )
    ax_time.set_xlabel("Time (s)")
    ax_time.set_ylabel("Descent rate (ft/s)")
    ax_time.set_title("Descent Rate vs Time")
    ax_time.grid(True, alpha=0.3)
    ax_time.legend()
    fig_time.tight_layout()

    output_descent_time_png = outdir / "lr_descent_rate_vs_time.png"
    fig_time.savefig(output_descent_time_png, dpi=180)
    plt.close(fig_time)

    fig_alt, ax_alt = plt.subplots(figsize=(10, 6))
    ax_alt.plot(
        time_s,
        altitude_agl_ft,
        color="tab:blue",
        alpha=0.25,
        lw=1,
        label="Raw altitude",
    )
    ax_alt.plot(
        time_s,
        altitude_smooth_ft,
        color="navy",
        lw=2,
        label="Smoothed altitude",
    )
    ax_alt.set_xlabel("Time (s)")
    ax_alt.set_ylabel("Altitude AGL (ft)")
    ax_alt.set_title("Altitude vs Time")
    ax_alt.grid(True, alpha=0.3)
    ax_alt.legend()
    fig_alt.tight_layout()

    output_altitude_time_png = outdir / "lr_altitude_vs_time.png"
    fig_alt.savefig(output_altitude_time_png, dpi=180)
    plt.close(fig_alt)

    print(f"Saved plot to {output_png}")
    print(f"Saved plot to {output_descent_time_png}")
    print(f"Saved plot to {output_altitude_time_png}")
    print(f"Saved data to {output_csv}")
    print(f"Saved profile to {profile_csv}")


if __name__ == "__main__":
    main()