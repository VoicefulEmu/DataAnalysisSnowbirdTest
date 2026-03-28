from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from path_config import default_analysis_dir, default_lr_csv, default_summary_csv, ensure_standard_dirs


def moving_average(arr, window=9):
    return (
        pd.Series(arr)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


def crossing_time(t, y, threshold=0.5):
    valid = np.isfinite(t) & np.isfinite(y)
    t = t[valid]
    y = y[valid]
    if len(t) < 2:
        return np.nan
    idx = np.where((y[:-1] < threshold) & (y[1:] >= threshold))[0]
    return float(t[idx[0] + 1]) if len(idx) else np.nan


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Blue Raven low-rate CSV for flight profile."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=None,
        help="Path to SnowbirdPrim LR CSV. Defaults to latest LR CSV in exports/.",
    )
    parser.add_argument("--summary", default=None, help="Optional summary CSV path")
    parser.add_argument("--outdir", default=None, help="Optional output directory")
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
    df.columns = [str(c).strip() for c in df.columns]

    required = [
        "Flight_Time_(s)",
        "Baro_Altitude_AGL_(feet)",
        "Velocity_Up",
        "Velocity_DR",
        "Velocity_CR",
        "Apo_fired",
        "Main_fired",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required LR columns: {missing}")

    t = pd.to_numeric(df["Flight_Time_(s)"], errors="coerce")
    altitude_agl = pd.to_numeric(df["Baro_Altitude_AGL_(feet)"], errors="coerce")
    smooth_window = 600  # larger window for clean appearance
    altitude_agl_smooth = pd.Series(moving_average(altitude_agl, window=smooth_window))
    vertical_velocity = pd.to_numeric(df["Velocity_Up"], errors="coerce")
    velocity_dr = pd.to_numeric(df["Velocity_DR"], errors="coerce")
    velocity_cr = pd.to_numeric(df["Velocity_CR"], errors="coerce")
    horizontal_velocity = np.sqrt(velocity_dr**2 + velocity_cr**2)

    tt = t.to_numpy(dtype=float)
    vv = vertical_velocity.to_numpy(dtype=float)
    valid = np.isfinite(tt) & np.isfinite(vv)
    acceleration = np.full(len(df), np.nan)
    if valid.sum() > 3:
        acceleration[valid] = np.gradient(vv[valid], tt[valid])
    acceleration = pd.Series(acceleration)

    derived = pd.DataFrame(
        {
            "time_s": t,
            "altitude_agl_ft": altitude_agl,
            "altitude_agl_smooth_ft": altitude_agl_smooth,
            "vertical_velocity_ft_s": vertical_velocity,
            "horizontal_velocity_ft_s": horizontal_velocity,
            "acceleration_ft_s2": acceleration,
            "velocity_dr_ft_s": velocity_dr,
            "velocity_cr_ft_s": velocity_cr,
            "apo_fired": pd.to_numeric(df["Apo_fired"], errors="coerce"),
            "main_fired": pd.to_numeric(df["Main_fired"], errors="coerce"),
        }
    )

    # Descent rate (positive downward speed) from vertical velocity field
    derived["descent_rate_ft_s"] = np.where(
        derived["vertical_velocity_ft_s"] < 0,
        -derived["vertical_velocity_ft_s"],
        np.nan,
    )

    # Descent rate from altitude change over time (d(altitude)/dt), which is what you requested
    tt = derived["time_s"].to_numpy(dtype=float)
    alt = derived["altitude_agl_smooth_ft"].to_numpy(dtype=float)
    descent_rate_from_alt = np.full(len(derived), np.nan)
    valid_alt = np.isfinite(tt) & np.isfinite(alt)
    if valid_alt.sum() > 1:
        # central difference for interior points, first/backwards for edges handled by np.gradient
        alt_rate = np.gradient(alt[valid_alt], tt[valid_alt])
        descent_rate_from_alt[valid_alt] = np.where(alt_rate < 0, -alt_rate, np.nan)
    derived["descent_rate_from_alt_ft_s"] = descent_rate_from_alt

    # optional: additionally add a smoothed version of the altitude-based descent rate
    derived["descent_rate_from_alt_smoothed_ft_s"] = pd.Series(
        moving_average(derived["descent_rate_from_alt_ft_s"].fillna(0), window=smooth_window)
    ).replace(0, np.nan)

    if "Baro_Altitude_ASL_(feet)" in df.columns:
        derived["altitude_asl_ft"] = pd.to_numeric(
            df["Baro_Altitude_ASL_(feet)"], errors="coerce"
        )
    if "Baro_Press_(atm)" in df.columns:
        derived["baro_press_atm"] = pd.to_numeric(
            df["Baro_Press_(atm)"], errors="coerce"
        )
    if "Inertial_Altitude" in df.columns:
        derived["inertial_altitude_ft"] = pd.to_numeric(
            df["Inertial_Altitude"], errors="coerce"
        )
    if "Tilt_Angle_(deg)" in df.columns:
        derived["tilt_angle_deg"] = pd.to_numeric(
            df["Tilt_Angle_(deg)"], errors="coerce"
        )
    if "Future_Angle_(deg)" in df.columns:
        derived["future_angle_deg"] = pd.to_numeric(
            df["Future_Angle_(deg)"], errors="coerce"
        )
    if "Roll_Angle_(deg)" in df.columns:
        derived["roll_angle_deg"] = pd.to_numeric(
            df["Roll_Angle_(deg)"], errors="coerce"
        )

    metrics = {}
    if altitude_agl_smooth.notna().any():
        i_apogee = altitude_agl_smooth.idxmax()
        metrics["apogee_time_s"] = float(t.loc[i_apogee])
        metrics["apogee_altitude_agl_ft"] = float(altitude_agl_smooth.loc[i_apogee])
    if horizontal_velocity.notna().any():
        i_h = horizontal_velocity.idxmax()
        metrics["max_horizontal_velocity_time_s"] = float(t.loc[i_h])
        metrics["max_horizontal_velocity_ft_s"] = float(horizontal_velocity.loc[i_h])
    if acceleration.notna().any():
        i_a = acceleration.abs().idxmax()
        metrics["max_abs_acceleration_time_s"] = float(t.loc[i_a])
        metrics["max_abs_acceleration_ft_s2"] = float(acceleration.loc[i_a])
    metrics["apo_fired_time_s"] = crossing_time(
        derived["time_s"].to_numpy(float), derived["apo_fired"].to_numpy(float)
    )
    metrics["main_fired_time_s"] = crossing_time(
        derived["time_s"].to_numpy(float), derived["main_fired"].to_numpy(float)
    )

    # Calculate horizontal velocity at drogue deployment (apo fired)
    if np.isfinite(metrics.get("apo_fired_time_s", np.nan)):
        apo_time = metrics["apo_fired_time_s"]
        # Interpolate horizontal velocity at apo fired time
        valid_hvel = derived["horizontal_velocity_ft_s"].notna()
        if valid_hvel.any():
            hvel_times = derived.loc[valid_hvel, "time_s"].to_numpy()
            hvel_values = derived.loc[valid_hvel, "horizontal_velocity_ft_s"].to_numpy()
            if len(hvel_times) > 1:
                metrics["horizontal_velocity_at_apo_fired_ft_s"] = float(
                    np.interp(apo_time, hvel_times, hvel_values)
                )

    pd.DataFrame([metrics]).to_csv(outdir / "lr_metrics.csv", index=False)
    derived.to_csv(outdir / "lr_derived.csv", index=False)

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)

    axes[0].plot(
        derived["time_s"],
        derived["altitude_agl_ft"],
        alpha=0.35,
        label="Raw AGL altitude",
    )
    axes[0].plot(
        derived["time_s"],
        derived["altitude_agl_smooth_ft"],
        lw=2,
        label="Smoothed AGL altitude",
    )
    if "apogee_time_s" in metrics:
        axes[0].axvline(metrics["apogee_time_s"], color="r", ls="--", label="Apogee")
    axes[0].set_ylabel("Altitude AGL (ft)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        derived["time_s"],
        derived["horizontal_velocity_ft_s"],
        color="tab:orange",
        label="Horizontal velocity",
    )
    if "max_horizontal_velocity_time_s" in metrics:
        axes[1].axvline(metrics["max_horizontal_velocity_time_s"], color="r", ls="--")
    if np.isfinite(metrics.get("apo_fired_time_s", np.nan)):
        axes[1].axvline(
            metrics["apo_fired_time_s"], color="purple", ls=":", label="Apo fired"
        )
    axes[1].set_ylabel("Horiz. velocity (ft/s)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(
        derived["time_s"],
        derived["acceleration_ft_s2"],
        color="tab:green",
        label="Acceleration from d(Velocity_Up)/dt",
    )
    if np.isfinite(metrics.get("apo_fired_time_s", np.nan)):
        axes[2].axvline(
            metrics["apo_fired_time_s"], color="purple", ls=":", label="Apo fired"
        )
    if np.isfinite(metrics.get("main_fired_time_s", np.nan)):
        axes[2].axvline(
            metrics["main_fired_time_s"], color="brown", ls=":", label="Main fired"
        )
    if "max_abs_acceleration_time_s" in metrics:
        axes[2].axvline(
            metrics["max_abs_acceleration_time_s"],
            color="r",
            ls="--",
            label="Max |accel|",
        )
    axes[2].set_ylabel("Acceleration (ft/s^2)")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.suptitle("SnowbirdPrim Blue Raven LR Flight Profile")
    fig.tight_layout()
    fig.savefig(outdir / "lr_flight_profile.png", dpi=180)
    plt.close(fig)

    # Descent rate vs altitude plot
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    # ax2.plot(
    #     derived["descent_rate_from_alt_ft_s"],
    #     derived["altitude_agl_ft"],
    #     marker=".",
    #     linestyle="-",
    #     color="tab:blue",
    #     alpha=0.5,
    #     label="Descent rate (from altitude derivative)"
    # )
    ax2.plot(
        derived["descent_rate_from_alt_smoothed_ft_s"],
        derived["altitude_agl_ft"],
        marker="",
        linestyle="-",
        color="tab:orange",
        lw=2,
        label="Smoothed descent rate"
    )
    ax2.set_xlabel("Descent rate (ft/s)")
    ax2.set_ylabel("Altitude AGL (ft)")
    ax2.set_title("Descent Rate vs Altitude")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(outdir / "lr_descent_rate_vs_altitude.png", dpi=180)
    plt.close(fig2)

    with open(outdir / "lr_detected_columns.txt", "w", encoding="utf-8") as f:
        for name in required:
            f.write(name + "\n")

    summary_arg = args.summary
    if summary_arg is None:
        default_summary = default_summary_csv()
        summary_arg = str(default_summary) if default_summary else None

    if summary_arg:
        summary_path = Path(summary_arg)
        if summary_path.exists():
            text = summary_path.read_text(encoding="utf-8", errors="replace")
            (outdir / "summary_copy.txt").write_text(text, encoding="utf-8")

    print(f"Saved LR outputs to {outdir}")


if __name__ == "__main__":
    main()
