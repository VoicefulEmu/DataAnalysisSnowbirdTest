from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from path_config import EXPORTS_DIR, PHOTOS_DIR, default_hr_csv, ensure_standard_dirs

START_TIME_S = 30.0
END_TIME_S = 250.0
ZOOM_START_TIME_S = 65.0
ZOOM_END_TIME_S = 170.0
DROGUE_DEPLOYMENT_TIME_S = 38.6
MAIN_DEPLOYMENT_TIME_S = 217.9


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot HR acceleration/gyro windows and export filtered CSV slices."
    )
    parser.add_argument(
        "--hr-csv",
        default=None,
        help="Optional HR CSV path. Defaults to latest HR CSV in exports/.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Optional prefix for generated 30-250s and 65-170s CSV names in exports/.",
    )
    args = parser.parse_args()

    ensure_standard_dirs()
    input_csv = Path(args.hr_csv) if args.hr_csv else default_hr_csv()
    output_prefix = args.output_prefix or input_csv.stem
    output_csv = EXPORTS_DIR / f"{output_prefix}_30s_250s.csv"
    output_csv_zoom = EXPORTS_DIR / f"{output_prefix}_65s_170s.csv"
    output_png = PHOTOS_DIR / "hr_accel_gyro_30s_250s.png"
    output_png_zoom = PHOTOS_DIR / "hr_accel_gyro_65s_170s.png"

    df = pd.read_csv(input_csv, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    required_columns = [
        "Flight_Time_(s)",
        "Accel_X",
        "Accel_Y",
        "Accel_Z",
        "Gyro_X",
        "Gyro_Y",
        "Gyro_Z",
    ]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["Flight_Time_(s)"] = pd.to_numeric(df["Flight_Time_(s)"], errors="coerce")
    for col in ["Accel_X", "Accel_Y", "Accel_Z", "Gyro_X", "Gyro_Y", "Gyro_Z"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    mask = df["Flight_Time_(s)"].between(START_TIME_S, END_TIME_S, inclusive="both")
    window = df.loc[mask].copy()

    if window.empty:
        raise ValueError(
            f"No rows found in Flight_Time_(s) window {START_TIME_S} to {END_TIME_S}."
        )

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    fig.suptitle("SnowbirdPrim HR: Acceleration and Gyro (30s to 250s)", fontsize=14)

    axes[0].plot(window["Flight_Time_(s)"], window["Accel_X"], label="Accel_X", lw=1)
    axes[0].plot(window["Flight_Time_(s)"], window["Accel_Y"], label="Accel_Y", lw=1)
    axes[0].plot(window["Flight_Time_(s)"], window["Accel_Z"], label="Accel_Z", lw=1)
    axes[0].set_ylabel("Acceleration (g)")
    axes[0].grid(True, alpha=0.3)

    # Mark deployment events so dynamics can be visually compared to known pyro times.
    if START_TIME_S <= DROGUE_DEPLOYMENT_TIME_S <= END_TIME_S:
        axes[0].axvline(
            DROGUE_DEPLOYMENT_TIME_S,
            color="purple",
            linestyle=":",
            linewidth=1.5,
            label=f"Drogue Deploy ({DROGUE_DEPLOYMENT_TIME_S:.1f}s)",
        )
    if START_TIME_S <= MAIN_DEPLOYMENT_TIME_S <= END_TIME_S:
        axes[0].axvline(
            MAIN_DEPLOYMENT_TIME_S,
            color="brown",
            linestyle=":",
            linewidth=1.5,
            label=f"Main Deploy ({MAIN_DEPLOYMENT_TIME_S:.1f}s)",
        )
    axes[0].legend(ncol=3)

    axes[1].plot(window["Flight_Time_(s)"], window["Gyro_X"], label="Gyro_X", lw=1)
    axes[1].plot(window["Flight_Time_(s)"], window["Gyro_Y"], label="Gyro_Y", lw=1)
    axes[1].plot(window["Flight_Time_(s)"], window["Gyro_Z"], label="Gyro_Z", lw=1)
    axes[1].set_xlabel("Flight Time (s)")
    axes[1].set_ylabel("Gyro (deg/s)")
    axes[1].grid(True, alpha=0.3)

    if START_TIME_S <= DROGUE_DEPLOYMENT_TIME_S <= END_TIME_S:
        axes[1].axvline(
            DROGUE_DEPLOYMENT_TIME_S,
            color="purple",
            linestyle=":",
            linewidth=1.5,
            label=f"Drogue Deploy ({DROGUE_DEPLOYMENT_TIME_S:.1f}s)",
        )
    if START_TIME_S <= MAIN_DEPLOYMENT_TIME_S <= END_TIME_S:
        axes[1].axvline(
            MAIN_DEPLOYMENT_TIME_S,
            color="brown",
            linestyle=":",
            linewidth=1.5,
            label=f"Main Deploy ({MAIN_DEPLOYMENT_TIME_S:.1f}s)",
        )
    axes[1].legend(ncol=3)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    window.to_csv(output_csv, index=False)

    zoom_mask = df["Flight_Time_(s)"].between(
        ZOOM_START_TIME_S, ZOOM_END_TIME_S, inclusive="both"
    )
    zoom_window = df.loc[zoom_mask].copy()
    if zoom_window.empty:
        raise ValueError(
            f"No rows found in Flight_Time_(s) window {ZOOM_START_TIME_S} to {ZOOM_END_TIME_S}."
        )

    fig2, axes2 = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    fig2.suptitle("SnowbirdPrim HR: Acceleration and Gyro (65s to 170s)", fontsize=14)

    axes2[0].plot(zoom_window["Flight_Time_(s)"], zoom_window["Accel_X"], label="Accel_X", lw=1)
    axes2[0].plot(zoom_window["Flight_Time_(s)"], zoom_window["Accel_Y"], label="Accel_Y", lw=1)
    axes2[0].plot(zoom_window["Flight_Time_(s)"], zoom_window["Accel_Z"], label="Accel_Z", lw=1)
    axes2[0].set_ylabel("Acceleration (g)")
    axes2[0].grid(True, alpha=0.3)
    axes2[0].legend(ncol=3)

    axes2[1].plot(zoom_window["Flight_Time_(s)"], zoom_window["Gyro_X"], label="Gyro_X", lw=1)
    axes2[1].plot(zoom_window["Flight_Time_(s)"], zoom_window["Gyro_Y"], label="Gyro_Y", lw=1)
    axes2[1].plot(zoom_window["Flight_Time_(s)"], zoom_window["Gyro_Z"], label="Gyro_Z", lw=1)
    axes2[1].set_xlabel("Flight Time (s)")
    axes2[1].set_ylabel("Gyro (deg/s)")
    axes2[1].grid(True, alpha=0.3)
    axes2[1].legend(ncol=3)

    plt.tight_layout()
    plt.savefig(output_png_zoom, dpi=300, bbox_inches="tight")
    plt.close(fig2)

    zoom_window.to_csv(output_csv_zoom, index=False)

    print(f"Saved plot: {output_png}")
    print(f"Saved filtered data: {output_csv}")
    print(f"Saved zoom plot: {output_png_zoom}")
    print(f"Saved zoom filtered data: {output_csv_zoom}")
    print(f"Rows in window: {len(window)}")
    print(f"Rows in zoom window: {len(zoom_window)}")
    print(
        f"Deployment markers: drogue={DROGUE_DEPLOYMENT_TIME_S:.1f}s, "
        f"main={MAIN_DEPLOYMENT_TIME_S:.1f}s"
    )
    print(
        f"Time range in filtered data: {window['Flight_Time_(s)'].min():.3f}s to "
        f"{window['Flight_Time_(s)'].max():.3f}s"
    )


if __name__ == "__main__":
    main()
