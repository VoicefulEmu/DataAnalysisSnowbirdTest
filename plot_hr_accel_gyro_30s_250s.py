from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_PATH = Path(__file__).resolve().parent
INPUT_CSV = BASE_PATH / "SnowbirdPrim HR_03-21-2026_10_52_17.csv"
OUTPUT_PNG = BASE_PATH / "hr_accel_gyro_30s_250s.png"
OUTPUT_CSV = BASE_PATH / "SnowbirdPrim HR_03-21-2026_10_52_17_30s_250s.csv"
OUTPUT_PNG_ZOOM = BASE_PATH / "hr_accel_gyro_65s_170s.png"
OUTPUT_CSV_ZOOM = BASE_PATH / "SnowbirdPrim HR_03-21-2026_10_52_17_65s_170s.csv"

START_TIME_S = 30.0
END_TIME_S = 250.0
ZOOM_START_TIME_S = 65.0
ZOOM_END_TIME_S = 170.0
DROGUE_DEPLOYMENT_TIME_S = 38.6
MAIN_DEPLOYMENT_TIME_S = 217.9


def main() -> None:
    df = pd.read_csv(INPUT_CSV, low_memory=False)
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
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    window.to_csv(OUTPUT_CSV, index=False)

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
    plt.savefig(OUTPUT_PNG_ZOOM, dpi=300, bbox_inches="tight")
    plt.close(fig2)

    zoom_window.to_csv(OUTPUT_CSV_ZOOM, index=False)

    print(f"Saved plot: {OUTPUT_PNG}")
    print(f"Saved filtered data: {OUTPUT_CSV}")
    print(f"Saved zoom plot: {OUTPUT_PNG_ZOOM}")
    print(f"Saved zoom filtered data: {OUTPUT_CSV_ZOOM}")
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
