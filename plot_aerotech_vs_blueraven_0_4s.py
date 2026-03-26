import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

base_path = Path(r"c:\Users\lamor\OneDrive\src\DataAnalysisSnowbirdTest")
aerotech_file = base_path / "AeroTech_O5500X-PS.eng"
blueraven_hr_file = base_path / "SnowbirdPrim HR_03-21-2026_10_52_17.csv"

# Read AeroTech .eng curve
with open(aerotech_file, "r", encoding="utf-8", errors="ignore") as f:
    lines = [line.strip() for line in f.readlines() if line.strip()]

# first line is header; remaining lines are time thrust.
curve_lines = []
for line in lines[1:]:
    parts = line.split()
    if len(parts) < 2:
        continue
    try:
        t = float(parts[0])
        y = float(parts[1])
    except ValueError:
        continue
    curve_lines.append((t, y))

if not curve_lines:
    raise SystemExit("No AeroTech thrust points found in file")

at_df = pd.DataFrame(curve_lines, columns=["time_s", "thrust_lbf"]).sort_values("time_s")

# Read BlueRaven HR data
hr = pd.read_csv(blueraven_hr_file)
hr["Flight_Time_(s)"] = pd.to_numeric(hr["Flight_Time_(s)"], errors="coerce")
for col in ["Accel_X", "Accel_Y", "Accel_Z"]:
    if col in hr.columns:
        hr[col] = pd.to_numeric(hr[col], errors="coerce")

hr = hr.dropna(subset=["Flight_Time_(s)", "Accel_X", "Accel_Y", "Accel_Z"])
hr = hr.sort_values("Flight_Time_(s)")
hr_subset = hr[(hr["Flight_Time_(s)"] >= 0) & (hr["Flight_Time_(s)"] <= 4.0)].copy()

hr_subset["accel_mag_g"] = np.sqrt(hr_subset["Accel_X"] ** 2 + hr_subset["Accel_Y"] ** 2 + hr_subset["Accel_Z"] ** 2)

# Plot
fig, ax1 = plt.subplots(figsize=(11, 6))
ax1.plot(at_df["time_s"], at_df["thrust_lbf"], color="tab:blue", lw=2, label="AeroTech thrust (lbf)")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Thrust (lbf)", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")
ax1.set_xlim(0, 4)
ax1.grid(alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(hr_subset["Flight_Time_(s)"], hr_subset["accel_mag_g"], color="tab:red", lw=1, label="BlueRaven Accel mag (g)")
ax2.set_ylabel("Accel magnitude (g)", color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc="upper right")

plt.title("AeroTech thrust curve vs BlueRaven accel magnitude (0-4 s)")
plt.tight_layout()
out_file = base_path / "aerotech_vs_blueraven_0_4s.png"
plt.savefig(out_file, dpi=200)
print(f"Saved plot to {out_file}")
plt.show()
