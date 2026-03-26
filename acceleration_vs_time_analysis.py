import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set up file paths
base_path = Path(r"c:\Users\lamor\OneDrive\src\DataAnalysisSnowbirdTest")
hr_file = base_path / "SnowbirdPrim HR_03-21-2026_10_52_17.csv"
lr_file = base_path / "SnowbirdPrim LR_03-21-2026_10_52_17.csv"

# Load data
print("Loading High Rate (HR) data...")
hr_data = pd.read_csv(hr_file)

print("Loading Low Rate (LR) data...")
lr_data = pd.read_csv(lr_file)

# Calculate acceleration magnitude from HR data
print("\nCalculating acceleration from HR data...")
hr_data['Accel_Magnitude'] = np.sqrt(
    hr_data['Accel_X']**2 + hr_data['Accel_Y']**2 + hr_data['Accel_Z']**2
)

# Calculate acceleration from velocity derivatives in LR data
# NOTE: Velocity_Up is inertial frame velocity, not descent rate!
print("Calculating acceleration from velocity derivatives in LR data...")
lr_data['Accel_from_VelUp'] = np.gradient(lr_data['Velocity_Up'], lr_data['Flight_Time_(s)'])
lr_data['Accel_from_VelDR'] = np.gradient(lr_data['Velocity_DR'], lr_data['Flight_Time_(s)'])
lr_data['Accel_from_VelCR'] = np.gradient(lr_data['Velocity_CR'], lr_data['Flight_Time_(s)'])

# Calculate total inertial velocity magnitude
lr_data['Velocity_Magnitude'] = np.sqrt(lr_data['Velocity_Up']**2 + lr_data['Velocity_DR']**2 + lr_data['Velocity_CR']**2)

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Snowbird Rocket - Acceleration Analysis\n(Note: Velocity_Up is inertial frame, not descent rate)', 
             fontsize=14, fontweight='bold')

# Plot 1: HR Acceleration Magnitude vs Time
ax1 = axes[0, 0]
ax1.plot(hr_data['Flight_Time_(s)'], hr_data['Accel_Magnitude'], 'b-', linewidth=0.8, label='Accel Magnitude')
ax1.set_xlabel('Flight Time (s)')
ax1.set_ylabel('Acceleration (g)')
ax1.set_title('HR Data - Acceleration Magnitude vs Time')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: HR Individual Acceleration Components
ax2 = axes[0, 1]
ax2.plot(hr_data['Flight_Time_(s)'], hr_data['Accel_X'], label='Accel_X', linewidth=0.8)
ax2.plot(hr_data['Flight_Time_(s)'], hr_data['Accel_Y'], label='Accel_Y', linewidth=0.8)
ax2.plot(hr_data['Flight_Time_(s)'], hr_data['Accel_Z'], label='Accel_Z', linewidth=0.8)
ax2.set_xlabel('Flight Time (s)')
ax2.set_ylabel('Acceleration (g)')
ax2.set_title('HR Data - Individual Acceleration Components')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Plot 3: LR Velocity Magnitude (Inertial)
ax3 = axes[0, 2]
ax3.plot(lr_data['Flight_Time_(s)'], lr_data['Velocity_Magnitude'], 'purple', linewidth=1.2)
ax3.set_xlabel('Flight Time (s)')
ax3.set_ylabel('Velocity Magnitude (ft/s)')
ax3.set_title('LR Data - Total Velocity Magnitude (Inertial)')
ax3.grid(True, alpha=0.3)

# Plot 4: LR Acceleration from Velocity_Up Derivative
ax4 = axes[1, 0]
ax4.plot(lr_data['Flight_Time_(s)'], lr_data['Accel_from_VelUp'], 'r-', linewidth=1.5)
ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax4.set_xlabel('Flight Time (s)')
ax4.set_ylabel('Acceleration (ft/s²)')
ax4.set_title('LR Data - Acceleration from Velocity_Up')
ax4.grid(True, alpha=0.3)

# Plot 5: LR Individual Velocity Components
ax5 = axes[1, 1]
ax5.plot(lr_data['Flight_Time_(s)'], lr_data['Velocity_Up'], label='Velocity_Up', linewidth=1)
ax5.plot(lr_data['Flight_Time_(s)'], lr_data['Velocity_DR'], label='Velocity_DR', linewidth=1)
ax5.plot(lr_data['Flight_Time_(s)'], lr_data['Velocity_CR'], label='Velocity_CR', linewidth=1)
ax5.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax5.set_xlabel('Flight Time (s)')
ax5.set_ylabel('Velocity (ft/s)')
ax5.set_title('LR Data - Velocity Components (Inertial Frame)')
ax5.grid(True, alpha=0.3)
ax5.legend()

# Plot 6: LR Altitude vs Time (Ground Truth)
ax6 = axes[1, 2]
ax6.plot(lr_data['Flight_Time_(s)'], lr_data['Baro_Altitude_AGL_(feet)'], 'orange', linewidth=2)
ax6.set_xlabel('Flight Time (s)')
ax6.set_ylabel('Altitude AGL (feet)')
ax6.set_title('LR Data - Barometric Altitude (Ground Truth)')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(base_path / 'acceleration_vs_time_analysis.png', dpi=300, bbox_inches='tight')
print(f"\nGraph saved to: {base_path / 'acceleration_vs_time_analysis.png'}")

# Print summary statistics
print("\n" + "="*60)
print("ACCELERATION STATISTICS - HIGH RATE (HR) DATA")
print("="*60)
print(f"Accel Magnitude - Min: {hr_data['Accel_Magnitude'].min():.4f} g")
print(f"Accel Magnitude - Max: {hr_data['Accel_Magnitude'].max():.4f} g")
print(f"Accel Magnitude - Mean: {hr_data['Accel_Magnitude'].mean():.4f} g")
print(f"Accel_X - Min: {hr_data['Accel_X'].min():.4f} g, Max: {hr_data['Accel_X'].max():.4f} g")
print(f"Accel_Y - Min: {hr_data['Accel_Y'].min():.4f} g, Max: {hr_data['Accel_Y'].max():.4f} g")
print(f"Accel_Z - Min: {hr_data['Accel_Z'].min():.4f} g, Max: {hr_data['Accel_Z'].max():.4f} g")
print(f"Total HR data points: {len(hr_data)}")

print("\n" + "="*60)
print("ACCELERATION STATISTICS - LOW RATE (LR) DATA")
print("="*60)
print(f"Accel from Velocity_Up - Min: {lr_data['Accel_from_VelUp'].min():.2f} ft/s²")
print(f"Accel from Velocity_Up - Max: {lr_data['Accel_from_VelUp'].max():.2f} ft/s²")
print(f"Accel from Velocity_DR  - Min: {lr_data['Accel_from_VelDR'].min():.2f} ft/s²")
print(f"Accel from Velocity_DR  - Max: {lr_data['Accel_from_VelDR'].max():.2f} ft/s²")
print(f"\nVelocity Components (INERTIAL FRAME - not descent rate):")
print(f"Velocity_Up (vertical inertial)   - Range: {lr_data['Velocity_Up'].min():.0f} to {lr_data['Velocity_Up'].max():.0f} ft/s")
print(f"Velocity_DR (dead reckoning)      - Range: {lr_data['Velocity_DR'].min():.0f} to {lr_data['Velocity_DR'].max():.0f} ft/s")
print(f"Velocity_CR (cross-range)         - Range: {lr_data['Velocity_CR'].min():.0f} to {lr_data['Velocity_CR'].max():.0f} ft/s")
print(f"Total velocity magnitude          - Max: {lr_data['Velocity_Magnitude'].max():.0f} ft/s")
print(f"Total LR data points: {len(lr_data)}")

print("\n" + "="*60)
print("KEY OBSERVATION - Why velocity keeps accelerating:")
print("="*60)
print("Velocity_Up, Velocity_DR, and Velocity_CR are velocity components in")
print("the INERTIAL reference frame, not ground-relative descent rates.")
print("These increase in magnitude because:")
print("  1. The rocket tumbles/oscillates after parachute deployment")
print("  2. The inertial frame velocity continues changing with acceleration")
print("  3. Actual descent rate is shown by the ALTITUDE (see Plot 6)")
print()
print(f"Altitude at time 219.36s (parachute fires):  {lr_data.iloc[11050]['Baro_Altitude_AGL_(feet)']:.1f} ft")
print(f"Altitude at end of flight (335.24s):        {lr_data.iloc[-1]['Baro_Altitude_AGL_(feet)']:.1f} ft")
print(f"Descent duration with main chute:           {lr_data.iloc[-1]['Flight_Time_(s)'] - 219.36:.1f} seconds")

print("\n" + "="*60)
print("FLIGHT TIME RANGE")
print("="*60)
print(f"HR Data - Flight Time: {hr_data['Flight_Time_(s)'].min():.3f}s to {hr_data['Flight_Time_(s)'].max():.3f}s")
print(f"LR Data - Flight Time: {lr_data['Flight_Time_(s)'].min():.3f}s to {lr_data['Flight_Time_(s)'].max():.3f}s")

plt.show()
