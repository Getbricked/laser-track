import laser_track, crossbar_track
import pandas as pd
import numpy as np
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt

# === 1. Load both CSVs ===
laser_df = pd.read_csv("laser_dot_positions.csv")
bar_df = pd.read_csv("crossbar_positions.csv")

# === 2. Merge on Frame column ===
merged_df = pd.merge(laser_df, bar_df, on="Frame", suffixes=("_laser", "_bar"))

# === 3. Compute new x and y ===
merged_df["x"] = merged_df["X_laser"] - merged_df["X_bar"]
merged_df["y"] = merged_df["Y_laser"] - merged_df["Y_bar"]

# === 4. Keep only required columns ===
final_df = merged_df[["Frame", "x", "y"]]

# === 5. Save to new CSV ===
final_df.to_csv("laser_relative_to_bar.csv", index=False)

print("Saved to laser_relative_to_bar.csv")

# Plot x vs Frame and save as PNG
plt.figure(figsize=(10, 4))
plt.plot(final_df["Frame"], final_df["x"], marker="o", linestyle="-", color="b")
plt.xlabel("Frame")
plt.ylabel("x")
plt.title("x vs Frame")
plt.grid(True)
plt.tight_layout()
plt.savefig("x_vs_frame.png")
plt.show()

# Plot y vs Frame and save as PNG
plt.figure(figsize=(10, 4))
plt.plot(final_df["Frame"], final_df["y"], marker="o", linestyle="-", color="g")
plt.xlabel("Frame")
plt.ylabel("y (relative to bar)")
plt.title("y (relative to bar) vs Frame")
plt.grid(True)
plt.tight_layout()
plt.savefig("y_vs_frame.png")
plt.show()
# Plot the laser trajectory as a smooth curve (curveline)

x = final_df["x"].values
y = final_df["y"].values

# Create smooth curve using spline interpolation
if len(x) > 3:
    t = np.linspace(0, 1, len(x))
    t_smooth = np.linspace(0, 1, 300)
    spline_x = make_interp_spline(t, x, k=3)
    spline_y = make_interp_spline(t, y, k=3)
    x_smooth = spline_x(t_smooth)
    y_smooth = spline_y(t_smooth)
else:
    x_smooth = x
    y_smooth = y

plt.figure(figsize=(6, 9))
plt.plot(x_smooth, y_smooth, color="blue", linewidth=2)

plt.xlabel("x")
plt.ylabel("y (relative to bar)")
plt.title("Laser Trajectory Relative to Bar")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("laser_trajectory.png")
plt.show()
