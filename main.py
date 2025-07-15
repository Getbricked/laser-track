import laser_track, crossbar_track
import pandas as pd

# === 1. Load both CSVs ===
laser_df = pd.read_csv("laser_dot_positions.csv")
bar_df = pd.read_csv("crossbar_positions.csv")

# === 2. Merge on Frame column ===
merged_df = pd.merge(laser_df, bar_df, on="Frame", suffixes=("_laser", "_bar"))

# === 3. Compute new x and y ===
merged_df["x"] = merged_df["X_laser"]
merged_df["y"] = merged_df["Y_laser"] - merged_df["Y_bar"]

# === 4. Keep only required columns ===
final_df = merged_df[["Frame", "x", "y"]]

# === 5. Save to new CSV ===
final_df.to_csv("laser_relative_to_bar.csv", index=False)

print("Saved to laser_relative_to_bar.csv")

import matplotlib.pyplot as plt

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
