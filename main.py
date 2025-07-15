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
