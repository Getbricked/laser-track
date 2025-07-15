import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === 1. Set video path ===
video_path = "video.mp4"  # Replace with your actual file path

# === 2. Open the video ===
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError("Cannot open video file")

# === 3. Define HSV range for orange-red crossbar ===
lower_orange = np.array([5, 100, 100])
upper_orange = np.array([25, 255, 255])

bar_positions = []
frame_count = 0

# === 4. Frame-by-frame processing ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Take the largest contour (assumed to be the bar)
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 50:  # Ignore small noise
            M = cv2.moments(largest)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                bar_positions.append((frame_count, cx, cy))

    frame_count += 1

cap.release()

# === 5. Save positions to CSV ===
df = pd.DataFrame(bar_positions, columns=["Frame", "X", "Y"])
df.to_csv("crossbar_positions.csv", index=False)

# === 6. Plot movement graph ===
plt.figure(figsize=(10, 6))
plt.plot(df["X"], df["Y"], marker="o", linestyle="-")
plt.title("Crossbar Center Position Over Time")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.grid(True)
plt.savefig("crossbar_movement.png")
plt.close()
