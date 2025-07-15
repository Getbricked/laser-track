import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# === 1. Load the video ===
video_path = "video.mp4"  # Change if needed
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise IOError("Cannot open video")

# === 2. Track the laser dot ===
positions = []
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to HSV for color segmentation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define red laser dot color ranges
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])

    # Combine red masks
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 5:  # Filter noise
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                positions.append((frame_count, cx, cy))

    frame_count += 1

cap.release()

# === 3. Save tracked positions ===
df = pd.DataFrame(positions, columns=["Frame", "X", "Y"])
df.to_csv("laser_dot_positions.csv", index=False)

# === 4. Plot and save the graph ===
plt.figure(figsize=(10, 6))
plt.plot(df["X"], df["Y"], marker="o", linestyle="-")
plt.title("Laser Dot Movement")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.grid(True)
plt.savefig("laser_dot_movement.png")
plt.close()
