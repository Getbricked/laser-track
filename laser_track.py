import cv2
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# After loading the video, get its width and height
video_path = input("Enter the path to the video file: ")
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError("Cannot open video")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
center_x = frame_width // 2
center_y = frame_height // 2

print(f"Center point: {center_x}x{center_y}")

positions = []
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 5:
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                # Shift coordinates so center is (0,0)
                shifted_x = cx - center_x
                shifted_y = center_y - cy  # Reverse subtraction for correct direction
                positions.append((frame_count, shifted_x, shifted_y))
    frame_count += 1

cap.release()

df = pd.DataFrame(positions, columns=["Frame", "X", "Y"])
df.to_csv("laser_dot_positions.csv", index=False)

plt.figure(figsize=(10, 6))
plt.plot(df["X"], df["Y"], linestyle="-")  # Removed marker="o"
plt.title("Laser Dot Movement (Centered at the middle of the video)")
plt.xlabel("X Position (centered)")
plt.ylabel("Y Position (centered)")
plt.grid(True)
plt.savefig("laser_dot_movement.png")
plt.close()
