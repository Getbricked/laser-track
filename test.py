import cv2
import numpy as np

video_path = input("Enter video file path: ")
cap = cv2.VideoCapture(video_path)

# Get frame dimensions
ret, frame = cap.read()
if not ret:
    print("Failed to read video")
    exit()
height, width = frame.shape[:2]
center_x, center_y = width // 2, height // 2
print(f"Frame center: x={center_x}, y={center_y}")

# Reset video to first frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            rel_x = cx - center_x
            rel_y = center_y - cy  # Reverse subtraction for correct direction
            print(f"Laser position (center=0,0): x={rel_x}, y={rel_y}")

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 50:
            (x, y, w, h) = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                "Red Laser",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

    # Draw center point
    cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)
    cv2.imshow("Red Laser Tracking", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
