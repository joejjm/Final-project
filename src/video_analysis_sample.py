import cv2
import numpy as np

# Path to your video file
VIDEO_PATH = '../data/sample_video.mp4'  # Replace with your video file

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Cannot open video file {VIDEO_PATH}")
    exit()

ret, prev_frame = cap.read()
if not ret:
    print("Error: Cannot read the first frame.")
    cap.release()
    exit()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Compute absolute difference between current frame and previous frame
    diff = cv2.absdiff(prev_gray, gray)
    # Threshold the difference
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    # Show the result
    cv2.imshow('Frame Difference', thresh)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
    prev_gray = gray

cap.release()
cv2.destroyAllWindows()
