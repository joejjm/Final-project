import cv2
import numpy as np
import os

def characterize_changes(frame_dir, threshold=30):
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
    prev_frame = None
    changes = []
    for idx, fname in enumerate(frame_files):
        frame_path = os.path.join(frame_dir, fname)
        frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        if prev_frame is not None:
            diff = cv2.absdiff(prev_frame, frame)
            change_score = np.sum(diff > threshold)
            changes.append((fname, change_score))
        prev_frame = frame
    return changes

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python frame_change_characterizer.py <frame_dir>")
        sys.exit(1)
    changes = characterize_changes(sys.argv[1])
    for fname, score in changes:
        print(f"{fname}: {score} pixels changed")
