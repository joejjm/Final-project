import os
import cv2
import pandas as pd
from ultralytics import YOLO
import numpy as np

# --- CONFIG ---
VIDEO_DIR = 'video_files/dante'
OUTPUT_DIR = 'data/mechanical_features'
MODEL_PATH = 'yolov8m.pt'

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_keypoints_and_features(video_path, model):
    print(f"Processing video file: {video_path}")
    video_basename = os.path.basename(video_path)
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    keypoints_list = []
    features_list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Run YOLO and capture printed output
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        results = model(frame)
        sys.stdout = old_stdout
        yolo_output = mystdout.getvalue()
        if yolo_output.strip():
            for line in yolo_output.strip().splitlines():
                print(f"{video_basename}: {line}")
        # Example: extract all detected keypoints for person (class 0)
        for box in results[0].boxes:
            cls = int(box.cls[0])
            if cls == 0 and hasattr(box, 'keypoints'):
                keypoints = box.keypoints[0].cpu().numpy() if hasattr(box.keypoints[0], 'cpu') else box.keypoints[0]
                keypoints_list.append(keypoints)
                # Example mechanical feature: shoulder-elbow-wrist angle (if keypoints available)
                if keypoints.shape[0] >= 7:
                    shoulder = keypoints[5]
                    elbow = keypoints[6]
                    wrist = keypoints[7]
                    angle = compute_angle(shoulder, elbow, wrist)
                    features_list.append({'frame': frame_idx, 'elbow_angle': angle})
                else:
                    features_list.append({'frame': frame_idx, 'elbow_angle': np.nan})
        frame_idx += 1
    cap.release()
    return keypoints_list, features_list

def compute_angle(a, b, c):
    # Compute angle at point b given three points a, b, c
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def process_all_videos():
    model = YOLO(MODEL_PATH)
    for fname in sorted(os.listdir(VIDEO_DIR)):
        if fname.endswith('.mp4'):
            video_path = os.path.join(VIDEO_DIR, fname)
            print(f'---\nNow working on: {video_path}')
            keypoints, features = extract_keypoints_and_features(video_path, model)
            # Save features as CSV
            features_df = pd.DataFrame(features)
            out_csv = os.path.join(OUTPUT_DIR, fname.replace('.mp4', '_features.csv'))
            print(f'Saving features to: {out_csv}')
            features_df.to_csv(out_csv, index=False)
            print(f'Finished: {video_path}\n---')

if __name__ == '__main__':
    process_all_videos()
