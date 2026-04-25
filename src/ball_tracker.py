import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from supertracker import ByteTrack, Detections

SPORTS_BALL_CLASS = 32  # COCO class id for 'sports ball'


def track_ball(video_path, output_csv, model_path='yolov8m.pt', conf_threshold=0.01,
               write_video=True):
    """
    Track the baseball in a video using YOLO + ByteTrack (supertracker).

    Outputs a CSV with columns: frame, x, y (ball centre pixel coords).
    Optionally writes an annotated MP4 alongside the CSV.
    """
    model = YOLO(model_path)

    # Normalize output names so new tracker artifacts do not overwrite older files.
    output_root, _ = os.path.splitext(output_csv)
    if not output_root.endswith('_nm'):
        output_root = f"{output_root}_nm"
    output_csv = f"{output_root}.csv"

    tracker = ByteTrack(
        track_activation_threshold=conf_threshold,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        frame_rate=30,
        minimum_consecutive_frames=1,
    )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = None
    video_out_path = None
    if write_video:
        video_out_path = f"{output_root}_tracking.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_out_path, fourcc, fps, (width, height))

    records = []
    frame_idx = 0
    primary_track_id = None
    last_pos = None
    last_bbox = None
    last_velocity = (0.0, 0.0)
    last_template = None
    missing_frames = 0
    max_missing_frames = int(fps)  # Keep continuity for up to ~1 second of misses.
    template_match_min_score = 0.35
    search_radius = 90

    def clamp(value, low, high):
        return max(low, min(high, value))

    def get_gray(frame_bgr):
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    def extract_template(gray_frame, bbox):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = clamp(x1, 0, gray_frame.shape[1] - 1)
        y1 = clamp(y1, 0, gray_frame.shape[0] - 1)
        x2 = clamp(x2, x1 + 1, gray_frame.shape[1])
        y2 = clamp(y2, y1 + 1, gray_frame.shape[0])
        patch = gray_frame[y1:y2, x1:x2]
        if patch.size == 0 or patch.shape[0] < 4 or patch.shape[1] < 4:
            return None
        return patch

    def template_reacquire(gray_frame, template, predicted_pos):
        if template is None or predicted_pos is None:
            return None

        th, tw = template.shape[:2]
        if th >= gray_frame.shape[0] or tw >= gray_frame.shape[1]:
            return None

        px, py = int(predicted_pos[0]), int(predicted_pos[1])
        x1 = clamp(px - search_radius, 0, gray_frame.shape[1] - tw)
        y1 = clamp(py - search_radius, 0, gray_frame.shape[0] - th)
        x2 = clamp(px + search_radius, tw, gray_frame.shape[1])
        y2 = clamp(py + search_radius, th, gray_frame.shape[0])

        if x2 <= x1 or y2 <= y1:
            return None

        search_img = gray_frame[y1:y2, x1:x2]
        if search_img.shape[0] < th or search_img.shape[1] < tw:
            return None

        result = cv2.matchTemplate(search_img, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val < template_match_min_score:
            return None

        rx1 = x1 + max_loc[0]
        ry1 = y1 + max_loc[1]
        rx2 = rx1 + tw
        ry2 = ry1 + th
        rcx = (rx1 + rx2) // 2
        rcy = (ry1 + ry2) // 2
        return (rx1, ry1, rx2, ry2, rcx, rcy, float(max_val))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = get_gray(frame)

        results = model(frame, conf=conf_threshold, verbose=False)

        # Build Detections manually — from_yolo expects numpy but ultralytics
        # returns PyTorch tensors, so we convert explicitly.
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confidence = boxes.conf.cpu().numpy()
            class_id = boxes.cls.cpu().numpy().astype(int)
            detections = Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
        else:
            detections = Detections.empty()

        # Keep only sports-ball detections for tracking
        if detections.class_id is not None and len(detections) > 0:
            ball_mask = detections.class_id == SPORTS_BALL_CLASS
            ball_detections = detections[ball_mask]
        else:
            ball_detections = Detections.empty()

        tracked = tracker.update_with_detections(ball_detections)

        cx, cy = None, None
        x1 = y1 = x2 = y2 = None
        source = 'none'
        score_text = ''

        if len(tracked) > 0:
            best_idx = None
            track_ids = tracked.tracker_id if tracked.tracker_id is not None else None

            if track_ids is not None and primary_track_id is not None:
                matches = np.where(track_ids == primary_track_id)[0]
                if len(matches) > 0:
                    best_idx = int(matches[0])

            if best_idx is None and last_pos is not None:
                centers = np.column_stack(((tracked.xyxy[:, 0] + tracked.xyxy[:, 2]) / 2,
                                           (tracked.xyxy[:, 1] + tracked.xyxy[:, 3]) / 2))
                distances = np.sqrt((centers[:, 0] - last_pos[0]) ** 2 + (centers[:, 1] - last_pos[1]) ** 2)
                best_idx = int(np.argmin(distances))

            if best_idx is None:
                best_idx = int(np.argmax(tracked.confidence)) if tracked.confidence is not None else 0

            x1, y1, x2, y2 = tracked.xyxy[best_idx].astype(int)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            source = 'det'
            missing_frames = 0

            if track_ids is not None:
                primary_track_id = int(track_ids[best_idx])

            current_conf = float(tracked.confidence[best_idx]) if tracked.confidence is not None else float('nan')
            score_text = f"conf={current_conf:.2f}"

            if last_pos is not None:
                vx = cx - last_pos[0]
                vy = cy - last_pos[1]
                last_velocity = (0.7 * last_velocity[0] + 0.3 * vx, 0.7 * last_velocity[1] + 0.3 * vy)

            last_bbox = (x1, y1, x2, y2)
            maybe_template = extract_template(frame_gray, last_bbox)
            if maybe_template is not None:
                last_template = maybe_template

        if cx is None and last_pos is not None:
            predicted_pos = (last_pos[0] + last_velocity[0], last_pos[1] + last_velocity[1])
            reacquired = template_reacquire(frame_gray, last_template, predicted_pos)
            if reacquired is not None:
                x1, y1, x2, y2, cx, cy, tm_score = reacquired
                source = 'tmpl'
                score_text = f"tm={tm_score:.2f}"
                missing_frames = 0

                vx = cx - last_pos[0]
                vy = cy - last_pos[1]
                last_velocity = (0.7 * last_velocity[0] + 0.3 * vx, 0.7 * last_velocity[1] + 0.3 * vy)

                last_bbox = (x1, y1, x2, y2)
                maybe_template = extract_template(frame_gray, last_bbox)
                if maybe_template is not None:
                    last_template = maybe_template

        if cx is None and last_pos is not None and missing_frames < max_missing_frames:
            pred_x = int(round(last_pos[0] + last_velocity[0]))
            pred_y = int(round(last_pos[1] + last_velocity[1]))
            pred_x = clamp(pred_x, 0, width - 1)
            pred_y = clamp(pred_y, 0, height - 1)
            cx, cy = pred_x, pred_y
            source = 'pred'
            score_text = f"miss={missing_frames + 1}"
            missing_frames += 1

        if cx is None:
            source = 'none'
            missing_frames += 1
            if missing_frames >= max_missing_frames:
                primary_track_id = None
                last_bbox = None
                last_template = None
                last_velocity = (0.0, 0.0)
                last_pos = None
        else:
            last_pos = (float(cx), float(cy))

        records.append({'frame': frame_idx, 'x': cx, 'y': cy})

        if out is not None and cx is not None and cy is not None:
            if source == 'det':
                box_color = (0, 255, 0)
                dot_color = (0, 0, 255)
            elif source == 'tmpl':
                box_color = (0, 255, 255)
                dot_color = (255, 255, 0)
            else:
                box_color = (200, 200, 200)
                dot_color = (255, 255, 255)

            if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)

            cv2.circle(frame, (int(cx), int(cy)), 5, dot_color, -1)
            label = f"{source} ({int(cx)},{int(cy)})"
            if score_text:
                label = f"{label} {score_text}"
            cv2.putText(frame, label, (int(cx) + 8, int(cy) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        if out is not None:
            out.write(frame)

        frame_idx += 1

    cap.release()
    if out is not None:
        out.release()

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} frames → {output_csv}")
    if write_video:
        print(f"Saved annotated video → {video_out_path}")

    return df


def batch_track(video_dir, output_dir, model_path='yolov8m.pt'):
    """Process every .mp4 in video_dir and write CSVs to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    for fname in sorted(os.listdir(video_dir)):
        if not fname.lower().endswith('.mp4'):
            continue
        video_path = os.path.join(video_dir, fname)
        output_csv = os.path.join(output_dir, fname.replace('.mp4', '_nm.csv'))
        print(f"\n--- Processing {fname} ---")
        track_ball(video_path, output_csv, model_path=model_path)


if __name__ == '__main__':
    batch_track('video_files/dante', 'data/ball_features')
