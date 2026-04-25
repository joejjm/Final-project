import os
import cv2
import pandas as pd
from ultralytics import YOLO

def extract_ball_features(video_path, output_csv, model_path='yolov8m.pt'):
    # ...existing code...
    # ...existing code...
    model = YOLO(model_path)
    # Lower confidence threshold for detection (even lower)
    model.conf = 0.01
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    xs, ys = [], []  # For tracked ball
    tracked_ball = None  # (x, y) of the currently tracked ball
    ball_memory = None   # Last known (x, y) of the ball
    ball_memory_frames = 0  # Number of consecutive frames ball has been missing
    BALL_MEMORY_MAX = 5     # Allow up to 5 frames of memory
    # ROI_X_MIN = 150  # User-defined ROI: only consider balls with x > 150 for initial assignment
    ROI_X_MIN = 0  # Relaxed: consider all x positions for initial assignment
    frames = []
    glove_heights = []
    glove_frame_indices = []
    cap_heights = []
    cap_frame_indices = []
    lost_glove_frames = []  # For debugging: frames where glove is not found
    glove_to_person_top = []  # Running calculation for glove-to-person-top distance
    # YOLO-only glove detection: no trackers or templates
    # Define expected glove color (brown, in BGR)
    expected_bgr = (50, 80, 120)  # Adjust as needed for your glove
    color_thresh = 60  # Acceptable color distance threshold
    # Get video properties for output
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    # Prepare output video path
    video_out_path = output_csv.replace('.csv', '_tracking.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_out_path, fourcc, fps, (width, height))
    YOLO_GLOVE_CLASS = 39  # 39 is 'baseball glove' in COCO
    last_glove_y = None
    DEBUG = False
    # Color map for COCO classes (person, glove, ball, etc.)
    COCO_COLORS = {
        0: (0, 255, 255),   # person: yellow
        32: (0, 255, 0),    # sports ball: green
        35: (255, 0, 255),  # baseball glove: magenta
        26: (0, 0, 255),    # baseball cap: red
        14: (255, 128, 0),  # bench: orange
    }
    COCO_NAMES = {
        0: 'person', 32: 'sports ball', 35: 'baseball glove', 26: 'baseball cap', 14: 'bench'
    }
    frame_number = 0
    frame_height, frame_width = None, None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_height is None or frame_width is None:
            frame_height, frame_width = frame.shape[0], frame.shape[1]
        results = model(frame)
        found_ball = False
        video_basename = os.path.basename(video_path)
        found_ball2 = False
        cx, cy = None, None
        cx2, cy2 = None, None
        glove_y = None
        glove_info = ''
        # Debug: Print all detected objects and their classes for this frame
        print(f"{video_basename} | Frame {frame_number}: Detected objects:")
        for box in results[0].boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            conf = float(box.conf[0]) if hasattr(box, 'conf') else None
            print(f"{video_basename} |   Class {cls} | BBox: ({x1},{y1},{x2},{y2}) | Conf: {conf}")
        frame_number += 1
        template_y = None
        template_info = ''
        # Draw all detected objects with color-coded boxes
        # Filter for largest person and class 35 (baseball bat) detections
        largest_person = None
        largest_person_area = 0
        # Track all detected balls (class 32)
        ball_coords = []
        prioritized_ball = None
        prioritized_dist = None
        largest_35 = None
        largest_35_area = 0
        # Find all person and glove (class 35) bboxes
        person_boxes = []
        glove_boxes = []
        for box in results[0].boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            area = (x2 - x1) * (y2 - y1)
            if cls == 0:
                person_boxes.append((x1, y1, x2, y2, area))
            if cls == 35:
                glove_boxes.append((x1, y1, x2, y2, area))
        # Find the glove and person pair where the glove is inside the person bbox (and use the largest glove if multiple)
        glove_on_person = None
        person_with_glove = None
        max_glove_area = 0
        for gx1, gy1, gx2, gy2, garea in glove_boxes:
            for px1, py1, px2, py2, parea in person_boxes:
                if gx1 >= px1 and gx2 <= px2 and gy1 >= py1 and gy2 <= py2:
                    if garea > max_glove_area:
                        glove_on_person = (gx1, gy1, gx2, gy2)
                        person_with_glove = (px1, py1, px2, py2)
                        max_glove_area = garea
        # Fallback to largest glove and largest person if no glove is inside any person
        if glove_on_person is not None and person_with_glove is not None:
            largest_35 = glove_on_person
            largest_person = person_with_glove
        else:
            largest_35 = max(glove_boxes, key=lambda b: b[4], default=None)
            largest_person = max(person_boxes, key=lambda b: b[4], default=None)
            if largest_35 is not None:
                largest_35 = largest_35[:4]
            if largest_person is not None:
                largest_person = largest_person[:4]
        # Find the baseball cap (class 26) that is inside the largest person bbox (if any)
        cap_y = None
        for box in results[0].boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            if cls == 26 and largest_person is not None:
                px1, py1, px2, py2 = largest_person
                # Check if cap bbox is inside person bbox (or overlaps significantly)
                if x1 >= px1 and x2 <= px2 and y1 >= py1 and y2 <= py2:
                    cap_y = int((y1 + y2) / 2)
                    break
        # Calculate glove-to-person-top distance if both are found and glove is inside person
        if glove_on_person is not None and person_with_glove is not None:
            glove_top = glove_on_person[1]
            person_top = person_with_glove[1]
            glove_to_person_top.append(glove_top - person_top)
        else:
            glove_to_person_top.append(None)
        # Draw all detections (for debugging), highlight largest person and class 35 as before
        for box in results[0].boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            color = COCO_COLORS.get(cls, (128, 128, 128))
            name = COCO_NAMES.get(cls, f'class {cls}')
            highlight = False
            if cls == 32:
                bx = (x1 + x2) / 2
                by = (y1 + y2) / 2
                ball_coords.append((bx, by))
                # Prioritize balls in right-middle region for first 10 frames
                if frame_idx < 10:
                    # Define right-middle region (e.g., right 40% and middle 40% vertically)
                    x_center = frame_width * 0.7
                    y_min = frame_height * 0.3
                    y_max = frame_height * 0.7
                    if bx > x_center and y_min < by < y_max:
                        dist = abs(bx - frame_width * 0.85) + abs(by - frame_height * 0.5)
                        if prioritized_ball is None or dist < prioritized_dist:
                            prioritized_ball = (bx, by)
                            prioritized_dist = dist
            if cls == 0 and (x1, y1, x2, y2) == largest_person:
                highlight = True
            if cls == 35 and (x1, y1, x2, y2) == largest_35:
                highlight = True
            thickness = 3 if highlight else 1
            # Always draw all detections for debugging
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame, f"{name} ({conf:.2f})", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
            # Always draw the top edge of class 35 bounding box
            if cls == 35:
                cv2.line(frame, (x1, y1), (x2, y1), color, 2)
            # Draw a line for the cap center if found
            if cls == 26 and cap_y is not None and y1 <= cap_y <= y2:
                cv2.line(frame, (0, cap_y), (frame.shape[1], cap_y), color, thickness)
                cv2.putText(frame, f'Cap y: {cap_y}', (10, cap_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
        # Simple two-object tracking: assign detected balls to ball1/ball2 based on nearest previous position
        # ROI-based single ball tracking (relaxed: no x > ROI_X_MIN restriction)
        if len(ball_coords) == 0:
            # No ball detected this frame
            # Try pseudo-detection: find large white blob
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, white_mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            pseudo_ball = None
            max_area = 0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 100:  # Minimum area threshold for a big blob
                    x, y, w, h = cv2.boundingRect(cnt)
                    cx, cy = x + w // 2, y + h // 2
                    if area > max_area:
                        pseudo_ball = (cx, cy, area, (x, y, w, h))
                        max_area = area
            used_pseudo = False
            if pseudo_ball is not None:
                cx, cy, area, (x, y, w, h) = pseudo_ball
                # If memory exists, require pseudo to be close to memory
                if ball_memory is not None:
                    dist = ((cx - ball_memory[0]) ** 2 + (cy - ball_memory[1]) ** 2) ** 0.5
                    if dist < 80:  # Accept if within 80 pixels of memory
                        xs.append(cx)
                        ys.append(cy)
                        tracked_ball = (cx, cy)
                        ball_memory = (cx, cy)
                        ball_memory_frames = 0
                        used_pseudo = True
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                        cv2.putText(frame, "PSEUDO BALL", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                else:
                    # No memory, just use the pseudo ball
                    xs.append(cx)
                    ys.append(cy)
                    tracked_ball = (cx, cy)
                    ball_memory = (cx, cy)
                    ball_memory_frames = 0
                    used_pseudo = True
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                    cv2.putText(frame, "PSEUDO BALL", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            if not used_pseudo:
                if ball_memory is not None and ball_memory_frames < BALL_MEMORY_MAX:
                    # Use last known ball position
                    xs.append(ball_memory[0])
                    ys.append(ball_memory[1])
                    tracked_ball = ball_memory
                    ball_memory_frames += 1
                    cv2.circle(frame, (int(ball_memory[0]), int(ball_memory[1])), 5, (0, 165, 255), -1)
                    coord_text = f"MEMORY: x={int(ball_memory[0])}, y={int(ball_memory[1])}"
                    cv2.putText(frame, coord_text, (int(ball_memory[0]) + 10, int(ball_memory[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                else:
                    xs.append(None)
                    ys.append(None)
                    tracked_ball = None
                    ball_memory = None
                    ball_memory_frames = 0
        elif tracked_ball is None:
            # No ball being tracked yet: prioritize right-middle region for first 10 frames
            if prioritized_ball is not None:
                cx, cy = prioritized_ball
            elif largest_person is not None:
                px1, py1, px2, py2 = largest_person
                person_cx = (px1 + px2) / 2
                person_cy = (py1 + py2) / 2
                # Find the ball closest to the center of the largest person
                cx, cy = min(ball_coords, key=lambda b: abs(b[0] - person_cx) + abs(b[1] - person_cy))
            else:
                # Fallback: pick the ball with maximum x
                cx, cy = max(ball_coords, key=lambda b: b[0])
            tracked_ball = (cx, cy)
            xs.append(cx)
            ys.append(cy)
            ball_memory = (cx, cy)
            ball_memory_frames = 0
            cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
            coord_text = f"ball: x={int(cx)}, y={int(cy)}"
            cv2.putText(frame, coord_text, (int(cx) + 10, int(cy) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            # Track the ball closest to previous tracked_ball
            cx_prev, cy_prev = tracked_ball
            cx, cy = min(ball_coords, key=lambda b: abs(b[0] - cx_prev) + abs(b[1] - cy_prev))
            tracked_ball = (cx, cy)
            xs.append(cx)
            ys.append(cy)
            ball_memory = (cx, cy)
            ball_memory_frames = 0
            cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
            coord_text = f"ball: x={int(cx)}, y={int(cy)}"
            cv2.putText(frame, coord_text, (int(cx) + 10, int(cy) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            if cls == YOLO_GLOVE_CLASS:
                glove_y = int((y1 + y2) / 2)
                cv2.line(frame, (0, glove_y), (frame.shape[1], glove_y), color, thickness)
                cv2.putText(frame, f'Glove y: {glove_y}', (10, glove_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
                glove_info = f'YOLO: bbox=({x1},{y1},{x2},{y2}), y={glove_y}'
        if not found_ball:
            xs.append(None)
            ys.append(None)
        # Only YOLO glove detection is used now
        if glove_y is None:
            lost_glove_frames.append(frame_idx)
        if glove_y is not None:
            glove_heights.append(glove_y)
            glove_frame_indices.append(frame_idx)
        # Save cap height for this frame if found
        if cap_y is not None:
            cap_heights.append(cap_y)
            cap_frame_indices.append(frame_idx)
        out.write(frame)
        # Print glove-to-person-top calculation for each frame
        if glove_on_person is not None and person_with_glove is not None:
            glove_top = glove_on_person[1]
            person_top = person_with_glove[1]
            print(f'{video_basename} | Frame {frame_idx}: glove_top={glove_top}, person_top={person_top}, vertical_diff={glove_top - person_top}')
        else:
            print(f'{video_basename} | Frame {frame_idx}: glove/person not both found or not associated.')
        if DEBUG:
            print(f'Frame {frame_idx}: {glove_info} | {template_info}')
            cv2.imshow('Debug Glove Tracking', frame)
            key = cv2.waitKey(0)
            if key == 27:  # ESC to quit
                break
        frame_idx += 1
    if DEBUG:
        cv2.destroyAllWindows()
    # Compute glove height at peak leg lift (minimum y across all frames)
    if glove_heights:
        min_y = min(glove_heights)
    else:
        min_y = float('nan')
    # Ensure ball coordinate lists are the same length
    max_len = max(len(xs), len(ys))
    def pad_list(lst, n):
        return lst + [None] * (n - len(lst))
    xs = pad_list(xs, max_len)
    ys = pad_list(ys, max_len)
    # After tracking, filter out x/y <= 150
    for i in range(len(xs)):
        if xs[i] is not None and xs[i] <= 150:
            xs[i] = None
            ys[i] = None
    df = pd.DataFrame({'frame': range(max_len), 'x': xs, 'y': ys})
    df['glove_height_peak_leg_lift'] = min_y
    # Add cap heights to the dataframe (NaN if not found for a frame)
    cap_height_series = pd.Series([None]*len(xs))
    for idx, y in zip(cap_frame_indices, cap_heights):
        if idx < len(cap_height_series):
            cap_height_series[idx] = y
    df['cap_height'] = cap_height_series
    # Ensure glove_to_person_top matches DataFrame length
    if len(glove_to_person_top) < len(xs):
        glove_to_person_top.extend([None] * (len(xs) - len(glove_to_person_top)))
    elif len(glove_to_person_top) > len(xs):
        glove_to_person_top = glove_to_person_top[:len(xs)]
    df['glove_to_person_top'] = glove_to_person_top
    df.to_csv(output_csv, index=False)
    print(f"Saved features to {output_csv}")
    print(f"Saved tracking video to {video_out_path}")
    # User prompt after debug video is created
    if lost_glove_frames:
        print(f"Frames where glove was lost: {lost_glove_frames}")
    # (No longer create a separate glove peak video; glove tracking is now shown in every frame)

def batch_extract(video_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    first_video_processed = False
    for fname in sorted(os.listdir(video_dir)):
        if fname.endswith('.mp4'):
            video_path = os.path.join(video_dir, fname)
            output_csv = os.path.join(output_dir, fname.replace('.mp4', '.csv'))
            # Use yolov8m.pt for higher resolution detection
            extract_ball_features(video_path, output_csv, model_path='yolov8m.pt')
                    

if __name__ == '__main__':
    # Only process dante-pitch-13.mp4 for troubleshooting
    video_path = 'video_files/dante/dante-pitch-13.mp4'
    output_csv = 'data/ball_features/dante-pitch-13.csv'
    extract_ball_features(video_path, output_csv, model_path='yolov8m.pt')