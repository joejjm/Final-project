import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
# main.py
# Entry point for video frame extraction and change characterization




from frame_extractor import extract_frames
from frame_change_characterizer import characterize_changes
import os

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

# For LLM image analysis
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

def describe_image(frame_filename):
    # Use the public GitHub raw URL for the frame image
    github_user = "joejjm"
    github_repo = "Final-project"
    github_branch = "main"
    github_path = f"data/frames/{frame_filename}"
    img_url = f"https://raw.githubusercontent.com/{github_user}/{github_repo}/{github_branch}/{github_path}"
    llm = ChatOpenAI(model="gpt-4o")
    message = HumanMessage(
        content=[
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": img_url}}
        ]
    )
    msg = llm.invoke([message])
    print(f"Description for {img_url}:")
    print(msg.content)

if __name__ == "__main__":
    video_path = "video_files/John-pitch-1.mp4"
    frames_dir = "data/frames"
    os.makedirs(frames_dir, exist_ok=True)
    print(f"Extracting frames from {video_path} to {frames_dir}...")
    extract_frames(video_path, frames_dir)

    print("Characterizing changes between frames...")
    changes = characterize_changes(frames_dir)
    for fname, score in changes:
        print(f"{fname}: {score} pixels changed")

    # Generate and plot change signal over time
    frame_indices = list(range(1, len(changes) + 1))
    change_signal = [score for _, score in changes]
    plt.figure(figsize=(12, 6))
    plt.plot(frame_indices, change_signal, marker='o')
    plt.title('Change Signal Over Time (Pixels Changed per Frame)')
    plt.xlabel('Frame Index')
    plt.ylabel('Number of Pixels Changed')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('data/change_signal_plot.png')
    plt.show()

    # Load YOLOv8 model (pretrained on COCO)
    print("Loading YOLOv8 model...")
    model = YOLO('yolov8n.pt')


    # Track the ball (class 'sports ball' in COCO, class id 32) and save annotated frames
    ball_positions = []
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    annotated_dir = os.path.join('data', 'annotated_frames')
    os.makedirs(annotated_dir, exist_ok=True)
    for idx, fname in enumerate(frame_files):
        frame_path = os.path.join(frames_dir, fname)
        img = cv2.imread(frame_path)
        results = model(frame_path)
        found_ball = False
        for box in results[0].boxes:
            if int(box.cls[0]) == 32:  # 32 is 'sports ball' in COCO
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                ball_positions.append((idx + 1, cx, cy))
                # Draw bounding box and center
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
                found_ball = True
                break
        if not found_ball:
            ball_positions.append((idx + 1, None, None))
        # Save annotated frame
        out_path = os.path.join(annotated_dir, fname)
        cv2.imwrite(out_path, img)

    # Create video from annotated frames
    video_out_path = os.path.join('data', 'ball_tracking_output.mp4')
    frame_example = cv2.imread(os.path.join(annotated_dir, frame_files[0]))
    height, width, _ = frame_example.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30  # Adjust if needed
    out = cv2.VideoWriter(video_out_path, fourcc, fps, (width, height))
    for fname in frame_files:
        img = cv2.imread(os.path.join(annotated_dir, fname))
        out.write(img)
    out.release()
    print(f"Annotated video saved to {video_out_path}")

    # Plot ball position over time
    xs = [p[1] for p in ball_positions]
    ys = [p[2] for p in ball_positions]
    frame_indices = [p[0] for p in ball_positions]
    plt.figure(figsize=(12, 6))
    plt.plot(frame_indices, xs, label='Ball X Position')
    plt.plot(frame_indices, ys, label='Ball Y Position')
    plt.title('Ball Position Over Time (YOLOv8)')
    plt.xlabel('Frame Index')
    plt.ylabel('Pixel Position')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('data/ball_position_plot.png')
    plt.show()

    # Optionally, keep the LLM description code or comment it out
    # if len(changes) >= 2:
    #     sorted_changes = sorted(changes, key=lambda x: x[1], reverse=True)
    #     top_frames = [sorted_changes[0][0], sorted_changes[1][0]]
    #     for frame in top_frames:
    #         describe_image(frame)
