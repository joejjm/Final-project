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

def describe_image(img_path):
    # This function assumes img_path is a local file path
    # For OpenAI Vision, you need a public URL. You can use a placeholder or upload to a service if needed.
    # Here, we just show the code structure.
    img_url = f"file://{os.path.abspath(img_path)}"  # Replace with actual public URL if needed
    llm = ChatOpenAI(model="gpt-4o")
    message = HumanMessage(
        content=[
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": img_url}}
        ]
    )
    msg = llm.invoke([message])
    print(f"Description for {img_path}:")
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

    # Option 3: Send two frames to LLM for description
    # Pick two frames with the largest change
    if len(changes) >= 2:
        sorted_changes = sorted(changes, key=lambda x: x[1], reverse=True)
        top_frames = [sorted_changes[0][0], sorted_changes[1][0]]
        for frame in top_frames:
            frame_path = os.path.join(frames_dir, frame)
            describe_image(frame_path)
