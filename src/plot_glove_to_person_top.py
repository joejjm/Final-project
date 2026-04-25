import pandas as pd
import matplotlib.pyplot as plt

# Load the predictions/features CSV
csv_path = "data/pitch_classifier_predictions.csv"
df = pd.read_csv(csv_path)

# Use the summary feature if available, else vertical_feature
if "glove_to_person_top_summary" in df.columns:
    feature = df["glove_to_person_top_summary"]
else:
    feature = df["vertical_feature"]

videos = df["video"] if "video" in df.columns else df.index.astype(str)
labels = df["actual_label"] if "actual_label" in df.columns else None

plt.figure(figsize=(12, 6))
plt.bar(videos, feature, color="skyblue")
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.ylabel("glove_to_person_top (summary)")
plt.xlabel("Video")
plt.title("glove_to_person_top_summary for Each Video")
if labels is not None:
    for i, (v, val, lab) in enumerate(zip(videos, feature, labels)):
        plt.text(i, val, str(lab), ha='center', va='bottom', fontsize=7, rotation=90)
plt.tight_layout()
plt.show()
