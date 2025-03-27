import os
from collections import defaultdict

# Define the dataset directory
video_dir = "/work/users/a/k/akkineni/Matchtime/MatchTime/models/dataset"

# Dictionary to store event counts
event_counts = defaultdict(int)

# Iterate over video files in the dataset directory
video_files = [f for f in os.listdir(video_dir) if f.endswith(".mkv")]

for video_file in video_files:
    # Extract event label
    event_label = video_file.split("_")[-1].replace(".mkv", "")
    event_counts[event_label] += 1

# Print the counts for each event
for event, count in event_counts.items():
    print(f"{event}: {count} files")
