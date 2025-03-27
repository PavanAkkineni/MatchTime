import os
import json
import re
import cv2

def convert_game_time_to_seconds(game_time):
    """
    Convert "2 - 47:01" format to total seconds.
    """
    match = re.match(r"(\d+) - (\d+):(\d+)", game_time)
    if match:
        half, minutes, seconds = map(int, match.groups())
        total_seconds = minutes * 60 + seconds  # Assuming 45-minute halves
        return total_seconds, half
    return None, None
    
def extract_valid_annotations(folder_path):
    """
    Extract annotations where all time fields are valid and within 10 seconds of each other.
    "no event" is only taken if the previous 3 and next 3 annotations also have "no event" or "".
    """
    valid_annotations = []
    label_counts = {"substitution": 0, "y-card": 0, "soccer-ball": 0}
    max_per_label = 60

    for game_folder in os.listdir(folder_path):
        if game_folder in ["2015-02-21 - 18-00 Chelsea 1 - 1 Burnley","2015-02-21 - 18-00 Swansea 2 - 1 Manchester United"]:
            
            game_path = os.path.join(folder_path, game_folder)
            json_file = os.path.join(game_path, "Labels-caption.json")
    
            if os.path.exists(json_file):
                with open(json_file, "r") as file:
                    data = json.load(file)
                    annotations = data.get("annotations", [])
    
                    for i, annotation in enumerate(annotations):
                        game_time = annotation.get("gameTime", "")
                        start_time = annotation.get("contrastive_aligned_gameTime", "")
                        end_time = annotation.get("event_aligned_gameTime", "")
                        label = annotation.get("label", "")
                        
                        if label == "":
                            label = "no event"
    
                        # Skip "no event" if surrounding annotations do not meet the requirement
                        if label == "no event":
                            prev_labels = [annotations[j].get("label", "") for j in range(max(0, i-3), i)]
                            next_labels = [annotations[j].get("label", "") for j in range(i+1, min(len(annotations), i+4))]
    
                            # Ensure surrounding 3 labels (both before and after) are either "no event" or ""
                            if not all(lbl == "no event" or lbl == "" for lbl in prev_labels + next_labels):
                                continue  # Skip this annotation
    
                        # Process valid annotations
                        if game_time and start_time and end_time and label in label_counts:
                            game_seconds, half = convert_game_time_to_seconds(game_time)
                            start_seconds, _ = convert_game_time_to_seconds(start_time)
                            end_seconds, _ = convert_game_time_to_seconds(end_time)
    
                            if None not in (game_seconds, start_seconds, end_seconds):
                                avg_time = (game_seconds + start_seconds + end_seconds) // 3
                                start_seconds = max(0, avg_time - 12)  # Ensure non-negative start time
                                end_seconds = avg_time + 12
    
                                if label_counts[label] < max_per_label:
                                    valid_annotations.append({
                                        "game": game_folder,
                                        "start_time": start_time,
                                        "start_seconds": start_seconds,
                                        "end_time": end_time,
                                        "end_seconds": end_seconds,
                                        "label": label,
                                        "description": annotation.get("description", ""),
                                        "half": half
                                    })
                                    label_counts[label] += 1

    return valid_annotations


def extract_video_clip(video_path, start_seconds, end_seconds, output_path):
    """
    Extract and save video clip.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'VP90')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(start_seconds * fps)
    end_frame = int(end_seconds * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = start_frame
    while frame_count <= end_frame:
        success, frame = cap.read()
        if not success:
            break
        out.write(frame)
        frame_count += 1
    
    cap.release()
    out.release()
    print(f"Extracted video saved at: {output_path}")

# Paths
dataset_path = "/work/users/a/k/akkineni/Matchtime/MatchTime/dataset/MatchTime/train/england_epl_2014-2015"
videos_path = "/work/users/a/k/akkineni/Matchtime/MatchTime/videos_224p/england_epl_2014-2015"
output_folder_25sec = "/work/users/a/k/akkineni/Matchtime/MatchTime/models/25sec"
output_folder_1min30sec = "/work/users/a/k/akkineni/Matchtime/MatchTime/models/720p/1min30sec"


valid_annotations = extract_valid_annotations(dataset_path)
print(valid_annotations)

# Save extracted video clips and annotations
annotations_dict = {}
for annotation in valid_annotations:
    print(annotations_dict)
    game_folder = annotation["game"]
    start_seconds = annotation["start_seconds"]
    end_seconds = annotation["end_seconds"]
    start1 = max(1, start_seconds-14)
    end1 = end_seconds+55
    label = annotation["label"]
    half = annotation["half"]
    gt = annotation["description"]
    
    video_file = f"{half}_224p.mkv"
    video_game_folder = os.path.join(videos_path, game_folder)
    video_path = os.path.join(video_game_folder, video_file)
    print(video_path)
    if os.path.exists(video_path):
        output_filename = f"{game_folder}_{start_seconds}_{end_seconds}_{label}.mkv"
        output_path = os.path.join(output_folder_25sec, output_filename)
        extract_video_clip(video_path, start_seconds, end_seconds, output_path)
        #output_filename1 = f"{game_folder}_{start1}_{end1}_{label}.mkv"
        #output_path1 = os.path.join(output_folder_1min30sec, output_filename1)
        #extract_video_clip(video_path, start_seconds, end_seconds, output_path)
        #extract_video_clip(video_path, start1, end1, output_path1)
        
        annotations_dict[output_filename] = gt
    
# Save annotations as JSON
annotations_json_path = os.path.join(output_folder_25sec, "annotations.json")
with open(annotations_json_path, "w") as json_file:
    json.dump(annotations_dict, json_file, indent=4)

print(f"Annotations saved at: {annotations_json_path}")
