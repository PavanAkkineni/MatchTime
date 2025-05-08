import cv2
import os
import re

def convert_game_time_to_seconds(game_time):
    """
    Convert "1 - 13:30" format to total seconds.
    """
    match = re.match(r"(\d+) - (\d+):(\d+)", game_time)
    if match:
        half, minutes, seconds = map(int, match.groups())
        total_seconds =   minutes * 60 + seconds  # Assuming 45-minute halves
        return total_seconds
    return None

def extract_frames(video_path, start_time, end_time, fps, output_folder):
    """
    Extract frames at a given FPS between start_time and end_time.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = video_fps / fps  # How often to capture frames

    start_frame = int(start_time * video_fps)
    end_frame = int(end_time * video_fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_count = start_frame
    while frame_count <= end_frame:
        success, frame = cap.read()
        if not success:
            break  # Stop if video ends
        
        output_path = f"{output_folder}/frame_{frame_count}.jpg"
        cv2.imwrite(output_path, frame)
        print(f"Saved: {output_path}")

        frame_count += int(frame_interval)  # Move forward to next frame

    cap.release()

# Example JSON Data
event_data = {
    "event_aligned_gameTime": "1 - 13:30",  # Start Time
    "contrastive_aligned_gameTime": "1 - 12:55"  # End Time
}

# Convert to seconds
start_seconds = convert_game_time_to_seconds(event_data["contrastive_aligned_gameTime"])
end_seconds = convert_game_time_to_seconds(event_data["event_aligned_gameTime"])

# Video Path
video_path = "/work/users/a/k/akkineni/Matchtime/MatchTime/videos_224p/england_epl_2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/1_224p.mkv"
output_folder = "/work/users/a/k/akkineni/Matchtime/MatchTime/getframes"

# Extract frames at 2 frames per second
extract_frames(video_path, start_seconds, end_seconds, fps=1, output_folder=output_folder)
