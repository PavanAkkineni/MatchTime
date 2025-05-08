import os
import cv2
import json
import csv
import torch
from ultralytics import YOLO
from collections import Counter
from PIL import Image, ImageDraw, ImageFont
import easyocr
import numpy as np

# Paths
video_folder = "/work/users/a/k/akkineni/Matchtime/MatchTime/models/1min30sec_new"
label_json = "/work/users/a/k/akkineni/Matchtime/MatchTime/dataset/MatchTime/train/england_epl_2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/Labels-caption.json"
output_csv = os.path.join(video_folder, "ocr_player_summary_easyocr.csv")
ocr_output_dir = os.path.join(video_folder, "ocr")
os.makedirs(ocr_output_dir, exist_ok=True)

# Load player info
with open(label_json, 'r') as f:
    label_data = json.load(f)

player_mapping = {}
for side in ['home', 'away']:
    for player in label_data['lineup'][side]['players']:
        shirt = str(player.get("shirt_number", "")).strip()
        name = player.get("short_name", "").strip()
        if shirt:
            player_mapping[shirt] = name

# Initialize models
yolo_model = YOLO("yolov8n.pt")
ocr_reader = easyocr.Reader(['en'], gpu=False)

# Process first 11 videos
video_files = sorted([f for f in os.listdir(video_folder) if f.endswith(".mkv")])[:11]
fieldnames = ["video_file", "top_names", "top_jersey_numbers", "players_mapped_from_jerseys"]

for idx, video_file in enumerate(video_files):
    video_path = os.path.join(video_folder, video_file)
    print(f"🔍 Processing: {video_file}")
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count // fps

    detected_texts = []
    video_ocr_dir = os.path.join(ocr_output_dir, os.path.splitext(video_file)[0])
    os.makedirs(video_ocr_dir, exist_ok=True)

    for sec in range(duration):
        cap.set(cv2.CAP_PROP_POS_FRAMES, sec * fps)
        ret, frame = cap.read()
        if not ret:
            continue

        results_yolo = yolo_model(frame)
        for result_id, result in enumerate(results_yolo):
            boxes = result.boxes.xyxy
            for box_id, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box[:4])
                person_crop = frame[y1:y2, x1:x2]
                if person_crop.size == 0:
                    continue

                ocr_result = ocr_reader.readtext(person_crop)
                annotated_crop = person_crop.copy()

                for detection in ocr_result:
                    text = detection[1]
                    if text:
                        cleaned_text = text.strip()
                        if cleaned_text.isdigit() and len(cleaned_text) <= 3:
                            detected_texts.append(cleaned_text)
                        elif cleaned_text in player_mapping.values():
                            detected_texts.append(cleaned_text)

                        # Draw text on the crop
                        try:
                            pil_img = Image.fromarray(cv2.cvtColor(annotated_crop, cv2.COLOR_BGR2RGB))
                            draw = ImageDraw.Draw(pil_img)
                            font = ImageFont.load_default()
                            draw.text((5, 5), cleaned_text, fill=(255, 0, 0), font=font)
                            annotated_crop = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                        except Exception as e:
                            print(f"Could not annotate image: {e}")

                # Save cropped image
                save_path = os.path.join(video_ocr_dir, f"{sec:03d}_{result_id}_{box_id}.jpg")
                cv2.imwrite(save_path, annotated_crop)

    cap.release()

    jersey_counter = Counter([t for t in detected_texts if t.isdigit()])
    name_counter = Counter([t for t in detected_texts if not t.isdigit()])

    top_jerseys = [j[0] for j in jersey_counter.most_common(2)]
    top_names = [n[0] for n in name_counter.most_common(2)]
    top_players_from_jerseys = [player_mapping.get(j, "Unknown") for j in top_jerseys]

    row = {
        "video_file": video_file,
        "top_names": ", ".join(top_names) if top_names else "None",
        "top_jersey_numbers": ", ".join(top_jerseys) if top_jerseys else "None",
        "players_mapped_from_jerseys": ", ".join(top_players_from_jerseys) if top_players_from_jerseys else "None"
    }

    # Write to CSV after each video
    write_mode = 'w' if idx == 0 else 'a'
    with open(output_csv, write_mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_mode == 'w':
            writer.writeheader()
        writer.writerow(row)

print(f"\n✅ All done. Results saved to CSV and images saved to: {ocr_output_dir}")
