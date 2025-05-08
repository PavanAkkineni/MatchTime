import os
import cv2
import shutil
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from ultralytics import YOLO
from keras_facenet import FaceNet

# Load models
model = YOLO("/work/users/a/k/akkineni/Matchtime/MatchTime/models/yolov11n-face.pt")
embedder = FaceNet()

# Paths
player_db_root = "/work/users/a/k/akkineni/Matchtime/MatchTime/models/player_images/england_epl_2015-2016"
video_folder = "/work/users/a/k/akkineni/Matchtime/MatchTime/models/1min30sec_new56"
extracted_base = "/work/users/a/k/akkineni/Matchtime/MatchTime/models/extracted_frames"
faces_base = "/work/users/a/k/akkineni/Matchtime/MatchTime/models/face_identified_images"
faces_datected = "/work/users/a/k/akkineni/Matchtime/MatchTime/models/detected_faces"
cluster_base = "/work/users/a/k/akkineni/Matchtime/MatchTime/models/uniquefaces_new"
csv_output_path = "/work/users/a/k/akkineni/Matchtime/MatchTime/models/top_players_per_video04.csv"

def safe_clear_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def load_known_faces(db_path):
    known_encodings, known_labels = [], []
    for filename in os.listdir(db_path):
        filepath = os.path.join(db_path, filename)
        image = cv2.imread(filepath)
        if image is None: continue
        results = model(image)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face = image[y1:y2, x1:x2]
                if face.shape[0] > 0 and face.shape[1] > 0:
                    resized = cv2.resize(face, (160, 160))
                    embedding = embedder.embeddings(np.expand_dims(resized, axis=0))[0]
                    known_encodings.append(embedding)
                    known_labels.append(os.path.splitext(filename)[0])
    return np.array(known_encodings), known_labels

def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = fps
    count = 0
    extracted_paths = []
    os.makedirs(output_folder, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame_path = os.path.join(output_folder, f"frame_{count}.jpg")
            cv2.imwrite(frame_path, frame)
            extracted_paths.append(frame_path)
        count += 1
    cap.release()
    return extracted_paths

def recognize_faces(image_paths, known_encodings, known_labels, save_folder, detect_ouput):
    face_counts = defaultdict(int)
    face_score = defaultdict(float)
    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(detect_ouput, exist_ok=True)

    for image_path in image_paths:
        frame = cv2.imread(image_path)
        results = model(frame)

        for r in results:
            if len(r.boxes) > 4:
                continue
            for i, box in enumerate(r.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face = frame[y1:y2, x1:x2]
                if face.shape[0] > 0 and face.shape[1] > 0:
                    resized = cv2.resize(face, (160, 160))
                    embedding = embedder.embeddings(np.expand_dims(resized, axis=0))[0]
                    similarities = cosine_similarity(known_encodings, [embedding]).flatten()
                    best_idx = np.argmax(similarities)
                    score = similarities[best_idx]
                    face_filename = f"{os.path.basename(image_path).replace('.jpg', '')}_face{i}.jpg"
                    if score > 0.4:
                        if score > 0.4:
                        
                            label = known_labels[best_idx]
                            if label in ['Marriner A.','Taylor A.','East R','Moss J','Clattenburg M.','Madley R.','Jones M.','Atkinson M.','Pawson C.','Oliver M.','Dyche S','Mourinho J.','Atkinson M.','Pardew A.','Wenger A.','Clattenburg M.','monk g','swarbrick','van gaal','Koeman R.','Rodgers B.','Friend K.','Dean M','Bilic S.']:
                                continue
                            face_counts[label] += 1
                            face_score[label] += score
                            face_filename = f"{label}_{os.path.basename(image_path).replace('.jpg', '')}_face{i}.jpg"
                            cv2.imwrite(os.path.join(save_folder, face_filename), face)
                        cv2.imwrite(os.path.join(detect_ouput, face_filename), face)

    return face_counts, face_score

def cluster_faces(input_folder, output_folder):
    face_paths = sorted([
        os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.jpg')
    ])
    embeddings, originals, ids = [], [], []

    for path in face_paths:
        image = cv2.imread(path)
        if image is None or image.shape[0] == 0 or image.shape[1] == 0:
            continue
        resized = cv2.resize(image, (160, 160))
        embedding = embedder.embeddings(np.expand_dims(resized, axis=0))[0]
        embeddings.append(embedding)
        originals.append(image)
        ids.append(os.path.basename(path))

    if not embeddings:
        print(f"No valid faces in {input_folder}")
        return

    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.4, metric='cosine', linkage='average')
    labels = clustering.fit_predict(np.array(embeddings))

    for label, img, name in zip(labels, originals, ids):
        cluster_dir = os.path.join(output_folder, f"cluster{label+1}")
        os.makedirs(cluster_dir, exist_ok=True)
        cv2.imwrite(os.path.join(cluster_dir, name), img)

    print(f"Clustered into {len(set(labels))} groups: {output_folder}")

import pandas.errors 

def update_csv_live(video_name, top_players, face_scores, csv_output_path):
    rows_to_add = []
    for player, count in top_players:
        rows_to_add.append({
            "Video Name": video_name,
            "Player Name": player,
            "Appearances": count,
            "Confidence Score": round(face_scores[player], 2)
        })

    try:
        existing_df = pd.read_csv(csv_output_path)
        updated_df = pd.concat([existing_df, pd.DataFrame(rows_to_add)], ignore_index=True)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        # Start a new file if it doesn't exist or is empty
        updated_df = pd.DataFrame(rows_to_add)

    updated_df.to_csv(csv_output_path, index=False)


def process_all_videos(filename):

    video_path = filename
    video_name = os.path.basename(video_path)
    print(f"\n▶️ Processing video: {video_name}")

    match_name = video_name.split("_")[0]
    print("match_name",match_name)
    player_db_path = os.path.join(player_db_root, match_name)
    print("player_db_path",player_db_path)
    
    if not os.path.exists(player_db_path):
        print(f"⚠️ Skipping {video_name} (No player DB found for: {match_name})")
        return

    print(f"🧠 Loading known players from: {player_db_path}")
    
    known_encodings, known_labels = load_known_faces(player_db_path)
    
    frame_output = os.path.join(extracted_base, video_name)
    face_output = os.path.join(faces_base, video_name)
    detect_ouput = os.path.join(faces_datected, video_name)
    cluster_output = os.path.join(cluster_base, video_name)

    safe_clear_folder(frame_output)
    safe_clear_folder(face_output)
    safe_clear_folder(cluster_output)
    safe_clear_folder(detect_ouput)

    extracted_images = extract_frames(video_path, frame_output)
    face_counts, face_scores = recognize_faces(extracted_images, known_encodings, known_labels, face_output, detect_ouput)
    cluster_faces(detect_ouput, cluster_output)

    top_players = sorted(face_counts.items(), key=lambda x: x[1], reverse=True)[:2]
    update_csv_live(video_name, top_players, face_scores, csv_output_path)

    print(f"✅ Done with {video_name}")

if __name__ == "__main__":
    process_all_videos()
