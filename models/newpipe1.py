import cv2
import os
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from keras_facenet import FaceNet  # Face embedding model

# Load YOLOv8 face detection model
model = YOLO("/work/users/a/k/akkineni/Matchtime/MatchTime/models/yolov11n-face.pt")  # Using large model for best accuracy

# Load FaceNet for face embedding extraction
embedder = FaceNet()

# Path to the folder containing known faces
db_path = "/work/users/a/k/akkineni/Matchtime/MatchTime/models/player_images/england_epl_2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley"

# Load known face encodings
def load_known_faces(db_path):
    known_encodings = []
    known_labels = []
    
    for filename in os.listdir(db_path):
        filepath = os.path.join(db_path, filename)
        image = cv2.imread(filepath)
        results = model(image)
        
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face = image[y1:y2, x1:x2]
                
                if face.shape[0] > 0 and face.shape[1] > 0:  # Check if face is valid
                    face = cv2.resize(face, (160, 160))  # Resize for FaceNet
                    face = np.expand_dims(face, axis=0)  # Add batch dimension
                    face_embedding = embedder.embeddings(face)[0]  # Get fixed-size embedding
                    known_encodings.append(face_embedding)
                    known_labels.append(os.path.splitext(filename)[0])
    
    return np.array(known_encodings), known_labels

# Function to extract frames at 1 FPS
def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get the FPS of the video
    frame_interval = fps # Extract 1 frame per second
    frame_count = 0  # Total frame counter
    extracted_count = 0  # Counter for extracted frames
    extracted_images = []
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:  # Extract 1 frame per second
            frame_filename = os.path.join(output_folder, f"frame_{extracted_count}.jpg")
            cv2.imwrite(frame_filename, frame)
            extracted_images.append(frame_filename)
            extracted_count += 1  # Increment only for saved frames
        
        frame_count += 1  # Increment for every frame processed
    
    cap.release()
    return extracted_images

# Function to detect and recognize faces
def recognize_faces(image_paths, known_encodings, known_labels):
    face_counts = defaultdict(int)
    face_score = defaultdict(float)
    
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
                    # Prepare face for embedding
                    resized_face = cv2.resize(face, (160, 160))
                    face_input = np.expand_dims(resized_face, axis=0)
                    face_embedding = embedder.embeddings(face_input)[0]
                    
                    # Compute similarity
                    similarities = cosine_similarity(known_encodings, [face_embedding]).flatten()
                    best_match_idx = np.argmax(similarities)
                    best_match_score = similarities[best_match_idx]
                    
                    if best_match_score > 0.4:
                        label = known_labels[best_match_idx]
                        face_counts[label] += 1
                        face_score[label] += best_match_score
                        
                        # Save cropped face (not resized)
                        face_filename = f"{label}_{os.path.basename(image_path).replace('.jpg', '')}_face{i}.jpg"
                        face_save_path = os.path.join(face_output_folder, face_filename)
                        cv2.imwrite(face_save_path, face)

    return face_counts, face_score


# Main function
def process_video(video_path, output_folder):
    known_encodings, known_labels = load_known_faces(db_path)
    extracted_images = extract_frames(video_path, output_folder)
    face_counts = recognize_faces(extracted_images, known_encodings, known_labels)
    
    return face_counts

# Run the process
video_path = "/work/users/a/k/akkineni/Matchtime/MatchTime/models/1min30sec/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley_1532_1629_substitution.mkv"
output_folder = "/work/users/a/k/akkineni/Matchtime/MatchTime/models/extracted_frames"
face_output_folder = "/work/users/a/k/akkineni/Matchtime/MatchTime/models/face_identified_images"

for file in os.listdir(output_folder):
    os.remove(os.path.join(output_folder, file))
for file in os.listdir(face_output_folder):
    os.remove(os.path.join(face_output_folder, file))
face_counts,face_score = process_video(video_path, output_folder)

    
# Display results
print("Face appearance counts:")
for label, count in face_counts.items():
    print(f"{label}: {count} times")
for label, score in face_score.items():
    print(f"{label}: {score} times")
#exit()

import os
import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict

def cluster_detected_faces(face_output_folder, output_base_folder, embedder):
    print("Clustering identified faces into unique persons...")

    face_paths = sorted([
        os.path.join(face_output_folder, f)
        for f in os.listdir(face_output_folder)
        if f.endswith('.jpg')
    ])

    embeddings = []
    image_ids = []
    original_faces = []

    for face_path in face_paths:
        image = cv2.imread(face_path)
        if image is None:
            continue

        if image.shape[0] > 0 and image.shape[1] > 0:
            # Resize just for embedding (don't save this resized image)
            resized = cv2.resize(image, (160, 160))
            face_input = np.expand_dims(resized, axis=0)
            embedding = embedder.embeddings(face_input)[0]

            embeddings.append(embedding)
            image_ids.append(os.path.basename(face_path))
            original_faces.append(image)

    if not embeddings:
        print("No valid face images found for clustering.")
        return

    embeddings = np.array(embeddings)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.4,
        metric='cosine',
        linkage='average'
    )
    labels = clustering.fit_predict(embeddings)

    print(f"Clustering complete: {len(set(labels))} unique persons found.")

    # Save faces into cluster folders
    for label, face_img, img_name in zip(labels, original_faces, image_ids):
        cluster_dir = os.path.join(output_base_folder, f"cluster{label+1}")
        os.makedirs(cluster_dir, exist_ok=True)
        save_path = os.path.join(cluster_dir, img_name)
        cv2.imwrite(save_path, face_img)

cluster_output_folder = "/work/users/a/k/akkineni/Matchtime/MatchTime/models/uniquefaces"
cluster_detected_faces(face_output_folder, cluster_output_folder, embedder)

