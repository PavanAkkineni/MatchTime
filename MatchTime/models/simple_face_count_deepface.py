import cv2
import os
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from deepface import DeepFace  # Advanced face comparison

# Load YOLOv8 face detection model
model = YOLO("yolov11n-face.pt")  # Using large model for best accuracy

# Paths for storing extracted faces
detected_faces_folder = "/work/users/a/k/akkineni/Matchtime/MatchTime/models/detected_faces"
database_faces_folder = "/work/users/a/k/akkineni/Matchtime/MatchTime/models/database_faces"

# Ensure directories exist
#os.makedirs(detected_faces_folder, exist_ok=True)
#os.makedirs(database_faces_folder, exist_ok=True)

# Path to the folder containing known faces
db_path = "/work/users/a/k/akkineni/Matchtime/MatchTime/models/player_images/england_epl_2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley"

# Function to extract faces from player database and save them as images
def extract_database_faces(db_path, output_folder):
    known_labels = []
    
    for filename in os.listdir(db_path):
        filepath = os.path.join(db_path, filename)
        image = cv2.imread(filepath)
        results = model(image)
        
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face = image[y1:y2, x1:x2]
                
                if face.shape[0] > 0 and face.shape[1] > 0:  # Ensure valid face
                    face_filename = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.jpg")
                    cv2.imwrite(face_filename, face)
                    known_labels.append(face_filename)
    
    return known_labels

# Function to extract frames at 1 FPS
def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get the FPS of the video
    frame_interval = fps  # Extract 1 frame per second
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

# Function to detect faces and save them as images
def detect_and_save_faces(image_paths, output_folder):
    face_images = []
    
    for image_path in image_paths:
        frame = cv2.imread(image_path)
        results = model(frame)
        
        for r in results:
            if len(r.boxes) > 3:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face = frame[y1:y2, x1:x2]
                
                if face.shape[0] > 0 and face.shape[1] > 0:
                    face_filename = os.path.join(output_folder, f"{os.path.basename(image_path)}_face.jpg")
                    cv2.imwrite(face_filename, face)
                    face_images.append(face_filename)
    
    return face_images

# Function to compare detected faces with database faces using DeepFace
def match_faces(detected_faces, database_faces):
    face_counts = defaultdict(int)
    face_scores = defaultdict(float)
    
    for detected_face in detected_faces:
        best_match = None
        best_score = 0
        best_label = "Unknown"
        
        for db_face in database_faces:
            result = DeepFace.verify(detected_face, db_face, model_name='Facenet', enforce_detection=False)
            similarity_score = 1 - result['distance']  # Higher score means better match
            
            if similarity_score > best_score and similarity_score > 0.7:  # Confidence threshold
                best_score = similarity_score
                best_match = db_face
                best_label = os.path.splitext(os.path.basename(db_face))[0]
        
        if best_match:
            face_counts[best_label] += 1
            face_scores[best_label] += best_score
            print(f"Detected {best_label} with confidence: {best_score:.2f}")
    
    return face_counts, face_scores

# Main function
def process_video(video_path, output_folder):
    database_faces = extract_database_faces(db_path, database_faces_folder)
    extracted_images = extract_frames(video_path, output_folder)
    detected_faces = detect_and_save_faces(extracted_images, detected_faces_folder)
    face_counts, face_scores = match_faces(detected_faces, database_faces)
    
    return face_counts, face_scores

# Run the process
video_path = "/work/users/a/k/akkineni/Matchtime/MatchTime/models/test_dataset_130min/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley_2120_2220_soccer-ball.mkv"
output_folder = "/work/users/a/k/akkineni/Matchtime/MatchTime/models/extracted_frames"
face_counts, face_scores = process_video(video_path, output_folder)

# Display results
print("Face appearance counts:")
for label, count in face_counts.items():
    print(f"{label}: {count} times")
for label, score in face_scores.items():
    print(f"{label}: {score:.2f} confidence")
