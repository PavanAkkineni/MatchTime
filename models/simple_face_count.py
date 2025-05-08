import cv2
import os
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from keras_facenet import FaceNet  # Face embedding model

# Load YOLOv8 face detection model
model = YOLO("yolov11n-face.pt")  # Using large model for best accuracy

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
        face_detected = False
        for r in results:
            if len(r.boxes)>4:
                continue
            for box in r.boxes:
            
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face = frame[y1:y2, x1:x2]
                
                if face.shape[0] > 0 and face.shape[1] > 0:
                    face = cv2.resize(face, (160, 160))  # Resize for FaceNet
                    face = np.expand_dims(face, axis=0)
                    face_embedding = embedder.embeddings(face)[0]  # Get fixed-size embedding
                    
                    # Compute cosine similarity
                    similarities = cosine_similarity(known_encodings, [face_embedding]).flatten()
                    best_match_idx = np.argmax(similarities)
                    best_match_score = similarities[best_match_idx]
                    
                    if best_match_score > 0.4:  # Adjusted threshold for accuracy
                        label = known_labels[best_match_idx]
                        face_counts[label] += 1
                        face_score[label] +=best_match_score
                        face_detected = True
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save annotated image
        #output_image_path = image_path.replace(".jpg", "_yolo_annotated.jpg")
        #cv2.imwrite(output_image_path, frame)
        if face_detected:
            output_image_path = os.path.join(face_output_folder, os.path.basename(image_path).replace(".jpg", "_face_identified.jpg"))
            cv2.imwrite(output_image_path, frame)
    
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
