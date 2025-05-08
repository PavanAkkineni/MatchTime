import cv2
import os
import numpy as np
from deepface import DeepFace
from retinaface import RetinaFace
from collections import defaultdict

# Paths
database_path = "/work/users/a/k/akkineni/Matchtime/MatchTime/models/player_images/england_epl_2014-2015/2015-02-21 - 18-00 Swansea 2 - 1 Manchester United"
realtime_images_path = "/work/users/a/k/akkineni/Matchtime/MatchTime/models/extracted_frames"
output_path = "/work/users/a/k/akkineni/Matchtime/MatchTime/models/face_atted_faces"
for file in os.listdir(output_path):
    os.remove(os.path.join(output_path, file))
    
# Track face identification counts and confidence scores
face_identification_count = defaultdict(int)
face_confidence_scores = defaultdict(list)

# Load database images
database_faces = {}
for file in os.listdir(database_path):
    if file.endswith(".jfif") or file.endswith(".jpg") or file.endswith(".png"):
        label = os.path.splitext(file)[0]  # Extract label from filename
        img_path = os.path.join(database_path, file)
        try:
            face_embedding = DeepFace.represent(img_path, model_name='ArcFace', enforce_detection=False)
            if face_embedding:
                database_faces[label] = face_embedding[0]['embedding']
        except Exception as e:
            print(f"Error processing {file}: {e}")

# Process real-time images
for image_file in os.listdir(realtime_images_path):
    image_path = os.path.join(realtime_images_path, image_file)
    image = cv2.imread(image_path)
    
    # Detect faces
    faces = RetinaFace.detect_faces(image_path)
    
    if faces:  # Only save images if faces are detected
        for key in faces.keys():
            face_info = faces[key]
            x1, y1, x2, y2 = face_info['facial_area']
            face_crop = image[y1:y2, x1:x2]
              # Confidence score of detection
            
            # Generate embedding for detected face
            try:
                face_embedding = DeepFace.represent(face_crop, model_name='ArcFace', enforce_detection=False)
                best_match = None
                best_distance = float('inf')
                
                # Compare with database
                for label, db_embedding in database_faces.items():
                    distance = np.linalg.norm(np.array(db_embedding) - np.array(face_embedding[0]['embedding']))
                    if distance < best_distance:
                        best_distance = distance
                        best_match = label
                        
                confidence = max(0, 1 - (best_distance / 1.2))
                # Draw bounding box and label
                #print(best_distance)
                face_identification_count[best_match] += 1
                face_confidence_scores[best_match].append(distance)
                if best_match and best_distance < 2:  # Threshold
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label_text = f"{best_match} ({best_distance:.2f})"
                    cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # Track identification count and confidence scores
                    
                    output_image_path = os.path.join(output_path, image_file)
                    cv2.imwrite(output_image_path, image)
                    print(f"Processed and saved: {output_image_path}")
            
            except Exception as e:
                print(f"Error processing face in {image_file}: {e}")
        
        # Save output image only if faces are detected
        #output_image_path = os.path.join(output_path, image_file)
        #cv2.imwrite(output_image_path, image)
        #print(f"Processed and saved: {output_image_path}")

# Print identification summary
# Print identification summary
print(face_identification_count)
print("\nFace Identification Summary:")
if face_identification_count:
    for face, count in face_identification_count.items():
        avg_confidence = sum(face_confidence_scores[face])//len(face_confidence_scores[face])
        print(f"{face}: Identified {count} times with an average confidence score of {avg_confidence:.2f}")
else:
    print("No faces were identified in the images.")

print("Face recognition task completed!")