from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="XmDbBFGVZJDAUZ3ENTZj"
)


import cv2
import json

# Load image
image_path = "/work/users/a/k/akkineni/Matchtime/MatchTime/getframes/frame_20250.jpg"
output_path = "/work/users/a/k/akkineni/Matchtime/MatchTime/annotated_frame_20250.jpg"

image = cv2.imread(image_path)

# Sample detection result from inference
result  = CLIENT.infer("/work/users/a/k/akkineni/Matchtime/MatchTime/getframes/frame_20250.jpg", model_id="soccer-players-ckbru/15")

# Define colors for different classes
class_colors = {
    "Ref": (255, 0, 0),       # Red for referees
    "Player": (0, 255, 0),    # Green for players
    "Ball": (0, 0, 255),      # Blue for ball
}

# Draw bounding boxes
for obj in result["predictions"]:
    x, y, w, h = int(obj["x"]), int(obj["y"]), int(obj["width"]), int(obj["height"])
    class_name = obj["class"]
    color = class_colors.get(class_name, (255, 255, 255))  # Default white if class not found

    # Convert from center-based to rectangle coordinates
    x1, y1 = int(x - w / 2), int(y - h / 2)
    x2, y2 = int(x + w / 2), int(y + h / 2)

    # Draw bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    # Add label
    label = f"{class_name} ({obj['confidence']:.2f})"
    cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Save the annotated image
cv2.imwrite(output_path, image)

print(f"Annotated image saved at: {output_path}")

