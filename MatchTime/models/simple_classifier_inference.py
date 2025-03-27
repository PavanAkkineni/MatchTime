import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from tqdm import tqdm

# Define label mapping
LABEL_MAPPING = {
    "corner": 0,
    "soccer-ball": 1,
    "injury": 2,
    "no event": 3
}
REVERSE_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}

# Directory containing video files
VIDEO_DIR = "/work/users/a/k/akkineni/Matchtime/MatchTime/models/dataset"
MODEL_SAVE_PATH = "/work/users/a/k/akkineni/Matchtime/MatchTime/models/mlp_models"
MODEL_FILE = os.path.join(MODEL_SAVE_PATH, "mlp_epoch_3.pth")  # Assuming latest epoch

# Ensure save directory exists
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Transform for frames
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Pre-trained feature extractor (ResNet18)
resnet = models.resnet18(pretrained=True)
resnet = nn.Sequential(*list(resnet.children())[:-1])  # Remove classification layer
resnet.eval()  # Set to evaluation mode

def extract_frame_for_inference(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))  # Get FPS
    count = 0
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_rate == 0:  # Take 1 frame per second
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)  # Convert numpy array to PIL image
            frame = transform(frame).unsqueeze(0)  # Add batch dimension
            frames.append(frame)
        count += 1
    
    cap.release()
    return frames

# Define MLP Classifier
class MLPClassifier(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_classes=4):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x  # No softmax, since CrossEntropyLoss expects raw logits

# Load trained model
def load_model():
    model = MLPClassifier()
    model.load_state_dict(torch.load(MODEL_FILE))
    model.eval()
    return model

def infer_video(video_path):
    model = load_model()
    frames = extract_frame_for_inference(video_path)
    
    if not frames:
        print("No frames extracted from the video.")
        return None
    
    with torch.no_grad():
        predictions = []
        for frame in frames:
            features = resnet(frame).squeeze()
            output = model(features.unsqueeze(0))  # Add batch dimension
            predicted_label = torch.argmax(output, dim=1).item()
            predictions.append(predicted_label)
    
    final_label = max(set(predictions), key=predictions.count)  # Majority vote
    return REVERSE_LABEL_MAPPING[final_label]

# Example usage:
result = infer_video("/work/users/a/k/akkineni/Matchtime/MatchTime/models/test_dataset/2015-02-21 - 18-00 Swansea 2 - 1 Manchester United_1621_1718_soccer-ball.mkv")
print("Predicted label:", result)
