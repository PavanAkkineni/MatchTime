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

# Directory containing video files
VIDEO_DIR = "/work/users/a/k/akkineni/Matchtime/MatchTime/models/dataset"
MODEL_SAVE_PATH = "/work/users/a/k/akkineni/Matchtime/MatchTime/models/mlp_models"

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

def extract_frames(video_path, label):
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
            frame = transform(frame)
            frames.append((frame, label))
        count += 1
    
    cap.release()
    return frames

# Prepare dataset
class VideoDataset(Dataset):
    def __init__(self, video_dir):
        self.data = []
        for file in tqdm(os.listdir(video_dir)):
            if file.endswith(".mkv"):
                label_str = file.split("_")[-1].replace(".mkv", "")
                if label_str in LABEL_MAPPING:
                    label = LABEL_MAPPING[label_str]
                    video_path = os.path.join(video_dir, file)
                    self.data.extend(extract_frames(video_path, label))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        frame, label = self.data[idx]
        return frame, torch.tensor(label, dtype=torch.long)

# Load dataset
dataset = VideoDataset(VIDEO_DIR)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

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

# Model, loss, optimizer
mlp_model = MLPClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)

def train_model(model, dataloader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for frames, labels in tqdm(dataloader):
            with torch.no_grad():
                features = resnet(frames).squeeze()
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
        
        # Save model after each epoch
        torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, f"mlp_epoch_{epoch+1}.pth"))

# Train the MLP model
train_model(mlp_model, dataloader, criterion, optimizer)
