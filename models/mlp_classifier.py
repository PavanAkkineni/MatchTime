import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from matchvoice_model_classfier import RestrictTokenGenerationLogitsProcessor, matchvoice_model, LayerNorm
import numpy as np
from torchvision import transforms
import clip
from PIL import Image
import torch, os, cv2, argparse


# Define the MLP Classifier
class MLPClassifier(nn.Module):
    def __init__(self, input_dim=512, num_classes=4):
        super(MLPClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        logits = self.classifier(x)  # Get raw scores
        labels = torch.argmax(logits, dim=1)  # Get the index of the highest score
        return labels



class VideoDataset(Dataset):
    def __init__(self, video_path, size=224, fps=2):
        self.video_path = video_path
        self.size = size
        self.fps = fps
        self.transforms = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])
        # Load video using OpenCV
        self.cap = cv2.VideoCapture(self.video_path)
        self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Calculate frames to capture based on FPS
        self.frame_indices = [int(x * self.cap.get(cv2.CAP_PROP_FPS) / self.fps) for x in range(int(self.length / self.cap.get(cv2.CAP_PROP_FPS) * self.fps))]

    def __len__(self):
        return len(self.frame_indices)

    def __getitem__(self, idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_indices[idx])
        ret, frame = self.cap.read()
        if not ret:
            print("Error in reading frame")
            return None
        # Convert color from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Apply transformations
        frame = self.transforms(Image.fromarray(frame))
        return frame.to(torch.float16)

    def close(self):
        self.cap.release()

def encode_features(data_loader, encoder, device):
    all_features = None  # 初始化为None，用于第一次赋值
    for frames in data_loader:
        features = encoder(frames.to(device))
        if all_features is None:
            all_features = features  # 第一次迭代，直接赋值
        else:
            all_features = torch.cat((all_features, features), dim=0)  # 后续迭代，在第0维（行）上连接
    return all_features

def predict_single_video_CLIP(video_path, predict_model, visual_encoder, size, fps, device):
    # Loading features
    try:
        dataset = VideoDataset(video_path, size=size, fps=fps)
        data_loader = DataLoader(dataset, batch_size=40, shuffle=False, pin_memory=True, num_workers=0)
        # print("Start encoding!")
        features = encode_features(data_loader, visual_encoder, device)
        dataset.close()
        print("Features of this video loaded with shape of:", features.shape)
    except:
        print("Error with loading:", video_path)

    
    return features




class VideoEmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

# Training function
def train_classifier(model, dataloader, criterion, optimizer, device, save_path, epochs=10):
    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for embeddings, labels in dataloader:
            embeddings, labels = embeddings.to(device), labels.to(device)

            optimizer.zero_grad()
            embeddings = embeddings.mean(dim=0, keepdim=True).to(device)
            embeddings = embeddings.to(torch.float32).to(device)
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(dataloader)}")

        # Save the model after each epoch
        model_save_path = os.path.join(save_path, f"mlp_classifier_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved at: {model_save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process video files for feature extraction.')
    parser.add_argument('--video_path', type=str, default="./examples/eng.mkv", help='Path to the soccer game video clip.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Device to extract.')
    parser.add_argument('--size', type=int, default=224, help='Size to which each video frame is resized.')
    parser.add_argument('--fps', type=int, default=2, help='Frames per second to sample from the video.')
    parser.add_argument("--tokenizer_name", type=str, default="meta-llama/Meta-Llama-3-8B", help="LLM checkpoints, use path in your computer is fine as well")
    parser.add_argument("--model_ckpt", type=str, default="/work/users/a/k/akkineni/Matchtime/MatchTime/ckpt/CLIP_matchvoice.pth")
    parser.add_argument("--num_query_tokens", type=int, default=32)
    parser.add_argument("--num_video_query_token", type=int, default=32)
    parser.add_argument("--num_features", type=int, default=512)

    args = parser.parse_args()

    # 创建并配置模型
    model, preprocess = clip.load("ViT-B/32", device="cuda:0")
    model.eval()
    # print(model.dtype)
    clip_image_encoder = model.encode_image
    predict_model = matchvoice_model(llm_ckpt=args.tokenizer_name,tokenizer_ckpt=args.tokenizer_name,num_video_query_token=args.num_video_query_token, num_features=args.num_features, device=args.device, inference=True)
    # Load checkpoints
    other_parts_state_dict = torch.load(args.model_ckpt)
    new_model_state_dict = predict_model.state_dict()
    for key, value in other_parts_state_dict.items():
        if key in new_model_state_dict:
            new_model_state_dict[key] = value
    predict_model.load_state_dict(new_model_state_dict)
    predict_model.eval()
    
    video_dir = "/work/users/a/k/akkineni/Matchtime/MatchTime/models/dataset"
    
    # Mapping labels to integers
    LABEL_MAPPING = {
        "corner": 0,
        "soccer-ball": 1,
        "injury": 2,
        "no event": 3
    }
    
    # Function to extract label from filename
    def extract_label(filename):
        label = filename.split("_")[-1].replace(".mkv", "")
        return LABEL_MAPPING.get(label, None)
    
    # Initialize lists to store embeddings and labels
    all_video_embeds = []
    all_labels = []
    
    # Iterate over video files in the dataset directory
    video_files = [f for f in os.listdir(video_dir) if f.endswith(".mkv")]
    
    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        label = extract_label(video_file)
    
        if label is None or label == "penalty":
            print(f"Skipping {video_file}: Label not found in mapping")
            continue
    
        print(f"Processing video: {video_file} | Label: {label}")
    
        # Extract video features
       
        features = predict_single_video_CLIP(video_path=video_path, predict_model=predict_model, visual_encoder=clip_image_encoder, device=args.device, size=args.size, fps=args.fps)
        
    
        print(f"Extracted features for {video_file}: {features.shape}")
        print(label)
        
        # Append features and labels
        print(features.unsqueeze(dim=0).shape)
        all_video_embeds.append(features.unsqueeze(dim=0))
        all_labels.append(torch.tensor(label, dtype=torch.long))
        break
    
    # Stack all embeddings and labels
    if len(all_video_embeds) == 0:
        raise ValueError("No valid video embeddings found.")
    
    video_embeds = torch.cat(all_video_embeds, dim=0).detach().cpu()  # Ensure it's on CPU
    labels = torch.stack(all_labels).squeeze().long()  # Convert label list to tensor
    
    # Prepare dataset and dataloader
    dataset = VideoEmbeddingDataset(video_embeds, labels)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Define model, loss, and optimizer
    mlp_classifier = MLPClassifier(input_dim=512, num_classes=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mlp_classifier.parameters(), lr=0.001)
    
    # Ensure save directory exists
    save_model_path = "/work/users/a/k/akkineni/Matchtime/MatchTime/models/mlp_models"
    os.makedirs(save_model_path, exist_ok=True)
    
    # Train classifier and save model after every epoch
    train_classifier(mlp_classifier, dataloader, criterion, optimizer, device=args.device, save_path=save_model_path, epochs=10)

    
    
