import os
import json
import csv
import torch
import clip
import cv2
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from models.matchvoice_model import matchvoice_model

# --------- HARDCODED CONFIG ---------
json_path = "/work/users/a/k/akkineni/Matchtime/MatchTime/models/1min30sec_new/annotations.json"
video_root = "/work/users/a/k/akkineni/Matchtime/MatchTime/models/1min30sec_new"
output_csv = "/work/users/a/k/akkineni/Matchtime/MatchTime/models/1min30sec_new/video_commentaries.csv"
device = "cuda:0"
size = 224
fps = 1
model_ckpt = "./ckpt/CLIP_matchvoice.pth"
tokenizer_name = "meta-llama/Meta-Llama-3-8B-Instruct"
num_video_query_token = 32
num_features = 512
# -----------------------------------


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
        self.cap = cv2.VideoCapture(self.video_path)
        self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_indices = [int(x * self.cap.get(cv2.CAP_PROP_FPS) / self.fps)
                              for x in range(int(self.length / self.cap.get(cv2.CAP_PROP_FPS) * self.fps))]

    def __len__(self):
        return len(self.frame_indices)

    def __getitem__(self, idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_indices[idx])
        ret, frame = self.cap.read()
        if not ret:
            print("Error in reading frame")
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = self.transforms(Image.fromarray(frame))
        return frame.to(torch.float16)

    def close(self):
        self.cap.release()


def encode_features(data_loader, encoder, device):
    all_features = None
    for frames in data_loader:
        features = encoder(frames.to(device))
        if all_features is None:
            all_features = features
        else:
            all_features = torch.cat((all_features, features), dim=0)
    return all_features


def predict_single_video_CLIP(video_path, predict_model, visual_encoder, size, fps, device):
    try:
        dataset = VideoDataset(video_path, size=size, fps=fps)
        data_loader = DataLoader(dataset, batch_size=40, shuffle=False, pin_memory=True, num_workers=0)
        features = encode_features(data_loader, visual_encoder, device)
        dataset.close()
        print("✅ Features of this video loaded with shape of:", features.shape)
    except Exception as e:
        print("❌ Error loading video:", video_path, "|", str(e))
        return []

    tot_frames = features.shape[0]
    commentary = []
    i = 0
    while tot_frames > 24:
        sample = {
            "features": features[i:i + 24].unsqueeze(dim=0),
            "labels": None,
            "attention_mask": None,
            "input_ids": None
        }
        comment = predict_model(sample)
        commentary.append(comment)
        i += 24
        tot_frames -= 24

    if i < features.shape[0]:
        sample = {
            "features": features[i:].unsqueeze(dim=0),
            "labels": None,
            "attention_mask": None,
            "input_ids": None
        }
        comment = predict_model(sample)
        commentary.append(comment)

    print("🗣️ Commentary generated:", commentary)
    return commentary


if __name__ == '__main__':

    with open(json_path, 'r') as f:
        video_meta = json.load(f)

    # Create and configure CLIP model
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    clip_image_encoder = model.encode_image

    # Load matchvoice model
    predict_model = matchvoice_model(
        llm_ckpt=tokenizer_name,
        tokenizer_ckpt=tokenizer_name,
        num_video_query_token=num_video_query_token,
        num_features=num_features,
        device=device,
        inference=True
    )
    other_parts_state_dict = torch.load(model_ckpt)
    new_model_state_dict = predict_model.state_dict()
    for key, value in other_parts_state_dict.items():
        if key in new_model_state_dict:
            new_model_state_dict[key] = value
    predict_model.load_state_dict(new_model_state_dict)
    predict_model.eval()

    # Write header for CSV
    fieldnames = [
        "video_name", "label", "description", "anonymized",
        "commentary_1", "commentary_2", "commentary_3", "commentary_4"
    ]
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    # Process each video
    if os.path.exists(output_csv):
        existing_df = pd.read_csv(output_csv)
        processed_videos = set(existing_df['video_name'].astype(str).tolist())
        
    else:
        processed_videos = set()
    for video_name, meta in video_meta.items():
        if video_name in processed_videos:
            continue
        video_path = os.path.join(video_root, video_name)
        print(f"▶️ Processing: {video_path}")
        comments = predict_single_video_CLIP(video_path, predict_model, clip_image_encoder, size, fps, device)

        # Pad to 4 comments
        comments = comments[:4] + ["N/A"] * (4 - len(comments))

        row = {
            "video_name": video_name,
            "label": meta.get("label", ""),
            "description": meta.get("description", ""),
            "anonymized": meta.get("anonymized", ""),
            "commentary_1": comments[0],
            "commentary_2": comments[1],
            "commentary_3": comments[2],
            "commentary_4": comments[3]
        }

        with open(output_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)

    print(f"\n✅ All videos processed. Results saved to: {output_csv}")
