import os
import cv2
import torch
import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
from torchvision import models, transforms

# Paths
VIDEO_DIR = "/work/users/a/k/akkineni/Matchtime/MatchTime/models/dataset"
MODEL_SAVE_PATH = "/work/users/a/k/akkineni/Matchtime/MatchTime/models/mlp_models/random_forest_model.pkl"

# Label mapping
LABEL_MAPPING = {
    "corner": 0,
    "soccer-ball": 1,
    "injury": 2,
    "no event": 3
}

# Load ResNet18
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = models.resnet18(pretrained=True).eval().to(device)
feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])

# Transform for ResNet
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Optical flow extraction
def extract_optical_flow(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, prev = cap.read()
    if not ret:
        return None

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    motion_mags = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_mags.append(np.mean(mag))
        prev_gray = gray

    cap.release()
    return np.array(motion_mags)

# Visual feature extraction
def extract_visual_features(video_path):
    cap = cv2.VideoCapture(video_path)
    feats = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img_tensor = transform(frame).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = feature_extractor(img_tensor).squeeze().cpu().numpy()
        feats.append(feat)

    cap.release()
    return np.mean(feats, axis=0)

# Combine visual and motion features
def process_video(video_path):
    visual_feat = extract_visual_features(video_path)
    motion_feat = extract_optical_flow(video_path)

    if motion_feat is None:
        return None

    motion_summary = [np.mean(motion_feat), np.std(motion_feat), np.max(motion_feat)]
    return np.concatenate([visual_feat, motion_summary])

# Load videos and extract features + labels
X, y = [], []
video_files = glob(os.path.join(VIDEO_DIR, "*.mkv"))

for video_path in tqdm(video_files):
    file = os.path.basename(video_path)
    label_str = file.split("_")[-1].replace(".mkv", "")
    if label_str not in LABEL_MAPPING:
        continue
    label_id = LABEL_MAPPING[label_str]
    feat = process_video(video_path)
    if feat is not None:
        X.append(feat)
        y.append(label_id)

X, y = np.array(X), np.array(y)

# Train/test split and classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save model
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
joblib.dump(clf, MODEL_SAVE_PATH)

# Evaluation
y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred, target_names=LABEL_MAPPING.keys())
report
