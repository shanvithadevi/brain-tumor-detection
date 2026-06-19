import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np

# Define model (same as training)
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 30 * 30, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 30 * 30)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

# Load model
model = CNNModel()
model.load_state_dict(torch.load("training_model.h5"))
model.eval()

# Streamlit UI
st.title("🧠 Brain Tumor Detection (PyTorch)")
st.write("Upload an MRI image to check if it has a tumor.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

def compute_features(img):
    # Convert to OpenCV format
    img_cv = np.array(img)
    img_cv = cv2.resize(img_cv, (128,128))

    # Color histogram
    hist = cv2.calcHist([img_cv],[0],None,[256],[0,256])
    hist = cv2.normalize(hist, hist).flatten()
    color_score = np.mean(hist)

    # Edge detection
    edges = cv2.Canny(img_cv, 100, 200)
    edge_score = np.mean(edges) / 255.0

    # Texture (variance of pixel intensities)
    texture_score = np.var(img_cv) / 255.0

    # Weighted combination
    feature_score = (0.4*color_score + 0.3*edge_score + 0.3*texture_score) / 100
    return feature_score

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded MRI", use_column_width=True)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        prediction = model(img_tensor).item()

    # CNN confidence
    cnn_confidence = prediction if prediction > 0.5 else (1 - prediction)

    # Feature similarity score
    feature_score = compute_features(img)

    # Final compatibility percentage (weighted)
    compatibility = (0.7*cnn_confidence + 0.3*feature_score) * 100

    if prediction > 0.5:
        st.error(f"⚠️ Tumor Detected — Compatibility: {compatibility:.2f}% (appearance, color, shape)")
    else:
        st.success(f"✅ No Tumor Detected — Compatibility: {compatibility:.2f}% (appearance, color, shape)")
