# app_detecting_invisible_threats.py
# Streamlit App: Detecting Invisible Threats - Explainable AI Demo

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# ---------- App Header ----------
st.set_page_config(page_title="Detecting Invisible Threats", layout="wide")
st.title("🧠 Detecting Invisible Threats: Adversarial Attack Identification Using Explainable AI")
st.markdown("Upload an image, apply adversarial perturbation (FGSM), and visualize model explanations (Grad-CAM).")

# ---------- Model Definition ----------
class SimpleCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SimpleCNN().to(device)
model.eval()

# ---------- Helper Functions ----------
def preprocess_image(img, use_color=False):
