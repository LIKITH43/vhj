# app_detecting_invisible_threats.py
# Streamlit App: Detecting Invisible Threats - Explainable AI Demo

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# -------------------- App Header --------------------
st.set_page_config(page_title="Detecting Invisible Threats", layout="wide")
st.title("🧠 Detecting Invisible Threats: Adversarial Attack Identification Using Explainable AI")
st.markdown("Upload an image, apply adversarial perturbation (FGSM), and visualize model explanations (**Grad-CAM**).")

# Determine device
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------- Model Definition --------------------
# NOTE: This simple CNN is designed for 28x28 grayscale images (like MNIST)
# Adjust dimensions if you use a different dataset/image size.
class SimpleCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # 9216 is (64 features * 12 * 12) from 28x28 input after two pooling layers
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

# Load model (Initialize a new model; typically you'd load pre-trained weights)
model = SimpleCNN(in_channels=1, num_classes=10).to(device)
model.eval()

# -------------------- Helper Functions --------------------

@st.cache_data
def preprocess_image(img):
    """Preprocesses a PIL image for the SimpleCNN (28x28, Grayscale, Normalized)."""
    # Define transformations for a simple grayscale 28x28 input model (like MNIST)
    transform = transforms.Compose([
        transforms.Resize((28, 28)), 
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        # Use standard MNIST mean/std for normalization
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Apply transform, add batch dimension, and move to device
    img_tensor = transform(img).unsqueeze(0).to(device)
    return img_tensor

def fgsm_attack(image_tensor, epsilon, data_grad):
    """
    Performs a Fast Gradient Sign Method (FGSM) attack.
    """
    # Collect the sign of the data gradients
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image_tensor + epsilon * sign_data_grad
    # Adding clipping to maintain image range [0, 1]
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def apply_gradcam(model, input_tensor, target_layer):
    """
    Computes and visualizes Grad-CAM.
    NOTE: This is a simplified, manual implementation of Grad-CAM.
    """
    # 1. Forward Pass & Hook Setup
    
    # Stores the output feature map of the target layer
    features = [] 
    # Stores the gradients of the output w.r.t the target layer
    gradients = []

    def save_features(module, input, output):
        features.append(output)

    def save_gradients(module, grad_input, grad_output):
        # We need the gradient of the loss w.r.t the feature map output
        gradients.append(grad_output[0])

    # Hook the target layer (e.g., the last convolutional layer: model.conv2)
    hook_handle_f = target_layer.register_forward_hook(save_features)
    hook_handle_b = target_layer.register_backward_hook(save_gradients)

    # 2. Get Model Prediction
    output = model(input_tensor)
    
    # Get the predicted class index
    pred_idx = output.argmax(dim=1).item()
    
    # 3. Zero Gradients & Backward Pass for the predicted class
    model.zero_grad()
    # Create a tensor of zeros, then put a 1 at the predicted class index
    one_hot = torch.zeros_like(output)
    one_hot[0][pred_idx] = 1
    # Calculate the gradient of the predicted score w.r.t the input
    output.backward(gradient=one_hot, retain_graph=True)

    # 4. Compute Grad-CAM
    
    # Get feature map and gradients from the hooks
    feature_map = features[0].squeeze(0)  # Shape: [Channels, H, W]
    grad_val = gradients[0].squeeze(0)    # Shape: [Channels, H, W]

    # Global Average Pooling (GAP) of gradients: Alpha weights
    alpha = grad_val.mean(dim=(1, 2), keepdim=True) # Shape: [Channels, 1, 1]

    # Compute weighted feature map: L_c = ReLU(sum_k(alpha_k * A_k))
    cam = torch.sum(alpha * feature_map, dim=0).relu() # Shape: [H, W]
    
    # 5. Cleanup
    hook_handle_f.remove()
    hook_handle_b.remove()

    # 6. Post-process CAM
    # Normalize CAM to range [0, 1]
    cam = cam - cam.min()
    if cam.max() != 0:
        cam = cam / cam.max()

    # Resize CAM to the input image dimensions (28x28)
    cam = transforms.Resize(input_tensor.shape[-2:])(Image.fromarray(cam.cpu().numpy()))
    cam = np.array(cam)
    
    return cam, pred_idx

# -------------------- Streamlit UI & Logic --------------------

# Sidebar for controls
st.sidebar.header("Controls")
uploaded_file = st.sidebar.file_uploader("Upload an Image (e.g., a hand-written digit):", type=["png", "jpg", "jpeg"])
epsilon = st.sidebar.slider("FGSM Epsilon ($\epsilon$)", min_value=0.0, max_value=0.5, value=0.1, step=0.01)
st.sidebar.markdown("""
_Epsilon controls the strength of the adversarial noise. A higher $\epsilon$ means stronger noise._
""")

# Define the target layer for Grad-CAM (the last convolutional layer)
TARGET_LAYER = model.conv2

if uploaded_file is not None:
    try:
        # Load and preprocess original image
        original_image_pil = Image.open(uploaded_file).convert("L") # Convert to grayscale for simplicity
        
        # Display image details in sidebar
        st.sidebar.image(original_image_pil, caption="Original Image Preview", use_column_width=True)
        st.sidebar.info(f"Image Size: {original_image_pil.size}")

        original_input_tensor = preprocess_image(original_image_pil)
        
        # Make tensor require gradient for the attack
        original_input_tensor.requires_grad = True

        st.subheader("📊 Model Analysis")

        col1, col2 = st.columns(2)

        # --- 1. Original Image Analysis ---
        with col1:
            st.markdown("### Original Image")
            
            # Get original prediction
            output_orig = model(original_input_tensor)
            _, pred_orig_idx = torch.max(output_orig.data, 1)
            
            st.success(f"**Predicted Class:** {pred_orig_idx.item()}")

            # Compute Grad-CAM for original image
            cam_orig, _ = apply_gradcam(model, original_input_tensor, TARGET_LAYER)

            # Visualization
            fig_orig, ax_orig = plt.subplots()
            ax_orig.imshow(original_image_pil.resize((28, 28)), cmap='gray')
            ax_orig.imshow(cam_orig, cmap='jet', alpha=0.5)
            ax_orig.set_title(f"Grad-CAM (Prediction: {pred_orig_idx.item()})")
            ax_orig.axis('off')
            st.pyplot(fig_orig)

        # --- 2. Adversarial Image Analysis (FGSM) ---
        with col2:
            st.markdown(f"### Adversarial Image ($\epsilon$={epsilon})")
            
            # a. Calculate Gradients
            # Zero all existing gradients
            model.zero_grad()
            
            # Calculate Loss (using the correct prediction for the target class)
            target = torch.tensor([pred_orig_idx.item()]).to(device)
            loss = F.nll_loss(output_orig, target) # Use NLL for classification
            loss.backward()

            # Collect the gradient of the loss w.r.t. the input data
            data_grad = original_input_tensor.grad.data
            
            # b. Apply FGSM Attack
            perturbed_data = fgsm_attack(original_input_tensor, epsilon, data_grad)
            
            # c. Get Adversarial Prediction
            output_adv = model(perturbed_data)
            _, pred_adv_idx = torch.max(output_adv.data, 1)

            if pred_adv_idx.item() != pred_orig_idx.item():
                 st.error(f"**Predicted Class:** {pred_adv_idx.item()} (Attack **Successful!**)")
            else:
                 st.warning(f"**Predicted Class:** {pred_adv_idx.item()} (Attack Failed)")

            # d. Compute Grad-CAM for adversarial image
            cam_adv, _ = apply_gradcam(model, perturbed_data, TARGET_LAYER)
            
            # Convert tensor back to PIL for display
            # Denormalize: X * std + mean
            perturbed_image_np = (perturbed_data.squeeze(0).squeeze(0).cpu().detach().numpy() * 0.3081) + 0.1307
            perturbed_image_pil = Image.fromarray((perturbed_image_np.clip(0, 1) * 255).astype(np.uint8))
            
            # Visualization
            fig_adv, ax_adv = plt.subplots()
            ax_adv.imshow(perturbed_image_pil, cmap='gray')
            ax_adv.imshow(cam_adv, cmap='jet', alpha=0.5)
            ax_adv.set_title(f"Grad-CAM (Prediction: {pred_adv_idx.item()})")
            ax_adv.axis('off')
            st.pyplot(fig_adv)
            
        st.markdown("---")
        st.markdown("### 🔍 Conclusion")
        st.markdown(
            "Notice how the **Grad-CAM heatmaps** might shift between the original and adversarial images. "
            "In a successful attack, the model's 'attention' (the bright yellow areas) may move to irrelevant "
            "or noisy parts of the image, showing it's relying on the *adversarial noise* rather than "
            "the true features to make its incorrect prediction."
        )

    except Exception as e:
        st.error(f"An error occurred during processing. Ensure your model architecture is correct for the input data. Error: {e}")

else:
    st.info("⬆️ Please upload an image in the sidebar to begin the analysis.")
