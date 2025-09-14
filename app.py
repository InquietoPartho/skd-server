
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import joblib
from PIL import Image
import plotly.graph_objects as go
import cv2
import matplotlib.pyplot as plt

# ---------------------- Page Setup ----------------------
st.set_page_config(page_title="🩺 Skin Disease Classifier", layout="centered")

# ---------------------- Custom CSS ----------------------
st.markdown("""
    <style>
        .main { background-color: #f0f4f8; }
        .block-container { padding: 2rem 1rem; }
        .stButton>button { border-radius: 12px; font-weight: 600; background-color: #4a90e2; color: white; }
        .stFileUploader { margin-bottom: 1.5rem; }
        footer { visibility: hidden; }
        .stImage > img { border-radius: 12px; box-shadow: 0 6px 20px rgba(0,0,0,0.15); object-fit: cover; }
        h1,h2,h3,h4 { text-align: center !important; font-family: 'Helvetica Neue', sans-serif; color: #333; }
        .stMarkdown { font-family: 'Helvetica Neue', sans-serif; color: #444; }
    </style>
""", unsafe_allow_html=True)

# ---------------------- Title ----------------------
st.markdown("<h1>👩‍⚕️ AI-Powered Skin Disease Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Upload a skin lesion image for diagnosis with explainability (Grad‑CAM & Saliency)</p>", unsafe_allow_html=True)

# ---------------------- Load Model ----------------------
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 6)
    model.load_state_dict(torch.load("resnet18_skin_disease.pth", map_location='cpu'))
    model.eval()
    return model

@st.cache_data
def load_label_encoder():
    return joblib.load("label_encoder.pkl")

model = load_model()
le = load_label_encoder()

class_names = [
    "Atopic Dermatitis",
    "Contact Dermatitis",
    "Eczema",
    "Scabies",
    "Seborrheic Dermatitis",
    "Tinea Corporis"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ---------------------- Preprocessing ----------------------
def preprocess_image(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    return transform(image).unsqueeze(0)

# ---------------------- Explainability Functions ----------------------
def generate_gradcam(model, image_path, class_names, device, label=None):
    model.eval()
    gradients, activations = [], []

    def save_grad(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def save_act(module, input, output):
        activations.append(output)

    handle1 = model.layer4[1].conv2.register_forward_hook(save_act)
    handle2 = model.layer4[1].conv2.register_backward_hook(save_grad)

    raw_image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess_image(raw_image).to(device)
    input_tensor.requires_grad_()

    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item() if label is None else label
    output[0, pred_class].backward()

    pooled_gradients = torch.mean(gradients[0], dim=[0,2,3])
    activation = activations[0][0]
    for i in range(len(pooled_gradients)):
        activation[i] *= pooled_gradients[i]

    heatmap = activation.detach().cpu().mean(0).numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = cv2.resize(heatmap, (224,224))
    heatmap_color = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)

    img_cv = np.array(raw_image.resize((224,224)))
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img_cv, 0.6, heatmap_color, 0.4, 0)

    handle1.remove()
    handle2.remove()

    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return Image.fromarray(overlay_rgb)

def generate_saliency_map(model, image_path, class_names, device):
    model.eval()
    raw_image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess_image(raw_image).to(device)
    input_tensor.requires_grad_()

    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()
    output[0, pred_class].backward()

    saliency = input_tensor.grad.abs().squeeze().cpu().numpy()
    saliency = np.max(saliency, axis=0)
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
    saliency_img = (saliency*255).astype(np.uint8)
    saliency_img = cv2.applyColorMap(saliency_img, cv2.COLORMAP_HOT)
    saliency_img = cv2.cvtColor(saliency_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(saliency_img)

# ---------------------- Sidebar Upload ----------------------
with st.sidebar:
    st.markdown("## 📤 Upload Skin Image")
    uploaded_file = st.file_uploader("Choose an image (JPG, PNG)", type=["jpg","jpeg","png"])
    st.markdown("---")
    st.info("Model: ResNet‑18 | 6 Classes", icon="ℹ️")

# ---------------------- Main App ----------------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.markdown("### 🖼️ Uploaded Image")
    st.image(image, use_container_width=False, width=320)

    input_tensor = preprocess_image(image).to(device)
    input_tensor.requires_grad_()

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1).cpu().numpy().flatten()
        pred_idx = int(np.argmax(probabilities))
        confidence = probabilities[pred_idx]

    st.markdown(f"<h3>✅ Prediction: <code>{class_names[pred_idx]}</code>  |  Confidence: <code>{confidence:.2f}</code></h3>", unsafe_allow_html=True)

    # ---------------------- Confidence Bar Plot ----------------------
    st.markdown("### 📊 Class Confidence Levels")
    sorted_indices = np.argsort(probabilities)[::-1]
    sorted_classes = [class_names[i] for i in sorted_indices]
    sorted_probs = probabilities[sorted_indices]

    colors = ['#3498db','#1abc9c','#9b59b6','#f39c12','#e74c3c','#2ecc71']
    fig = go.Figure(go.Bar(
        x=sorted_probs,
        y=sorted_classes,
        orientation='h',
        marker=dict(color=colors),
        text=[f"{p:.2f}" for p in sorted_probs],
        textposition="auto",
        hoverinfo='x+y+text',
    ))
    fig.update_layout(
        xaxis=dict(title="Confidence", range=[0,1]),
        yaxis=dict(title="Class"),
        height=400,
        margin=dict(l=40,r=40,t=40,b=40),
        showlegend=False,
        plot_bgcolor='rgba(255,255,255,0)'
    )
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)

    # ---------------------- Explainability Visuals ----------------------
    st.markdown("### 🔬 Explainability (Grad‑CAM & Saliency Map)")
    with st.spinner("Generating explainability visuals..."):
        gradcam_img = generate_gradcam(model, uploaded_file, class_names, device)
        saliency_img = generate_saliency_map(model, uploaded_file, class_names, device)

    col1, col2 = st.columns(2)
    img_w = 300
    with col1:
        st.markdown("#### Grad-CAM")
        st.image(gradcam_img, use_container_width=False, width=img_w)
    with col2:
        st.markdown("#### Saliency Map")
        st.image(saliency_img, use_container_width=False, width=img_w)
else:
    st.markdown("<h3>⚠️ Please upload a skin image to begin diagnosis.</h3>", unsafe_allow_html=True)
