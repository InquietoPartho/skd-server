# import streamlit as st
# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# import torchvision.models as models
# import numpy as np
# import joblib
# from PIL import Image
# import os
# from utils.explainability import generate_gradcam, generate_saliency_map

# # ---------------------- Configuration ----------------------
# st.set_page_config(page_title="Skin Disease Classifier", layout="centered")
# st.title("👩‍⚕ Skin Disease Detection with Explainability")

# # Load model
# @st.cache_resource
# def load_model():
#     model = models.resnet18(pretrained=False)
#     model.fc = nn.Linear(model.fc.in_features, 6)
#     model.load_state_dict(torch.load("resnet18_skin_disease.pth", map_location='cpu'))
#     model.eval()
#     return model

# # Load LabelEncoder
# @st.cache_data
# def load_label_encoder():
#     le = joblib.load("label_encoder.pkl")
#     return le

# model = load_model()
# le = load_label_encoder()
# class_names = le.classes_
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)

# # Image preprocessing
# def preprocess_image(image):
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5]*3, [0.5]*3)
#     ])
#     return transform(image).unsqueeze(0)

# # ---------------------- Upload Section ----------------------
# uploaded_file = st.file_uploader("Upload a skin image", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     input_tensor = preprocess_image(image).to(device)
#     input_tensor.requires_grad_()

#     with torch.no_grad():
#         output = model(input_tensor)
#         pred_class = torch.argmax(output, 1).item()
#         confidence = torch.softmax(output, dim=1)[0][pred_class].item()

#     st.success(f"*Prediction:* {class_names[pred_class]}  |  Confidence: {confidence:.2f}")

#     # ---------------------- Explainability Section ----------------------
#     st.subheader("🔮 Grad-CAM Explanation")
#     with st.spinner("Generating Grad-CAM..."):
#         generate_gradcam(model, uploaded_file, class_names, device, label=None)

#     st.subheader("🔍 Saliency Map")
#     with st.spinner("Generating Saliency Map..."):
#         generate_saliency_map(model, uploaded_file, class_names, device)

# else:
#     st.info("Please upload a skin image to begin diagnosis.")


# import streamlit as st
# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# import torchvision.models as models
# import numpy as np
# import joblib
# from PIL import Image
# import plotly.graph_objects as go
# from utils.explainability import generate_gradcam, generate_saliency_map

# # ---------------------- Page Setup ----------------------
# st.set_page_config(page_title="🩺 Skin Disease Classifier", layout="centered")

# # Custom CSS
# st.markdown("""
#     <style>
#         .main {background-color: #f9f9f9;}
#         .block-container {padding-top: 2rem;}
#         .stButton>button {border-radius: 10px; font-weight: 600;}
#         .stFileUploader {margin-bottom: 1.5rem;}
#         footer {visibility: hidden;}
#     </style>
# """, unsafe_allow_html=True)

# # Title
# st.markdown("<h1 style='text-align: center;'>👩‍⚕️ AI-Powered Skin Disease Classifier</h1>", unsafe_allow_html=True)
# st.markdown("<p style='text-align: center; font-size: 18px;'>Upload a skin lesion image to diagnose and explain using Grad-CAM and Saliency Maps</p>", unsafe_allow_html=True)

# # ---------------------- Load Model ----------------------
# @st.cache_resource
# def load_model():
#     model = models.resnet18(pretrained=False)
#     model.fc = nn.Linear(model.fc.in_features, 6)
#     model.load_state_dict(torch.load("resnet18_skin_disease.pth", map_location='cpu'))
#     model.eval()
#     return model

# @st.cache_data
# def load_label_encoder():
#     return joblib.load("label_encoder.pkl")

# model = load_model()
# le = load_label_encoder()
# class_names = le.classes_
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)

# # ---------------------- Preprocessing ----------------------
# def preprocess_image(image):
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5]*3, [0.5]*3)
#     ])
#     return transform(image).unsqueeze(0)

# # ---------------------- Sidebar Upload ----------------------
# with st.sidebar:
#     st.markdown("## 📤 Upload Skin Image")
#     uploaded_file = st.file_uploader("Upload an image (JPG, PNG)", type=["jpg", "jpeg", "png"])
#     st.markdown("---")
#     st.info("Model: ResNet18 | Classes: 6", icon="ℹ️")

# # ---------------------- Main App ----------------------
# if uploaded_file:
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

#     input_tensor = preprocess_image(image).to(device)
#     input_tensor.requires_grad_()

#     with torch.no_grad():
#         output = model(input_tensor)
#         pred_class = torch.argmax(output, 1).item()
#         confidence = torch.softmax(output, dim=1)[0][pred_class].item()

#     st.success(f"✅ **Prediction:** `{class_names[pred_class]}`  |  **Confidence:** `{confidence:.2f}`")

#     # ---------------------- Interactive Plotly Confidence Bar ----------------------
#     st.markdown("### 📊 Class Probabilities")

#     probabilities = torch.softmax(output, dim=1).cpu().numpy().flatten()
#     sorted_idx = np.argsort(probabilities)[::-1]
#     sorted_probs = probabilities[sorted_idx]
#     sorted_classes = class_names[sorted_idx]

#     colors = []
#     for idx in sorted_idx:
#         if idx == pred_class:
#             colors.append("royalblue")
#         elif probabilities[idx] >= 0.5:
#             colors.append("green")
#         elif probabilities[idx] >= 0.2:
#             colors.append("orange")
#         else:
#             colors.append("red")

#     fig = go.Figure(go.Bar(
#         x=sorted_probs,
#         y=sorted_classes,
#         orientation='h',
#         marker_color=colors,
#         text=[f"{p:.2f}" for p in sorted_probs],
#         textposition='auto',
#         hoverinfo='x+y+text',
#     ))

#     fig.update_layout(
#         xaxis=dict(title="Confidence", range=[0, 1]),
#         yaxis=dict(title="Class"),
#         height=400,
#         margin=dict(l=40, r=40, t=40, b=40),
#         showlegend=False
#     )

#     fig.update_yaxes(autorange="reversed")  # Highest at top
#     st.plotly_chart(fig, use_container_width=True)

#     # ---------------------- Explainability Section ----------------------
#     st.markdown("### 🔬 Explainability")
#     col1, col2 = st.columns(2)

#     with col1:
#         st.markdown("#### 🔮 Grad-CAM")
#         with st.spinner("Generating Grad-CAM..."):
#             generate_gradcam(model, uploaded_file, class_names, device, label=None)

#     with col2:
#         st.markdown("#### 🌡️ Saliency Map")
#         with st.spinner("Generating Saliency Map..."):
#             generate_saliency_map(model, uploaded_file, class_names, device)

# else:
#     st.warning("⚠️ Please upload a skin image to start diagnosis.", icon="📷")

import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import joblib
from PIL import Image
import plotly.graph_objects as go
from utils.explainability import generate_gradcam, generate_saliency_map

# ---------------------- Page Setup ----------------------
st.set_page_config(page_title="🩺 Skin Disease Classifier", layout="centered")

# ---------------------- Custom CSS ----------------------
st.markdown("""
    <style>
        .main {
            background-color: #f0f4f8;
        }
        .block-container {
            padding: 2rem 1rem;
        }
        .stButton>button {
            border-radius: 12px;
            font-weight: 600;
            background-color: #4a90e2;
            color: white;
        }
        .stFileUploader {
            margin-bottom: 1.5rem;
        }
        footer {
            visibility: hidden;
        }
        .stImage > img {
            border-radius: 12px;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
            object-fit: cover;
        }
        h1, h2, h3, h4 {
            text-align: center !important;
            font-family: 'Helvetica Neue', sans-serif;
            color: #333;
        }
        .stMarkdown {
            font-family: 'Helvetica Neue', sans-serif;
            color: #444;
        }
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

# Use full medical class names
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

# ---------------------- Sidebar Upload ----------------------
with st.sidebar:
    st.markdown("## 📤 Upload Skin Image")
    uploaded_file = st.file_uploader("Choose an image (JPG, PNG)", type=["jpg", "jpeg", "png"])
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

    colors = ['#3498db', '#1abc9c', '#9b59b6', '#f39c12', '#e74c3c', '#2ecc71']

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
        xaxis=dict(title="Confidence", range=[0, 1]),
        yaxis=dict(title="Class"),
        height=400,
        margin=dict(l=40, r=40, t=40, b=40),
        showlegend=False,
        plot_bgcolor='rgba(255,255,255,0)',
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
