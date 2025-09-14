import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import streamlit as st  # ✅ Required for Streamlit output

# Grad-CAM
def generate_gradcam(model, image_path, class_names, device, label=None):
    model.eval()
    gradients = []
    activations = []

    def save_grad(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def save_act(module, input, output):
        activations.append(output)

    handle1 = model.layer4[1].conv2.register_forward_hook(save_act)
    handle2 = model.layer4[1].conv2.register_backward_hook(save_grad)

    raw_image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    input_tensor = preprocess(raw_image).unsqueeze(0).to(device)
    input_tensor.requires_grad_()

    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item() if label is None else label
    class_score = output[0, pred_class]
    model.zero_grad()
    class_score.backward()

    pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])
    activation = activations[0][0]
    for i in range(len(pooled_gradients)):
        activation[i] *= pooled_gradients[i]

    heatmap = activation.detach().cpu().mean(0).numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    img_cv = np.array(raw_image.resize((224, 224)))
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img_cv, 0.6, heatmap_color, 0.4, 0)

    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(heatmap)
    cv2.circle(overlay, maxLoc, 30, (0, 0, 255), 2)

    handle1.remove()
    handle2.remove()

    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title(f"Grad-CAM: {class_names[pred_class]}")
    plt.axis("off")
    st.pyplot(plt)  # ✅ Display in Streamlit


# Saliency Map
def generate_saliency_map(model, image_path, class_names, device):
    model.eval()
    raw_image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    input_tensor = preprocess(raw_image).unsqueeze(0).to(device)
    input_tensor.requires_grad_()

    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()
    output[0, pred_class].backward()

    saliency = input_tensor.grad.abs().squeeze().cpu().numpy()
    saliency = np.max(saliency, axis=0)
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(raw_image.resize((224, 224)))
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(saliency, cmap='hot')
    plt.title(f"Saliency Map: {class_names[pred_class]}")
    plt.axis("off")
    st.pyplot(plt)  # ✅ Display in Streamlit
