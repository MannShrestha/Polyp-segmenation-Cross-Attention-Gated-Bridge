import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import uuid

from model.swin_unet import SwinAttentionUnet


# ---------------------------------------------------------
# ORANGE THEME CSS (section-title reduced for smaller column headers)
# ---------------------------------------------------------
st.markdown("""
<style>

body {
    background-color: #fff7f0;
}

h1, h4 {
    text-align: center;
    font-family: 'Segoe UI', sans-serif;
    color: #ff8c00;   /* medium orange */
}

/* smaller section titles for the column headers */
.section-title {
    font-size: 14px;      /* reduced from 18px */
    font-weight: 600;
    margin-bottom: 6px;
    text-align: center;
    color: #e67e22;
}

.stButton>button {
    background-color: #e67e22 !important;
    color: white !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    font-size: 16px !important;
    border: none !important;
}

.stButton>button:hover {
    background-color: #d35400 !important;
    color: #ffa64d !important;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# 1. Inference Transform
# ---------------------------------------------------------
def get_inference_transform():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ---------------------------------------------------------
# 2. Load Model
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    model = SwinAttentionUnet(num_classes=1, pretrained=False)
    weights_path = "./model/SAU_91_12.pth"
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


model = load_model()
transform = get_inference_transform()


# ---------------------------------------------------------
# 3. Create Save Folder
# ---------------------------------------------------------
SAVE_DIR = "./saved_predictions"
os.makedirs(SAVE_DIR, exist_ok=True)


# ---------------------------------------------------------
# 4. Streamlit UI Header (user-provided)
# ---------------------------------------------------------
# st.markdown("<h1 style='text-align: center;'>Polyp Segmentation</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color:#ffa64d;'>Polyp Segmentation</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'>Mann Shrestha</h5>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'>Department of Computer Science, University of  East London</h5>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'>u3059676@uel.ac.uk</h5>", unsafe_allow_html=True)


uploaded_file = st.file_uploader("Upload an endoscopy image", type=["png", "jpg", "jpeg", "tif"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.session_state["uploaded_img"] = np.array(img)


# Threshold slider
threshold = st.slider("Mask Threshold", 0.0, 1.0, 0.5, 0.01)

# Opacity slider
opacity = st.slider("Overlay Opacity", 0.0, 1.0, 0.45, 0.05)

# Overlay color selector
overlay_color = st.selectbox(
    "Overlay Color",
    ["Green", "Red", "Blue", "Orange"]
)


# ---------------------------------------------------------
# 5. Predict Button
# ---------------------------------------------------------
if st.button("Predict Segmentation"):

    if "uploaded_img" not in st.session_state:
        st.error("Please upload an image first.")
    else:
        img_np = st.session_state["uploaded_img"]

        # Resize original image to 224×224
        img_resized = cv2.resize(img_np, (224, 224))

        # Apply transforms
        augmented = transform(image=img_np)
        tensor_img = augmented["image"].unsqueeze(0)

        # Inference
        with torch.no_grad():
            logits = model(tensor_img)
            probs = torch.sigmoid(logits)

        mask = (probs > threshold).float()[0, 0].cpu().numpy()

        # ---------------------------------------------------------
        # OVERLAY COLOR MAP
        # ---------------------------------------------------------
        overlay_mask = (mask > 0.5).astype(np.uint8)
        color_layer = np.zeros_like(img_resized)

        if overlay_color == "Green":
            color_layer[..., 1] = overlay_mask * 255
        elif overlay_color == "Red":
            color_layer[..., 2] = overlay_mask * 255
        elif overlay_color == "Blue":
            color_layer[..., 0] = overlay_mask * 255
        elif overlay_color == "Orange":
            color_layer[..., 2] = overlay_mask * 255
            color_layer[..., 1] = overlay_mask * 150

        overlay = cv2.addWeighted(img_resized, 1.0, color_layer, opacity, 0)

        # ---------------------------------------------------------
        # HEATMAP (Turbo)
        # ---------------------------------------------------------
        heatmap_raw = (probs[0, 0].cpu().numpy() * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_raw, cv2.COLORMAP_TURBO)
        # heatmap_color = cv2.applyColorMap(heatmap_raw, cv2.COLORMAP_INFERNO)
        # heatmap_color = cv2.applyColorMap(heatmap_raw, cv2.COLORMAP_MAGMA)
        # heatmap_color = cv2.applyColorMap(heatmap_raw, cv2.COLORMAP_VIRIDIS)
        # heatmap_color = cv2.applyColorMap(heatmap_raw, cv2.COLORMAP_CIVIDIS)





        # Heatmap overlay
        heatmap_overlay = cv2.addWeighted(img_resized, 0.6, heatmap_color, 0.4, 0.5)

        # ---------------------------------------------------------
        # FORCE ALL OUTPUTS TO SAME SIZE (224×224)
        # ---------------------------------------------------------
        img_resized_224 = cv2.resize(img_resized, (224, 224))
        mask_224 = cv2.resize(mask, (224, 224))
        overlay_224 = cv2.resize(overlay, (224, 224))
        heatmap_224 = cv2.resize(heatmap_color, (224, 224))
        heatmap_overlay_224 = cv2.resize(heatmap_overlay, (224, 224))

        # ---------------------------------------------------------
        # SIDE-BY-SIDE DISPLAY (small section titles)
        # ---------------------------------------------------------
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.markdown("<div class='section-title'>Original</div>", unsafe_allow_html=True)
            st.image(img_resized_224, use_container_width=True)

        with col2:
            st.markdown("<div class='section-title'>Predicted</div>", unsafe_allow_html=True)
            st.image(mask_224, clamp=True)

        with col3:
            st.markdown("<div class='section-title'>Overlay</div>", unsafe_allow_html=True)
            st.image(overlay_224, use_container_width=True)

        with col4:
            st.markdown("<div class='section-title'>Heatmap (Turbo)</div>", unsafe_allow_html=True)
            st.image(heatmap_224, use_container_width=True)

        with col5:
            st.markdown("<div class='section-title'>Heatmap Overlay</div>", unsafe_allow_html=True)
            st.image(heatmap_overlay_224, use_container_width=True)

        # ---------------------------------------------------------
        # CREATE COMBINED IMAGE WITH SMALL LABELS + THIN COLORED BORDERS
        # ---------------------------------------------------------
        panel_w = 224
        panel_h = 224
        num_panels = 5

        combined_w = panel_w * num_panels
        combined_h = panel_h

        combined_image = np.ones((combined_h, combined_w, 3), dtype=np.uint8) * 255

        mask_rgb = cv2.cvtColor((mask_224 * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        images = [
            img_resized_224,
            mask_rgb,
            overlay_224,
            heatmap_224,
            heatmap_overlay_224
        ]

        labels = ["Original", "Mask", "Overlay", "Heatmap", "Heatmap Overlay"]

        border_colors = [
            (255, 140, 0),   # Orange
            (255, 0, 0),     # Red
            (0, 255, 0),     # Green
            (180, 0, 255),   # Purple
            (0, 180, 255)    # Blue
        ]

        # SMALL LABEL FONT (smaller) and thin border thickness
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        text_thickness = 1
        border_thickness = 1
        label_y = 14

        for i in range(num_panels):
            x_start = i * panel_w
            x_end = x_start + panel_w

            combined_image[:, x_start:x_end] = images[i]

            cv2.rectangle(
                combined_image,
                (x_start, 0),
                (x_end, panel_h),
                border_colors[i],
                border_thickness
            )

            text = labels[i]
            text_size = cv2.getTextSize(text, font, font_scale, text_thickness)[0]
            text_x = x_start + (panel_w - text_size[0]) // 2

            cv2.putText(
                combined_image,
                text,
                (text_x, label_y),
                font,
                font_scale,
                (255, 255, 255),
                text_thickness,
                cv2.LINE_AA
            )

        # Save combined image
        unique_id = str(uuid.uuid4())[:8]
        combined_path = os.path.join(SAVE_DIR, f"{unique_id}_combined.png")
        Image.fromarray(combined_image).save(combined_path)

        st.success("Combined result saved successfully.")
        st.image(combined_image, caption="Combined Output", use_container_width=True)
