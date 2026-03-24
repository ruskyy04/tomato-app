import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image
import numpy as np
import cv2
import io
import tempfile
import matplotlib.pyplot as plt

st.set_page_config(page_title="Tomato Disease Detector", layout="wide")

# ---------------------------
# UI: Header
# ---------------------------
st.title("🌿 Tomato Disease Detection — Prediction · Severity · Management")
st.write("Upload a leaf image → detect disease, show black & white severity mask, and get management steps.")

# ---------------------------
# Cached model loader
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_model_from_path(path: str, num_classes: int = 11, device_str: str = "cpu"):
    device = torch.device(device_str)
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    state = torch.load(path, map_location=device)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    new_state = {}
    for k, v in state.items() if isinstance(state, dict) else []:
        nk = k.replace("module.", "") if k.startswith("module.") else k
        new_state[nk] = v

    if new_state:
        model.load_state_dict(new_state)
    else:
        model.load_state_dict(state)

    model.to(device)
    model.eval()
    return model, device

# ---------------------------
# App: Left column (upload)
# ---------------------------
col1, col2 = st.columns([1, 1.2])

with col1:
    st.header("1) Upload")
    uploaded_img = st.file_uploader("Choose a tomato leaf image", type=["jpg", "jpeg", "png"])

    st.write("---")
    uploaded_model = st.file_uploader("Upload model (.pth) — optional", type=["pth", "pt"])

    st.write("---")
    st.header("Settings")
    use_gpu = st.checkbox("Use GPU if available", value=False)
    device_str = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"

    model_path = model.pth


# ---------------------------
# Class names
# ---------------------------
class_names = [
    "Bacterial_spot",
    "Early_blight",
    "Late_blight",
    "Leaf_Mold",
    "Septoria_leaf_spot",
    "Spider_mites Two-spotted_spider_mite",
    "Target_Spot",
    "Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_mosaic_virus",
    "healthy",
    "powdery_mildew"
]

# ---------------------------
# Transforms
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------------------
# Helper: Load model
# ---------------------------
def load_model_choice(uploaded_model_file, model_path_text, device_str_local):
    if uploaded_model_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp:
            tmp.write(uploaded_model_file.read())
            tmp.flush()
            return load_model_from_path(tmp.name, num_classes=len(class_names), device_str=device_str_local)

    try:
        return load_model_from_path(model_path_text, num_classes=len(class_names), device_str=device_str_local)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None

# ---------------------------
# Helper: Model prediction
# ---------------------------
def predict_with_model(model, device, pil_img):
    img_t = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(img_t)
        probs = torch.softmax(out, dim=1)[0].cpu().numpy()
    top_idx = int(np.argmax(probs))
    return top_idx, probs

# ---------------------------
# EXACT TensorFlow-style severity mask
# ---------------------------
def compute_severity_mask(np_img_rgb):
    hsv = cv2.cvtColor(np_img_rgb, cv2.COLOR_RGB2HSV)

    # ORIGINAL HSV THRESHOLDS (your TF code)
    lower_brown = np.array([10, 40, 20])
    upper_brown = np.array([40, 255, 255])

    mask = cv2.inRange(hsv, lower_brown, upper_brown)

    total_pixels = np_img_rgb.shape[0] * np_img_rgb.shape[1]
    infected_pixels = np.count_nonzero(mask)
    severity_percent = (infected_pixels / total_pixels) * 100

    if severity_percent < 30:
        severity_level = "🟢 Mild Infection"
    elif 30 <= severity_percent < 60:
        severity_level = "🟡 Moderate Infection"
    else:
        severity_level = "🔴 Severe Infection"

    return mask, severity_percent, severity_level

# ---------------------------
# Main run button
# ---------------------------
with col1:
    run_button = st.button("🔎 Run Prediction & Severity")

# ---------------------------
# Right column: results
# ---------------------------
with col2:
    st.header("Results")
    placeholder_image = st.empty()
    placeholder_mask = st.empty()
    placeholder_info = st.empty()

# ---------------------------
# When user clicks RUN
# ---------------------------
if run_button:
    if uploaded_img is None:
        st.warning("Please upload an image.")
    else:
        model_obj, device_obj = load_model_choice(uploaded_model, model_path, device_str)

        if model_obj is None:
            st.error("Model failed to load.")
        else:
            image_data = uploaded_img.read()
            pil_img = Image.open(io.BytesIO(image_data)).convert("RGB")
            np_img = np.array(pil_img)

            # Prediction
            with st.spinner("Predicting disease..."):
                idx, probs = predict_with_model(model_obj, device_obj, pil_img)
                label = class_names[idx]
                confidence = float(probs[idx] * 100)

            # Severity
            mask, severity_pct, severity_level = compute_severity_mask(np_img)

            # Show original image
            placeholder_image.image(np_img, caption="Uploaded Image", use_column_width=True)

            # Show pure B/W mask — EXACT TF OUTPUT
            placeholder_mask.image(
                mask,
                caption="🖤 Infected Area Mask (Pure Black & White)",
                use_column_width=True
            )

            # Show predictions
            with placeholder_info.container():
                st.subheader("🩺 Prediction")
                st.metric("Predicted Disease", label, f"{confidence:.2f}% confidence")

                st.write("Top-5 predictions:")
                top5_idx = np.argsort(probs)[-5:][::-1]
                for i in top5_idx:
                    st.write(f"- **{class_names[i]}** — {probs[i]*100:.2f}%")

                st.write("---")
                st.subheader("🌡️ Severity")
                st.write(f"Infected area: **{severity_pct:.2f}%** — **{severity_level}**")
                st.progress(min(1.0, severity_pct / 100))

                st.write("---")
                st.subheader("🌿 Management Recommendations")

                disease_info = {
                    "Bacterial_spot": {
                        "management": [
                            "Use certified disease-free seeds.",
                            "Avoid overhead irrigation.",
                            "Destroy crop residues after harvest."
                        ],
                        "pesticides": [
                            {"name": "Copper Oxychloride (Blitox 50 WP)", "dosage": "2.5 g/L"},
                            {"name": "Streptocycline", "dosage": "100 ppm"}
                        ]
                    },
                    "Early_blight": {
                        "management": [
                            "Remove and destroy infected leaves.",
                            "Rotate with non-solanaceous crops.",
                            "Avoid excess nitrogen."
                        ],
                        "pesticides": [
                            {"name": "Mancozeb", "dosage": "2.5 g/L"},
                            {"name": "Chlorothalonil", "dosage": "2 g/L"}
                        ]
                    },
                    "Late_blight": {
                        "management": [
                            "Avoid water stagnation.",
                            "Remove affected plants.",
                            "Ensure air flow."
                        ],
                        "pesticides": [
                            {"name": "Ridomil Gold", "dosage": "2 g/L"},
                            {"name": "Copper Hydroxide", "dosage": "3 g/L"}
                        ]
                    },
                    "Leaf_Mold": {
                        "management": [
                            "Improve ventilation.",
                            "Avoid overhead watering.",
                            "Keep humidity moderate."
                        ],
                        "pesticides": [
                            {"name": "Chlorothalonil", "dosage": "2 g/L"},
                            {"name": "Mancozeb", "dosage": "2.5 g/L"}
                        ]
                    },
                    "Septoria_leaf_spot": {
                        "management": [
                            "Remove infected leaves.",
                            "Avoid touching wet plants.",
                            "Use preventive fungicides."
                        ],
                        "pesticides": [
                            {"name": "Copper fungicide", "dosage": "2.5 g/L"},
                            {"name": "Bravo", "dosage": "2 g/L"}
                        ]
                    },
                    "Spider_mites Two-spotted_spider_mite": {
                        "management": [
                            "Spray water under leaves.",
                            "Introduce ladybugs.",
                            "Avoid excess nitrogen."
                        ],
                        "pesticides": [
                            {"name": "Abamectin 1.8 EC", "dosage": "0.5 ml/L"},
                            {"name": "Fenpyroximate 5 EC", "dosage": "1 ml/L"}
                        ]
                    },
                    "Target_Spot": {
                        "management": [
                            "Use fungicides early.",
                            "Avoid overhead irrigation.",
                            "Crop rotation helps."
                        ],
                        "pesticides": [
                            {"name": "Azoxystrobin", "dosage": "2 ml/L"},
                            {"name": "Mancozeb", "dosage": "2.5 g/L"}
                        ]
                    },
                    "Tomato_Yellow_Leaf_Curl_Virus": {
                        "management": [
                            "Remove infected plants.",
                            "Use yellow sticky traps.",
                            "Use resistant varieties."
                        ],
                        "pesticides": [
                            {"name": "Imidacloprid", "dosage": "0.3 ml/L"},
                            {"name": "Thiamethoxam", "dosage": "0.25 g/L"}
                        ]
                    },
                    "Tomato_mosaic_virus": {
                        "management": [
                            "Disinfect tools.",
                            "Avoid smoking while handling plants.",
                            "Use virus-free seeds."
                        ],
                        "pesticides": [
                            {"name": "Control aphids using Imidacloprid", "dosage": "0.3 ml/L"}
                        ]
                    },
                    "healthy": {
                        "management": [
                            "Maintain good nutrition.",
                            "Regular inspection.",
                            "Ensure proper sunlight."
                        ],
                        "pesticides": []
                    },
                    "powdery_mildew": {
                        "management": [
                            "Improve airflow.",
                            "Remove heavily infected leaves."
                        ],
                        "pesticides": [
                            {"name": "Sulfur fungicide", "dosage": "As per label"},
                            {"name": "Neem oil", "dosage": "As per label"}
                        ]
                    }
                }

                info = disease_info.get(label, None)

                if info:
                    for tip in info["management"]:
                        st.write(f"- {tip}")

                    st.write("")
                    if info["pesticides"]:
                        for p in info["pesticides"]:
                            st.write(f"- **{p['name']}** — {p['dosage']}")
                    else:
                        st.write("- None required.")
                else:
                    st.write("⚠ No management info for this disease.")

                st.write("---")
                st.caption("Mask uses simple HSV brown-spot detection (same as your TensorFlow script).")

# Footer
st.write("")
st.write("💡 Tip: Put your model next to the script for fastest loading.")
