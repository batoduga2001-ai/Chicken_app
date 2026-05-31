import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# -----------------------------
# LOAD MODEL (make sure best.pt is in same folder)
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Chicken Detection App", page_icon="🐔")

st.title("🐔 Chicken Detection App")
st.write("Upload an image and the model will detect chickens.")

# -----------------------------
# UPLOAD IMAGE
# -----------------------------
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file)

    st.subheader("Original Image")
    st.image(image, use_container_width=True)

    # Save temp image
    image_path = "temp.jpg"
    image.save(image_path)

    # -----------------------------
    # RUN YOLO PREDICTION
    # -----------------------------
    with st.spinner("Detecting..."):
        results = model(image_path)

    # Get first result
    result = results[0]

    # Annotated image
    annotated_img = result.plot()

    # Convert BGR → RGB (important fix)
    annotated_img = annotated_img[..., ::-1]

    # -----------------------------
    # SHOW RESULT
    # -----------------------------
    st.subheader("Detected Image")
    st.image(annotated_img, use_container_width=True)

    # Optional: show boxes info
    st.subheader("Detections")
    boxes = result.boxes

    if len(boxes) > 0:
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            st.write(f"Class: {model.names[cls]} | Confidence: {conf:.2f}")
    else:
        st.write("No objects detected.")