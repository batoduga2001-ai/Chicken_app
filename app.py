import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import os

# -----------------------------
# PAGE CONFIG (must be FIRST Streamlit command)
# -----------------------------
st.set_page_config(page_title="Chicken Detection App", page_icon="🐔")

st.title("🐔 Chicken Detection App")
st.write("Upload an image and the model will detect chickens.")

# -----------------------------
# LOAD MODEL (cached)
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # make sure best.pt is in same folder

model = load_model()

# -----------------------------
# UPLOAD IMAGE
# -----------------------------
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file)

    st.subheader("Original Image")
    st.image(image, use_container_width=True)

    # -----------------------------
    # Save image safely (TEMP FILE FIX)
    # -----------------------------
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        image_path = tmp.name

    # -----------------------------
    # RUN YOLO PREDICTION
    # -----------------------------
    with st.spinner("Detecting..."):
        results = model(image_path)

    result = results[0]

    # -----------------------------
    # Annotated image
    # -----------------------------
    annotated_img = result.plot()

    # YOLO already returns RGB sometimes depending on version
    # so we avoid double flipping issues
    annotated_img = np.array(annotated_img)

    # -----------------------------
    # SHOW RESULT
    # -----------------------------
    st.subheader("Detected Image")
    st.image(annotated_img, use_container_width=True)

    # -----------------------------
    # DETECTIONS INFO
    # -----------------------------
    st.subheader("Detections")

    boxes = result.boxes

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            st.write(f"🐔 Class: {model.names[cls]} | Confidence: {conf:.2f}")
    else:
        st.write("No objects detected.")

    # cleanup temp file
    os.remove(image_path)