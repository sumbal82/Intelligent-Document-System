import streamlit as st
import cv2
import numpy as np
from PIL import Image

from ocr import extract_text


# ==============================
# PAGE SETTINGS
# ==============================

st.set_page_config(
    page_title="Intelligent Document System",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Intelligent Document System")
st.write("Upload an image and extract its text using OCR.")


# ==============================
# IMAGE UPLOAD
# ==============================

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"]
)


# ==============================
# PROCESS IMAGE
# ==============================

if uploaded_file is not None:

    try:

        # Read uploaded file
        file_bytes = uploaded_file.getvalue()

        # Convert bytes to numpy array
        image_array = np.frombuffer(
            file_bytes,
            dtype=np.uint8
        )

        # Decode image
        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if image is None:
            st.error("❌ Could not read this image.")
            st.stop()

        # Convert BGR -> RGB for Streamlit
        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # ==============================
        # DISPLAY IMAGE
        # ==============================

        st.subheader("🖼️ Uploaded Image")

        st.image(
            image_rgb,
            use_container_width=True
        )

        # ==============================
        # OCR
        # ==============================

        with st.spinner("🔍 Extracting text..."):

            text, results = extract_text(image)

        st.subheader("📝 Extracted Text")

        if text.strip():

            st.text_area(
                "OCR Result",
                value=text,
                height=250
            )

        else:

            st.warning(
                "No readable text was detected."
            )

    except Exception as e:

        st.error("❌ Error while processing image.")

        st.exception(e)
