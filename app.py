import streamlit as st
import cv2
import numpy as np
from PIL import Image

from ocr import extract_text
from detection import detect_objects
from vqa import answer_question


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Intelligent Document System",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Intelligent Document System")
st.write(
    "Upload an image, extract text, detect objects, "
    "and ask questions about the image."
)


# ============================================================
# SESSION STATE
# ============================================================

if "image_id" not in st.session_state:
    st.session_state.image_id = None

if "image" not in st.session_state:
    st.session_state.image = None

if "ocr_text" not in st.session_state:
    st.session_state.ocr_text = ""

if "detections" not in st.session_state:
    st.session_state.detections = []

if "annotated_image" not in st.session_state:
    st.session_state.annotated_image = None


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload your document/image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ],
    accept_multiple_files=False
)


# ============================================================
# PROCESS NEW IMAGE
# ============================================================

if uploaded_file is not None:

    try:

        file_bytes = uploaded_file.getvalue()

        # Unique ID for current image
        current_id = (
            uploaded_file.name,
            len(file_bytes),
            hash(file_bytes)
        )

        # ----------------------------------------------------
        # ONLY PROCESS IF IMAGE CHANGED
        # ----------------------------------------------------

        if current_id != st.session_state.image_id:

            # Clear previous image data
            st.session_state.image_id = current_id
            st.session_state.image = None
            st.session_state.ocr_text = ""
            st.session_state.detections = []
            st.session_state.annotated_image = None

            # ------------------------------------------------
            # DECODE IMAGE
            # ------------------------------------------------

            image_array = np.frombuffer(
                file_bytes,
                dtype=np.uint8
            )

            image_bgr = cv2.imdecode(
                image_array,
                cv2.IMREAD_COLOR
            )

            if image_bgr is None:

                st.error(
                    "❌ Could not read this image."
                )

                st.stop()

            # ------------------------------------------------
            # RGB IMAGE
            # ------------------------------------------------

            image_rgb = cv2.cvtColor(
                image_bgr,
                cv2.COLOR_BGR2RGB
            )

            st.session_state.image = image_rgb

            # ------------------------------------------------
            # OCR
            # ------------------------------------------------

            with st.spinner(
                "🔍 Extracting text..."
            ):

                text, _ = extract_text(
                    image_bgr
                )

            st.session_state.ocr_text = text

            # ------------------------------------------------
            # YOLO
            # ------------------------------------------------

            with st.spinner(
                "🎯 Detecting objects..."
            ):

                detections, annotated = (
                    detect_objects(image_bgr)
                )

            st.session_state.detections = detections

            st.session_state.annotated_image = (
                annotated
            )

        # ====================================================
        # DISPLAY IMAGES
        # ====================================================

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("🖼️ Original Image")

            st.image(
                st.session_state.image,
                use_container_width=True
            )

        with col2:

            st.subheader("🎯 YOLO Detection")

            annotated_rgb = cv2.cvtColor(
                st.session_state.annotated_image,
                cv2.COLOR_BGR2RGB
            )

            st.image(
                annotated_rgb,
                use_container_width=True
            )


        # ====================================================
        # OCR RESULT
        # ====================================================

        st.divider()

        st.subheader("📝 Extracted Text")

        if st.session_state.ocr_text.strip():

            st.text_area(
                "OCR Result",
                value=st.session_state.ocr_text,
                height=250
            )

        else:

            st.info(
                "No readable text was detected."
            )


        # ====================================================
        # YOLO RESULTS
        # ====================================================

        st.subheader("🎯 Detected Objects")

        detections = st.session_state.detections

        if detections:

            for item in detections:

                name = item.get(
                    "class",
                    "Unknown"
                )

                confidence = float(
                    item.get(
                        "confidence",
                        0
                    )
                )

                st.write(
                    f"**{name}** — "
                    f"{confidence:.1%}"
                )

        else:

            st.info(
                "No objects were detected."
            )


        # ====================================================
        # QUESTION ANSWERING
        # ====================================================

        st.divider()

        st.subheader(
            "💬 Ask a Question About This Image"
        )

        question = st.text_input(
            "Your question",
            placeholder=(
                "Example: What is written in the document?"
            )
        )

        if st.button(
            "🤖 Get Answer",
            type="primary"
        ):

            if not question.strip():

                st.warning(
                    "Please enter a question first."
                )

            else:

                with st.spinner(
                    "🤖 Analyzing the image..."
                ):

                    answer = answer_question(
                        st.session_state.image,
                        question,
                        st.session_state.ocr_text,
                        st.session_state.detections
                    )

                if answer:

                    st.success(
                        answer
                    )

                else:

                    st.warning(
                        "I could not determine "
                        "a reliable answer."
                    )


    except Exception as error:

        st.error(
            "❌ Something went wrong while "
            "processing this image."
        )

        st.exception(error)
