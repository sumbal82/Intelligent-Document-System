import streamlit as st
import cv2
import numpy as np
import tempfile
import os

from PIL import Image

from preprocessing import preprocess_image
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
    "Upload an image → Preprocess → Extract Text → "
    "Detect Objects → Ask Questions"
)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "image_id": None,
    "original_image": None,
    "processed_image": None,
    "ocr_text": "",
    "detections": [],
    "annotated_image": None,
    "processed": False
}

for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# STEP 1 — UPLOAD
# ============================================================

st.header("📤 Step 1 — Upload Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ],
    accept_multiple_files=False
)


# ============================================================
# PROCESS IMAGE
# ============================================================

if uploaded_file is not None:

    file_bytes = uploaded_file.getvalue()

    current_id = (
        uploaded_file.name,
        len(file_bytes),
        hash(file_bytes)
    )

    # --------------------------------------------------------
    # New image
    # --------------------------------------------------------

    if current_id != st.session_state.image_id:

        # Clear old data

        st.session_state.image_id = current_id
        st.session_state.original_image = None
        st.session_state.processed_image = None
        st.session_state.ocr_text = ""
        st.session_state.detections = []
        st.session_state.annotated_image = None
        st.session_state.processed = False

        try:

            # ------------------------------------------------
            # Save uploaded image temporarily
            # ------------------------------------------------

            suffix = os.path.splitext(
                uploaded_file.name
            )[1]

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as temp_file:

                temp_file.write(file_bytes)
                temp_path = temp_file.name


            # ------------------------------------------------
            # Read image
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
                    "❌ Could not read the uploaded image."
                )

                st.stop()


            # =================================================
            # ORIGINAL IMAGE
            # =================================================

            image_rgb = cv2.cvtColor(
                image_bgr,
                cv2.COLOR_BGR2RGB
            )

            st.session_state.original_image = image_rgb


            # =================================================
            # STEP 2 — PREPROCESSING
            # =================================================

            st.header("🔧 Step 2 — Image Preprocessing")

            with st.spinner(
                "🔧 Preprocessing image..."
            ):

                processed_pil = preprocess_image(
                    temp_path
                )

            st.session_state.processed_image = (
                processed_pil
            )

            st.success(
                "✅ Image preprocessing completed."
            )

            col1, col2 = st.columns(2)

            with col1:

                st.subheader("Original Image")

                st.image(
                    st.session_state.original_image,
                    use_container_width=True
                )

            with col2:

                st.subheader("Preprocessed Image")

                st.image(
                    processed_pil,
                    use_container_width=True
                )


            # =================================================
            # STEP 3 — OCR
            # =================================================

            st.header("📝 Step 3 — Text Extraction")

            with st.spinner(
                "📝 Extracting text with OCR..."
            ):

                # Convert processed PIL → RGB numpy
                processed_rgb = np.array(
                    processed_pil
                )

                # RGB → BGR for OpenCV
                processed_bgr = cv2.cvtColor(
                    processed_rgb,
                    cv2.COLOR_RGB2BGR
                )

                text, _ = extract_text(
                    processed_bgr
                )

            st.session_state.ocr_text = (
                text.strip()
                if text
                else ""
            )

            if st.session_state.ocr_text:

                st.success(
                    "✅ Text extraction completed."
                )

                st.text_area(
                    "Extracted Text",
                    value=st.session_state.ocr_text,
                    height=250
                )

            else:

                st.warning(
                    "⚠️ No readable text was detected."
                )


            # =================================================
            # STEP 4 — YOLO
            # =================================================

            st.header("🎯 Step 4 — YOLO Object Detection")

            with st.spinner(
                "🎯 Detecting objects..."
            ):

                detections, annotated = (
                    detect_objects(temp_path)
                )

            st.session_state.detections = (
                detections
            )

            st.session_state.annotated_image = (
                annotated
            )

            if annotated is not None:

                annotated_rgb = cv2.cvtColor(
                    annotated,
                    cv2.COLOR_BGR2RGB
                )

                st.image(
                    annotated_rgb,
                    caption="YOLO Detection Result",
                    use_container_width=True
                )

            if detections:

                st.success(
                    f"✅ Detected {len(detections)} object(s)."
                )

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


            # =================================================
            # PROCESSING COMPLETE
            # =================================================

            st.session_state.processed = True

            st.success(
                "🎉 Image processing completed successfully!"
            )

            # Delete temporary file

            try:
                os.remove(temp_path)
            except Exception:
                pass


        except Exception as error:

            st.error(
                "❌ Error while processing image."
            )

            st.exception(error)

            st.session_state.processed = False


# ============================================================
# STEP 5 — CHATBOT
# ============================================================

if (
    st.session_state.processed
    and st.session_state.original_image is not None
):

    st.divider()

    st.header(
        "💬 Step 5 — Ask Questions About This Image"
    )

    st.write(
        "Image processing is complete. "
        "Now ask a question about the uploaded image."
    )

    question = st.text_input(
        "Your question",
        placeholder=(
            "Example: What is the title of this document?"
        ),
        key="question_input"
    )

    if st.button(
        "🤖 Get Answer",
        type="primary"
    ):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "🤖 Analyzing image and answering..."
            ):

                answer = answer_question(
                    st.session_state.original_image,
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
                    "I could not determine a reliable answer."
                )
