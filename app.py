import os
import tempfile

import streamlit as st
import cv2
import numpy as np
from PIL import Image

from preprocessing import preprocess_image
from ocr import extract_text
from detection import detect_objects
from chatbot import answer_from_text
from vqa import answer_question as vqa_answer


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
    "Upload → Preprocessing → OCR → YOLO → Question Answering"
)


# ============================================================
# SESSION STATE
# ============================================================

if "image_id" not in st.session_state:
    st.session_state.image_id = None

if "original_image" not in st.session_state:
    st.session_state.original_image = None

if "processed_image" not in st.session_state:
    st.session_state.processed_image = None

if "ocr_text" not in st.session_state:
    st.session_state.ocr_text = ""

if "detections" not in st.session_state:
    st.session_state.detections = []

if "annotated_image" not in st.session_state:
    st.session_state.annotated_image = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📤 Upload Document/Image",
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

    try:

        file_bytes = uploaded_file.getvalue()

        current_id = (
            uploaded_file.name,
            len(file_bytes),
            hash(file_bytes)
        )

        # ====================================================
        # NEW IMAGE
        # ====================================================

        if current_id != st.session_state.image_id:

            # Clear previous image context
            st.session_state.image_id = current_id
            st.session_state.original_image = None
            st.session_state.processed_image = None
            st.session_state.ocr_text = ""
            st.session_state.detections = []
            st.session_state.annotated_image = None
            st.session_state.chat_history = []

            # =================================================
            # READ IMAGE
            # =================================================

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

            # Original RGB image
            original_rgb = cv2.cvtColor(
                image_bgr,
                cv2.COLOR_BGR2RGB
            )

            st.session_state.original_image = (
                original_rgb
            )

            # =================================================
            # STEP 1 — PREPROCESSING
            # =================================================

            with st.spinner(
                "🔧 Step 1/3: Preprocessing image..."
            ):

                temp_path = None

                try:

                    with tempfile.NamedTemporaryFile(
                        suffix=".png",
                        delete=False
                    ) as temp_file:

                        temp_file.write(file_bytes)
                        temp_path = temp_file.name

                    processed_pil = preprocess_image(
                        temp_path
                    )

                finally:

                    if (
                        temp_path
                        and os.path.exists(temp_path)
                    ):
                        os.remove(temp_path)

            processed_rgb = np.array(
                processed_pil
            )

            st.session_state.processed_image = (
                processed_rgb
            )

            # Processed BGR for OCR and YOLO
            processed_bgr = cv2.cvtColor(
                processed_rgb,
                cv2.COLOR_RGB2BGR
            )

            # =================================================
            # STEP 2 — OCR
            # =================================================

            with st.spinner(
                "📝 Step 2/3: Extracting text..."
            ):

                text, _ = extract_text(
                    processed_bgr
                )

            st.session_state.ocr_text = text

            # =================================================
            # STEP 3 — YOLO
            # =================================================

            with st.spinner(
                "🎯 Step 3/3: Detecting objects..."
            ):

                detections, annotated = (
                    detect_objects(
                        processed_bgr
                    )
                )

            st.session_state.detections = (
                detections
            )

            st.session_state.annotated_image = (
                annotated
            )


        # ====================================================
        # STEP 1 DISPLAY — PREPROCESSING
        # ====================================================

        st.divider()

        st.header("1️⃣ Image & Preprocessing")

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
                st.session_state.processed_image,
                use_container_width=True
            )


        # ====================================================
        # STEP 2 DISPLAY — OCR
        # ====================================================

        st.divider()

        st.header("2️⃣ Text Extraction")

        if st.session_state.ocr_text.strip():

            st.text_area(
                "Extracted Text",
                value=st.session_state.ocr_text,
                height=250
            )

        else:

            st.warning(
                "No readable text was detected."
            )


        # ====================================================
        # STEP 3 DISPLAY — YOLO
        # ====================================================

        st.divider()

        st.header("3️⃣ YOLO Object Detection")

        if st.session_state.annotated_image is not None:

            annotated_rgb = cv2.cvtColor(
                st.session_state.annotated_image,
                cv2.COLOR_BGR2RGB
            )

            st.image(
                annotated_rgb,
                use_container_width=True
            )

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
        # STEP 4 — CHATBOT
        # ====================================================

        st.divider()

        st.header(
            "4️⃣ 💬 Ask Questions About This Image"
        )

        question = st.text_input(
            "Your question",
            placeholder=(
                "Example: What is the title of this document?"
            )
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
                    "🤖 Finding the answer..."
                ):

                    # First use OCR text
                    text_answer = answer_from_text(
                        question,
                        st.session_state.ocr_text
                    )

                    not_found_message = (
                        "Sorry, this information was not "
                        "found in the document."
                    )

                    # If OCR cannot answer,
                    # use image VQA
                    if (
                        not text_answer.strip()
                        or text_answer.strip()
                        == not_found_message
                    ):

                        final_answer = vqa_answer(
                            Image.fromarray(
                                st.session_state.original_image
                            ),
                            question,
                            st.session_state.ocr_text,
                            st.session_state.detections
                        )

                    else:

                        final_answer = text_answer

                # Save conversation
                st.session_state.chat_history.append(
                    (
                        question,
                        final_answer
                    )
                )


        # ====================================================
        # CHAT HISTORY
        # ====================================================

        if st.session_state.chat_history:

            st.subheader("💬 Conversation")

            for question_text, answer_text in (
                st.session_state.chat_history
            ):

                st.markdown(
                    f"**You:** {question_text}"
                )

                st.success(
                    answer_text
                )


    except Exception as error:

        st.error(
            "❌ Something went wrong while "
            "processing this image."
        )

        st.exception(error)
