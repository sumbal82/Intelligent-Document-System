import os
import tempfile

import cv2
import numpy as np
import streamlit as st

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


# ============================================================
# PAGE TITLE
# ============================================================

st.title("📄 Intelligent Document System")

st.write(
    "Upload an image → Preprocess → Extract Text → "
    "Detect Objects → Ask Any Question About the Image"
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
    "processed": False,
    "ocr_error": None,
    "detection_error": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# IMAGE DISPLAY HELPER
# ============================================================

def make_display_image(image):
    """
    Convert PIL / NumPy image into a clean uint8 RGB array
    that Streamlit can reliably display.
    """

    if image is None:
        return None

    if hasattr(image, "convert"):
        image = image.convert("RGB")
        image = np.asarray(image)

    else:
        image = np.asarray(image)

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2RGB
        )

    elif image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    return np.ascontiguousarray(image)


# ============================================================
# STEP 1 — UPLOAD IMAGE
# ============================================================

st.header("📤 Step 1 — Upload Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"],
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
    # ONLY PROCESS A NEW IMAGE
    # --------------------------------------------------------

    if current_id != st.session_state.image_id:

        st.session_state.image_id = current_id
        st.session_state.original_image = None
        st.session_state.processed_image = None
        st.session_state.ocr_text = ""
        st.session_state.detections = []
        st.session_state.annotated_image = None
        st.session_state.processed = False
        st.session_state.ocr_error = None
        st.session_state.detection_error = None

        temp_path = None

        try:

            # =================================================
            # SAVE TEMPORARY IMAGE
            # =================================================

            suffix = os.path.splitext(
                uploaded_file.name
            )[1].lower()

            if suffix not in [
                ".jpg",
                ".jpeg",
                ".png",
                ".webp"
            ]:
                suffix = ".jpg"

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as temp_file:

                temp_file.write(file_bytes)
                temp_path = temp_file.name


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
                    "❌ The uploaded file could not be read as an image."
                )

                st.session_state.processed = False

            else:

                # =================================================
                # ORIGINAL IMAGE
                # =================================================

                image_rgb = cv2.cvtColor(
                    image_bgr,
                    cv2.COLOR_BGR2RGB
                )

                image_rgb = make_display_image(
                    image_rgb
                )

                st.session_state.original_image = image_rgb


                # =================================================
                # STEP 2 — PREPROCESSING
                # =================================================

                st.header(
                    "🔧 Step 2 — Image Preprocessing"
                )

                try:

                    with st.spinner(
                        "🔧 Preprocessing image..."
                    ):

                        processed_pil = preprocess_image(
                            temp_path
                        )

                    processed_display = make_display_image(
                        processed_pil
                    )

                    st.session_state.processed_image = (
                        processed_display
                    )

                    st.success(
                        "✅ Image preprocessing completed."
                    )

                    col1, col2 = st.columns(2)

                    with col1:

                        st.subheader("Original Image")

                        st.image(
                            st.session_state.original_image,
                            width="stretch",
                            output_format="PNG"
                        )

                    with col2:

                        st.subheader("Preprocessed Image")

                        st.image(
                            st.session_state.processed_image,
                            width="stretch",
                            output_format="PNG"
                        )

                except Exception as error:

                    st.error(
                        "❌ Image preprocessing failed."
                    )

                    st.exception(error)

                    st.session_state.processed = False


                # =================================================
                # STEP 3 — OCR
                # =================================================

                st.header(
                    "📝 Step 3 — Text Extraction"
                )

                try:

                    with st.spinner(
                        "📝 Extracting text with EasyOCR..."
                    ):

                        processed_bgr = cv2.cvtColor(
                            processed_display,
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

                        st.info(
                            "ℹ️ No readable text was detected."
                        )

                except Exception as error:

                    # OCR should NOT kill the entire application.
                    st.session_state.ocr_text = ""
                    st.session_state.ocr_error = str(error)

                    st.warning(
                        "⚠️ OCR could not be completed for this image. "
                        "The other image features can still continue."
                    )


                # =================================================
                # STEP 4 — YOLO
                # =================================================

                st.header(
                    "🎯 Step 4 — YOLO Object Detection"
                )

                try:

                    with st.spinner(
                        "🎯 Detecting objects..."
                    ):

                        detections, annotated = (
                            detect_objects(temp_path)
                        )

                    st.session_state.detections = (
                        detections or []
                    )

                    st.session_state.annotated_image = (
                        annotated
                    )

                    if annotated is not None:

                        annotated_rgb = cv2.cvtColor(
                            annotated,
                            cv2.COLOR_BGR2RGB
                        )

                        annotated_rgb = make_display_image(
                            annotated_rgb
                        )

                        st.image(
                            annotated_rgb,
                            caption="YOLO Detection Result",
                            width="stretch",
                            output_format="PNG"
                        )

                    if st.session_state.detections:

                        st.success(
                            f"✅ Detected "
                            f"{len(st.session_state.detections)} object(s)."
                        )

                        for item in st.session_state.detections:

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
                            "ℹ️ No objects were detected."
                        )

                except Exception as error:

                    st.session_state.detections = []
                    st.session_state.detection_error = str(error)

                    st.warning(
                        "⚠️ Object detection could not be completed."
                    )


                # =================================================
                # PROCESSING COMPLETE
                # =================================================

                st.session_state.processed = True

                st.success(
                    "🎉 Image processing completed successfully!"
                )


        except Exception as error:

            st.error(
                "❌ Unexpected error while processing the image."
            )

            st.exception(error)

            st.session_state.processed = False


        finally:

            # =================================================
            # DELETE TEMPORARY FILE
            # =================================================

            if temp_path is not None:

                try:
                    os.remove(temp_path)
                except Exception:
                    pass


# ============================================================
# STEP 5 — IMAGE CHATBOT
# ============================================================

if (
    st.session_state.processed
    and
    st.session_state.original_image is not None
):

    st.divider()

    st.header(
        "💬 Step 5 — Ask Any Question About This Image"
    )

    st.write(
        "Ask any question related to the uploaded image. "
        "Questions are not predefined."
    )

    question = st.text_input(
        "Your question",
        placeholder="Ask anything about this image...",
        key="question_input"
    )

    if st.button(
        "🤖 Get Answer",
        type="primary"
    ):

        if not question.strip():

            st.warning(
                "⚠️ Please enter a question."
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
                    "⚠️ I could not determine a reliable answer."
                )
