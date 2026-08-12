import os
import streamlit as st
from PIL import Image


st.set_page_config(
    page_title="Intelligent Document Understanding System",
    page_icon="📄",
    layout="wide"
)


st.title("📄 Intelligent Document Understanding System")
st.write(
    "Upload an image, extract text, detect objects, "
    "and ask questions about the image."
)


# ============================================================
# SESSION STATE
# ============================================================

if "processed" not in st.session_state:
    st.session_state.processed = False

if "text" not in st.session_state:
    st.session_state.text = ""

if "detections" not in st.session_state:
    st.session_state.detections = []

if "image_path" not in st.session_state:
    st.session_state.image_path = ""

if "detection_image" not in st.session_state:
    st.session_state.detection_image = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "uploaded_id" not in st.session_state:
    st.session_state.uploaded_id = ""


# ============================================================
# UPLOAD
# ============================================================

st.header("📤 Upload Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    current_id = (
        uploaded_file.name
        + "_"
        + str(uploaded_file.size)
    )

    # --------------------------------------------------------
    # NEW IMAGE
    # --------------------------------------------------------

    if current_id != st.session_state.uploaded_id:

        st.session_state.uploaded_id = current_id
        st.session_state.processed = False
        st.session_state.text = ""
        st.session_state.detections = []
        st.session_state.detection_image = None
        st.session_state.chat_history = []

        os.makedirs("input", exist_ok=True)

        extension = os.path.splitext(
            uploaded_file.name
        )[1].lower()

        if extension not in [".jpg", ".jpeg", ".png"]:
            extension = ".jpg"

        image_path = os.path.join(
            "input",
            "current_image" + extension
        )

        with open(image_path, "wb") as file:
            file.write(uploaded_file.getbuffer())

        st.session_state.image_path = image_path


    # --------------------------------------------------------
    # SHOW IMAGE
    # --------------------------------------------------------

    uploaded_file.seek(0)

    st.image(
        uploaded_file,
        caption="Uploaded Image",
        use_container_width=True
    )


    # ========================================================
    # PROCESS BUTTON
    # ========================================================

    if not st.session_state.processed:

        if st.button(
            "🚀 Process Image",
            use_container_width=True
        ):

            image_path = st.session_state.image_path


            # =================================================
            # PREPROCESSING
            # =================================================

            try:

                from preprocessing import preprocess_image

                with st.spinner(
                    "🔄 Preprocessing image..."
                ):

                    processed_image = preprocess_image(
                        image_path
                    )

                processed_path = os.path.join(
                    "input",
                    "preprocessed_current.jpg"
                )

                processed_image.save(
                    processed_path
                )

                st.success(
                    "✅ Preprocessing completed"
                )

            except Exception as error:

                st.error(
                    "❌ Preprocessing failed: "
                    + str(error)
                )

                st.stop()


            # =================================================
            # OCR
            # =================================================

            try:

                from ocr import extract_text

                with st.spinner(
                    "📝 Extracting text with EasyOCR..."
                ):

                    image = Image.open(
                        image_path
                    ).convert("RGB")

                    extracted_text = extract_text(
                        image
                    )

                if extracted_text is None:
                    extracted_text = ""

                st.session_state.text = str(
                    extracted_text
                ).strip()

                st.success(
                    "✅ OCR completed"
                )

            except Exception as error:

                st.error(
                    "❌ OCR failed: "
                    + str(error)
                )

                st.stop()


            # =================================================
            # YOLO
            # =================================================

            try:

                from detection import detect_objects

                with st.spinner(
                    "🔍 Detecting objects with YOLO..."
                ):

                    result = detect_objects(
                        image_path
                    )

                detections = result[0]
                annotated_image = result[1]

                if detections is None:
                    detections = []

                st.session_state.detections = (
                    detections
                )


                if annotated_image is not None:

                    try:

                        output_path = os.path.join(
                            "input",
                            "detected_current.jpg"
                        )

                        import cv2

                        cv2.imwrite(
                            output_path,
                            annotated_image
                        )

                        st.session_state.detection_image = (
                            output_path
                        )

                    except Exception:

                        st.session_state.detection_image = None


                st.success(
                    "✅ YOLO detection completed"
                )

            except Exception as error:

                st.error(
                    "❌ YOLO detection failed: "
                    + str(error)
                )

                st.stop()


            # =================================================
            # COMPLETE
            # =================================================

            st.session_state.processed = True

            st.success(
                "🎉 Image processing completed!"
            )

            st.rerun()


# ============================================================
# RESULTS
# ============================================================

if st.session_state.processed:

    st.divider()

    st.header("📄 Extracted Text")

    if st.session_state.text:

        st.text_area(
            "EasyOCR Result",
            st.session_state.text,
            height=250
        )

    else:

        st.info(
            "No readable text was detected."
        )


    # ========================================================
    # DETECTED OBJECTS
    # ========================================================

    st.header("🔍 Detected Objects")

    if st.session_state.detections:

        for obj in st.session_state.detections:

            name = str(
                obj.get(
                    "class",
                    "Unknown"
                )
            )

            confidence = obj.get(
                "confidence",
                0
            )

            try:

                confidence = float(
                    confidence
                )

            except Exception:

                confidence = 0.0

            st.write(
                name
                + " — Confidence: "
                + str(
                    round(
                        confidence,
                        2
                    )
                )
            )

    else:

        st.info(
            "No trained objects were detected."
        )


    # ========================================================
    # YOLO IMAGE
    # ========================================================

    if st.session_state.detection_image:

        st.header(
            "🖼️ YOLO Detection Result"
        )

        st.image(
            st.session_state.detection_image,
            caption="Detected Objects",
            use_container_width=True
        )


    # ========================================================
    # CHATBOT
    # ========================================================

    st.divider()

    st.header(
        "💬 Image Question Answering"
    )

    st.info(
        "Ask a question about the uploaded image."
    )


    # ========================================================
    # CHAT HISTORY
    # ========================================================

    for chat in st.session_state.chat_history:

        st.markdown("### ❓ Question")

        st.write(
            chat["question"]
        )

        st.markdown("### 💡 Answer")

        st.write(
            chat["answer"]
        )

        st.divider()


    # ========================================================
    # QUESTION FORM
    # ========================================================

    with st.form(
        "question_form",
        clear_on_submit=True
    ):

        question = st.text_input(
            "Ask your question:",
            placeholder="Example: What is shown in the image?"
        )

        ask_button = st.form_submit_button(
            "🔎 Ask Question",
            use_container_width=True
        )


    # ========================================================
    # QUESTION ANSWERING
    # ========================================================

    if ask_button:

        question = question.strip()

        if not question:

            st.warning(
                "Please enter a question."
            )

        else:

            try:

                # VQA is imported ONLY when a question
                # is actually asked. This prevents the
                # model from loading during app startup.

                from vqa import answer_question

                with st.spinner(
                    "🤖 AI is analyzing the image..."
                ):

                    answer = answer_question(
                        st.session_state.image_path,
                        question,
                        document_text=(
                            st.session_state.text
                        ),
                        detections=(
                            st.session_state.detections
                        )
                    )

                if answer is None:

                    answer = (
                        "I could not determine a "
                        "reliable answer from the image."
                    )

                answer = str(
                    answer
                ).strip()

                if not answer:

                    answer = (
                        "I could not determine a "
                        "reliable answer from the image."
                    )

                st.session_state.chat_history.append(
                    {
                        "question": question,
                        "answer": answer
                    }
                )

                st.rerun()

            except Exception as error:

                st.error(
                    "❌ Question answering failed: "
                    + str(error)
                )
