import streamlit as st
import os
import cv2
from PIL import Image

from preprocessing import preprocess_image
from ocr import extract_text
from detection import detect_objects
from vqa import answer_question


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Intelligent Document Understanding System",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Intelligent Document Understanding System")

st.write(
    "Upload any image, extract text, detect objects, "
    "and ask any question about the image."
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


# ============================================================
# UPLOAD IMAGE
# ============================================================

st.header("📤 Upload Document / Image")

uploaded_file = st.file_uploader(
    "Upload any image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    os.makedirs("input", exist_ok=True)

    # Keep original file extension
    extension = os.path.splitext(
        uploaded_file.name
    )[1].lower()

    if extension not in [".jpg", ".jpeg", ".png"]:
        extension = ".jpg"

    image_path = os.path.join(
        "input",
        "test" + extension
    )

    # Save uploaded image
    with open(image_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    st.session_state.image_path = image_path

    # Display uploaded image directly
    # instead of reading it again from disk
    uploaded_file.seek(0)

    st.image(
        uploaded_file,
        caption="Uploaded Image",
        use_container_width=True
    )

    st.success("✅ Image uploaded successfully")


    # ========================================================
    # PROCESS IMAGE
    # ========================================================

    if st.button(
        "🚀 Process Image",
        use_container_width=True
    ):

        # Reset previous results
        st.session_state.processed = False
        st.session_state.text = ""
        st.session_state.detections = []
        st.session_state.detection_image = None
        st.session_state.chat_history = []


        # ====================================================
        # PREPROCESSING
        # ====================================================

        try:

            with st.spinner(
                "🔄 Preprocessing image..."
            ):

                processed_image = preprocess_image(
                    image_path
                )

                processed_path = os.path.join(
                    "input",
                    "preprocessed_test.jpg"
                )

                processed_image.save(
                    processed_path
                )

            st.image(
                processed_image,
                caption="Preprocessed Image",
                use_container_width=True
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


        # ====================================================
        # OCR
        # ====================================================

        try:

            with st.spinner(
                "📝 Extracting text using EasyOCR..."
            ):

                image = cv2.imread(
                    image_path
                )

                if image is None:
                    raise Exception(
                        "Could not read uploaded image."
                    )

                image = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB
                )

                pil_image = Image.fromarray(
                    image
                )

                extracted_text = extract_text(
                    pil_image
                )

            if extracted_text is None:
                extracted_text = ""

            st.session_state.text = str(
                extracted_text
            ).strip()

            # Save OCR text
            with open(
                "input/extracted_text.txt",
                "w",
                encoding="utf-8"
            ) as file:

                file.write(
                    st.session_state.text
                )

            st.success(
                "✅ EasyOCR text extraction completed"
            )

        except Exception as error:

            st.error(
                "❌ OCR failed: "
                + str(error)
            )

            st.stop()


        # ====================================================
        # YOLO OBJECT DETECTION
        # ====================================================

        try:

            with st.spinner(
                "🔍 Detecting objects using YOLO..."
            ):

                detections, annotated_image = detect_objects(
                    image_path
                )

            if detections is None:
                detections = []

            st.session_state.detections = detections

            if annotated_image is not None:

                output_path = os.path.join(
                    "input",
                    "detected_test.jpg"
                )

                cv2.imwrite(
                    output_path,
                    annotated_image
                )

                st.session_state.detection_image = (
                    output_path
                )

            st.success(
                "✅ YOLO object detection completed"
            )

        except Exception as error:

            st.error(
                "❌ YOLO detection failed: "
                + str(error)
            )

            st.stop()


        # ====================================================
        # PROCESSING COMPLETE
        # ====================================================

        st.session_state.processed = True

        st.success(
            "🎉 Image processing completed successfully!"
        )

        st.rerun()


# ============================================================
# RESULTS
# ============================================================

if st.session_state.processed:

    st.divider()


    # ========================================================
    # EXTRACTED TEXT
    # ========================================================

    st.header("📄 Extracted Text")

    if st.session_state.text:

        st.text_area(
            "Text extracted by EasyOCR",
            st.session_state.text,
            height=250
        )

    else:

        st.warning(
            "⚠️ No readable text was extracted."
        )


    # ========================================================
    # DETECTED OBJECTS
    # ========================================================

    st.header("🔍 Detected Objects")

    if st.session_state.detections:

        for obj in st.session_state.detections:

            name = obj.get(
                "class",
                "Unknown"
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
                f"**{name}** — "
                f"Confidence: {confidence:.2f}"
            )

    else:

        st.info(
            "ℹ️ No trained objects were detected."
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
        "💬 Document & Image Chatbot"
    )

    st.info(
        "Ask any question related to the uploaded image."
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
    # QUESTION INPUT
    # ========================================================

    with st.form(
        "question_form",
        clear_on_submit=True
    ):

        question = st.text_input(
            "Ask anything about this image:",
            placeholder="Type any question..."
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
                "⚠️ Please enter a question."
            )

        else:

            try:

                q = question.lower().strip()


                # =================================================
                # FAST OCR QUESTIONS
                # =================================================

                text_keywords = [
                    "what text",
                    "what is written",
                    "what's written",
                    "read the text",
                    "read text",
                    "text in image",
                    "text on image",
                    "written in image",
                    "written on image",
                    "what does it say",
                    "what does the image say",
                    "extract text",
                    "show text",
                    "give me the text"
                ]

                is_text_question = any(
                    keyword in q
                    for keyword in text_keywords
                )


                if is_text_question:

                    if st.session_state.text:

                        answer = (
                            st.session_state.text
                        )

                    else:

                        answer = (
                            "No readable text was "
                            "detected in the image."
                        )


                # =================================================
                # FAST OBJECT QUESTIONS
                # =================================================

                elif any(
                    keyword in q
                    for keyword in [
                        "what objects",
                        "which objects",
                        "objects in image",
                        "objects are detected",
                        "what is detected",
                        "what are detected",
                        "detect objects",
                        "show detected objects"
                    ]
                ):

                    if st.session_state.detections:

                        names = []

                        for obj in st.session_state.detections:

                            name = str(
                                obj.get(
                                    "class",
                                    "Unknown"
                                )
                            )

                            if name not in names:

                                names.append(
                                    name
                                )

                        answer = (
                            "Detected objects: "
                            + ", ".join(names)
                        )

                    else:

                        answer = (
                            "No trained objects were "
                            "detected in the image."
                        )


                # =================================================
                # FAST OBJECT COUNT
                # =================================================

                elif (
                    "how many objects" in q
                    or "number of objects" in q
                    or "count objects" in q
                ):

                    answer = (
                        f"I detected "
                        f"{len(st.session_state.detections)} "
                        f"object(s)."
                    )


                # =================================================
                # FAST TEXT CHECK
                # =================================================

                elif (
                    "is there any text" in q
                    or "does the image have text" in q
                    or "does it contain text" in q
                ):

                    if st.session_state.text:

                        answer = (
                            "Yes, text was detected "
                            "in the image."
                        )

                    else:

                        answer = (
                            "No readable text was "
                            "detected in the image."
                        )


                # =================================================
                # GENERAL VISUAL QUESTIONS
                # =================================================

                else:

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


                # =================================================
                # CLEAN ANSWER
                # =================================================

                if answer is None:

                    answer = ""

                answer = str(
                    answer
                ).strip()


                if not answer:

                    answer = (
                        "I could not determine the answer "
                        "from the uploaded image."
                    )


                # =================================================
                # SAVE CHAT
                # =================================================

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
