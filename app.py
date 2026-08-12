import os
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Intelligent Document Understanding System",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Intelligent Document Understanding System")
st.write("Upload an image and process it using OCR and YOLO.")

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

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    os.makedirs("input", exist_ok=True)

    image_path = os.path.join(
        "input",
        "current_image.jpg"
    )

    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.session_state.image_path = image_path
    st.session_state.processed = False
    st.session_state.text = ""
    st.session_state.detections = []
    st.session_state.detection_image = None

    st.image(
        uploaded_file,
        caption="Uploaded Image",
        width="stretch"
    )

    if st.button(
        "🚀 Process Image",
        width="stretch"
    ):

        # OCR
        try:

            from ocr import extract_text

            with st.spinner("📝 Extracting text..."):

                image = Image.open(
                    image_path
                ).convert("RGB")

                text = extract_text(image)

            if text is None:
                text = ""

            st.session_state.text = str(text).strip()

            st.success("✅ OCR completed")

        except Exception as e:

            st.error(
                "OCR error: " + str(e)
            )

        # YOLO
        try:

            from detection import detect_objects

            with st.spinner("🔍 Detecting objects..."):

                detections, annotated_image = detect_objects(
                    image_path
                )

            if detections is None:
                detections = []

            st.session_state.detections = detections

            if annotated_image is not None:

                output_path = os.path.join(
                    "input",
                    "detected_current.jpg"
                )

                try:

                    import cv2

                    cv2.imwrite(
                        output_path,
                        annotated_image
                    )

                    st.session_state.detection_image = output_path

                except Exception:

                    st.session_state.detection_image = None

            st.success("✅ YOLO detection completed")

        except Exception as e:

            st.error(
                "YOLO error: " + str(e)
            )

        st.session_state.processed = True

if st.session_state.processed:

    st.divider()

    st.header("📄 Extracted Text")

    if st.session_state.text:

        st.text_area(
            "OCR Result",
            st.session_state.text,
            height=250
        )

    else:

        st.info(
            "No readable text was detected."
        )

    st.header("🔍 Detected Objects")

    if st.session_state.detections:

        for obj in st.session_state.detections:

            name = str(
                obj.get("class", "Unknown")
            )

            confidence = obj.get(
                "confidence",
                0
            )

            st.write(
                name
                + " — Confidence: "
                + str(round(float(confidence), 2))
            )

    else:

        st.info(
            "No trained objects were detected."
        )

    if st.session_state.detection_image:

        st.header("🖼️ Detection Result")

        st.image(
            st.session_state.detection_image,
            caption="YOLO Detection",
            width="stretch"
        )

    st.divider()

    st.header("💬 Ask Questions")

    question = st.text_input(
        "Ask a question about the image"
    )

    if st.button("🔎 Ask Question"):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            try:

                from vqa import answer_question

                with st.spinner(
                    "🤖 Analyzing image..."
                ):

                    answer = answer_question(
                        st.session_state.image_path,
                        question,
                        document_text=st.session_state.text,
                        detections=st.session_state.detections
                    )

                if answer is None:
                    answer = "I could not determine the answer."

                st.success(
                    str(answer)
                )

            except Exception as e:

                st.error(
                    "VQA error: " + str(e)
                )
