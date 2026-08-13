import streamlit as st
from PIL import Image

from preprocessing import preprocess_image
from ocr import extract_text
from detection import detect_objects
from chatbot import answer_question


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Intelligent Document Understanding System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main application */
    .stApp {
        background-color: #f5f7fb;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #172554, #2563eb);
        padding: 30px;
        border-radius: 18px;
        color: white;
        margin-bottom: 25px;
    }

    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 34px;
        font-weight: 700;
    }

    .main-header p {
        color: #e0e7ff;
        margin-top: 10px;
        margin-bottom: 0;
        font-size: 16px;
    }

    /* Section card */
    .section-card {
        background-color: white;
        padding: 22px;
        border-radius: 16px;
        margin-top: 20px;
        margin-bottom: 20px;
        border: 1px solid #e2e8f0;
    }

    .section-title {
        color: #172554;
        font-size: 23px;
        font-weight: 700;
        margin-top: 4px;
        margin-bottom: 8px;
    }

    .section-description {
        color: #64748b;
        font-size: 14px;
        margin-bottom: 5px;
    }

    /* Step badge */
    .step {
        display: inline-block;
        background-color: #dbeafe;
        color: #1d4ed8;
        padding: 7px 13px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 13px;
        margin-bottom: 10px;
    }

    /* Success box */
    .success-box {
        background-color: #ecfdf5;
        border-left: 5px solid #10b981;
        padding: 13px 16px;
        border-radius: 8px;
        color: #065f46;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    /* Question card */
    .question-card {
        background: linear-gradient(135deg, #eef2ff, #f8fafc);
        padding: 24px;
        border-radius: 16px;
        border: 1px solid #c7d2fe;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        border: none;
        background-color: #2563eb;
        color: white;
        font-weight: 700;
        padding: 12px;
        font-size: 16px;
    }

    .stButton > button:hover {
        background-color: #1d4ed8;
        color: white;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: white;
        padding: 15px;
        border-radius: 14px;
        border: 2px dashed #93c5fd;
    }

    /* Remove unnecessary top spacing */
    div[data-testid="stVerticalBlock"] {
        gap: 0.5rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-header">
        <h1>📄 Intelligent Document Understanding System</h1>
        <p>
            Upload images, preprocess them, extract text,
            detect trained objects and ask questions about the image.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "processed_images" not in st.session_state:
    st.session_state.processed_images = {}

if "processing_started" not in st.session_state:
    st.session_state.processing_started = False


# ============================================================
# STEP 1 — UPLOAD
# ============================================================

st.markdown(
    """
    <div class="section-card">
        <div class="step">STEP 1</div>
        <div class="section-title">📤 Upload Image(s)</div>
        <div class="section-description">
            Upload one or multiple images for analysis.
            The system can process documents, logos, signatures,
            stamps, symbols and shapes.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


uploaded_files = st.file_uploader(
    "Choose image files",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "webp"
    ],
    accept_multiple_files=True,
    label_visibility="collapsed"
)


# ============================================================
# DISPLAY UPLOADED IMAGES
# ============================================================

if uploaded_files:

    st.markdown("### 🖼️ Uploaded Images")

    number_of_columns = min(len(uploaded_files), 4)

    columns = st.columns(number_of_columns)

    for index, uploaded_file in enumerate(uploaded_files):

        try:

            image = Image.open(
                uploaded_file
            ).convert("RGB")

            with columns[index % number_of_columns]:

                st.image(
                    image,
                    caption=uploaded_file.name,
                    width="stretch"
                )

        except Exception as error:

            st.error(
                f"Could not open {uploaded_file.name}: {error}"
            )


    # ========================================================
    # PROCESS BUTTON
    # ========================================================

    st.markdown("")

    if st.button(
        "🚀 Process Image(s)",
        type="primary"
    ):

        # Clear previous results
        st.session_state.processed_images = {}

        st.session_state.processing_started = True

        progress = st.progress(0)

        total = len(uploaded_files)

        status_text = st.empty()


        # ====================================================
        # PROCESS EACH IMAGE
        # ====================================================

        for index, uploaded_file in enumerate(uploaded_files):

            status_text.write(
                f"Processing {uploaded_file.name}..."
            )

            try:

                image = Image.open(
                    uploaded_file
                ).convert("RGB")

            except Exception as error:

                st.error(
                    f"Could not open {uploaded_file.name}: {error}"
                )

                continue


            # =================================================
            # PREPROCESSING
            # =================================================

            try:

                processed_image = preprocess_image(
                    image
                )

            except Exception as error:

                st.warning(
                    f"Preprocessing failed for "
                    f"{uploaded_file.name}: {error}"
                )

                processed_image = image


            # =================================================
            # OCR
            # =================================================

            try:

                extracted_text = extract_text(
                    processed_image
                )

                if extracted_text is None:
                    extracted_text = ""

            except Exception as error:

                st.warning(
                    f"OCR failed for "
                    f"{uploaded_file.name}: {error}"
                )

                extracted_text = ""


            # =================================================
            # YOLO OBJECT DETECTION
            # =================================================

            try:

                detections, detected_image = detect_objects(
                    image,
                    confidence=0.10
                )

                if detections is None:
                    detections = []

            except Exception as error:

                st.warning(
                    f"YOLO failed for "
                    f"{uploaded_file.name}: {error}"
                )

                detections = []

                detected_image = image


            # =================================================
            # SAVE RESULTS
            # =================================================

            st.session_state.processed_images[
                uploaded_file.name
            ] = {
                "image": image,

                "processed_image": processed_image,

                "text": extracted_text,

                "detections": detections,

                "detected_image": detected_image
            }


            # =================================================
            # PROGRESS
            # =================================================

            progress.progress(
                int(
                    ((index + 1) / total) * 100
                )
            )


        status_text.empty()

        st.success(
            "✅ Processing completed successfully!"
        )


# ============================================================
# RESULTS
# ============================================================

if (
    st.session_state.processing_started
    and st.session_state.processed_images
):

    st.markdown("---")

    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">
                📊 Analysis Results
            </div>
            <div class="section-description">
                Select an image below to view preprocessing,
                OCR, YOLO detection and question-answering results.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # IMAGE SELECTOR
    # ========================================================

    image_names = list(
        st.session_state.processed_images.keys()
    )

    selected_image = st.selectbox(
        "Select image to view results",
        image_names
    )

    data = st.session_state.processed_images[
        selected_image
    ]


    # ========================================================
    # STEP 2 — PREPROCESSING
    # ========================================================

    st.markdown(
        """
        <div class="section-card">
            <div class="step">STEP 2</div>
            <div class="section-title">
                ⚙️ Image Preprocessing
            </div>
            <div class="section-description">
                The uploaded image is prepared before text extraction.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="success-box">
            ✓ Image preprocessing completed
        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # ORIGINAL + PREPROCESSED
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("#### 🖼️ Original Image")

        st.image(
            data["image"],
            width="stretch"
        )

    with col2:

        st.markdown("#### ⚙️ Preprocessed Image")

        st.image(
            data["processed_image"],
            width="stretch"
        )


    # ========================================================
    # STEP 3 — OCR
    # ========================================================

    st.markdown(
        """
        <div class="section-card">
            <div class="step">STEP 3</div>
            <div class="section-title">
                📝 Text Extraction
            </div>
            <div class="section-description">
                Text is extracted from the image using OCR.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


    extracted_text = data["text"]


    if extracted_text and str(extracted_text).strip():

        st.markdown(
            """
            <div class="success-box">
                ✓ OCR processing completed successfully
            </div>
            """,
            unsafe_allow_html=True
        )

        st.text_area(
            "Extracted Text",
            value=str(extracted_text),
            height=250
        )

    else:

        st.warning(
            "⚠️ No readable text was extracted from this image."
        )


    # ========================================================
    # STEP 4 — YOLO
    # ========================================================

    st.markdown(
        """
        <div class="section-card">
            <div class="step">STEP 4</div>
            <div class="section-title">
                🔍 YOLO Object Detection
            </div>
            <div class="section-description">
                Detect trained logos, signatures, stamps,
                symbols and shapes.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


    detections = data["detections"]


    if detections:

        st.markdown(
            """
            <div class="success-box">
                ✓ Object detection completed successfully
            </div>
            """,
            unsafe_allow_html=True
        )


        st.image(
            data["detected_image"],
            caption="YOLO Detection Result",
            width="stretch"
        )


        st.markdown("### 🔎 Detected Objects")


        for detection in detections:

            class_name = detection.get(
                "class",
                "Unknown"
            )

            confidence = detection.get(
                "confidence",
                0
            )

            box = detection.get(
                "box",
                []
            )


            try:

                confidence_text = f"{float(confidence):.2f}"

            except Exception:

                confidence_text = str(confidence)


            st.write(
                f"🔹 **{class_name}** "
                f"— Confidence: **{confidence_text}**"
            )


            if box:

                st.caption(
                    f"Bounding Box: {box}"
                )


    else:

        st.info(
            "ℹ️ No trained YOLO objects were detected "
            "in this image."
        )


    # ========================================================
    # STEP 5 — QUESTION ANSWERING
    # ========================================================

    st.markdown(
        """
        <div class="question-card">
            <div class="step">STEP 5</div>
            <div class="section-title">
                💬 Ask Questions About the Image
            </div>
            <div class="section-description">
                Ask any question related to the selected image.
                Questions are not predefined.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


    question = st.text_input(
        "Your question",
        placeholder="Ask anything about this image..."
    )


    if st.button(
        "💡 Ask Question",
        type="primary"
    ):

        if not question.strip():

            st.warning(
                "Please enter a question first."
            )

        else:

            with st.spinner(
                "Analyzing image and generating answer..."
            ):

                try:

                    answer = answer_question(
                        data["image"],
                        question,
                        data["text"],
                        data["detections"]
                    )

                except Exception as error:

                    answer = (
                        "Unable to generate answer: "
                        f"{error}"
                    )


            st.markdown("### 🤖 Answer")

            st.success(answer)
