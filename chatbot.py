from vqa import answer_image_question


def answer_question(image, question, extracted_text, detections):
    """
    Answer any user question about the uploaded image.

    Uses:
    1. Original image through VQA
    2. OCR extracted text as additional context
    3. YOLO detections as additional context

    Questions are not predefined.
    """

    question = question.strip()

    if not question:
        return "Please enter a question about the image."

    # =====================================================
    # YOLO CONTEXT
    # =====================================================

    detected_objects = []

    if detections:

        for detection in detections:

            class_name = detection.get(
                "class",
                "Unknown"
            )

            if class_name not in detected_objects:
                detected_objects.append(class_name)


    # =====================================================
    # OCR CONTEXT
    # =====================================================

    ocr_text = ""

    if extracted_text:

        ocr_text = str(
            extracted_text
        ).strip()


    # =====================================================
    # ASK VQA
    # =====================================================

    try:

        answer = answer_image_question(
            image,
            question
        )

    except Exception as error:

        answer = ""

    
    # =====================================================
    # CLEAN VQA ANSWER
    # =====================================================

    if answer:

        answer = str(answer).strip()

        if answer:

            return answer


    # =====================================================
    # FALLBACK — YOLO
    # =====================================================

    question_lower = question.lower()

    object_question_words = [
        "object",
        "objects",
        "detect",
        "detected",
        "detection",
        "logo",
        "stamp",
        "signature",
        "sign",
        "symbol",
        "shape"
    ]

    if any(
        word in question_lower
        for word in object_question_words
    ):

        if detected_objects:

            return (
                "The detected objects are: "
                + ", ".join(detected_objects)
                + "."
            )


    # =====================================================
    # FALLBACK — OCR
    # =====================================================

    if ocr_text:

        text_question_words = [
            "text",
            "written",
            "write",
            "word",
            "words",
            "read",
            "name",
            "date",
            "number",
            "title",
            "document"
        ]

        if any(
            word in question_lower
            for word in text_question_words
        ):

            return (
                "The readable text extracted from "
                "the image is:\n\n"
                + ocr_text
            )


    # =====================================================
    # FINAL RESPONSE
    # =====================================================

    return (
        "I could not determine the answer "
        "from the image."
    )
