import re


def answer_from_text(question, document_text):
    """
    Answer questions using OCR extracted text.
    """

    if not question or not question.strip():
        return "Please enter a question."

    if not document_text or not document_text.strip():
        return "No readable text was found in this document."

    question_lower = question.lower().strip()

    lines = [
        line.strip()
        for line in document_text.splitlines()
        if line.strip()
    ]

    # ========================================================
    # TITLE QUESTIONS
    # ========================================================

    title_words = [
        "title",
        "document title",
        "name of the document",
        "document name"
    ]

    if any(word in question_lower for word in title_words):

        if lines:
            return lines[0]

    # ========================================================
    # GENERAL TEXT QUESTIONS
    # ========================================================

    if (
        "what is written" in question_lower
        or "what text" in question_lower
        or "text on the document" in question_lower
    ):

        return "\n".join(lines[:5])

    # ========================================================
    # SIGNATURE QUESTIONS
    # ========================================================

    if (
        "signature" in question_lower
        or "sign" in question_lower
    ):

        for line in lines:

            if (
                "sign" in line.lower()
                or "signature" in line.lower()
            ):
                return line

    # ========================================================
    # KEYWORD SEARCH
    # ========================================================

    keywords = re.findall(
        r"[a-zA-Z0-9]+",
        question_lower
    )

    stop_words = {
        "what",
        "is",
        "the",
        "a",
        "an",
        "of",
        "to",
        "who",
        "when",
        "where",
        "which",
        "tell",
        "me",
        "about",
        "this",
        "that",
        "was",
        "are",
        "do",
        "does",
        "did",
        "how",
        "why",
        "can",
        "you"
    }

    keywords = [
        word
        for word in keywords
        if word not in stop_words
    ]

    results = []

    for line in lines:

        line_lower = line.lower()

        score = 0

        for keyword in keywords:

            if keyword in line_lower:
                score += 1

        if score > 0:

            results.append(
                (score, line)
            )

    results.sort(
        key=lambda x: x[0],
        reverse=True
    )

    if results:

        return "\n".join(
            line
            for score, line in results[:3]
        )

    return (
        "Sorry, this information was not "
        "found in the document."
    )
