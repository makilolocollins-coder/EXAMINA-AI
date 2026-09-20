# ============================================================
# EXAMINA AI
# THREE-UPLOAD EXAM MARKING SYSTEM
#
# 1. Typed question paper
# 2. Handwritten marking scheme
# 3. Handwritten student answer
#
# OCR:
#   Typed      -> TrOCR printed
#   Handwriting-> Makky07/Trocr
#
# Marking:
#   Verified OCR -> OpenAI Responses API
# ============================================================

import os
import re
import json
from io import BytesIO

import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

APP_TITLE = "Examina AI"

HANDWRITING_MODEL = "Makky07/Trocr"

# Printed-text model used for the typed question paper.
PRINTED_MODEL = "microsoft/trocr-base-printed"

DEFAULT_MARKING_MODEL = "gpt-5.6-luna"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📝",
    layout="wide",
)


# ============================================================
# SESSION STATE
# ============================================================

if "question_text" not in st.session_state:
    st.session_state.question_text = ""

if "scheme_text" not in st.session_state:
    st.session_state.scheme_text = ""

if "answer_text" not in st.session_state:
    st.session_state.answer_text = ""

if "result" not in st.session_state:
    st.session_state.result = None


# ============================================================
# HELPERS
# ============================================================

def clean_text(text):
    """Clean OCR output without destroying useful structure."""

    if not text:
        return ""

    text = str(text)

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove excessive spaces while preserving newlines.
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def get_device():
    """Return CUDA if available, otherwise CPU."""

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"

    except Exception:
        pass

    return "cpu"


# ============================================================
# TR OCR LOADER
# ============================================================

@st.cache_resource(show_spinner=False)
def load_trocr(model_name):

    import torch

    from transformers import (
        TrOCRProcessor,
        VisionEncoderDecoderModel,
    )

    processor = TrOCRProcessor.from_pretrained(
        model_name,
        use_fast=False,
    )

    model = VisionEncoderDecoderModel.from_pretrained(
        model_name
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    return processor, model, device


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def prepare_image(uploaded_file):

    from PIL import Image

    image = Image.open(
        BytesIO(
            uploaded_file.getvalue()
        )
    )

    image = image.convert("RGB")

    return image


# ============================================================
# SINGLE IMAGE OCR
# ============================================================

def trocr_image(
    image,
    model_name,
    max_new_tokens=128,
):

    import torch

    processor, model, device = load_trocr(
        model_name
    )

    pixel_values = processor(
        images=image,
        return_tensors="pt",
    ).pixel_values

    pixel_values = pixel_values.to(device)

    with torch.no_grad():

        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=max_new_tokens,
            num_beams=4,
        )

    text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0]

    return clean_text(text)


# ============================================================
# HANDWRITING LINE SEGMENTATION
# ============================================================

def segment_handwriting_lines(image):

    """
    Attempts to split a handwritten page into horizontal
    writing lines.

    This is important because TrOCR is primarily a text-line
    recognition model rather than a complete-page OCR engine.
    """

    import cv2
    import numpy as np
    from PIL import Image

    rgb = np.array(image)

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY,
    )

    # Mild blur reduces tiny noise.
    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0,
    )

    # Binary image.
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV
        + cv2.THRESH_OTSU,
    )

    height, width = binary.shape

    # Horizontal projection.
    projection = np.sum(
        binary > 0,
        axis=1,
    )

    # A row is considered part of writing when
    # enough foreground pixels occur.
    threshold = max(
        2,
        int(width * 0.005),
    )

    active = projection > threshold

    ranges = []

    start = None

    for y, value in enumerate(active):

        if value and start is None:

            start = y

        elif not value and start is not None:

            end = y

            if end - start >= 8:

                ranges.append(
                    (start, end)
                )

            start = None

    if start is not None:

        ranges.append(
            (start, height)
        )

    # Merge lines that are very close together.
    merged = []

    for start, end in ranges:

        if not merged:

            merged.append(
                [start, end]
            )

        else:

            previous = merged[-1]

            if start - previous[1] < 12:

                previous[1] = end

            else:

                merged.append(
                    [start, end]
                )

    # Padding around each line.
    padding_y = max(
        8,
        int(height * 0.01)
    )

    crops = []

    for start, end in merged:

        y1 = max(
            0,
            start - padding_y
        )

        y2 = min(
            height,
            end + padding_y
        )

        crop = rgb[
            y1:y2,
            0:width
        ]

        if crop.shape[0] < 15:
            continue

        crops.append(
            Image.fromarray(
                crop
            )
        )

    # If segmentation failed, use the complete image.
    if not crops:

        crops = [image]

    return crops


# ============================================================
# HANDWRITING OCR
# ============================================================

def handwriting_ocr(uploaded_file):

    image = prepare_image(
        uploaded_file
    )

    lines = segment_handwriting_lines(
        image
    )

    results = []

    progress = st.progress(
        0,
        text="Reading handwriting..."
    )

    total = len(lines)

    for index, line_image in enumerate(
        lines
    ):

        try:

            text = trocr_image(
                line_image,
                HANDWRITING_MODEL,
                max_new_tokens=128,
            )

            if text:

                results.append(
                    text
                )

        except Exception as error:

            results.append(
                f"[OCR ERROR: {error}]"
            )

        progress.progress(
            (index + 1) / total,
            text=(
                f"Reading handwriting "
                f"{index + 1}/{total}"
            )
        )

    progress.empty()

    return clean_text(
        "\n".join(results)
    )


# ============================================================
# TYPED QUESTION PAPER OCR
# ============================================================

def typed_ocr(uploaded_file):

    image = prepare_image(
        uploaded_file
    )

    # Printed TrOCR is still a line recognition model.
    # We therefore use the same segmentation approach,
    # but with the printed checkpoint.
    lines = segment_handwriting_lines(
        image
    )

    results = []

    progress = st.progress(
        0,
        text="Reading typed question paper..."
    )

    total = len(lines)

    for index, line_image in enumerate(
        lines
    ):

        try:

            text = trocr_image(
                line_image,
                PRINTED_MODEL,
                max_new_tokens=256,
            )

            if text:

                results.append(
                    text
                )

        except Exception as error:

            results.append(
                f"[OCR ERROR: {error}]"
            )

        progress.progress(
            (index + 1) / total,
            text=(
                f"Reading question paper "
                f"{index + 1}/{total}"
            )
        )

    progress.empty()

    return clean_text(
        "\n".join(results)
    )


# ============================================================
# QUESTION EXTRACTION
# ============================================================

def extract_questions(text):

    text = clean_text(text)

    if not text:
        return []

    # Common examination numbering:
    #
    # 1.
    # 1)
    # 1:
    # Q1.
    # Q1)
    # Question 1.
    #
    pattern = re.compile(
        r"""
        (?=
            ^|\n
        )
        \s*
        (?:
            question\s*
            |q\s*
        )?
        (\d{1,3})
        \s*
        [\.\):\-]
        \s*
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    matches = list(
        pattern.finditer(text)
    )

    if not matches:

        return [
            {
                "number": 1,
                "text": text,
            }
        ]

    questions = []

    for index, match in enumerate(matches):

        number = int(
            match.group(1)
        )

        start = match.end()

        if index + 1 < len(matches):

            end = matches[
                index + 1
            ].start()

        else:

            end = len(text)

        body = text[
            start:end
        ].strip()

        if body:

            questions.append(
                {
                    "number": number,
                    "text": body,
                }
            )

    return questions


# ============================================================
# MARK EXTRACTION
# ============================================================

def extract_marks(text):

    patterns = [

        r"\[\s*(\d+(?:\.\d+)?)\s*marks?\s*\]",

        r"\(\s*(\d+(?:\.\d+)?)\s*marks?\s*\)",

        r"\b(\d+(?:\.\d+)?)\s*marks?\b",

    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE,
        )

        if match:

            try:

                return float(
                    match.group(1)
                )

            except Exception:

                pass

    return None


# ============================================================
# OPENAI CLIENT
# ============================================================

def get_api_key():

    # Streamlit Cloud secrets first.
    try:

        if "OPENAI_API_KEY" in st.secrets:

            return st.secrets[
                "OPENAI_API_KEY"
            ]

    except Exception:

        pass

    # Local environment fallback.
    return os.getenv(
        "OPENAI_API_KEY"
    )


def get_marking_model():

    try:

        if "EXAMINA_MARKING_MODEL" in st.secrets:

            return st.secrets[
                "EXAMINA_MARKING_MODEL"
            ]

    except Exception:

        pass

    return os.getenv(
        "EXAMINA_MARKING_MODEL",
        DEFAULT_MARKING_MODEL,
    )


# ============================================================
# AI MARKING
# ============================================================

def mark_exam(
    question_paper,
    marking_scheme,
    student_answer,
):

    api_key = get_api_key()

    if not api_key:

        raise RuntimeError(
            "OPENAI_API_KEY is not configured."
        )

    from openai import OpenAI

    client = OpenAI(
        api_key=api_key
    )

    questions = extract_questions(
        question_paper
    )

    if not questions:

        raise RuntimeError(
            "No questions could be identified "
            "from the question paper."
        )

    # --------------------------------------------------------
    # Build question information.
    # --------------------------------------------------------

    question_data = []

    for question in questions:

        question_data.append(
            {
                "number": question[
                    "number"
                ],
                "question": question[
                    "text"
                ],
                "maximum_marks": (
                    extract_marks(
                        question["text"]
                    )
                ),
            }
        )

    payload = {

        "question_paper": question_paper,

        "questions": question_data,

        "marking_scheme": marking_scheme,

        "student_answer": student_answer,

    }

    prompt = f"""
You are the examination marking engine for Examina AI.

Your task is to mark the student's answers using ONLY
the supplied examination question paper and marking scheme.

IMPORTANT RULES:

1. Do not invent marking criteria.
2. The uploaded marking scheme is the primary authority.
3. Equivalent wording should receive credit when it expresses
   the same valid concept.
4. Award partial marks when the marking scheme supports
   partial credit.
5. Do not award marks merely because an answer sounds plausible.
6. Do not penalize grammar unless it changes the meaning.
7. Do not give more marks than the maximum available.
8. If the OCR appears ambiguous, identify the ambiguity.
9. Give a short explanation for every score.
10. Return ONLY valid JSON.

For every question return:

- question_number
- question
- maximum_marks
- marks_awarded
- decision
- reason
- matched_points
- missing_points
- feedback

The decision must be one of:

"full"
"partial"
"zero"

Then return:

- total_marks_awarded
- total_marks_available
- percentage
- overall_feedback

EXAMINATION DATA:

{json.dumps(payload, ensure_ascii=False, indent=2)}
"""

    response = client.responses.create(
        model=get_marking_model(),
        input=prompt,
    )

    raw = response.output_text

    # --------------------------------------------------------
    # Extract JSON safely.
    # --------------------------------------------------------

    try:

        result = json.loads(
            raw
        )

    except Exception:

        match = re.search(
            r"\{.*\}",
            raw,
            re.DOTALL,
        )

        if not match:

            raise RuntimeError(
                "The marking model did not return valid JSON.\n\n"
                + raw
            )

        result = json.loads(
            match.group(0)
        )

    return result


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(result):

    st.header(
        "Marking Result"
    )

    total = result.get(
        "total_marks_awarded",
        0,
    )

    maximum = result.get(
        "total_marks_available",
        0,
    )

    percentage = result.get(
        "percentage",
        0,
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Score",
            f"{total} / {maximum}",
        )

    with col2:

        st.metric(
            "Percentage",
            f"{percentage}%",
        )

    with col3:

        st.metric(
            "Questions",
            len(
                result.get(
                    "questions",
                    []
                )
            ),
        )

    st.divider()

    questions = result.get(
        "questions",
        []
    )

    for item in questions:

        number = item.get(
            "question_number",
            "?",
        )

        marks = item.get(
            "marks_awarded",
            0,
        )

        maximum_marks = item.get(
            "maximum_marks",
            0,
        )

        with st.expander(
            f"Question {number} — "
            f"{marks}/{maximum_marks}",
            expanded=True,
        ):

            st.markdown(
                "**Question**"
            )

            st.write(
                item.get(
                    "question",
                    "",
                )
            )

            st.markdown(
                "**Decision**"
            )

            st.write(
                item.get(
                    "decision",
                    "",
                )
            )

            st.markdown(
                "**Reason**"
            )

            st.write(
                item.get(
                    "reason",
                    "",
                )
            )

            matched = item.get(
                "matched_points",
                []
            )

            if matched:

                st.markdown(
                    "**Points credited**"
                )

                for point in matched:

                    st.write(
                        f"✓ {point}"
                    )

            missing = item.get(
                "missing_points",
                []
            )

            if missing:

                st.markdown(
                    "**Missing points**"
                )

                for point in missing:

                    st.write(
                        f"• {point}"
                    )

            feedback = item.get(
                "feedback",
                "",
            )

            if feedback:

                st.markdown(
                    "**Feedback**"
                )

                st.info(
                    feedback
                )

    overall = result.get(
        "overall_feedback",
        "",
    )

    if overall:

        st.subheader(
            "Overall Feedback"
        )

        st.info(
            overall
        )

    with st.expander(
        "View raw marking JSON"
    ):

        st.json(
            result
        )


# ============================================================
# MAIN APP
# ============================================================

def main():

    st.title(
        "📝 Examina AI"
    )

    st.subheader(
        "Three-Document Examination Marking"
    )

    st.write(
        "Upload the question paper, handwritten marking "
        "scheme, and handwritten student answer."
    )

    st.divider()

    # ========================================================
    # STATUS
    # ========================================================

    device = get_device()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Compute",
            device.upper()
        )

    with col2:

        st.metric(
            "Printed OCR",
            "TrOCR"
        )

    with col3:

        st.metric(
            "Handwriting OCR",
            "Makky07/Trocr"
        )

    # ========================================================
    # UPLOADS
    # ========================================================

    st.header(
        "1. Upload the three documents"
    )

    col1, col2, col3 = st.columns(3)

    # --------------------------------------------------------
    # QUESTION PAPER
    # --------------------------------------------------------

    with col1:

        st.subheader(
            "📄 Question Paper"
        )

        question_file = st.file_uploader(
            "Typed / printed question paper",
            type=[
                "png",
                "jpg",
                "jpeg",
                "webp",
            ],
            key="question_file",
        )

        if question_file:

            st.image(
                question_file,
                use_container_width=True,
            )

    # --------------------------------------------------------
    # MARKING SCHEME
    # --------------------------------------------------------

    with col2:

        st.subheader(
            "✍️ Marking Scheme"
        )

        scheme_file = st.file_uploader(
            "Handwritten marking scheme",
            type=[
                "png",
                "jpg",
                "jpeg",
                "webp",
            ],
            key="scheme_file",
        )

        if scheme_file:

            st.image(
                scheme_file,
                use_container_width=True,
            )

    # --------------------------------------------------------
    # STUDENT ANSWER
    # --------------------------------------------------------

    with col3:

        st.subheader(
            "📝 Student Answer"
        )

        answer_file = st.file_uploader(
            "Handwritten student answer",
            type=[
                "png",
                "jpg",
                "jpeg",
                "webp",
            ],
            key="answer_file",
        )

        if answer_file:

            st.image(
                answer_file,
                use_container_width=True,
            )

    # ========================================================
    # OCR
    # ========================================================

    st.header(
        "2. Extract the text"
    )

    ocr_button = st.button(
        "🔍 Read All Three Documents",
        type="primary",
        use_container_width=True,
    )

    if ocr_button:

        if not question_file:

            st.error(
                "Upload the question paper first."
            )

            st.stop()

        if not scheme_file:

            st.error(
                "Upload the handwritten marking scheme."
            )

            st.stop()

        if not answer_file:

            st.error(
                "Upload the handwritten student answer."
            )

            st.stop()

        # ----------------------------------------------------
        # QUESTION PAPER
        # ----------------------------------------------------

        with st.spinner(
            "Reading typed question paper..."
        ):

            try:

                st.session_state.question_text = (
                    typed_ocr(
                        question_file
                    )
                )

            except Exception as error:

                st.error(
                    "Typed question-paper OCR failed."
                )

                st.exception(error)

                st.stop()

        # ----------------------------------------------------
        # MARKING SCHEME
        # ----------------------------------------------------

        with st.spinner(
            "Reading handwritten marking scheme..."
        ):

            try:

                st.session_state.scheme_text = (
                    handwriting_ocr(
                        scheme_file
                    )
                )

            except Exception as error:

                st.error(
                    "Marking-scheme OCR failed."
                )

                st.exception(error)

                st.stop()

        # ----------------------------------------------------
        # STUDENT ANSWER
        # ----------------------------------------------------

        with st.spinner(
            "Reading handwritten student answer..."
        ):

            try:

                st.session_state.answer_text = (
                    handwriting_ocr(
                        answer_file
                    )
                )

            except Exception as error:

                st.error(
                    "Student-answer OCR failed."
                )

                st.exception(error)

                st.stop()

        st.success(
            "All three documents have been processed."
        )

    # ========================================================
    # OCR REVIEW
    # ========================================================

    if any(
        [
            st.session_state.question_text,
            st.session_state.scheme_text,
            st.session_state.answer_text,
        ]
    ):

        st.header(
            "3. Review and correct OCR"
        )

        st.info(
            "Correct any OCR mistakes here before marking. "
            "The corrected text is what the marking engine uses."
        )

        st.subheader(
            "📄 Question Paper"
        )

        st.session_state.question_text = (
            st.text_area(
                "Question paper",
                value=(
                    st.session_state.question_text
                ),
                height=300,
                key="question_editor",
            )
        )

        st.subheader(
            "✍️ Handwritten Marking Scheme"
        )

        st.session_state.scheme_text = (
            st.text_area(
                "Marking scheme",
                value=(
                    st.session_state.scheme_text
                ),
                height=300,
                key="scheme_editor",
            )
        )

        st.subheader(
            "📝 Handwritten Student Answer"
        )

        st.session_state.answer_text = (
            st.text_area(
                "Student answer",
                value=(
                    st.session_state.answer_text
                ),
                height=300,
                key="answer_editor",
            )
        )

        # ====================================================
        # MARKING
        # ====================================================

        st.header(
            "4. Mark the examination"
        )

        api_key_available = bool(
            get_api_key()
        )

        if not api_key_available:

            st.warning(
                "OPENAI_API_KEY has not been configured. "
                "OCR will still work, but AI marking cannot "
                "run until the API key is added."
            )

        mark_button = st.button(
            "🎯 Mark Student Answer",
            type="primary",
            use_container_width=True,
            disabled=not api_key_available,
        )

        if mark_button:

            if not st.session_state.question_text.strip():

                st.error(
                    "Question paper text is empty."
                )

                st.stop()

            if not st.session_state.scheme_text.strip():

                st.error(
                    "Marking scheme text is empty."
                )

                st.stop()

            if not st.session_state.answer_text.strip():

                st.error(
                    "Student answer text is empty."
                )

                st.stop()

            with st.spinner(
                "Marking the examination..."
            ):

                try:

                    result = mark_exam(

                        question_paper=(
                            st.session_state.question_text
                        ),

                        marking_scheme=(
                            st.session_state.scheme_text
                        ),

                        student_answer=(
                            st.session_state.answer_text
                        ),
                    )

                    st.session_state.result = result

                except Exception as error:

                    st.error(
                        "The marking engine failed."
                    )

                    st.exception(error)

    # ========================================================
    # RESULTS
    # ========================================================

    if st.session_state.result:

        st.divider()

        display_results(
            st.session_state.result
        )

    # ========================================================
    # RESET
    # ========================================================

    st.divider()

    if st.button(
        "🔄 Start New Examination",
        use_container_width=True,
    ):

        st.session_state.question_text = ""
        st.session_state.scheme_text = ""
        st.session_state.answer_text = ""
        st.session_state.result = None

        st.rerun()


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()
