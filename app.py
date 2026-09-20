# ============================================================
# EXAMINA AI
# THREE-DOCUMENT EXAMINATION MARKING LAB
#
# INPUTS
#   1. Typed / printed question paper
#   2. Handwritten marking scheme
#   3. Handwritten student answer
#
# PIPELINE
#   Question paper  -> PaddleOCR
#   Marking scheme  -> Makky07/Trocr
#   Student answer  -> Makky07/Trocr
#   OCR verification/editing
#   Question parsing
#   AI marking
#   Score + feedback
# ============================================================

import os
import re
import json
import base64
from pathlib import Path

import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Examina AI",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONSTANTS
# ============================================================

APP_NAME = "Examina AI"

HF_HANDWRITING_REPO = "Makky07/Trocr"

CACHE_DIR = (
    Path.home()
    / ".cache"
    / "examina_ai"
)

HANDWRITING_DIR = (
    CACHE_DIR
    / "handwriting"
)

TYPED_DIR = (
    CACHE_DIR
    / "typed"
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {

    "question_ocr": "",

    "marking_scheme_ocr": "",

    "student_answer_ocr": "",

    "marking_result": None,

    "question_file_name": None,

    "scheme_file_name": None,

    "answer_file_name": None,

}

for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================
# BASIC HELPERS
# ============================================================

def normalize_text(text):

    if text is None:
        return ""

    text = str(text)

    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


def safe_json_loads(text):

    if not text:
        return None

    try:

        return json.loads(text)

    except Exception:

        return None


# ============================================================
# DEVICE
# ============================================================

def get_device():

    try:

        import torch

        if torch.cuda.is_available():

            return "cuda"

    except Exception:

        pass

    return "cpu"


# ============================================================
# IMAGE → BASE64
# ============================================================

def image_to_base64(uploaded_file):

    data = uploaded_file.getvalue()

    encoded = base64.b64encode(
        data
    ).decode("utf-8")

    return encoded


# ============================================================
# TYPED OCR
# ============================================================

@st.cache_resource(
    show_spinner=False
)
def load_typed_ocr():

    from paddleocr import PaddleOCR

    return PaddleOCR()


def run_typed_ocr(uploaded_file):

    import numpy as np
    from PIL import Image

    model = load_typed_ocr()

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    image_np = np.array(
        image
    )

    result = model.predict(
        image_np
    )

    texts = []

    for page in result:

        try:

            data = page.json

            if callable(data):

                data = data()

            if isinstance(
                data,
                str
            ):

                data = json.loads(
                    data
                )

            page_texts = data.get(
                "rec_texts",
                []
            )

            for text in page_texts:

                if str(text).strip():

                    texts.append(
                        str(text).strip()
                    )

        except Exception:

            # Fallback for different
            # PaddleOCR result versions
            try:

                if hasattr(
                    page,
                    "rec_texts"
                ):

                    values = (
                        page.rec_texts
                    )

                    for value in values:

                        if str(value).strip():

                            texts.append(
                                str(value).strip()
                            )

            except Exception:

                pass

    return normalize_text(
        "\n".join(texts)
    )


# ============================================================
# HANDWRITING MODEL
# ============================================================

@st.cache_resource(
    show_spinner=False
)
def load_handwriting_model():

    import torch

    from huggingface_hub import (
        snapshot_download
    )

    from transformers import (
        TrOCRProcessor,
        VisionEncoderDecoderModel
    )

    HANDWRITING_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Download model only once
    # --------------------------------------------------------

    snapshot_download(
        repo_id=HF_HANDWRITING_REPO,
        local_dir=str(
            HANDWRITING_DIR
        )
    )

    processor = (
        TrOCRProcessor.from_pretrained(
            str(HANDWRITING_DIR),
            use_fast=False
        )
    )

    model = (
        VisionEncoderDecoderModel
        .from_pretrained(
            str(HANDWRITING_DIR)
        )
    )

    device = get_device()

    model.to(
        device
    )

    model.eval()

    return (
        processor,
        model,
        device
    )


# ============================================================
# HANDWRITING OCR
# ============================================================

def run_handwriting_ocr(
    uploaded_file
):

    import torch

    from PIL import Image

    (
        processor,
        model,
        device
    ) = load_handwriting_model()

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    pixel_values = processor(
        images=image,
        return_tensors="pt"
    ).pixel_values

    pixel_values = pixel_values.to(
        device
    )

    with torch.no_grad():

        generated_ids = (
            model.generate(
                pixel_values,
                max_new_tokens=512,
                num_beams=4
            )
        )

    text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return normalize_text(
        text
    )


# ============================================================
# QUESTION PARSING
# ============================================================

def parse_questions(question_text):

    """
    Attempts to split the question paper into
    numbered questions.

    Supports patterns such as:

        1.
        1)
        1:
        Question 1
        Question 1:
        Q1
        Q1.
    """

    question_text = normalize_text(
        question_text
    )

    if not question_text:

        return []

    pattern = re.compile(
        r"""
        (?=
            (?:
                ^|\n
            )
            \s*
            (?:
                Question\s*
                |Q\s*
            )?
            (\d{1,3})
            \s*
            [\.\):\-]
            \s*
        )
        """,
        re.IGNORECASE
        | re.VERBOSE
    )

    matches = list(
        pattern.finditer(
            question_text
        )
    )

    questions = []

    if not matches:

        return [
            {
                "number": 1,
                "text": question_text
            }
        ]

    for index, match in enumerate(
        matches
    ):

        number = int(
            match.group(1)
        )

        start = match.end()

        if index + 1 < len(matches):

            end = matches[
                index + 1
            ].start()

        else:

            end = len(
                question_text
            )

        body = question_text[
            start:end
        ].strip()

        if body:

            questions.append(
                {
                    "number": number,
                    "text": body
                }
            )

    return questions


# ============================================================
# MARKING SCHEME PARSING
# ============================================================

def parse_max_marks(text):

    """
    Attempts to extract marks from a question.

    Examples:

        [5 marks]
        [5]
        (5 marks)
        5 marks
        - 5 marks
    """

    patterns = [

        r"\[\s*(\d+(?:\.\d+)?)\s*marks?\s*\]",

        r"\(\s*(\d+(?:\.\d+)?)\s*marks?\s*\)",

        r"(\d+(?:\.\d+)?)\s*marks?\b",

    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
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
# MARKING ENGINE
# ============================================================

def get_openai_client():

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    if not api_key:

        return None

    try:

        from openai import OpenAI

        return OpenAI(
            api_key=api_key
        )

    except Exception:

        return None


def build_marking_prompt(
    question_text,
    marking_scheme_text,
    student_answer_text
):

    return f"""
You are the marking engine for Examina AI.

Your job is to mark a student's examination answer
STRICTLY against the supplied marking scheme.

Do not invent a marking scheme.

Do not award marks merely because an answer sounds
generally correct.

Use the supplied marking scheme as the primary
assessment criterion.

A student's wording does not need to exactly match
the marking scheme if the student's answer clearly
expresses the same scientifically or academically
valid idea.

Do not penalize grammar unless grammar changes the
meaning or the question explicitly assesses language.

Return ONLY valid JSON.

Required JSON structure:

{{
  "score": 0,
  "maximum_marks": 0,
  "percentage": 0,
  "decision": "full",
  "reason": "short explanation",
  "matched_points": [
      "..."
  ],
  "missing_points": [
      "..."
  ],
  "feedback": "short constructive feedback"
}}

Allowed decision values:

"full"
"partial"
"zero"

QUESTION:

{question_text}

MARKING SCHEME:

{marking_scheme_text}

STUDENT ANSWER:

{student_answer_text}
"""


def mark_single_question(
    question_text,
    marking_scheme_text,
    student_answer_text,
    maximum_marks=None
):

    client = get_openai_client()

    if client is None:

        raise RuntimeError(
            "OPENAI_API_KEY is not configured."
        )

    prompt = build_marking_prompt(
        question_text,
        marking_scheme_text,
        student_answer_text
    )

    if maximum_marks is None:

        maximum_marks = 0

    response = client.responses.create(

        model=os.getenv(
            "EXAMINA_MARKING_MODEL",
            "gpt-5.6-luna"
        ),

        input=prompt
    )

    raw = response.output_text

    result = safe_json_loads(
        raw
    )

    if result is None:

        # Try extracting JSON if the model
        # surrounded it with additional text.

        match = re.search(
            r"\{.*\}",
            raw,
            re.DOTALL
        )

        if match:

            result = safe_json_loads(
                match.group(0)
            )

    if result is None:

        raise RuntimeError(
            "Marking model returned invalid JSON."
        )

    # --------------------------------------------------------
    # Enforce maximum marks
    # --------------------------------------------------------

    if maximum_marks:

        result["maximum_marks"] = (
            maximum_marks
        )

        try:

            score = float(
                result.get(
                    "score",
                    0
                )
            )

            score = max(
                0,
                min(
                    score,
                    maximum_marks
                )
            )

            result["score"] = score

        except Exception:

            result["score"] = 0

    else:

        try:

            maximum_marks = float(
                result.get(
                    "maximum_marks",
                    0
                )
            )

        except Exception:

            maximum_marks = 0

    if maximum_marks:

        result["percentage"] = round(
            (
                float(
                    result.get(
                        "score",
                        0
                    )
                )
                / maximum_marks
            )
            * 100,
            2
        )

    return result


# ============================================================
# MULTI-QUESTION MARKING
# ============================================================

def mark_exam(
    question_text,
    marking_scheme_text,
    student_answer_text
):

    questions = parse_questions(
        question_text
    )

    if not questions:

        raise RuntimeError(
            "No questions could be extracted "
            "from the question paper."
        )

    results = []

    total_score = 0.0

    total_marks = 0.0

    # --------------------------------------------------------
    # For the first prototype, we send the complete
    # marking scheme and student answer to the marking
    # engine for each extracted question.
    #
    # This makes the system tolerant of handwritten
    # marking schemes whose exact formatting is unknown.
    # --------------------------------------------------------

    for question in questions:

        question_number = (
            question["number"]
        )

        question_body = (
            question["text"]
        )

        maximum_marks = parse_max_marks(
            question_body
        )

        result = mark_single_question(

            question_text=(
                question_body
            ),

            marking_scheme_text=(
                marking_scheme_text
            ),

            student_answer_text=(
                student_answer_text
            ),

            maximum_marks=maximum_marks
        )

        result["question_number"] = (
            question_number
        )

        result["question"] = (
            question_body
        )

        results.append(
            result
        )

        try:

            total_score += float(
                result.get(
                    "score",
                    0
                )
            )

        except Exception:

            pass

        if maximum_marks:

            total_marks += (
                maximum_marks
            )

        else:

            try:

                total_marks += float(
                    result.get(
                        "maximum_marks",
                        0
                    )
                )

            except Exception:

                pass

    percentage = 0

    if total_marks > 0:

        percentage = round(
            (
                total_score
                / total_marks
            )
            * 100,
            2
        )

    return {

        "questions": results,

        "total_score": total_score,

        "total_marks": total_marks,

        "percentage": percentage,

    }


# ============================================================
# DISPLAY MARKING RESULTS
# ============================================================

def display_marking_results(
    result
):

    st.success(
        "Marking completed."
    )

    total_score = result.get(
        "total_score",
        0
    )

    total_marks = result.get(
        "total_marks",
        0
    )

    percentage = result.get(
        "percentage",
        0
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total Score",
            f"{total_score:g}"
            if isinstance(
                total_score,
                float
            )
            else total_score
        )

    with col2:

        st.metric(
            "Maximum Marks",
            f"{total_marks:g}"
            if isinstance(
                total_marks,
                float
            )
            else total_marks
        )

    with col3:

        st.metric(
            "Percentage",
            f"{percentage}%"
        )

    st.divider()

    # --------------------------------------------------------
    # QUESTION RESULTS
    # --------------------------------------------------------

    st.subheader(
        "Question-by-Question Results"
    )

    for item in result.get(
        "questions",
        []
    ):

        number = item.get(
            "question_number",
            "?"
        )

        score = item.get(
            "score",
            0
        )

        maximum = item.get(
            "maximum_marks",
            0
        )

        decision = item.get(
            "decision",
            ""
        )

        with st.expander(
            f"Question {number} — "
            f"{score}/{maximum}",
            expanded=True
        ):

            st.markdown(
                "**Question**"
            )

            st.write(
                item.get(
                    "question",
                    ""
                )
            )

            st.markdown(
                "**Marking decision**"
            )

            st.write(
                decision
            )

            st.markdown(
                "**Reason**"
            )

            st.write(
                item.get(
                    "reason",
                    ""
                )
            )

            matched = item.get(
                "matched_points",
                []
            )

            if matched:

                st.markdown(
                    "**Matched marking points**"
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

            st.markdown(
                "**Feedback**"
            )

            st.info(
                item.get(
                    "feedback",
                    ""
                )
            )

    # --------------------------------------------------------
    # RAW JSON
    # --------------------------------------------------------

    with st.expander(
        "Show complete marking JSON"
    ):

        st.json(
            result
        )


# ============================================================
# OCR REVIEW SECTION
# ============================================================

def editable_ocr_section():

    st.header(
        "4. Review OCR"
    )

    st.info(
        "Check the extracted text carefully. "
        "You can correct OCR mistakes before marking."
    )

    # --------------------------------------------------------
    # QUESTION PAPER
    # --------------------------------------------------------

    st.subheader(
        "📄 Question Paper"
    )

    st.session_state.question_ocr = (
        st.text_area(
            "Question paper OCR",
            value=(
                st.session_state.question_ocr
            ),
            height=300,
            key="question_ocr_editor"
        )
    )

    # --------------------------------------------------------
    # MARKING SCHEME
    # --------------------------------------------------------

    st.subheader(
        "✍️ Marking Scheme"
    )

    st.session_state.marking_scheme_ocr = (
        st.text_area(
            "Marking scheme OCR",
            value=(
                st.session_state.marking_scheme_ocr
            ),
            height=300,
            key="scheme_ocr_editor"
        )
    )

    # --------------------------------------------------------
    # STUDENT ANSWER
    # --------------------------------------------------------

    st.subheader(
        "📝 Student Answer"
    )

    st.session_state.student_answer_ocr = (
        st.text_area(
            "Student answer OCR",
            value=(
                st.session_state.student_answer_ocr
            ),
            height=300,
            key="answer_ocr_editor"
        )
    )


# ============================================================
# MAIN APP
# ============================================================

def main():

    # ========================================================
    # HEADER
    # ========================================================

    st.title(
        "📝 Examina AI"
    )

    st.subheader(
        "Automated Examination Marking"
    )

    st.write(
        "Upload a typed question paper, a handwritten "
        "marking scheme, and a handwritten student answer."
    )

    st.divider()

    # ========================================================
    # SYSTEM STATUS
    # ========================================================

    device = get_device()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "OCR Device",
            device.upper()
        )

    with col2:

        st.metric(
            "Question OCR",
            "PaddleOCR"
        )

    with col3:

        st.metric(
            "Handwriting OCR",
            "Makky07/Trocr"
        )

    # ========================================================
    # API STATUS
    # ========================================================

    if os.getenv(
        "OPENAI_API_KEY"
    ):

        st.success(
            "Marking engine connected."
        )

    else:

        st.warning(
            "Marking engine is not configured. "
            "Add OPENAI_API_KEY to your Streamlit secrets "
            "before running the final marking step."
        )

    # ========================================================
    # THREE UPLOADS
    # ========================================================

    st.header(
        "1. Upload Examination Documents"
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
                "webp"
            ],
            key="question_upload"
        )

        if question_file:

            st.session_state.question_file_name = (
                question_file.name
            )

            st.image(
                question_file,
                use_container_width=True
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
                "webp"
            ],
            key="scheme_upload"
        )

        if scheme_file:

            st.session_state.scheme_file_name = (
                scheme_file.name
            )

            st.image(
                scheme_file,
                use_container_width=True
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
                "webp"
            ],
            key="answer_upload"
        )

        if answer_file:

            st.session_state.answer_file_name = (
                answer_file.name
            )

            st.image(
                answer_file,
                use_container_width=True
            )

    # ========================================================
    # OCR BUTTON
    # ========================================================

    st.header(
        "2. Read Documents"
    )

    run_ocr = st.button(
        "🔍 Extract Text From All Documents",
        type="primary",
        use_container_width=True
    )

    if run_ocr:

        if not question_file:

            st.error(
                "Please upload the question paper."
            )

            st.stop()

        if not scheme_file:

            st.error(
                "Please upload the marking scheme."
            )

            st.stop()

        if not answer_file:

            st.error(
                "Please upload the student answer."
            )

            st.stop()

        # ----------------------------------------------------
        # QUESTION PAPER OCR
        # ----------------------------------------------------

        with st.spinner(
            "Reading typed question paper..."
        ):

            try:

                question_text = (
                    run_typed_ocr(
                        question_file
                    )
                )

                st.session_state.question_ocr = (
                    question_text
                )

            except Exception as e:

                st.error(
                    "Question paper OCR failed."
                )

                st.exception(e)

                st.stop()

        # ----------------------------------------------------
        # MARKING SCHEME OCR
        # ----------------------------------------------------

        with st.spinner(
            "Reading handwritten marking scheme..."
        ):

            try:

                scheme_text = (
                    run_handwriting_ocr(
                        scheme_file
                    )
                )

                st.session_state.marking_scheme_ocr = (
                    scheme_text
                )

            except Exception as e:

                st.error(
                    "Marking scheme OCR failed."
                )

                st.exception(e)

                st.stop()

        # ----------------------------------------------------
        # STUDENT ANSWER OCR
        # ----------------------------------------------------

        with st.spinner(
            "Reading handwritten student answer..."
        ):

            try:

                answer_text = (
                    run_handwriting_ocr(
                        answer_file
                    )
                )

                st.session_state.student_answer_ocr = (
                    answer_text
                )

            except Exception as e:

                st.error(
                    "Student answer OCR failed."
                )

                st.exception(e)

                st.stop()

        st.success(
            "All three documents have been read."
        )

    # ========================================================
    # OCR REVIEW
    # ========================================================

    if (
        st.session_state.question_ocr
        or
        st.session_state.marking_scheme_ocr
        or
        st.session_state.student_answer_ocr
    ):

        editable_ocr_section()

    # ========================================================
    # MARK BUTTON
    # ========================================================

    st.header(
        "5. Mark Examination"
    )

    ready = all(
        [
            st.session_state.question_ocr.strip(),
            st.session_state.marking_scheme_ocr.strip(),
            st.session_state.student_answer_ocr.strip()
        ]
    )

    if not ready:

        st.info(
            "Complete OCR extraction and review the "
            "three documents before marking."
        )

    mark_button = st.button(
        "🎯 Mark Student Answer",
        type="primary",
        use_container_width=True,
        disabled=not ready
    )

    if mark_button:

        if not os.getenv(
            "OPENAI_API_KEY"
        ):

            st.error(
                "OPENAI_API_KEY is missing. "
                "Configure it in Streamlit Secrets."
            )

            st.stop()

        with st.spinner(
            "Analyzing answers against the marking scheme..."
        ):

            try:

                result = mark_exam(

                    question_text=(
                        st.session_state.question_ocr
                    ),

                    marking_scheme_text=(
                        st.session_state.marking_scheme_ocr
                    ),

                    student_answer_text=(
                        st.session_state.student_answer_ocr
                    )
                )

                st.session_state.marking_result = (
                    result
                )

            except Exception as e:

                st.error(
                    "Marking failed."
                )

                st.exception(e)

                st.stop()

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    if st.session_state.marking_result:

        st.divider()

        st.header(
            "6. Marking Result"
        )

        display_marking_results(
            st.session_state.marking_result
        )

    # ========================================================
    # RESET
    # ========================================================

    st.divider()

    if st.button(
        "🔄 Start New Examination",
        use_container_width=True
    ):

        for key in DEFAULT_STATE:

            st.session_state[key] = (
                DEFAULT_STATE[key]
            )

        st.rerun()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
