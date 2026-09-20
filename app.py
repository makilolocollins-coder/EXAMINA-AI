# ================================================================
# EXAMINA AI
# ================================================================
# Three-document examination marking system
#
# 1. Typed / printed question paper  -> Examina-AI PP-OCRv5
# 2. Handwritten marking scheme      -> Makky07/Trocr
# 3. Handwritten student answer      -> Makky07/Trocr
#
# OCR -> editable text -> question parsing -> AI marking
# ================================================================

import io
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image


# ================================================================
# APP CONFIG
# ================================================================

st.set_page_config(
    page_title="Examina AI",
    page_icon="📝",
    layout="wide",
)

APP_TITLE = "Examina AI"

HF_EXAMINA_REPO = "Makky07/Examina-AI"
HF_TROCR_REPO = "Makky07/Trocr"

MARKING_MODEL_DEFAULT = "gpt-5.6-luna"


# ================================================================
# SESSION STATE
# ================================================================

DEFAULT_STATE = {
    "question_text": "",
    "scheme_text": "",
    "answer_text": "",
    "marking_result": None,
    "ocr_status": "",
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ================================================================
# UTILITY
# ================================================================

def get_device():
    """
    TrOCR uses PyTorch.
    PaddleOCR will determine its own CPU/GPU backend.
    """

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"

    except Exception:
        pass

    return "cpu"


def image_from_upload(uploaded_file):
    """
    Convert Streamlit UploadedFile to PIL image.
    """

    uploaded_file.seek(0)

    image = Image.open(uploaded_file).convert("RGB")

    return image


def pil_to_numpy(image):
    """
    PIL RGB -> NumPy RGB.
    """

    return np.array(image)


# ================================================================
# HUGGING FACE DOWNLOAD
# ================================================================

@st.cache_resource(show_spinner=False)
def download_examina_models():
    """
    Download the two actual Examina-AI model files from Hugging Face.

    The repository contains:

    Detection.json
    Text detection

    Recognition.json
    Text Recognition
    """

    from huggingface_hub import hf_hub_download

    root = Path(tempfile.gettempdir()) / "examina_ai_models"

    detection_dir = root / "detection"
    recognition_dir = root / "recognition"

    detection_dir.mkdir(parents=True, exist_ok=True)
    recognition_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Detection model
    # ------------------------------------------------------------

    detection_model = hf_hub_download(
        repo_id=HF_EXAMINA_REPO,
        filename="Text detection",
    )

    detection_json = hf_hub_download(
        repo_id=HF_EXAMINA_REPO,
        filename="Detection.json",
    )

    detection_yml = hf_hub_download(
        repo_id=HF_EXAMINA_REPO,
        filename="Detection.yml.txt",
    )

    # ------------------------------------------------------------
    # Recognition model
    # ------------------------------------------------------------

    recognition_model = hf_hub_download(
        repo_id=HF_EXAMINA_REPO,
        filename="Text Recognition",
    )

    recognition_json = hf_hub_download(
        repo_id=HF_EXAMINA_REPO,
        filename="Recognition.json",
    )

    recognition_yml = hf_hub_download(
        repo_id=HF_EXAMINA_REPO,
        filename="Recognition.yml.txt",
    )

    # ------------------------------------------------------------
    # Copy into clean model directories.
    #
    # PaddleOCR/PaddleX can inspect the model directory.
    # ------------------------------------------------------------

    shutil.copy2(
        detection_model,
        detection_dir / "inference.pdiparams",
    )

    shutil.copy2(
        detection_json,
        detection_dir / "inference.json",
    )

    shutil.copy2(
        detection_yml,
        detection_dir / "inference.yml",
    )

    shutil.copy2(
        recognition_model,
        recognition_dir / "inference.pdiparams",
    )

    shutil.copy2(
        recognition_json,
        recognition_dir / "inference.json",
    )

    shutil.copy2(
        recognition_yml,
        recognition_dir / "inference.yml",
    )

    return str(detection_dir), str(recognition_dir)


# ================================================================
# EXAMINA-AI OCR
# ================================================================

@st.cache_resource(show_spinner=False)
def load_examina_ocr():
    """
    Load the user's actual PP-OCRv5 detection + recognition models.
    """

    from paddleocr import PaddleOCR

    detection_dir, recognition_dir = download_examina_models()

    ocr = PaddleOCR(
        text_detection_model_dir=detection_dir,
        text_recognition_model_dir=recognition_dir,

        # We don't need the additional document models
        # for this first Examina version.
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,

        # OCR parameters based on the configuration
        # shipped with the user's detection model.
        text_det_limit_side_len=960,
        text_det_thresh=0.3,
        text_det_box_thresh=0.6,
        text_det_unclip_ratio=1.5,

        text_recognition_batch_size=1,

        device="cpu",
    )

    return ocr


def extract_paddle_result_text(result):
    """
    Extract text from PaddleOCR 3.x result objects.

    This function is intentionally defensive because
    PaddleOCR result structures can differ between versions.
    """

    texts = []
    boxes = []

    # ------------------------------------------------------------
    # New PaddleOCR result object
    # ------------------------------------------------------------

    try:
        data = result.json

        if callable(data):
            data = data()

        if isinstance(data, str):
            data = json.loads(data)

        if isinstance(data, dict):

            res = data.get("res", data)

            possible_texts = (
                res.get("rec_texts")
                or res.get("texts")
                or []
            )

            possible_scores = (
                res.get("rec_scores")
                or res.get("scores")
                or []
            )

            possible_boxes = (
                res.get("rec_polys")
                or res.get("dt_polys")
                or res.get("boxes")
                or []
            )

            for i, text in enumerate(possible_texts):

                score = 1.0

                if i < len(possible_scores):
                    try:
                        score = float(possible_scores[i])
                    except Exception:
                        pass

                if str(text).strip():
                    texts.append(
                        {
                            "text": str(text).strip(),
                            "score": score,
                            "box": (
                                possible_boxes[i]
                                if i < len(possible_boxes)
                                else None
                            ),
                        }
                    )

            if texts:
                return texts

    except Exception:
        pass

    # ------------------------------------------------------------
    # Older / alternate result format
    # ------------------------------------------------------------

    try:

        if isinstance(result, dict):

            rec_texts = result.get("rec_texts", [])
            rec_scores = result.get("rec_scores", [])

            for i, text in enumerate(rec_texts):

                score = 1.0

                if i < len(rec_scores):
                    try:
                        score = float(rec_scores[i])
                    except Exception:
                        pass

                if str(text).strip():

                    texts.append(
                        {
                            "text": str(text).strip(),
                            "score": score,
                            "box": None,
                        }
                    )

            if texts:
                return texts

    except Exception:
        pass

    return []


def sort_ocr_items(items):
    """
    Sort OCR lines in reading order.

    Uses bounding boxes when available.
    """

    def key(item):

        box = item.get("box")

        if box is None:
            return (0, 0)

        try:

            arr = np.asarray(box)

            y = float(np.min(arr[:, 1]))
            x = float(np.min(arr[:, 0]))

            return (y, x)

        except Exception:

            return (0, 0)

    return sorted(items, key=key)


def examina_ocr_image(image):
    """
    Run the user's Examina-AI PP-OCRv5 models on a page.
    """

    ocr = load_examina_ocr()

    image_np = pil_to_numpy(image)

    result = ocr.predict(image_np)

    all_items = []

    for page_result in result:

        items = extract_paddle_result_text(page_result)

        all_items.extend(items)

    all_items = sort_ocr_items(all_items)

    return "\n".join(
        item["text"]
        for item in all_items
        if item["text"].strip()
    )


# ================================================================
# TROCR
# ================================================================

@st.cache_resource(show_spinner=False)
def download_trocr_repository():
    """
    Download the user's complete TrOCR repository.
    """

    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(
        repo_id=HF_TROCR_REPO,
        repo_type="model",
    )

    return local_dir


@st.cache_resource(show_spinner=False)
def load_trocr():
    """
    Load Makky07/Trocr correctly.

    IMPORTANT:
    The repository stores the model as:

        Trocr_model.bin

    rather than:

        pytorch_model.bin

    So we create a temporary Transformers-compatible copy.

    The tokenizer is explicitly loaded as RobertaTokenizer.
    """

    import torch

    from transformers import (
        RobertaTokenizer,
        ViTImageProcessor,
        VisionEncoderDecoderModel,
    )

    repo_dir = Path(download_trocr_repository())

    # ------------------------------------------------------------
    # Locate model files
    # ------------------------------------------------------------

    config_file = repo_dir / "config.json"
    model_file = repo_dir / "Trocr_model.bin"

    if not config_file.exists():
        raise FileNotFoundError(
            "Makky07/Trocr is missing config.json"
        )

    if not model_file.exists():
        raise FileNotFoundError(
            "Makky07/Trocr is missing Trocr_model.bin"
        )

    # ------------------------------------------------------------
    # Build a temporary standard Transformers model directory.
    # ------------------------------------------------------------

    model_dir = Path(tempfile.gettempdir()) / "makky07_trocr"

    model_dir.mkdir(parents=True, exist_ok=True)

    files_to_copy = [
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    ]

    for filename in files_to_copy:

        source = repo_dir / filename

        if source.exists():

            destination = model_dir / filename

            if not destination.exists():

                shutil.copy2(
                    source,
                    destination,
                )

    standard_weights = model_dir / "pytorch_model.bin"

    if not standard_weights.exists():

        shutil.copy2(
            model_file,
            standard_weights,
        )

    # ------------------------------------------------------------
    # Explicit tokenizer.
    #
    # This avoids the tokenizer backend problem we encountered.
    # ------------------------------------------------------------

    tokenizer = RobertaTokenizer.from_pretrained(
        str(model_dir),
        use_fast=False,
    )

    # ------------------------------------------------------------
    # Image processor
    # ------------------------------------------------------------

    image_processor = ViTImageProcessor.from_pretrained(
        str(model_dir)
    )

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------

    model = VisionEncoderDecoderModel.from_pretrained(
        str(model_dir)
    )

    device = get_device()

    model.to(device)
    model.eval()

    return (
        tokenizer,
        image_processor,
        model,
        device,
    )


# ================================================================
# HANDWRITING LINE SEGMENTATION
# ================================================================

def segment_handwriting_lines(image):
    """
    Segment a handwritten page into horizontal text lines.

    TrOCR recognizes a text-line image much better than an entire
    examination page.

    This is deliberately conservative.
    """

    import cv2

    img = pil_to_numpy(image)

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY,
    )

    # Light blur to reduce paper noise.
    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0,
    )

    # Binary inverse threshold.
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    # Connect letters within each line.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (25, 3),
    )

    connected = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
    )

    projection = np.sum(
        connected > 0,
        axis=1,
    )

    threshold = max(
        2,
        int(img.shape[1] * 0.005),
    )

    active = projection > threshold

    ranges = []

    start = None

    for i, value in enumerate(active):

        if value and start is None:
            start = i

        elif not value and start is not None:

            if i - start >= 8:
                ranges.append(
                    (start, i)
                )

            start = None

    if start is not None:

        if len(active) - start >= 8:
            ranges.append(
                (start, len(active))
            )

    # ------------------------------------------------------------
    # Merge very close ranges.
    # ------------------------------------------------------------

    merged = []

    for start, end in ranges:

        if not merged:

            merged.append(
                [start, end]
            )

        else:

            previous = merged[-1]

            if start - previous[1] <= 12:

                previous[1] = end

            else:

                merged.append(
                    [start, end]
                )

    # ------------------------------------------------------------
    # If segmentation fails, use full page.
    # ------------------------------------------------------------

    if not merged:

        return [image]

    crops = []

    padding = 12

    for start, end in merged:

        y1 = max(
            0,
            start - padding,
        )

        y2 = min(
            img.shape[0],
            end + padding,
        )

        crop = image.crop(
            (
                0,
                y1,
                img.shape[1],
                y2,
            )
        )

        # Ignore tiny crops.
        if crop.height >= 15:
            crops.append(crop)

    if not crops:

        return [image]

    return crops


# ================================================================
# HANDWRITING OCR
# ================================================================

def trocr_single_image(image):
    """
    Recognize one handwritten text line.
    """

    import torch

    (
        tokenizer,
        image_processor,
        model,
        device,
    ) = load_trocr()

    pixel_values = image_processor(
        images=image,
        return_tensors="pt",
    ).pixel_values

    pixel_values = pixel_values.to(device)

    with torch.no_grad():

        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=128,
            num_beams=4,
            early_stopping=True,
        )

    text = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0]

    return text.strip()


def handwriting_ocr(uploaded_file):
    """
    OCR a handwritten page using Makky07/Trocr.
    """

    image = image_from_upload(
        uploaded_file
    )

    lines = segment_handwriting_lines(
        image
    )

    recognized = []

    progress = st.progress(
        0,
        text="Reading handwriting..."
    )

    total = len(lines)

    for index, line in enumerate(lines):

        try:

            text = trocr_single_image(
                line
            )

            if text.strip():

                recognized.append(
                    text.strip()
                )

        except Exception as error:

            recognized.append(
                f"[OCR ERROR: {error}]"
            )

        progress.progress(
            (index + 1) / total,
            text=f"Reading handwritten line {index + 1}/{total}"
        )

    progress.empty()

    return "\n".join(recognized)


# ================================================================
# TYPED / PRINTED OCR
# ================================================================

def typed_ocr(uploaded_file):
    """
    OCR a typed/printed question paper with the user's
    Examina-AI PP-OCRv5 detection + recognition models.
    """

    image = image_from_upload(
        uploaded_file
    )

    return examina_ocr_image(
        image
    )


# ================================================================
# QUESTION PARSING
# ================================================================

QUESTION_PATTERN = re.compile(
    r"(?im)"
    r"(?:^|\n)"
    r"\s*"
    r"(?:"
    r"Q(?:uestion)?\s*"
    r"|"
    r""
    r")"
    r"(\d+)"
    r"\s*"
    r"[\.\):\-]"
)


def extract_questions(text):
    """
    Parse common examination numbering formats.

    Examples:
        1.
        1)
        Q1.
        Question 1:
    """

    if not text.strip():
        return []

    matches = list(
        QUESTION_PATTERN.finditer(text)
    )

    if not matches:

        return [
            {
                "question_number": 1,
                "question": text.strip(),
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

        question_text = text[
            start:end
        ].strip()

        questions.append(
            {
                "question_number": number,
                "question": question_text,
            }
        )

    return questions


# ================================================================
# MARK EXTRACTION
# ================================================================

def extract_marks(text):
    """
    Try to identify maximum marks from question text.
    """

    patterns = [
        r"\[\s*(\d+(?:\.\d+)?)\s*marks?\s*\]",
        r"\(\s*(\d+(?:\.\d+)?)\s*marks?\s*\)",
        r"(\d+(?:\.\d+)?)\s*marks?",
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


# ================================================================
# OPENAI MARKING
# ================================================================

def get_openai_key():

    try:

        if "OPENAI_API_KEY" in st.secrets:

            return st.secrets[
                "OPENAI_API_KEY"
            ]

    except Exception:
        pass

    return os.environ.get(
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

    return os.environ.get(
        "EXAMINA_MARKING_MODEL",
        MARKING_MODEL_DEFAULT,
    )


def build_marking_prompt(
    question_paper,
    marking_scheme,
    student_answer,
):

    parsed_questions = extract_questions(
        question_paper
    )

    question_information = []

    for item in parsed_questions:

        max_marks = extract_marks(
            item["question"]
        )

        question_information.append(
            {
                "question_number":
                    item["question_number"],
                "question":
                    item["question"],
                "maximum_marks":
                    max_marks,
            }
        )

    return f"""
You are the marking engine for Examina AI.

Your task is to mark a student's examination answer against
the supplied official marking scheme.

IMPORTANT RULES:

1. Mark only what the student actually wrote.
2. Do not award marks merely because an answer is plausible.
3. Do not penalize correct wording simply because it differs from
   the marking scheme.
4. Equivalent scientific or mathematical statements should receive
   credit when they clearly satisfy the marking point.
5. Do not award more than the maximum marks for a question.
6. If a marking point is missing, identify it.
7. If the student's answer is partially correct, award appropriate
   partial credit where the marking scheme supports it.
8. Do not invent marking points.
9. If the OCR appears uncertain, use the visible text supplied,
   not assumptions.
10. Return valid JSON only.

QUESTION PAPER:

{question_paper}

PARSED QUESTIONS:

{json.dumps(question_information, indent=2)}

OFFICIAL MARKING SCHEME:

{marking_scheme}

STUDENT ANSWER:

{student_answer}

Return exactly this JSON structure:

{{
  "questions": [
    {{
      "question_number": 1,
      "question": "...",
      "maximum_marks": 10,
      "marks_awarded": 7,
      "decision": "partial",
      "reason": "...",
      "matched_points": [
        "..."
      ],
      "missing_points": [
        "..."
      ],
      "feedback": "..."
    }}
  ],
  "total_marks_awarded": 0,
  "total_marks_available": 0,
  "percentage": 0,
  "overall_feedback": "..."
}}
"""


def mark_exam(
    question_paper,
    marking_scheme,
    student_answer,
):

    api_key = get_openai_key()

    if not api_key:

        raise RuntimeError(
            "OPENAI_API_KEY is not configured. "
            "Add it to Streamlit Secrets."
        )

    from openai import OpenAI

    client = OpenAI(
        api_key=api_key
    )

    prompt = build_marking_prompt(
        question_paper,
        marking_scheme,
        student_answer,
    )

    response = client.responses.create(
        model=get_marking_model(),
        input=prompt,
    )

    output_text = response.output_text.strip()

    # ------------------------------------------------------------
    # Remove accidental markdown JSON fences.
    # ------------------------------------------------------------

    output_text = re.sub(
        r"^```json\s*",
        "",
        output_text,
        flags=re.IGNORECASE,
    )

    output_text = re.sub(
        r"\s*```$",
        "",
        output_text,
    )

    result = json.loads(
        output_text
    )

    return result


# ================================================================
# RESULT DISPLAY
# ================================================================

def display_results(result):

    st.subheader(
        "📊 Examination Result"
    )

    total_awarded = result.get(
        "total_marks_awarded",
        0,
    )

    total_available = result.get(
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
            "Marks",
            f"{total_awarded} / {total_available}",
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
                    [],
                )
            ),
        )

    st.divider()

    for question in result.get(
        "questions",
        [],
    ):

        number = question.get(
            "question_number",
            "?",
        )

        awarded = question.get(
            "marks_awarded",
            0,
        )

        maximum = question.get(
            "maximum_marks",
            0,
        )

        with st.expander(
            f"Question {number}: {awarded}/{maximum}",
            expanded=False,
        ):

            st.write(
                "**Question:**"
            )

            st.write(
                question.get(
                    "question",
                    "",
                )
            )

            st.write(
                "**Decision:** "
                + str(
                    question.get(
                        "decision",
                        "",
                    )
                )
            )

            st.write(
                "**Reason:**"
            )

            st.write(
                question.get(
                    "reason",
                    "",
                )
            )

            matched = question.get(
                "matched_points",
                [],
            )

            missing = question.get(
                "missing_points",
                [],
            )

            if matched:

                st.write(
                    "**Matched marking points:**"
                )

                for point in matched:

                    st.success(
                        str(point)
                    )

            if missing:

                st.write(
                    "**Missing marking points:**"
                )

                for point in missing:

                    st.warning(
                        str(point)
                    )

            st.write(
                "**Feedback:**"
            )

            st.info(
                question.get(
                    "feedback",
                    "",
                )
            )

    st.divider()

    st.subheader(
        "Overall Feedback"
    )

    st.write(
        result.get(
            "overall_feedback",
            "",
        )
    )


# ================================================================
# HEADER
# ================================================================

st.title(
    "📝 Examina AI"
)

st.caption(
    "OCR-powered examination marking using your "
    "Examina-AI and TrOCR models."
)

st.markdown(
    """
### Examination workflow

**1.** Upload the typed question paper  
**2.** Upload the handwritten marking scheme  
**3.** Upload the handwritten student answer  
**4.** Run OCR  
**5.** Correct the OCR text if necessary  
**6.** Mark the examination
"""
)


# ================================================================
# MODEL STATUS
# ================================================================

with st.expander(
    "🔧 Model configuration",
    expanded=False,
):

    st.write(
        f"**Typed/printed OCR:** `{HF_EXAMINA_REPO}`"
    )

    st.write(
        "**Detection:** PP-OCRv5 server"
    )

    st.write(
        "**Recognition:** PP-OCRv5"
    )

    st.write(
        f"**Handwriting OCR:** `{HF_TROCR_REPO}`"
    )

    st.write(
        f"**Marking model:** `{get_marking_model()}`"
    )

    st.write(
        f"**PyTorch device:** `{get_device()}`"
    )


# ================================================================
# UPLOADS
# ================================================================

st.header(
    "1. Upload Examination Documents"
)

col1, col2, col3 = st.columns(3)

with col1:

    question_file = st.file_uploader(
        "Typed / Printed Question Paper",
        type=[
            "png",
            "jpg",
            "jpeg",
            "webp",
        ],
        key="question_upload",
    )

with col2:

    scheme_file = st.file_uploader(
        "Handwritten Marking Scheme",
        type=[
            "png",
            "jpg",
            "jpeg",
            "webp",
        ],
        key="scheme_upload",
    )

with col3:

    answer_file = st.file_uploader(
        "Handwritten Student Answer",
        type=[
            "png",
            "jpg",
            "jpeg",
            "webp",
        ],
        key="answer_upload",
    )


# ================================================================
# OCR BUTTON
# ================================================================

if st.button(
    "🔍 Run OCR",
    type="primary",
    use_container_width=True,
):

    if not question_file:

        st.error(
            "Please upload the question paper."
        )

    elif not scheme_file:

        st.error(
            "Please upload the handwritten marking scheme."
        )

    elif not answer_file:

        st.error(
            "Please upload the handwritten student answer."
        )

    else:

        try:

            with st.spinner(
                "Reading typed question paper..."
            ):

                st.session_state.question_text = (
                    typed_ocr(
                        question_file
                    )
                )

            with st.spinner(
                "Reading handwritten marking scheme..."
            ):

                st.session_state.scheme_text = (
                    handwriting_ocr(
                        scheme_file
                    )
                )

            with st.spinner(
                "Reading handwritten student answer..."
            ):

                st.session_state.answer_text = (
                    handwriting_ocr(
                        answer_file
                    )
                )

            st.session_state.ocr_status = (
                "OCR completed successfully."
            )

            st.success(
                st.session_state.ocr_status
            )

        except Exception as error:

            st.error(
                "OCR failed."
            )

            st.exception(error)


# ================================================================
# EDITABLE OCR
# ================================================================

st.header(
    "2. Review and Correct OCR"
)

st.caption(
    "You can edit any OCR error before marking."
)

col1, col2, col3 = st.columns(3)

with col1:

    st.session_state.question_text = st.text_area(
        "Question Paper OCR",
        value=st.session_state.question_text,
        height=450,
        key="question_editor",
    )

with col2:

    st.session_state.scheme_text = st.text_area(
        "Marking Scheme OCR",
        value=st.session_state.scheme_text,
        height=450,
        key="scheme_editor",
    )

with col3:

    st.session_state.answer_text = st.text_area(
        "Student Answer OCR",
        value=st.session_state.answer_text,
        height=450,
        key="answer_editor",
    )


# ================================================================
# PARSED QUESTIONS PREVIEW
# ================================================================

if st.session_state.question_text.strip():

    with st.expander(
        "🔎 Preview parsed questions",
        expanded=False,
    ):

        parsed = extract_questions(
            st.session_state.question_text
        )

        for item in parsed:

            marks = extract_marks(
                item["question"]
            )

            if marks is not None:

                st.write(
                    f"**Q{item['question_number']}** "
                    f"({marks:g} marks)"
                )

            else:

                st.write(
                    f"**Q{item['question_number']}**"
                )

            st.write(
                item["question"]
            )

            st.divider()


# ================================================================
# MARK BUTTON
# ================================================================

st.header(
    "3. Mark Examination"
)

api_available = bool(
    get_openai_key()
)

if not api_available:

    st.warning(
        "OpenAI API key not configured yet. "
        "OCR can still be tested, but marking requires "
        "OPENAI_API_KEY in Streamlit Secrets."
    )

mark_button = st.button(
    "🧠 Mark Examination",
    type="primary",
    use_container_width=True,
    disabled=(
        not api_available
        or not st.session_state.question_text.strip()
        or not st.session_state.scheme_text.strip()
        or not st.session_state.answer_text.strip()
    ),
)

if mark_button:

    try:

        with st.spinner(
            "AI is marking the examination..."
        ):

            result = mark_exam(
                st.session_state.question_text,
                st.session_state.scheme_text,
                st.session_state.answer_text,
            )

            st.session_state.marking_result = result

        st.success(
            "Examination marked successfully."
        )

    except Exception as error:

        st.error(
            "Marking failed."
        )

        st.exception(error)


# ================================================================
# DISPLAY RESULTS
# ================================================================

if st.session_state.marking_result:

    display_results(
        st.session_state.marking_result
    )


# ================================================================
# RESET
# ================================================================

st.divider()

if st.button(
    "🔄 Reset Exam",
    use_container_width=True,
):

    for key, value in DEFAULT_STATE.items():

        st.session_state[key] = value

    st.rerun()
