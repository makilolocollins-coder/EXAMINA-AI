# ======================================================================
# EXAMINA AI
# OCR-POWERED EXAMINATION MARKING
#
# Typed / Printed Question Paper:
#     Makky07/Examina-AI
#     PP-OCRv5 detection + recognition
#
# Handwritten Marking Scheme:
#     Makky07/Trocr
#
# Handwritten Student Answer:
#     Makky07/Trocr
#
# Workflow:
#     1. Upload question paper
#     2. Upload marking scheme
#     3. Upload student answer
#     4. Run OCR
#     5. Correct OCR text
#     6. Mark examination
# ======================================================================

import os
import re
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st
from PIL import Image


# ======================================================================
# CONFIGURATION
# ======================================================================

APP_TITLE = "Examina AI"

HF_EXAMINA_REPO = "Makky07/Examina-AI"
HF_TROCR_REPO = "Makky07/Trocr"

MARKING_MODEL_DEFAULT = "gpt-5.6-luna"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📝",
    layout="wide",
)


# ======================================================================
# SESSION STATE
# ======================================================================

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


# ======================================================================
# GENERAL HELPERS
# ======================================================================

def get_device() -> str:
    """
    TrOCR can use CUDA if available.
    PaddleOCR is deliberately kept on CPU because the Streamlit
    environment may not provide a Paddle CUDA runtime.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"

    except Exception:
        pass

    return "cpu"


def image_to_numpy(uploaded_file) -> np.ndarray:
    """
    Convert Streamlit uploaded image into RGB numpy array.
    """
    image = Image.open(uploaded_file).convert("RGB")
    return np.array(image)


def safe_mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ======================================================================
# HUGGING FACE - EXAMINA AI
# ======================================================================

@st.cache_resource(show_spinner=False)
def download_examina_models():
    """
    Download the user's Examina-AI PP-OCRv5 detection and recognition
    artifacts from Hugging Face.

    The repository contains the model artifacts under:
        Text detection
        Text Recognition

    PaddleOCR expects an inference directory containing:
        inference.pdiparams
        inference.yml
        inference.json
    """

    from huggingface_hub import hf_hub_download

    root = Path(tempfile.gettempdir()) / "examina_ai_models"

    detection_dir = safe_mkdir(root / "detection")
    recognition_dir = safe_mkdir(root / "recognition")

    # --------------------------------------------------------------
    # Detection
    # --------------------------------------------------------------

    detection_weights = hf_hub_download(
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

    shutil.copy2(
        detection_weights,
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

    # --------------------------------------------------------------
    # Recognition
    # --------------------------------------------------------------

    recognition_weights = hf_hub_download(
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

    shutil.copy2(
        recognition_weights,
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

    return {
        "root": root,
        "detection": detection_dir,
        "recognition": recognition_dir,
    }


# ======================================================================
# LOAD EXAMINA OCR
# ======================================================================

@st.cache_resource(show_spinner=False)
def load_examina_ocr():
    """
    Load the user's Examina-AI PP-OCRv5 pipeline.

    IMPORTANT:
        enable_mkldnn=False

    This disables Paddle's CPU MKL-DNN / oneDNN acceleration.

    The previous crash occurred inside:
        onednn_instruction.cc

    Therefore we deliberately avoid that execution path.
    """

    from paddleocr import PaddleOCR

    models = download_examina_models()

    detection_dir = str(models["detection"])
    recognition_dir = str(models["recognition"])

    ocr = PaddleOCR(
        # ----------------------------------------------------------
        # User's local Examina-AI models
        # ----------------------------------------------------------

        text_detection_model_dir=detection_dir,
        text_recognition_model_dir=recognition_dir,

        # ----------------------------------------------------------
        # Do not allow PaddleOCR to substitute its own model names
        # ----------------------------------------------------------

        text_detection_model_name=None,
        text_recognition_model_name=None,

        # ----------------------------------------------------------
        # Document preprocessing is unnecessary for this prototype
        # ----------------------------------------------------------

        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,

        # ----------------------------------------------------------
        # Streamlit Cloud CPU
        # ----------------------------------------------------------

        device="cpu",

        # ----------------------------------------------------------
        # CRITICAL FIX
        #
        # Disable MKL-DNN / oneDNN.
        #
        # The previous error was:
        #
        # NotImplementedError:
        # ConvertPirAttribute2RuntimeAttribute
        # ...
        # onednn_instruction.cc
        # ----------------------------------------------------------

        enable_mkldnn=False,

        # Keep CPU usage modest on Streamlit Cloud.
        cpu_threads=2,

        # ----------------------------------------------------------
        # Detection settings matching the user's model configuration
        # ----------------------------------------------------------

        text_det_limit_side_len=960,
        text_det_limit_type="max",
        text_det_thresh=0.3,
        text_det_box_thresh=0.6,
        text_det_unclip_ratio=1.5,

        # ----------------------------------------------------------
        # Recognition
        # ----------------------------------------------------------

        text_recognition_batch_size=1,
    )

    return ocr


# ======================================================================
# PADDLEOCR RESULT PARSING
# ======================================================================

def _to_python(value):
    """
    Convert Paddle / numpy values into normal Python values.
    """

    if value is None:
        return None

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {
            key: _to_python(val)
            for key, val in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _to_python(v)
            for v in value
        ]

    return value


def extract_paddle_result_text(result) -> List[Dict[str, Any]]:
    """
    Extract OCR text, confidence and bounding boxes from PaddleOCR 3.x
    result objects.

    The function is intentionally defensive because PaddleOCR's result
    object structure can vary between releases.
    """

    extracted = []

    try:
        result_dict = result.json
    except Exception:
        result_dict = None

    if callable(result_dict):
        try:
            result_dict = result_dict()
        except Exception:
            result_dict = None

    if result_dict is None:
        try:
            result_dict = result.to_json()
        except Exception:
            result_dict = None

    result_dict = _to_python(result_dict)

    if not isinstance(result_dict, dict):
        return extracted

    # --------------------------------------------------------------
    # Common PaddleOCR 3.x structure
    # --------------------------------------------------------------

    rec_texts = result_dict.get("rec_texts")
    rec_scores = result_dict.get("rec_scores")
    rec_polys = result_dict.get("rec_polys")

    if rec_texts is not None:

        if rec_scores is None:
            rec_scores = [None] * len(rec_texts)

        if rec_polys is None:
            rec_polys = [None] * len(rec_texts)

        for text, score, box in zip(
            rec_texts,
            rec_scores,
            rec_polys,
        ):
            if text is None:
                continue

            text = str(text).strip()

            if not text:
                continue

            extracted.append(
                {
                    "text": text,
                    "score": (
                        float(score)
                        if score is not None
                        else None
                    ),
                    "box": box,
                }
            )

        return extracted

    # --------------------------------------------------------------
    # Older / alternative nested structure
    # --------------------------------------------------------------

    if "data" in result_dict:

        data = result_dict["data"]

        if isinstance(data, dict):
            nested_texts = data.get("rec_texts", [])
            nested_scores = data.get("rec_scores", [])
            nested_boxes = data.get("rec_polys", [])

            for i, text in enumerate(nested_texts):

                if not text:
                    continue

                score = (
                    nested_scores[i]
                    if i < len(nested_scores)
                    else None
                )

                box = (
                    nested_boxes[i]
                    if i < len(nested_boxes)
                    else None
                )

                extracted.append(
                    {
                        "text": str(text).strip(),
                        "score": (
                            float(score)
                            if score is not None
                            else None
                        ),
                        "box": box,
                    }
                )

    return extracted


# ======================================================================
# OCR SORTING
# ======================================================================

def box_position(box) -> Tuple[float, float]:
    """
    Get approximate x/y position from a polygon/box.
    """

    if box is None:
        return 0.0, 0.0

    try:

        arr = np.asarray(box, dtype=float)

        if arr.ndim == 2 and arr.shape[1] >= 2:

            x = float(np.min(arr[:, 0]))
            y = float(np.min(arr[:, 1]))

            return x, y

        if arr.size >= 4:

            flat = arr.reshape(-1)

            return float(flat[0]), float(flat[1])

    except Exception:
        pass

    return 0.0, 0.0


def sort_ocr_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sort OCR lines top-to-bottom and left-to-right.
    """

    if not items:
        return []

    enriched = []

    for item in items:

        x, y = box_position(
            item.get("box")
        )

        enriched.append(
            (
                y,
                x,
                item,
            )
        )

    enriched.sort(
        key=lambda value: (
            round(value[0] / 15.0),
            value[1],
        )
    )

    return [
        item
        for _, _, item in enriched
    ]


# ======================================================================
# TYPED OCR
# ======================================================================

def examina_ocr_image(image_np: np.ndarray) -> str:
    """
    Run Examina-AI OCR on one typed/printed image.
    """

    ocr = load_examina_ocr()

    result = ocr.predict(image_np)

    all_items = []

    for item in result:

        extracted = extract_paddle_result_text(item)

        all_items.extend(extracted)

    all_items = sort_ocr_items(all_items)

    lines = []

    for item in all_items:

        text = item.get("text", "").strip()

        if text:
            lines.append(text)

    return "\n".join(lines)


def typed_ocr(uploaded_file) -> str:
    """
    OCR for typed / printed question paper.
    """

    image_np = image_to_numpy(uploaded_file)

    return examina_ocr_image(image_np)


# ======================================================================
# HUGGING FACE - TROCR
# ======================================================================

@st.cache_resource(show_spinner=False)
def download_trocr_repository():

    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=HF_TROCR_REPO
    )

    return Path(path)


# ======================================================================
# LOAD TROCR
# ======================================================================

@st.cache_resource(show_spinner=False)
def load_trocr():

    import torch

    from transformers import (
        RobertaTokenizer,
        ViTImageProcessor,
        VisionEncoderDecoderModel,
    )

    source_dir = download_trocr_repository()

    config_file = source_dir / "config.json"
    weights_file = source_dir / "Trocr_model.bin"

    if not config_file.exists():
        raise FileNotFoundError(
            f"TrOCR config.json was not found in {source_dir}"
        )

    if not weights_file.exists():
        raise FileNotFoundError(
            f"Trocr_model.bin was not found in {source_dir}"
        )

    # --------------------------------------------------------------
    # Transformers expects pytorch_model.bin
    # --------------------------------------------------------------

    model_dir = Path(
        tempfile.gettempdir()
    ) / "examina_trocr_model"

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    required_files = [
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    ]

    for filename in required_files:

        source = source_dir / filename

        if source.exists():

            shutil.copy2(
                source,
                model_dir / filename,
            )

    pytorch_weights = model_dir / "pytorch_model.bin"

    if (
        not pytorch_weights.exists()
        or pytorch_weights.stat().st_size
        != weights_file.stat().st_size
    ):

        shutil.copy2(
            weights_file,
            pytorch_weights,
        )

    # --------------------------------------------------------------
    # Explicit tokenizer
    #
    # The user's repository identifies RobertaTokenizer.
    # --------------------------------------------------------------

    tokenizer = RobertaTokenizer.from_pretrained(
        str(model_dir),
        use_fast=False,
    )

    processor = ViTImageProcessor.from_pretrained(
        str(model_dir)
    )

    model = VisionEncoderDecoderModel.from_pretrained(
        str(model_dir),
        local_files_only=True,
    )

    device = get_device()

    model = model.to(device)

    model.eval()

    return {
        "model": model,
        "tokenizer": tokenizer,
        "processor": processor,
        "device": device,
    }


# ======================================================================
# HANDWRITING LINE SEGMENTATION
# ======================================================================

def segment_handwriting_lines(
    image_np: np.ndarray,
) -> List[np.ndarray]:
    """
    Segment handwritten document into horizontal text lines.

    This is intentionally simple and robust for ordinary examination
    answer sheets.
    """

    import cv2

    if image_np is None:
        return []

    gray = cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2GRAY,
    )

    # Light denoising
    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0,
    )

    # Binary threshold
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV
        + cv2.THRESH_OTSU,
    )

    # Connect characters belonging to the same line
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (35, 3),
    )

    connected = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
    )

    horizontal_projection = np.sum(
        connected > 0,
        axis=1,
    )

    threshold = max(
        3,
        int(image_np.shape[1] * 0.003),
    )

    active = horizontal_projection > threshold

    ranges = []

    start = None

    for i, is_active in enumerate(active):

        if is_active and start is None:
            start = i

        elif not is_active and start is not None:

            end = i

            if end - start >= 8:
                ranges.append(
                    (
                        start,
                        end,
                    )
                )

            start = None

    if start is not None:

        end = len(active)

        if end - start >= 8:
            ranges.append(
                (
                    start,
                    end,
                )
            )

    # Merge close ranges
    merged = []

    for start, end in ranges:

        if not merged:

            merged.append(
                [
                    start,
                    end,
                ]
            )

        else:

            previous = merged[-1]

            if start - previous[1] <= 12:

                previous[1] = end

            else:

                merged.append(
                    [
                        start,
                        end,
                    ]
                )

    # Crop with margins
    lines = []

    height, width = image_np.shape[:2]

    for start, end in merged:

        top = max(
            0,
            start - 12,
        )

        bottom = min(
            height,
            end + 12,
        )

        crop = image_np[
            top:bottom,
            0:width,
        ]

        if crop.size > 0:
            lines.append(crop)

    return lines


# ======================================================================
# TROCR SINGLE LINE
# ======================================================================

def trocr_single_image(
    image_np: np.ndarray,
) -> str:

    import torch

    components = load_trocr()

    model = components["model"]
    tokenizer = components["tokenizer"]
    processor = components["processor"]
    device = components["device"]

    image = Image.fromarray(
        image_np
    ).convert("RGB")

    pixel_values = processor(
        images=image,
        return_tensors="pt",
    ).pixel_values

    pixel_values = pixel_values.to(
        device
    )

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


# ======================================================================
# HANDWRITING OCR
# ======================================================================

def handwriting_ocr(uploaded_file) -> str:

    image_np = image_to_numpy(
        uploaded_file
    )

    lines = segment_handwriting_lines(
        image_np
    )

    if not lines:

        # If segmentation fails, try the entire page.
        return trocr_single_image(
            image_np
        )

    recognized_lines = []

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

            if text:

                recognized_lines.append(
                    text
                )

        except Exception as exc:

            st.warning(
                f"Could not read handwriting line "
                f"{index + 1}: {exc}"
            )

        progress.progress(
            (index + 1) / total,
            text=(
                f"Reading handwriting "
                f"{index + 1}/{total}"
            ),
        )

    progress.empty()

    return "\n".join(
        recognized_lines
    )


# ======================================================================
# QUESTION PARSING
# ======================================================================

QUESTION_PATTERN = re.compile(
    r"(?im)"
    r"^\s*"
    r"(?:"
    r"Q(?:uestion)?\s*)?"
    r"(\d{1,3})"
    r"\s*"
    r"[\.\):\-]"
    r"\s*"
)


def parse_questions(text: str) -> List[Dict[str, Any]]:
    """
    Parse a question paper into numbered questions.

    Supported examples:

        1. Explain...
        2) Describe...
        Q3. Calculate...
        Question 4: Discuss...
    """

    if not text.strip():
        return []

    matches = list(
        QUESTION_PATTERN.finditer(text)
    )

    if not matches:
        return [
            {
                "number": 1,
                "question": text.strip(),
                "max_marks": None,
            }
        ]

    questions = []

    for i, match in enumerate(matches):

        number = int(
            match.group(1)
        )

        start = match.end()

        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)

        question_text = text[
            start:end
        ].strip()

        max_marks = extract_max_marks(
            question_text
        )

        questions.append(
            {
                "number": number,
                "question": question_text,
                "max_marks": max_marks,
            }
        )

    return questions


# ======================================================================
# MARK EXTRACTION
# ======================================================================

def extract_max_marks(text: str) -> Optional[float]:

    patterns = [
        r"\[\s*(\d+(?:\.\d+)?)\s*marks?\s*\]",
        r"\(\s*(\d+(?:\.\d+)?)\s*marks?\s*\)",
        r"(\d+(?:\.\d+)?)\s*marks?\b",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:

            try:
                return float(
                    match.group(1)
                )

            except Exception:
                pass

    return None


# ======================================================================
# OPENAI API KEY
# ======================================================================

def get_openai_api_key() -> Optional[str]:

    # Streamlit secrets
    try:

        if "OPENAI_API_KEY" in st.secrets:

            value = st.secrets[
                "OPENAI_API_KEY"
            ]

            if value:
                return str(value).strip()

    except Exception:
        pass

    # Environment variable
    value = os.getenv(
        "OPENAI_API_KEY"
    )

    if value:
        return value.strip()

    return None


# ======================================================================
# MARKING PROMPT
# ======================================================================

def build_marking_prompt(
    question_text: str,
    scheme_text: str,
    answer_text: str,
) -> str:

    return f"""
You are an examination marking assistant.

Your task is to mark a student's examination answer against the
provided marking scheme.

IMPORTANT RULES:

1. Mark only from the information provided.
2. Do not invent marking points.
3. Do not award marks simply because an answer sounds plausible.
4. Give credit when the student's wording is different but clearly
   expresses the same valid concept.
5. Do not penalize spelling or grammar unless they change the
   scientific/technical meaning.
6. Respect the maximum marks available.
7. If the OCR contains an obvious ambiguity, identify it rather than
   silently inventing text.
8. Give question-by-question reasoning.
9. Separate matched marking points from missing marking points.
10. Calculate the total marks and percentage.

QUESTION PAPER
==============
{question_text}

MARKING SCHEME
==============
{scheme_text}

STUDENT ANSWER
==============
{answer_text}

Return ONLY valid JSON using this structure:

{{
  "questions": [
    {{
      "question_number": 1,
      "question": "question text",
      "max_marks": 5,
      "marks_awarded": 4,
      "decision": "Partially correct",
      "reason": "Explanation of why the marks were awarded.",
      "matched_points": [
        "Point correctly addressed by student"
      ],
      "missing_points": [
        "Required point not addressed"
      ],
      "feedback": "Specific feedback to the student."
    }}
  ],
  "total_marks_awarded": 4,
  "total_marks_available": 5,
  "percentage": 80,
  "overall_feedback": "Overall examination feedback."
}}
""".strip()


# ======================================================================
# OPENAI MARKING
# ======================================================================

def mark_examination(
    question_text: str,
    scheme_text: str,
    answer_text: str,
    model_name: str,
) -> Dict[str, Any]:

    from openai import OpenAI

    api_key = get_openai_api_key()

    if not api_key:

        raise RuntimeError(
            "OPENAI_API_KEY is not configured. "
            "Add it to Streamlit Secrets."
        )

    client = OpenAI(
        api_key=api_key
    )

    prompt = build_marking_prompt(
        question_text,
        scheme_text,
        answer_text,
    )

    response = client.chat.completions.create(
        model=model_name,
        temperature=0,
        response_format={
            "type": "json_object"
        },
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise examination marking "
                    "assistant. Return valid JSON only."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError(
            "The marking model returned an empty response."
        )

    try:

        return json.loads(
            content
        )

    except json.JSONDecodeError as exc:

        # Try extracting a JSON object if the model added text.
        match = re.search(
            r"\{.*\}",
            content,
            flags=re.DOTALL,
        )

        if match:

            return json.loads(
                match.group(0)
            )

        raise RuntimeError(
            f"Marking model returned invalid JSON: {exc}"
        )


# ======================================================================
# DISPLAY MARKING RESULTS
# ======================================================================

def display_marking_results(
    result: Dict[str, Any]
):

    st.subheader(
        "📊 Examination Results"
    )

    questions = result.get(
        "questions",
        []
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
            "Marks Awarded",
            total_awarded,
        )

    with col2:
        st.metric(
            "Total Available",
            total_available,
        )

    with col3:
        st.metric(
            "Percentage",
            f"{percentage}%",
        )

    st.divider()

    for question in questions:

        number = question.get(
            "question_number",
            "?",
        )

        awarded = question.get(
            "marks_awarded",
            0,
        )

        maximum = question.get(
            "max_marks",
            0,
        )

        decision = question.get(
            "decision",
            "",
        )

        with st.expander(
            f"Question {number}: "
            f"{awarded}/{maximum} — {decision}",
            expanded=True,
        ):

            st.write(
                "**Question**"
            )

            st.write(
                question.get(
                    "question",
                    "",
                )
            )

            st.write(
                "**Reason**"
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

            if matched:

                st.write(
                    "**Matched marking points**"
                )

                for point in matched:

                    st.write(
                        f"✓ {point}"
                    )

            missing = question.get(
                "missing_points",
                [],
            )

            if missing:

                st.write(
                    "**Missing marking points**"
                )

                for point in missing:

                    st.write(
                        f"• {point}"
                    )

            feedback = question.get(
                "feedback",
                "",
            )

            if feedback:

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

        st.write(
            overall
        )


# ======================================================================
# SIDEBAR
# ======================================================================

with st.sidebar:

    st.header(
        "🔧 Model configuration"
    )

    st.write(
        "**Typed / Printed OCR**"
    )

    st.code(
        HF_EXAMINA_REPO,
        language="text",
    )

    st.write(
        "**Handwriting OCR**"
    )

    st.code(
        HF_TROCR_REPO,
        language="text",
    )

    st.write(
        "**Paddle CPU inference**"
    )

    st.code(
        "MKL-DNN: DISABLED",
        language="text",
    )

    st.caption(
        "MKL-DNN is disabled to avoid the "
        "oneDNN inference error encountered "
        "on Streamlit Cloud."
    )

    st.divider()

    marking_model = st.text_input(
        "Marking model",
        value=MARKING_MODEL_DEFAULT,
    )

    st.divider()

    st.caption(
        "Examina AI prototype"
    )


# ======================================================================
# HEADER
# ======================================================================

st.title(
    "📝 Examina AI"
)

st.caption(
    "OCR-powered examination marking using "
    "your Examina-AI and TrOCR models."
)


# ======================================================================
# WORKFLOW
# ======================================================================

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


# ======================================================================
# UPLOAD DOCUMENTS
# ======================================================================

st.header(
    "1. Upload Examination Documents"
)

col1, col2, col3 = st.columns(3)

with col1:

    question_file = st.file_uploader(
        "Typed / Printed Question Paper",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
        key="question_file",
    )

with col2:

    scheme_file = st.file_uploader(
        "Handwritten Marking Scheme",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
        key="scheme_file",
    )

with col3:

    answer_file = st.file_uploader(
        "Handwritten Student Answer",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
        key="answer_file",
    )


# ======================================================================
# OCR BUTTON
# ======================================================================

st.header(
    "2. OCR"
)

run_ocr = st.button(
    "🔍 Run OCR",
    type="primary",
    use_container_width=True,
)


if run_ocr:

    if not question_file:
        st.error(
            "Please upload the typed question paper."
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

            st.session_state.ocr_status = (
                "OCR is running..."
            )

            # ------------------------------------------------------
            # Typed question paper
            # ------------------------------------------------------

            with st.spinner(
                "Reading typed question paper with Examina-AI..."
            ):

                st.session_state.question_text = (
                    typed_ocr(
                        question_file
                    )
                )

            # ------------------------------------------------------
            # Handwritten marking scheme
            # ------------------------------------------------------

            with st.spinner(
                "Reading handwritten marking scheme with TrOCR..."
            ):

                st.session_state.scheme_text = (
                    handwriting_ocr(
                        scheme_file
                    )
                )

            # ------------------------------------------------------
            # Handwritten student answer
            # ------------------------------------------------------

            with st.spinner(
                "Reading handwritten student answer with TrOCR..."
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
                "OCR completed successfully."
            )

        except Exception as exc:

            st.session_state.ocr_status = (
                "OCR failed."
            )

            st.error(
                "OCR failed."
            )

            st.exception(
                exc
            )


# ======================================================================
# OCR EDITING
# ======================================================================

if (
    st.session_state.question_text
    or st.session_state.scheme_text
    or st.session_state.answer_text
):

    st.header(
        "3. Review and Correct OCR"
    )

    st.info(
        "You can edit any OCR result before marking. "
        "This is important because handwriting and image OCR "
        "can occasionally make transcription errors."
    )

    st.session_state.question_text = st.text_area(
        "Typed Question Paper OCR",
        value=st.session_state.question_text,
        height=350,
        key="question_editor",
    )

    st.session_state.scheme_text = st.text_area(
        "Handwritten Marking Scheme OCR",
        value=st.session_state.scheme_text,
        height=350,
        key="scheme_editor",
    )

    st.session_state.answer_text = st.text_area(
        "Handwritten Student Answer OCR",
        value=st.session_state.answer_text,
        height=500,
        key="answer_editor",
    )


# ======================================================================
# QUESTION PARSER PREVIEW
# ======================================================================

if st.session_state.question_text.strip():

    st.header(
        "4. Parsed Questions"
    )

    questions = parse_questions(
        st.session_state.question_text
    )

    if questions:

        for question in questions:

            marks_text = (
                f"{question['max_marks']} marks"
                if question["max_marks"] is not None
                else "marks not detected"
            )

            with st.expander(
                f"Question {question['number']} "
                f"({marks_text})"
            ):

                st.write(
                    question["question"]
                )

    else:

        st.warning(
            "No numbered questions were detected. "
            "The entire question paper will be treated as one question."
        )


# ======================================================================
# MARK EXAMINATION
# ======================================================================

st.header(
    "5. Mark Examination"
)

api_key_available = bool(
    get_openai_api_key()
)

if not api_key_available:

    st.warning(
        "OPENAI_API_KEY is not configured. "
        "OCR can still be tested, but examination marking "
        "requires an OpenAI API key in Streamlit Secrets."
    )

mark_button = st.button(
    "🧮 Mark Examination",
    type="primary",
    use_container_width=True,
)


if mark_button:

    if not st.session_state.question_text.strip():

        st.error(
            "The question paper text is empty."
        )

    elif not st.session_state.scheme_text.strip():

        st.error(
            "The marking scheme text is empty."
        )

    elif not st.session_state.answer_text.strip():

        st.error(
            "The student answer text is empty."
        )

    elif not get_openai_api_key():

        st.error(
            "OPENAI_API_KEY is missing."
        )

    else:

        try:

            with st.spinner(
                "Marking examination..."
            ):

                result = mark_examination(
                    question_text=(
                        st.session_state.question_text
                    ),
                    scheme_text=(
                        st.session_state.scheme_text
                    ),
                    answer_text=(
                        st.session_state.answer_text
                    ),
                    model_name=marking_model,
                )

            st.session_state.marking_result = result

            st.success(
                "Examination marked successfully."
            )

        except Exception as exc:

            st.error(
                "Examination marking failed."
            )

            st.exception(
                exc
            )


# ======================================================================
# RESULTS
# ======================================================================

if st.session_state.marking_result:

    display_marking_results(
        st.session_state.marking_result
    )


# ======================================================================
# RESET
# ======================================================================

st.divider()

if st.button(
    "♻️ Reset Examination",
    use_container_width=True,
):

    for key, value in DEFAULT_STATE.items():

        st.session_state[key] = value

    st.rerun()
