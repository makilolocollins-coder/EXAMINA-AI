import streamlit as st
import torch
from PIL import Image
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel
)
from huggingface_hub import hf_hub_download


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Makky07 TrOCR Tester",
    page_icon="✍️",
    layout="centered"
)

st.title("✍️ Makky07 TrOCR Handwriting Tester")

st.write(
    "Test the actual Makky07/Trocr handwritten OCR model."
)


# ============================================================
# MODEL SETTINGS
# ============================================================

MODEL_ID = "Makky07/Trocr"
WEIGHT_FILE = "Trocr_model.bin"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------------
    # PROCESSOR
    # --------------------------------------------------------

    processor = TrOCRProcessor.from_pretrained(
        MODEL_ID
    )

    # --------------------------------------------------------
    # MODEL CONFIG
    # --------------------------------------------------------

    model = VisionEncoderDecoderModel.from_pretrained(
        MODEL_ID,
        weights_only=False
    )

    # --------------------------------------------------------
    # DOWNLOAD ACTUAL WEIGHTS
    # --------------------------------------------------------

    weight_path = hf_hub_download(
        repo_id=MODEL_ID,
        filename=WEIGHT_FILE
    )

    # --------------------------------------------------------
    # LOAD STATE DICT
    # --------------------------------------------------------

    checkpoint = torch.load(
        weight_path,
        map_location="cpu"
    )

    # Some checkpoints store the state dictionary directly.
    # Others store it under a "state_dict" key.

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:

        state_dict = checkpoint["state_dict"]

    else:

        state_dict = checkpoint

    # --------------------------------------------------------
    # LOAD WEIGHTS
    # --------------------------------------------------------

    missing_keys, unexpected_keys = model.load_state_dict(
        state_dict,
        strict=False
    )

    # --------------------------------------------------------
    # MOVE MODEL
    # --------------------------------------------------------

    model.to(device)
    model.eval()

    return (
        processor,
        model,
        device,
        missing_keys,
        unexpected_keys
    )


# ============================================================
# LOAD
# ============================================================

try:

    with st.spinner(
        "Downloading and loading Makky07/Trocr..."
    ):

        (
            processor,
            model,
            device,
            missing_keys,
            unexpected_keys
        ) = load_model()

    st.success(
        f"Model loaded successfully on {device.upper()}"
    )

    if missing_keys:

        st.warning(
            f"Missing keys: {len(missing_keys)}"
        )

    if unexpected_keys:

        st.warning(
            f"Unexpected keys: {len(unexpected_keys)}"
        )


except Exception as e:

    st.error("Failed to load Makky07/Trocr.")

    st.exception(e)

    st.stop()


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.divider()

uploaded_file = st.file_uploader(
    "Upload a handwritten image",
    type=[
        "png",
        "jpg",
        "jpeg",
        "webp"
    ]
)


# ============================================================
# OCR
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.subheader("Uploaded Image")

    st.image(
        image,
        caption="Handwritten input",
        use_container_width=True
    )

    st.divider()

    if st.button(
        "🔍 Recognize Handwriting",
        type="primary"
    ):

        with st.spinner(
            "Recognizing handwriting..."
        ):

            try:

                # ------------------------------------------------
                # PREPROCESS
                # ------------------------------------------------

                pixel_values = processor(
                    images=image,
                    return_tensors="pt"
                ).pixel_values

                pixel_values = pixel_values.to(
                    device
                )

                # ------------------------------------------------
                # GENERATE
                # ------------------------------------------------

                with torch.no_grad():

                    generated_ids = model.generate(
                        pixel_values,
                        max_new_tokens=100,
                        num_beams=4,
                        early_stopping=True
                    )

                # ------------------------------------------------
                # DECODE
                # ------------------------------------------------

                recognized_text = processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )[0]

                # ------------------------------------------------
                # RESULT
                # ------------------------------------------------

                st.subheader(
                    "📝 Recognized Text"
                )

                st.text_area(
                    "OCR output",
                    recognized_text,
                    height=180
                )

            except Exception as e:

                st.error(
                    "OCR recognition failed."
                )

                st.exception(e)


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander(
    "🔧 Model information"
):

    st.write(
        "**Hugging Face model:** "
        "`Makky07/Trocr`"
    )

    st.write(
        "**Task:** Handwritten image → text"
    )

    st.write(
        "**Architecture:** VisionEncoderDecoderModel"
    )

    st.write(
        "**Encoder:** ViT"
    )

    st.write(
        "**Input:** 384 × 384"
    )

    st.write(
        "**Device:** "
        + device
    )

    st.write(
        "**Weights:** `Trocr_model.bin`"
    )
