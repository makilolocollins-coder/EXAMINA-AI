import streamlit as st
from PIL import Image
import torch

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)

# ============================================================
# CONFIGURATION
# ============================================================

HANDWRITING_MODEL = "Makky07/Trocr"

st.set_page_config(
    page_title="Makky07 TrOCR Handwriting Tester",
    page_icon="✍️",
    layout="wide",
)

# ============================================================
# PAGE HEADER
# ============================================================

st.title("✍️ Makky07 TrOCR Handwriting Tester")

st.write(
    "Upload a handwritten image and test the Makky07/Trocr "
    "handwriting recognition model."
)

st.info(
    "This app currently tests handwritten OCR only. "
    "Typed-text OCR is not included here."
)

# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.caption(f"Running on: `{device}`")

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    try:
        # ----------------------------------------------------
        # Processor
        # ----------------------------------------------------
        processor = TrOCRProcessor.from_pretrained(
            HANDWRITING_MODEL
        )

        # ----------------------------------------------------
        # Model
        #
        # This requires the Hugging Face repository to contain
        # either:
        #
        #   model.safetensors
        #
        # or:
        #
        #   pytorch_model.bin
        # ----------------------------------------------------
        model = VisionEncoderDecoderModel.from_pretrained(
            HANDWRITING_MODEL
        )

        model.to(device)
        model.eval()

        return processor, model, None

    except Exception as e:
        return None, None, str(e)


processor, model, model_error = load_model()

# ============================================================
# MODEL STATUS
# ============================================================

if model_error:

    st.error("❌ Failed to load the handwriting model.")

    st.warning(
        "The Hugging Face repository must contain the actual "
        "model weights, such as `model.safetensors` or "
        "`pytorch_model.bin`."
    )

    st.code(
        model_error,
        language="text"
    )

    st.markdown("### Required files")

    st.code(
        """Makky07/Trocr/
├── config.json
├── preprocessor_config.json
├── tokenizer_config.json
├── special_tokens_map.json
├── tokenizer.json
├── vocab.json
├── merges.txt
└── model.safetensors
""",
        language="text"
    )

    st.markdown(
        "Once the model weights have been uploaded to "
        "`Makky07/Trocr`, restart the Streamlit app."
    )

    st.stop()

# ============================================================
# SUCCESS
# ============================================================

st.success("✅ Makky07/Trocr loaded successfully.")

# ============================================================
# IMAGE UPLOAD
# ============================================================

st.subheader("Upload handwritten text")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=[
        "png",
        "jpg",
        "jpeg",
        "webp",
        "bmp",
    ],
)

# ============================================================
# OCR FUNCTION
# ============================================================

def recognize_handwriting(image):

    # Convert image to RGB
    image = image.convert("RGB")

    # Process image
    pixel_values = processor(
        images=image,
        return_tensors="pt"
    ).pixel_values

    pixel_values = pixel_values.to(device)

    # --------------------------------------------------------
    # Generate prediction
    # --------------------------------------------------------

    with torch.no_grad():

        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=256,
            num_beams=4,
            early_stopping=True,
        )

    # --------------------------------------------------------
    # Decode
    # --------------------------------------------------------

    generated_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return generated_text.strip()


# ============================================================
# RUN OCR
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("Uploaded image")

    st.image(
        image,
        caption="Handwritten input",
        use_container_width=True
    )

    st.divider()

    if st.button(
        "🔍 Recognize Handwriting",
        type="primary",
        use_container_width=True
    ):

        with st.spinner("Reading handwriting..."):

            try:

                result = recognize_handwriting(image)

                st.subheader("OCR Result")

                if result:

                    st.text_area(
                        "Recognized text",
                        value=result,
                        height=200,
                    )

                    st.success(
                        "✅ Handwriting recognition completed."
                    )

                else:

                    st.warning(
                        "The model did not return any text."
                    )

            except Exception as e:

                st.error(
                    "❌ An error occurred while running OCR."
                )

                st.exception(e)

# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander("Model information"):

    st.write(
        f"**Hugging Face model:** `{HANDWRITING_MODEL}`"
    )

    st.write(
        f"**Device:** `{device}`"
    )

    st.write(
        "**Purpose:** Handwritten text recognition"
    )

    st.write(
        "**Architecture:** TrOCR / VisionEncoderDecoderModel"
    )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Makky07 TrOCR Handwriting Tester"
)

#This is  repository, this same app should be able to load them.
