import streamlit as st
import torch
from PIL import Image
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Makky07 TrOCR Test",
    page_icon="✍️",
    layout="centered"
)

st.title("✍️ Makky07 TrOCR Handwriting Tester")
st.write("Test the handwritten OCR model before integrating it into Examina AI.")

# ============================================================
# MODEL
# ============================================================

MODEL_ID = "Makky07/Trocr"

@st.cache_resource
def load_model():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    st.write(f"Loading model on: `{device}`")

    # Load processor
    processor = TrOCRProcessor.from_pretrained(MODEL_ID)

    # Load model configuration
    model = VisionEncoderDecoderModel.from_pretrained(
        MODEL_ID,
        ignore_mismatched_sizes=False
    )

    model.to(device)
    model.eval()

    return processor, model, device


# ============================================================
# LOAD MODEL
# ============================================================

try:

    with st.spinner("Loading Makky07/Trocr..."):
        processor, model, device = load_model()

    st.success("Model loaded successfully.")

except Exception as e:

    st.error("Failed to load the model.")

    st.exception(e)

    st.stop()


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a handwritten image",
    type=["png", "jpg", "jpeg", "webp"]
)


# ============================================================
# OCR
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")

    st.image(
        image,
        caption="Handwritten input",
        use_container_width=True
    )

    st.divider()

    if st.button("🔍 Recognize Handwriting", type="primary"):

        with st.spinner("Recognizing handwriting..."):

            try:

                # ------------------------------------------------
                # PREPROCESS IMAGE
                # ------------------------------------------------

                pixel_values = processor(
                    images=image,
                    return_tensors="pt"
                ).pixel_values

                pixel_values = pixel_values.to(device)

                # ------------------------------------------------
                # GENERATE TEXT
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

                text = processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )[0]

                # ------------------------------------------------
                # DISPLAY
                # ------------------------------------------------

                st.subheader("OCR Result")

                if text.strip():

                    st.text_area(
                        "Recognized text",
                        text,
                        height=150
                    )

                else:

                    st.warning(
                        "The model did not produce readable text."
                    )

            except Exception as e:

                st.error("OCR failed.")

                st.exception(e)


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander("Model configuration"):

    st.write("**Model:** `Makky07/Trocr`")
    st.write("**Task:** Handwritten image → text")
    st.write("**Architecture:** VisionEncoderDecoderModel")
    st.write("**Encoder:** ViT")
    st.write("**Input size:** 384 × 384")
    st.write("**Device:**", device)
