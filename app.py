import streamlit as st
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText


# ============================================================
# CONFIG
# ============================================================

MODEL_ID = "stepfun-ai/GOT-OCR-2.0-hf"

st.set_page_config(
    page_title="Examina AI",
    page_icon="📝",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        use_fast=True
    )

    if device == "cuda":

        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )

    else:

        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )

        model.to("cpu")

    model.eval()

    return processor, model, device


# ============================================================
# OCR
# ============================================================

def perform_ocr(image, processor, model, device):

    # Resize extremely large images to reduce memory usage
    max_dimension = 2500

    if max(image.size) > max_dimension:

        ratio = max_dimension / max(image.size)

        new_size = (
            int(image.width * ratio),
            int(image.height * ratio)
        )

        image = image.resize(
            new_size,
            Image.Resampling.LANCZOS
        )

    # Prepare image
    inputs = processor(
        image,
        return_tensors="pt",
        crop_to_patches=True,
        max_patches=3,
        format=False
    )

    # Move tensors to device
    for key in inputs:

        if torch.is_tensor(inputs[key]):
            inputs[key] = inputs[key].to(device)

    # Generate
    with torch.inference_mode():

        output_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=2048
        )

    # Decode
    input_length = inputs["input_ids"].shape[1]

    result = processor.decode(
        output_ids[0][input_length:],
        skip_special_tokens=True
    )

    return result.strip()


# ============================================================
# HEADER
# ============================================================

st.title("📝 Examina AI")

st.subheader("Whole-Page Handwritten OCR")

st.write(
    "Upload a complete handwritten examination page "
    "and extract the written text."
)


# ============================================================
# LOAD MODEL
# ============================================================

with st.spinner("Loading GOT-OCR 2.0..."):

    try:

        processor, model, device = load_model()

        st.success(
            f"Model loaded successfully on {device.upper()}"
        )

    except Exception as e:

        st.error("Could not load GOT-OCR 2.0.")

        st.exception(e)

        st.stop()


# ============================================================
# UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload handwritten examination page",
    type=["jpg", "jpeg", "png", "webp"]
)


# ============================================================
# DISPLAY IMAGE
# ============================================================

if uploaded_file:

    try:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded examination page",
            use_container_width=True
        )

        st.write(
            f"Original image: "
            f"{image.width} × {image.height} pixels"
        )

    except Exception as e:

        st.error("Could not open the image.")

        st.exception(e)

        st.stop()


    # ========================================================
    # OCR BUTTON
    # ========================================================

    if st.button(
        "🔍 READ HANDWRITING",
        type="primary",
        use_container_width=True
    ):

        progress = st.progress(0)

        status = st.empty()

        try:

            status.info("Preparing the examination page...")
            progress.progress(20)

            status.info(
                "GOT-OCR 2.0 is analysing the handwritten page..."
            )
            progress.progress(40)

            result = perform_ocr(
                image,
                processor,
                model,
                device
            )

            progress.progress(100)

            status.success("OCR completed.")

            st.markdown("## Extracted Text")

            if result:

                st.text_area(
                    "Editable OCR Result",
                    value=result,
                    height=600
                )

                st.download_button(
                    "⬇️ Download Text",
                    data=result,
                    file_name="examina_handwriting.txt",
                    mime="text/plain",
                    use_container_width=True
                )

            else:

                st.warning(
                    "GOT-OCR did not return any text."
                )

        except Exception as e:

            progress.empty()

            status.error("OCR failed.")

            st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Examina AI • GOT-OCR 2.0"
)
