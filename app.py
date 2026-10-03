import streamlit as st
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_ID = "stepfun-ai/GOT-OCR-2.0-hf"

st.set_page_config(
    page_title="Examina AI - Handwritten OCR",
    page_icon="📝",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    st.info(f"Loading GOT-OCR 2.0 on {device.upper()}...")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        use_fast=True
    )

    if device == "cuda":

        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    else:

        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )

        model.to(device)

    model.eval()

    return processor, model, device


# ============================================================
# OCR FUNCTION
# ============================================================

def run_ocr(image, processor, model, device):

    # GOT-OCR supports dynamic image patching.
    # This is useful for full exam pages.
    inputs = processor(
        image,
        return_tensors="pt",
        format=False,
        crop_to_patches=True,
        max_patches=3
    )

    # Move tensors to the model device
    inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

    with torch.inference_mode():

        generated_ids = model.generate(
            **inputs,
            do_sample=False,
            tokenizer=processor.tokenizer,
            stop_strings="<|im_end|>",
            max_new_tokens=4096
        )

    # Remove the prompt tokens
    input_length = inputs["input_ids"].shape[1]

    output = processor.decode(
        generated_ids[0][input_length:],
        skip_special_tokens=True
    )

    return output.strip()


# ============================================================
# USER INTERFACE
# ============================================================

st.title("📝 Examina AI")
st.subheader("Whole-Page Handwritten OCR")

st.write(
    "Upload a handwritten examination page and GOT-OCR 2.0 "
    "will attempt to convert the handwriting into editable text."
)

st.warning(
    "For best results, upload a clear, well-lit image with the "
    "entire examination page visible."
)


# ============================================================
# MODEL
# ============================================================

try:

    processor, model, device = load_model()

    st.success(
        f"GOT-OCR 2.0 loaded successfully on {device.upper()}."
    )

except Exception as e:

    st.error("Failed to load GOT-OCR 2.0.")

    st.exception(e)

    st.stop()


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload handwritten exam page",
    type=["png", "jpg", "jpeg", "webp"]
)


# ============================================================
# PROCESS IMAGE
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.markdown("### Uploaded Page")

    st.image(
        image,
        caption="Handwritten examination page",
        use_container_width=True
    )

    st.write(
        f"Image size: {image.width} × {image.height} pixels"
    )

    if st.button(
        "🔍 Extract Handwriting",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "Reading the handwritten page... This may take some time."
        ):

            try:

                result = run_ocr(
                    image,
                    processor,
                    model,
                    device
                )

                st.markdown("### Extracted Text")

                if result:

                    st.text_area(
                        "OCR Result",
                        value=result,
                        height=500
                    )

                    st.download_button(
                        label="⬇️ Download OCR Text",
                        data=result,
                        file_name="examina_ocr_result.txt",
                        mime="text/plain",
                        use_container_width=True
                    )

                else:

                    st.warning(
                        "No text was detected. Try a clearer image."
                    )

            except Exception as e:

                st.error("OCR processing failed.")

                st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Examina AI • Handwritten OCR powered by GOT-OCR 2.0"
)
