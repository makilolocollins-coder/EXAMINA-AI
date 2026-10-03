import streamlit as st
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Examina AI",
    page_icon="📝",
    layout="wide"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_ID = "stepfun-ai/GOT-OCR-2.0-hf"


# ============================================================
# LOAD GOT-OCR 2.0
# ============================================================

@st.cache_resource(show_spinner="Loading GOT-OCR 2.0...")
def load_model():

    # Detect available hardware
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16

        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True
        )

    else:
        device = "cpu"
        dtype = torch.float32

        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        )

        model.to("cpu")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        use_fast=True
    )

    model.eval()

    return processor, model, device


# ============================================================
# OCR FUNCTION
# ============================================================

def extract_text(image, processor, model, device):

    # Process the complete page.
    # GOT-OCR can dynamically divide large pages into patches.
    inputs = processor(
        image,
        return_tensors="pt",
        format=False,
        crop_to_patches=True,
        max_patches=3
    )

    # Move tensors to the correct device
    processed_inputs = {}

    for key, value in inputs.items():

        if hasattr(value, "to"):
            processed_inputs[key] = value.to(device)
        else:
            processed_inputs[key] = value

    # Generate OCR result
    with torch.inference_mode():

        generated_ids = model.generate(
            **processed_inputs,
            do_sample=False,
            tokenizer=processor.tokenizer,
            stop_strings="<|im_end|>",
            max_new_tokens=4096
        )

    # Remove input/prompt tokens from generated output
    input_length = processed_inputs["input_ids"].shape[1]

    generated_text = processor.decode(
        generated_ids[0][input_length:],
        skip_special_tokens=True
    )

    return generated_text.strip()


# ============================================================
# APPLICATION HEADER
# ============================================================

st.title("📝 Examina AI")

st.subheader("Whole-Page Handwritten OCR")

st.write(
    "Upload a handwritten examination page and Examina AI "
    "will use GOT-OCR 2.0 to extract the written content."
)

st.info(
    "For best results, use a clear, well-lit image of the "
    "complete examination page."
)


# ============================================================
# LOAD MODEL
# ============================================================

try:

    processor, model, device = load_model()

    st.success(
        f"GOT-OCR 2.0 is ready • Running on {device.upper()}"
    )

except Exception as error:

    st.error("GOT-OCR 2.0 could not be loaded.")

    st.code(
        str(error),
        language="text"
    )

    st.stop()


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a handwritten examination page",
    type=[
        "png",
        "jpg",
        "jpeg",
        "webp"
    ]
)


# ============================================================
# IMAGE PREVIEW
# ============================================================

if uploaded_file is not None:

    try:

        image = Image.open(uploaded_file).convert("RGB")

        st.markdown("### Uploaded Examination Page")

        st.image(
            image,
            caption="Handwritten examination page",
            use_container_width=True
        )

        st.caption(
            f"Image dimensions: {image.width} × {image.height} pixels"
        )

    except Exception as error:

        st.error("The uploaded image could not be opened.")

        st.code(
            str(error),
            language="text"
        )

        st.stop()


    # ========================================================
    # OCR BUTTON
    # ========================================================

    if st.button(
        "🔍 Extract Handwriting",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "Reading the handwritten examination page..."
        ):

            try:

                result = extract_text(
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
                        height=600
                    )

                    st.download_button(
                        "⬇️ Download Text",
                        data=result,
                        file_name="examina_ocr_result.txt",
                        mime="text/plain",
                        use_container_width=True
                    )

                else:

                    st.warning(
                        "No text was detected. "
                        "Try uploading a clearer image."
                    )

            except Exception as error:

                st.error("OCR processing failed.")

                st.code(
                    str(error),
                    language="text"
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Examina AI | Whole-page OCR powered by GOT-OCR 2.0"
)
