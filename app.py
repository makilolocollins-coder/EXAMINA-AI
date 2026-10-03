import streamlit as st
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


# ============================================================
# CONFIG
# ============================================================

HANDWRITING_MODEL = "microsoft/trocr-base-handwritten"


st.set_page_config(
    page_title="Examina AI - Handwriting OCR",
    page_icon="✍️",
    layout="wide",
)


# ============================================================
# PAGE
# ============================================================

st.title("✍️ Examina AI")
st.subheader("Handwritten Answer OCR Tester")

st.write(
    "Upload a clear image containing handwritten text. "
    "The app will use Microsoft's TrOCR handwritten model "
    "to convert the handwriting into digital text."
)

st.info(
    "For best results, upload one handwritten text line or "
    "a small cropped section of an answer sheet."
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

st.caption(f"Device: `{device}`")


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_trocr():

    processor = TrOCRProcessor.from_pretrained(
        HANDWRITING_MODEL
    )

    model = VisionEncoderDecoderModel.from_pretrained(
        HANDWRITING_MODEL
    )

    model.to(device)
    model.eval()

    return processor, model


# ============================================================
# LOAD MODEL WITH ERROR HANDLING
# ============================================================

try:

    with st.spinner(
        "Loading handwritten OCR model from Hugging Face..."
    ):

        processor, model = load_trocr()

    st.success(
        "✅ Handwriting OCR model loaded successfully."
    )

except Exception as e:

    st.error("❌ Could not load the handwriting model.")

    st.write(
        "The app could not download or initialize "
        "Microsoft TrOCR."
    )

    st.exception(e)

    st.stop()


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.divider()

uploaded_file = st.file_uploader(
    "Upload handwritten text",
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

    # Generate text
    with torch.no_grad():

        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=128,
            num_beams=4,
            early_stopping=True,
        )

    # Decode
    generated_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return generated_text.strip()


# ============================================================
# DISPLAY IMAGE
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("Uploaded handwriting")

    st.image(
        image,
        caption="Input image",
        use_container_width=True
    )

    st.divider()

    # ========================================================
    # OCR BUTTON
    # ========================================================

    if st.button(
        "🔍 Recognize Handwriting",
        type="primary",
        use_container_width=True,
    ):

        try:

            with st.spinner(
                "Reading handwriting..."
            ):

                result = recognize_handwriting(image)

            st.subheader("Recognized Text")

            if result:

                st.text_area(
                    "OCR output",
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
                "❌ OCR failed."
            )

            st.exception(e)


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

with st.expander("Model information"):

    st.write(
        "**Model:** "
        "`microsoft/trocr-base-handwritten`"
    )

    st.write(
        "**Purpose:** Handwritten text recognition"
    )

    st.write(
        "**Architecture:** TrOCR / VisionEncoderDecoderModel"
    )

    st.write(
        "**Source:** Hugging Face"
    )

    st.write(
        "**Device:** "
        f"`{device}`"
    )


# ============================================================
# IMPORTANT NOTE
# ============================================================

st.warning(
    "TrOCR works best with individual handwritten text lines. "
    "For full exam sheets, the next version should detect and "
    "crop individual handwriting lines before sending them to "
    "TrOCR."
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Examina AI • Handwritten Answer OCR"
)
