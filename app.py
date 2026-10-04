import streamlit as st
import torch
from PIL import Image
from transformers import AutoProcessor, PaddleOCRVLForConditionalGeneration

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PaddleOCR-VL Handwriting Tester",
    page_icon="📝",
    layout="wide"
)

st.title("📝 PaddleOCR-VL Handwritten English OCR")
st.write(
    "Upload a handwritten examination page and test how well "
    "PaddleOCR-VL can transcribe it."
)

# ============================================================
# MODEL
# ============================================================

MODEL_ID = "PaddlePaddle/PaddleOCR-VL"


@st.cache_resource
def load_model():

    processor = AutoProcessor.from_pretrained(
        MODEL_ID
    )

    if torch.cuda.is_available():
        model = PaddleOCRVLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
            device_map="auto"
        )
    else:
        model = PaddleOCRVLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            dtype=torch.float32
        )

    model.eval()

    return processor, model


# ============================================================
# MODEL STATUS
# ============================================================

with st.spinner("Loading PaddleOCR-VL..."):

    try:
        processor, model = load_model()

        st.success("PaddleOCR-VL loaded successfully.")

        if torch.cuda.is_available():
            st.info(
                f"GPU detected: {torch.cuda.get_device_name(0)}"
            )
        else:
            st.warning(
                "No GPU detected. OCR may be very slow."
            )

    except Exception as e:

        st.error("Failed to load PaddleOCR-VL.")

        st.exception(e)

        st.stop()


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a handwritten examination page",
    type=["jpg", "jpeg", "png", "webp"]
)


# ============================================================
# OCR
# ============================================================

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Examination Page")

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            image,
            caption="Handwritten examination page",
            use_container_width=True
        )

    with col2:

        st.subheader("OCR Result")

        if st.button(
            "🔍 Run Handwriting OCR",
            type="primary"
        ):

            prompt = """
OCR:

Transcribe all handwritten English text in this examination page.

Instructions:

1. Read the entire page.
2. Preserve the original reading order.
3. Preserve question numbers.
4. Preserve paragraphs and line breaks where possible.
5. Transcribe the student's handwriting exactly.
6. Do not summarize.
7. Do not explain the answer.
8. Do not correct spelling or grammar.
9. If a word is genuinely unreadable, write [UNCLEAR].
10. Return only the transcription.
"""

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]

            try:

                with st.spinner(
                    "Reading handwritten examination page..."
                ):

                    text = processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )

                    inputs = processor(
                        text=[text],
                        images=[image],
                        return_tensors="pt"
                    )

                    # Move tensors to model device
                    if torch.cuda.is_available():

                        inputs = {
                            key: value.to("cuda")
                            if hasattr(value, "to")
                            else value
                            for key, value in inputs.items()
                        }

                    with torch.inference_mode():

                        generated_ids = model.generate(
                            **inputs,
                            max_new_tokens=2048,
                            do_sample=False
                        )

                    result = processor.batch_decode(
                        generated_ids,
                        skip_special_tokens=True
                    )[0]

                    # Remove prompt if returned
                    if text in result:
                        result = result.split(
                            text,
                            1
                        )[-1]

                    result = result.strip()

                st.success("OCR completed.")

                st.text_area(
                    "Transcribed handwriting",
                    value=result,
                    height=500
                )

                st.download_button(
                    "⬇️ Download transcription",
                    data=result,
                    file_name="examina_ocr_result.txt",
                    mime="text/plain"
                )

            except Exception as e:

                st.error("OCR failed.")

                st.exception(e)
