import streamlit as st
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "stepfun-ai/GOT-OCR-2.0-hf"

st.set_page_config(
    page_title="Examina AI OCR Test",
    page_icon="📝"
)

st.title("📝 Examina AI")
st.write("GOT-OCR 2.0 diagnostic test")


@st.cache_resource
def load_got():

    st.write("1. Loading processor...")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID
    )

    st.write("✅ Processor loaded")

    st.write("2. Loading model...")

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )

    st.write("✅ Model loaded")

    model.eval()

    return processor, model


try:

    processor, model = load_got()

    st.success("GOT-OCR 2.0 is loaded successfully.")

except Exception as e:

    st.error("MODEL LOADING FAILED")

    st.exception(e)

    st.stop()


uploaded = st.file_uploader(
    "Upload one handwritten page",
    type=["jpg", "jpeg", "png", "webp"]
)


if uploaded:

    image = Image.open(uploaded).convert("RGB")

    st.image(
        image,
        caption="Uploaded page",
        use_container_width=True
    )

    if st.button(
        "TEST OCR",
        type="primary"
    ):

        st.write("3. Preparing image...")

        # Keep the test image reasonably small
        max_size = 2000

        if max(image.size) > max_size:

            scale = max_size / max(image.size)

            image = image.resize(
                (
                    int(image.width * scale),
                    int(image.height * scale)
                ),
                Image.Resampling.LANCZOS
            )

        st.write(
            f"Test image size: {image.width} × {image.height}"
        )

        try:

            st.write("4. Running processor...")

            inputs = processor(
                image,
                return_tensors="pt"
            )

            st.write("✅ Processor completed")

            st.write("5. Preparing tensors...")

            for key in inputs:

                if torch.is_tensor(inputs[key]):

                    inputs[key] = inputs[key].to("cpu")

            st.write("✅ Tensors prepared")

            st.write("6. Starting GOT-OCR generation...")

            with torch.inference_mode():

                output = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=False
                )

            st.write("✅ Generation completed")

            st.write("7. Decoding result...")

            result = processor.decode(
                output[0],
                skip_special_tokens=True
            )

            st.success("OCR completed!")

            st.text_area(
                "OCR RESULT",
                result,
                height=500
            )

        except Exception as e:

            st.error("OCR PROCESSING FAILED")

            st.exception(e)
