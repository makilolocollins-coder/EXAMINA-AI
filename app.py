import streamlit as st
from PIL import Image
from surya.inference import SuryaInferenceManager
from surya.recognition import RecognitionPredictor


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Examina AI - Handwritten OCR",
    page_icon="✍️",
    layout="wide",
)


# ============================================================
# TITLE
# ============================================================

st.title("✍️ Examina AI")

st.subheader("Full-Page Handwritten Answer OCR")

st.write(
    "Upload a complete handwritten exam page and Surya OCR 2 "
    "will attempt to recognize the handwriting, preserve the "
    "reading order, and separate the page into text blocks."
)

st.info(
    "For the best results, upload a clear, well-lit photograph "
    "or scan of one complete exam page."
)


# ============================================================
# LOAD SURYA
# ============================================================

@st.cache_resource
def load_surya():

    # Surya automatically manages the OCR inference backend.
    manager = SuryaInferenceManager()

    recognizer = RecognitionPredictor(manager)

    return manager, recognizer


# ============================================================
# MODEL INITIALIZATION
# ============================================================

try:

    with st.spinner(
        "Loading Surya OCR 2..."
    ):

        manager, recognizer = load_surya()

    st.success(
        "✅ Surya OCR 2 is ready."
    )

except Exception as e:

    st.error(
        "❌ Surya OCR could not be initialized."
    )

    st.write(
        "This usually means that the Surya package or its "
        "inference backend could not be installed or started."
    )

    st.exception(e)

    st.stop()


# ============================================================
# FILE UPLOAD
# ============================================================

st.divider()

uploaded_file = st.file_uploader(
    "Upload a handwritten exam page",
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

def run_full_page_ocr(image):

    # Surya expects a PIL image.
    image = image.convert("RGB")

    # --------------------------------------------------------
    # Full-page OCR
    #
    # No manual line cropping is required here.
    # Surya OCR 2 performs full-page recognition.
    # --------------------------------------------------------

    results = recognizer(
        [image],
        full_page=True
    )

    return results[0]


# ============================================================
# DISPLAY UPLOADED IMAGE
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    # --------------------------------------------------------
    # IMAGE INFORMATION
    # --------------------------------------------------------

    width, height = image.size

    st.subheader("Uploaded page")

    st.caption(
        f"Image size: {width} × {height} pixels"
    )

    st.image(
        image,
        caption="Handwritten exam page",
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # OCR BUTTON
    # --------------------------------------------------------

    if st.button(
        "🔍 Read Entire Page",
        type="primary",
        use_container_width=True,
    ):

        try:

            with st.spinner(
                "Reading the handwritten page..."
            ):

                page_result = run_full_page_ocr(image)

            st.success(
                "✅ Page processing completed."
            )

            # ==================================================
            # RESULTS
            # ==================================================

            st.subheader("Recognized Answer")

            blocks = getattr(
                page_result,
                "blocks",
                []
            )

            if not blocks:

                st.warning(
                    "No readable text blocks were detected."
                )

            else:

                # ------------------------------------------------
                # Sort blocks by reading order
                # ------------------------------------------------

                blocks = sorted(
                    blocks,
                    key=lambda block: getattr(
                        block,
                        "reading_order",
                        0
                    )
                )

                recognized_parts = []

                for block in blocks:

                    # --------------------------------------------
                    # Get text
                    # --------------------------------------------

                    text = getattr(
                        block,
                        "html",
                        ""
                    )

                    if text is None:
                        text = ""

                    text = str(text).strip()

                    if not text:
                        continue

                    recognized_parts.append(
                        text
                    )

                # ------------------------------------------------
                # Combine blocks
                # ------------------------------------------------

                final_text = "\n\n".join(
                    recognized_parts
                )

                if final_text:

                    st.text_area(
                        "OCR output",
                        value=final_text,
                        height=450,
                    )

                    st.success(
                        f"✅ Recognized "
                        f"{len(recognized_parts)} text blocks."
                    )

                else:

                    st.warning(
                        "Surya detected page content, "
                        "but no readable text was returned."
                    )

            # ==================================================
            # BLOCK DETAILS
            # ==================================================

            if blocks:

                st.divider()

                with st.expander(
                    "View detected page blocks"
                ):

                    for index, block in enumerate(
                        blocks,
                        start=1
                    ):

                        label = getattr(
                            block,
                            "label",
                            "Unknown"
                        )

                        confidence = getattr(
                            block,
                            "confidence",
                            None
                        )

                        block_text = getattr(
                            block,
                            "html",
                            ""
                        )

                        if block_text is None:
                            block_text = ""

                        block_text = str(
                            block_text
                        ).strip()

                        st.markdown(
                            f"### Block {index}"
                        )

                        st.write(
                            f"**Type:** {label}"
                        )

                        if confidence is not None:

                            try:

                                st.write(
                                    f"**Confidence:** "
                                    f"{float(confidence):.3f}"
                                )

                            except Exception:

                                pass

                        if block_text:

                            st.code(
                                block_text,
                                language="text"
                            )

            # ==================================================
            # RAW RESULT
            # ==================================================

            with st.expander(
                "View raw Surya result"
            ):

                try:

                    st.json(
                        page_result.model_dump()
                    )

                except Exception:

                    st.write(
                        page_result
                    )

        except Exception as e:

            st.error(
                "❌ Full-page OCR failed."
            )

            st.exception(e)


# ============================================================
# INFORMATION
# ============================================================

st.divider()

with st.expander(
    "About this OCR system"
):

    st.write(
        "**OCR engine:** Surya OCR 2"
    )

    st.write(
        "**Mode:** Full-page OCR"
    )

    st.write(
        "**Input:** Complete handwritten exam page"
    )

    st.write(
        "**Output:** Recognized text blocks in reading order"
    )

    st.write(
        "**Model source:** Datalab / Surya OCR 2"
    )


# ============================================================
# EXAMINA AI PIPELINE
# ============================================================

with st.expander(
    "Examina AI processing pipeline"
):

    st.markdown(
        """
        **Step 1 — Student uploads answer**

        ↓

        **Step 2 — Surya OCR 2 reads the complete page**

        ↓

        **Step 3 — Text blocks are reconstructed in reading order**

        ↓

        **Step 4 — Recognized answer is prepared for marking**

        ↓

        **Step 5 — Marking scheme is compared with the answer**

        ↓

        **Step 6 — Score and feedback are generated**
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Examina AI • Full-Page Handwritten OCR"
)
