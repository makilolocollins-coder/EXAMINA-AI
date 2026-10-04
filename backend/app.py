from fastapi import FastAPI, UploadFile, File
from PIL import Image

import io


app = FastAPI(
    title="Examina AI OCR API",
    version="1.0.0"
)


@app.get("/")
def home():

    return {
        "status": "online",
        "service": "Examina AI OCR",
        "message": "OCR backend is running"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.post("/ocr")
async def ocr(
    file: UploadFile = File(...)
):

    contents = await file.read()


    try:

        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

    except Exception:

        return {
            "success": False,
            "text": "",
            "error": "Invalid image file"
        }


    print(
        f"Received examination page: "
        f"{image.width} x {image.height}"
    )


    # ========================================================
    # TEMPORARY RESPONSE
    #
    # GOT-OCR WILL BE CONNECTED HERE LATER.
    # ========================================================

    return {

        "success": True,

        "text":
            "The Examina AI OCR backend received "
            "your examination page successfully.",

        "width": image.width,

        "height": image.height

    }
