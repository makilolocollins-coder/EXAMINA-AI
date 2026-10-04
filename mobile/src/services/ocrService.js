const OCR_SERVER_URL = "https://YOUR-OCR-SERVER-URL";


export async function performOCR(imageUri) {

  const formData = new FormData();

  formData.append("file", {
    uri: imageUri,
    name: "exam_page.jpg",
    type: "image/jpeg"
  });


  const response = await fetch(
    `${OCR_SERVER_URL}/ocr`,
    {
      method: "POST",

      headers: {
        Accept: "application/json",
        "Content-Type": "multipart/form-data"
      },

      body: formData
    }
  );


  if (!response.ok) {

    const errorText = await response.text();

    throw new Error(
      `OCR server returned ${response.status}: ${errorText}`
    );
  }


  return await response.json();
}
