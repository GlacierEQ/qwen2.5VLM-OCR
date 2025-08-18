# Qwen2.5-VL OCR Deployment

This project provides a Dockerized setup to run **Qwen2.5-VL-3B-Instruct-AWQ** with **vLLM** and a **Streamlit frontend** for performing OCR on PDF documents.

---

## Features

* Upload PDF files via Streamlit UI
* Rasterizes PDF pages to images
* Sends images to Qwen2.5-VL for text extraction
* View extracted text per page in the browser
* Download combined OCR output as a `.txt` file

---

## Requirements

* Docker & Docker Compose installed
* GPU with CUDA support (optional but recommended)
* Internet connection to download the model from Hugging Face
* vLLM API running (inside Docker)

---

## Project Structure

```
vllmocrexp/
├── docker-compose.yml
├── streamlit/
│   ├── app.py
│   ├── requirements.txt
│   └── Dockerfile
├── models/          # optional, local model storage
└── tmp/             # temporary PDFs/images (ignored in git)
```

---

## Setup & Deployment

1. **Clone the repo**

```bash
git clone <your-repo-url>
cd vllmocrexp
```

2. **Build and start the containers**

```bash
docker compose build --no-cache
docker compose up
```

* `qwen-vlm` container runs the Qwen2.5-VL model with vLLM
* `streamlit-ui` container runs the frontend at: [http://localhost:8501](http://localhost:8501)

3. **Upload a PDF in the Streamlit UI**

* Each page will be displayed as an image
* OCR text is extracted for each page
* Full text can be downloaded as `ocr_output.txt`

---

## Clean-up

To remove temporary files, Docker volumes, and images:

```bash
docker compose down -v
rm -rf tmp/
```

---

## License

MIT License.
