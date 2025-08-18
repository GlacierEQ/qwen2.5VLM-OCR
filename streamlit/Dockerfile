FROM python:3.11-slim
WORKDIR /app
# Install system deps
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    git \
    && rm -rf /var/lib/apt/lists/*
# Install packaging first to resolve flash-attn dependency
RUN pip install --no-cache-dir packaging
# Install python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
