FROM python:3.10

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Download/copy model weights
# (In production, mount as volume or download from S3)
COPY models/ /app/models/

EXPOSE 8000 8501 6006

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
