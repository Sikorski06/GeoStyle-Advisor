FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models/ models/
COPY src/ src/
COPY data/raw/test_data.mp4 data/raw/test_data.mp4
COPY data/processed/ data/processed/

EXPOSE 8501

CMD ["streamlit", "run", "src/app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]