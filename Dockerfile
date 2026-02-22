# Bazowy obraz o obniżonym rozmiarze
FROM python:3.11-slim

# Ustawienie zmiennych środowiskowych
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8501

# Katalog roboczy
WORKDIR /app

# Instalacja zależności systemowych wymaganych przez OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Kopiowanie pliku zależności i instalacja
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kopiowanie struktury projektu
# (katalogi data/ i config/ zostaną skopiowane, ale zawartość logów i raw ignorujemy w .dockerignore)
COPY . .

# Wystawienie portu dla Streamlit
EXPOSE 8501

# Komenda uruchomieniowa wymuszająca nasłuch na wszystkich interfejsach
CMD ["streamlit", "run", "src/app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]