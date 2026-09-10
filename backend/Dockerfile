# NOTE: build this image from the REPO ROOT so both backend/ and
# frontend/ are inside the build context, e.g.:
#   docker build -f backend/Dockerfile -t document-intelligence .
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Mirror the repo layout (backend/ + frontend/ as siblings) so the relative
# path used in app/main.py to locate the frontend folder resolves the same
# way in the container as it does locally.
WORKDIR /app

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend/app backend/app
COPY frontend frontend

WORKDIR /app/backend
ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
