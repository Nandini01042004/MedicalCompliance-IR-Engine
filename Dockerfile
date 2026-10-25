# Industry-Grade Multi-Target Container for Medical Compliance Engine
FROM python:3.12-slim

WORKDIR /app

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and data
COPY . .

# Generate corpus and qrels
RUN python data/build_corpus.py && python data/build_qrels.py

EXPOSE 8000 8501

CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
