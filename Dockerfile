# SiamRAG — CPU-only, deploy-ready (not published; see README "Deploying")
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.hf-cache \
    DATA_DIR=/app/data

WORKDIR /app

# torch CPU first (keeps the image small; no CUDA wheels)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# bake the embedding model + sample PDFs into the image so first boot is fast
COPY scripts/ scripts/
COPY app/ app/
RUN python scripts/make_sample_pdfs.py && \
    python -c "from sentence_transformers import SentenceTransformer; \
               SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu')"

COPY data/ data/
COPY eval.py .

EXPOSE 8000 7860

# API on :8000. Gradio UI: docker run -e SIAMRAG_API=http://127.0.0.1:8000 ... python -m app.ui
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
