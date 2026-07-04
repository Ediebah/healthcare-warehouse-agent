# Container for the Clinical Insight Agent (Streamlit). Portable across Cloud Run / Hugging Face
# Spaces (Docker) / Render / Railway / Fly. The OpenAI key is supplied at RUNTIME as the
# OPENAI_API_KEY env var — it is never baked into the image.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    PORT=8080

WORKDIR /app

# Install the lean runtime deps first (cached layer) — requirements.txt, NOT requirements-dev (no dbt).
COPY requirements.txt .
RUN pip install -r requirements.txt

# App code + the committed slim demo warehouse. The full 200MB DB is excluded via .dockerignore,
# so warehouse._resolve_db_path() falls back to data/healthcare_demo.duckdb.
COPY . .

EXPOSE 8080

# Shell form so ${PORT} expands at runtime (Cloud Run / Render / Railway inject PORT; default 8080).
# CORS/XSRF are disabled because Streamlit runs behind the platform's HTTPS proxy (standard recipe).
CMD streamlit run app.py \
    --server.port=${PORT:-8080} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false
