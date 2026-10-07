FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY codexa_api ./codexa_api
RUN pip install --no-cache-dir .

# The packs and the embedding model: downloaded from the project's bucket by CI before the build.
COPY data ./data
COPY models ./models

ENV CODEXA_DATA_DIR=/app/data \
    CODEXA_MODEL_DIR=/app/models/embeddinggemma-300m-ONNX \
    PORT=8080
CMD ["sh", "-c", "exec uvicorn codexa_api.main:app --host 0.0.0.0 --port $PORT"]
