# Two-stage so the local model weights are baked in rather than downloaded on
# first request -- a cold start that silently pulls 530 MB would make the first
# latency measurement meaningless.
FROM python:3.13-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    AF_MODEL_DIR=/models \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

# PyMuPDF ships manylinux wheels, so no build toolchain is needed.
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e .

# Pre-download the embedding model. The reranker is NOT fetched: it ships
# disabled by default after measuring as a 4.4-point nDCG regression on SciFact
# (see results/retrieval-scifact.md), so pulling 90 MB for a code path that is
# off would be waste.
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('BAAI/bge-base-en-v1.5', cache_folder='/models')" \
    && find /models -name '*.onnx' -delete

COPY Makefile README.md ./
COPY results ./results

EXPOSE 8000

# One worker: the job table is in-process state, so a second worker would poll a
# job it is not running. Scaling out means moving to a real queue, and the
# corpus does not warrant one.
CMD ["uvicorn", "agentforge.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
