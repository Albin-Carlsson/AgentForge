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
# CPU-only PyTorch first: the container has no GPU, and the default wheel adds
# several GB of CUDA libraries (the same thing that filled CI's disk).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e .

# Pre-download both models the shipped config uses: the embedder and the
# reranker, which ships on after measuring +0.181 nDCG@10 on the arXiv tier
# (results/retrieval-arxiv40.md).
RUN python -c "\
from sentence_transformers import CrossEncoder, SentenceTransformer; \
SentenceTransformer('BAAI/bge-base-en-v1.5', cache_folder='/models'); \
CrossEncoder('BAAI/bge-reranker-base', cache_folder='/models')" \
    && find /models -name '*.onnx' -delete

# Serve from the baked models only. Without this the hub is still contacted at
# runtime, a network dependency the baked models exist to remove; a missing
# model now fails loudly instead of downloading.
ENV HF_HUB_OFFLINE=1

COPY Makefile README.md ./
COPY results ./results

EXPOSE 8000

# One worker: the job table is in-process state, so a second worker would poll a
# job it is not running. Scaling out means moving to a real queue, and the
# corpus does not warrant one.
CMD ["uvicorn", "agentforge.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
