FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src/ ./src/

RUN pip install --no-cache-dir -e . && \
    pip cache purge

FROM python:3.12-slim

WORKDIR /app

# Copy installed packages and app code from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

ENV DOTMD_INDEX_DIR=/dotmd-index
ENV DOTMD_DATA_DIR=/data

VOLUME ["/data", "/dotmd-index"]

ENTRYPOINT ["dotmd"]
