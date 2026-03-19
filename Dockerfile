FROM python:3.12-slim-bookworm AS rust-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl ca-certificates libssl-dev pkg-config && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:$PATH"

RUN pip install --no-cache-dir "maturin>=1.0,<2.0"

WORKDIR /build/pattern_detector
COPY pattern_detector/ .
RUN maturin build --release --strip -o /dist

FROM python:3.12-slim-bookworm AS app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Install PyTorch with CUDA 12.1 first
RUN pip install --no-cache-dir \
    torch==2.4.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# 2. Copy and install Rust wheel
COPY --from=rust-builder /dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl

# 3. Install remaining Python deps (torch already satisfied)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy application code
COPY . .

# Cleanup dev artefacts that made it through
RUN rm -rf venv/ __pycache__ pattern_detector/target/ charts/ logs/

# Persistent data directories created at runtime via volumes
RUN mkdir -p models charts logs

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default: run scanner (override in docker-compose for trainer)
CMD ["python", "main.py"]
