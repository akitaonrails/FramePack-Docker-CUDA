FROM nvidia/cuda:12.6.0-devel-ubuntu22.04 AS base

# Set user/group IDs to match host user (default 1000 for first user)
ARG UID=1000
ARG GID=1000
# Control compilation parallelism (4 jobs ~16GB RAM, 8 jobs ~32GB RAM, 64 jobs ~500GB RAM)
ARG MAX_JOBS=4

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/app/venv \
    PATH="/app/venv/bin:$PATH" \
    USER=appuser \
    MAX_JOBS=$MAX_JOBS

# Layer 1: Install system dependencies (most stable, cached longest)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ninja-build \
    sudo \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Layer 2: Create user and basic directory structure early
RUN groupadd -g $GID appuser && \
    useradd -u $UID -g $GID -m -s /bin/bash appuser && \
    echo "appuser ALL=(ALL) NOPASSWD: /bin/chown" >> /etc/sudoers && \
    mkdir -p /app /app/outputs /app/hf_download && \
    chown -R $UID:$GID /app

# Layer 3: Setup Python environment and clone repository (stable)
USER $UID:$GID
WORKDIR /app
RUN python3.10 -m venv $VIRTUAL_ENV && \
    python -m pip install --upgrade pip wheel setuptools && \
    git clone https://github.com/lllyasviel/FramePack /tmp/framepack && \
    cp -r /tmp/framepack/* /app/ && \
    rm -rf /tmp/framepack

# Layer 4: Install PyTorch ecosystem with specific CUDA version
RUN pip install --no-cache-dir \
    torch==2.6.0 \
    torchvision==0.21.0 \
    torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Layer 5: Install FramePack requirements (from the official requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

# Layer 6: Install performance enhancements - triton and xformers
RUN pip install --no-cache-dir triton==3.2.0 xformers

# Layer 7: Install sageattention (needs compilation from source)
RUN python -m pip install --no-cache-dir sageattention==1.0.6

# Layer 8: Install flash-attn (needs special compilation)
RUN python -m pip -v install --no-cache-dir flash-attn --no-build-isolation

# Verify PyTorch installation works correctly and check operator availability
RUN python -c "import torch; import torchvision; import torchaudio; print(f'PyTorch: {torch.__version__}, TorchVision: {torchvision.__version__}, TorchAudio: {torchaudio.__version__}'); print('CUDA available:', torch.cuda.is_available()); import torchvision.ops; print('torchvision.ops loaded successfully')"

# ===== APPLICATION STAGE =====
FROM base AS application

ARG UID=1000
ARG GID=1000

# Layer 9: Copy entrypoint script (changes occasionally)
USER root
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && \
    chown $UID:$GID /entrypoint.sh

# Final configuration
USER $UID:$GID
WORKDIR /app
EXPOSE 7860

# Configure volumes
VOLUME /app/hf_download
VOLUME /app/outputs

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "demo_gradio.py", "--share", "--server", "0.0.0.0"]
