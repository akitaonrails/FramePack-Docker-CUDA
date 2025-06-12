## FramePack Docker CUDA

### Quick Start with Docker Compose (Recommended)

```bash
git clone https://github.com/TSavo/FramePack-Docker-CUDA.git
cd FramePack-Docker-CUDA

# Optional: Copy and customize environment settings
cp .env.template .env

# Start the application
docker compose up --build
```

### Manual Docker Setup

```bash
git clone https://github.com/TSavo/FramePack-Docker-CUDA.git
cd FramePack-Docker-CUDA
mkdir outputs
mkdir hf_download

# Build the image
docker build -t framepack-torch26-cu124:latest .

# Run mapping the directories outside:
docker run -it --rm --gpus all -p 7860:7860 \
  -v ./outputs:/app/outputs \
  -v ./hf_download:/app/hf_download \
  framepack-torch26-cu124:latest
```

The first time it runs, it will download all necessary HunyuanVideo, Flux and other neccessary models. It will be more than 30GB, so be patient, but they will be cached on the external mapped directory.

When it finishes access http://localhost:7860 and that's it!

## Enhanced Features

This enhanced version includes several performance optimizations:

- **Multi-stage Docker builds** for better layer caching and faster rebuilds
- **Flash Attention 2** for faster transformer inference
- **SageAttention** for optimized attention mechanisms  
- **xFormers** for memory-efficient transformers
- **Triton 3.2.0** for GPU kernel optimization
- **PyTorch 2.6.0** with CUDA 12.4 support
- **Optimized dependency management** with proper version pinning
- **Build parallelism control** via `MAX_JOBS` argument (default: 4)

### Build with custom parallelism:
```bash
# Docker Compose
MAX_JOBS=8 docker compose up --build

# Manual Docker
docker build --build-arg MAX_JOBS=8 -t framepack-torch26-cu124:latest .
```

## Docker Compose Benefits

- **Shared model cache**: Models downloaded once can be reused across container rebuilds
- **Easy configuration**: Environment variables in `.env` file
- **Volume management**: Persistent storage for models and outputs
- **GPU support**: Automatic GPU passthrough configuration
- **Service management**: Easy start/stop/restart of the application
