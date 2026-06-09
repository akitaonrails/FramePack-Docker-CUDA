# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

This repo contains **no application source code**. It is a Docker packaging layer that builds a CUDA-enabled container for [FramePack](https://github.com/lllyasviel/FramePack) (an image-to-video generation tool). The FramePack source is `git clone`d into the image during the build (Dockerfile Layer 3) — to read or change application logic you must look at the upstream repo, not this one.

The entire repo is four meaningful files: `Dockerfile`, `docker-compose.yml`, `entrypoint.sh`, and `run.sh` (plus `.env.template`). All work here is editing how the container is built and run.

## Common commands

```bash
# Build + run (recommended)
docker compose up --build

# Control compile parallelism (4 jobs ~16GB RAM, 8 ~32GB, 64 ~500GB)
MAX_JOBS=8 docker compose up --build

# Manual build / run
docker build -t framepack-torch28-cu128:latest .
./run.sh                      # docker run with --gpus all, port 7860, volume mounts
```

The Gradio UI is served on **http://localhost:7860**. First run downloads >30GB of models (HunyuanVideo, Flux, etc.) into the `hf_download` volume; this persists across rebuilds.

## Architecture / build design

- **Two-stage build** (`base` → `application`), but the real intent is **layer ordering for cache reuse**: system deps → user creation → venv + clone → PyTorch → requirements → perf libs (triton/xformers/sageattention/flash-attn). Heavy compiled deps (flash-attn, sageattention) are last so editing the entrypoint doesn't trigger a recompile. Preserve this ordering when adding dependencies — put cheap/frequently-changing things late.
- **Version pinning is load-bearing.** PyTorch 2.8.0 + torchvision 0.23.0 + torchaudio 2.8.0 are installed from the `cu128` wheel index and must move together; `triton==3.4.0` and `xformers==0.0.32.post2` are the builds matched to torch 2.8.0 (xformers also from the cu128 index). The base image (`cuda:12.8.1-devel`) and the cu128 wheels both target CUDA 12.8 — this is required for Blackwell (sm_120) GPUs. The older cu124 stack does **not** run on RTX 50xx at all.
- **`TORCH_CUDA_ARCH_LIST` arg** controls which GPU archs flash-attn compiles for (default `12.0+PTX`, i.e. Blackwell only — keeps the build fast on this RTX 5090 host). Override it (e.g. `8.0;8.6;8.9;9.0`) when building for Ampere/Ada/Hopper, or flash-attn falls back to PyTorch SDPA at runtime on unbuilt archs.
- **UID/GID passthrough** (`ARG UID/GID`, default 1000) exists so files written to mounted `outputs/` and `hf_download/` are owned by the host user, not root. The container runs as `appuser`, and `entrypoint.sh` uses a narrowly-scoped passwordless `sudo chown` (the only sudo command permitted, per `/etc/sudoers`) to fix volume ownership at startup. Keep this UID/GID threading intact in both `Dockerfile` and `docker-compose.yml` when modifying user/permission logic.
- **Volumes:** `hf_download` (model cache, a named bind volume in compose pointing at `HF_MODELS_PATH`) and `outputs` (generated videos, bind-mounted to `./outputs`). Both are gitignored.

## Gotchas

- `CMD` runs `demo_gradio.py --share`, which creates a public Gradio tunnel. Remove `--share` if you don't want a public URL.
- `entrypoint.sh` previously hard-failed startup on a stale SDXL/Real-ESRGAN `MODEL_CHECKLIST` (issue #1) — that check has been removed. FramePack downloads its own models to `hf_download` on first run; do not re-add a pre-flight model check.
- flash-attn compiles from source (sdist) and is the dominant build-time cost (tens of minutes). sageattention 1.0.6 is a pure-python wheel (no compile). Both are optional at runtime — FramePack falls back to PyTorch SDPA if an attention backend is missing or lacks kernels for the GPU arch.
