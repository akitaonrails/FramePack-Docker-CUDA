#!/bin/bash
set -e

# Ensure directories have correct ownership
if [ "$(stat -c '%u' /app/outputs)" != "$UID" ]; then
    echo "Fixing permissions for /app/outputs"
    sudo chown -R $UID:$GID /app/outputs
fi

if [ "$(stat -c '%u' /app/hf_download)" != "$UID" ]; then
    echo "Fixing permissions for /app/hf_download"
    sudo chown -R $UID:$GID /app/hf_download
fi

# NOTE: No model pre-download check here. Upstream FramePack downloads its own
# models (HunyuanVideo, Flux, etc.) to /app/hf_download on first run via
# HuggingFace. The previous SDXL/Real-ESRGAN checklist was leftover from a
# different project and blocked startup for everyone (see issue #1).

exec "$@"
