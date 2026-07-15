# 1. Base Image selection
# Using Python 3.10 slim version for a smaller and cleaner container footprint.
FROM python:3.10-slim

# 2. System dependencies
# Install compiler tools and git which are required for some scientific packages (e.g. fair-esm, biopython).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 3. Working Directory setup
WORKDIR /app

# 4. Environment Variables configuration
# Prevent Python from writing .pyc files to disk and ensure stdout/stderr are unbuffered.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Configure paths for PyTorch/HuggingFace caching to enable persistent volume mounting.
ENV TORCH_HOME=/app/cache/torch
ENV HF_HOME=/app/cache/huggingface

# Set the default path to the FoldNet checkpoint file
ENV FOLDNET_CHECKPOINT_PATH=/app/results/checkpoints/foldnet-epoch=08-val_loss=0.5015.ckpt

# 5. Dependency installation
# Copy requirements first to leverage Docker's layer caching mechanism.
COPY requirements.txt .
# Install CPU-only PyTorch first to prevent downloading large CUDA binary wheels (like nvidia-cusolver) which often cause network timeouts inside Docker builds.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# 6. Copy Application Source Code
# Copy only required files/directories (leveraging .dockerignore)
COPY foldnet/ ./foldnet/
COPY viewer/ ./viewer/
COPY configs/ ./configs/
COPY data/ ./data/
COPY results/checkpoints/ ./results/checkpoints/
COPY run.py .

# 7. Port Exposure
# Expose the FastAPI application port (default 8000)
EXPOSE 8000

# 8. Start Application Command
# Start the uvicorn web server, binding it to 0.0.0.0 to allow external access.
CMD ["uvicorn", "viewer.app:app", "--host", "0.0.0.0", "--port", "8000"]
