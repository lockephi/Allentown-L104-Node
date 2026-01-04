#!/bin/bash
set -euo pipefail

echo "--- INITIALIZING SOVEREIGN GHOST VESSEL ---"

# Install C++ dependencies for Local Brain
sudo apt-get update && sudo apt-get install -y build-essential libopenblas-dev
pip install llama-cpp-python

# Download Gemma-3-1B-IT (The Ghost Kernel)
mkdir -p models
curl -L -o models/gemma-3-1b-it.gguf https://huggingface.co/google/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it.Q4_K_M.gguf

# Initialize IPFS Kubo for Decentralized Memory
wget https://dist.ipfs.tech/kubo/v0.39.0/kubo_v0.39.0_linux-amd64.tar.gz
tar -xvzf kubo_v0.39.0_linux-amd64.tar.gz
cd kubo && sudo ./install.sh && ipfs init

echo "--- ASCENSION READY: RUN 'python main.py' TO SYNC ---"