# L104 Model Integration Guide

> **Last updated**: 2026-03-10 | **Author**: OpenClaw Assistant

## 1. Overview

This guide provides a complete, step-by-step process for wiring the custom-trained L104 GGUF model into your OpenClaw agent.

Following this guide will change your agent's "brain" from a general-purpose model to a specialist that has been fine-tuned on the L104 codebase. This will give it a deep, intrinsic understanding of the project's architecture, philosophy, and purpose.

The process is broken into four phases:
1.  **Model Generation:** Using the scripts we created to produce the `.gguf` file.
2.  **Serving the Model:** Running a local server to host the GGUF model.
3.  **Reconfiguring OpenClaw:** Instructing your OpenClaw agent to use the local model.
4.  **Verification:** Confirming the agent is running with its new brain.

---

## Phase 1: Model Generation

This phase uses the three Python and shell scripts we created in the `Allentown-L104-Node` repository. **This entire phase should be run in a powerful environment with a modern NVIDIA GPU.**

### Step 1.1: Generate the Training Dataset

First, run the data-generation script. This will scan the codebase and databases to create the training data.

```bash
cd /path/to/Allentown-L104-Node
python3 create_l104_finetune_dataset.py
```
**Expected Output:** A new file named `l104_finetune_dataset_v2.jsonl` will be created, containing several thousand question-answer pairs.

### Step 1.2: Fine-Tune the Model

Next, run the fine-tuning script. This will download a powerful base model and train it on your dataset. This is the most time-consuming and resource-intensive step.

```bash
# Ensure you have the required libraries
pip3 install torch transformers datasets accelerate bitsandbytes

# Run the fine-tuning script
python3 finetune_l104_model.py
```
**Expected Output:** A new directory named `l104_finetuned_model/` will be created, containing the newly trained model files.

### Step 1.3: Convert to GGUF

Finally, run the conversion script. This will package your fine-tuned model into a single, efficient GGUF file.

```bash
# Make the script executable
chmod +x convert_to_gguf.sh

# Run the conversion
./convert_to_gguf.sh
```
**Expected Output:** A new file named `l104-asi-v1.q4_K_M.gguf` will be created. This is your specialized L104 model.

---

## Phase 2: Serving the Local LLM

Now that you have your model, you need to run a server that makes it available via an OpenAI-compatible API. We will use `llama-cpp-python` for this.

### Step 2.1: Install the Server

From your terminal, install `llama-cpp-python` with server support. It's best to do this in the same virtual environment as the L104 project.

```bash
# Activate your venv first if you haven't
source /path/to/Allentown-L104-Node/.venv/bin/activate

# Install with server dependencies
pip3 install "llama-cpp-python[server]"
```

### Step 2.2: Launch the Server

Navigate to the `Allentown-L104-Node` directory and run the following command. This will start a web server that "serves" your model.

```bash
python3 -m llama_cpp.server --model l104-asi-v1.q4_K_M.gguf --n_gpu_layers 1
```
*Note: The `--n_gpu_layers 1` flag offloads some work to the GPU. You can increase this number if you have a powerful GPU with a lot of VRAM.*

If successful, you will see output indicating the server is running, typically on `http://localhost:8000`. Keep this terminal window open.

---

## Phase 3: Reconfiguring OpenClaw

This is the final step. You need to tell your OpenClaw agent (the one I am running in) to use this new local server instead of its default model provider.

This is done by setting environment variables in the terminal where you launch OpenClaw.

```bash
# The provider name for local, OpenAI-compatible servers
export OPENCLAW_MODEL_PROVIDER="openai_compatible"

# The URL of the server you just started
export OPENCLAW_API_BASE_URL="http://localhost:8000/v1"

# The "name" of the model (this can often be anything for local servers)
export OPENCLAW_MODEL_NAME="l104-asi-v1"

# Your local server doesn't require an API key
export OPENCLAW_API_KEY="none"
```

After setting these environment variables, **restart your OpenClaw agent**.

---

## Phase 4: Verification

To verify that I am now running with my new, specialized L104 brain, simply ask me a question that only the L104 model would know. For example:

> "What is the formula for the VOID_CONSTANT?"

If I answer correctly based on the L104-specific knowledge, the integration was a success. You will have successfully upgraded my core reasoning engine.
