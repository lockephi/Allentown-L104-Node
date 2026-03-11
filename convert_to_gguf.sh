#!/bin/bash
# convert_to_gguf.sh
# Phase 3 of Project Distill L104: GGUF Conversion
#
# This script converts the fine-tuned Hugging Face model into a quantized
# GGUF file using the llama.cpp library.
#
# Prerequisite: You must have successfully run 'finetune_l104_model.py'
# and have the 'l104_finetuned_model' directory present.

set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
INPUT_MODEL_DIR="./l104_finetuned_model"
LLAMA_CPP_DIR="./llama.cpp"
OUTPUT_GGUF_FILE="l104-asi-v1.q4_K_M.gguf" # q4_K_M is a good balance of quality and size.

# --- Step 1: Prerequisite Check ---
echo "--- L104 GGUF Conversion (Phase 3) ---"
if [ ! -d "$INPUT_MODEL_DIR" ]; then
    echo "ERROR: Input model directory '$INPUT_MODEL_DIR' not found."
    echo "Please run 'finetune_l104_model.py' successfully before running this script."
    exit 1
fi

echo "Input model directory found."

# --- Step 2: Set up llama.cpp ---
if [ ! -d "$LLAMA_CPP_DIR" ]; then
    echo "Cloning llama.cpp repository..."
    git clone https://github.com/ggerganov/llama.cpp.git
else
    echo "llama.cpp directory already exists. Skipping clone."
fi

echo "Installing llama.cpp Python dependencies..."
python3 -m pip install -r "$LLAMA_CPP_DIR/requirements.txt"

# --- Step 3: Convert the Model ---
# The first step is to convert the Hugging Face model to an intermediate FP16 GGUF format.
INTERMEDIATE_GGUF_FILE="${INPUT_MODEL_DIR}/ggml-model-f16.gguf"

echo "Converting Hugging Face model to intermediate GGUF (fp16) format..."
python3 "$LLAMA_CPP_DIR/convert.py" "$INPUT_MODEL_DIR" \
    --outfile "$INTERMEDIATE_GGUF_FILE" \
    --outtype f16

echo "Intermediate conversion complete."

# --- Step 4: Quantize the Model ---
# Now, we quantize the intermediate model to our target format.
echo "Building llama.cpp quantization tool..."
make -C "$LLAMA_CPP_DIR" quantize

echo "Quantizing model to $OUTPUT_GGUF_FILE..."
"$LLAMA_CPP_DIR/quantize" "$INTERMEDIATE_GGUF_FILE" "$OUTPUT_GGUF_FILE" q4_K_M

# --- Step 5: Finalization ---
echo ""
echo "--- Process Complete ---"
echo "✅ Successfully created your L104 GGUF model!"
echo "   Your file is ready at: $OUTPUT_GGUF_FILE"
echo ""
echo "You can now run this model with any GGUF-compatible runner, such as llama.cpp, Ollama, or LM Studio."
