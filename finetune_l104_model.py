# finetune_l104_model.py
# Phase 2 of Project Distill L104: Fine-Tuning
#
# This script uses the generated dataset to fine-tune a pre-trained
# language model. The result is a new model specialized in understanding
# the L104 architecture and philosophy.
#
# !! WARNING !!
# This script is computationally intensive and requires a CUDA-enabled GPU
# with significant VRAM. It will download a multi-gigabyte base model.
# It is designed to be run by a user in a dedicated machine learning environment.

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from pathlib import Path

# --- Configuration ---
# We use a smaller, powerful base model that is efficient to fine-tune.
# Phi-3-mini is a great candidate. Other options: 'mistralai/Mistral-7B-v0.1', 'gemma-2b'
BASE_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
DATASET_FILE = "l104_finetune_dataset.jsonl"
OUTPUT_MODEL_DIR = "./l104_finetuned_model"

def format_dataset_entry(entry):
    """
    Formats each JSON object from our dataset into a single string
    that's suitable for instruction fine-tuning.
    """
    # This is a standard instruction format.
    return f"<|user|>\n{entry['question']}<|end|>\n<|assistant|>\n{entry['answer']}<|end|>\n"

def main():
    """Main function to run the fine-tuning process."""
    print("--- L104 Model Fine-Tuning (Phase 2) ---")
    
    # --- Step 1: Check Environment ---
    if not torch.cuda.is_available():
        print("WARNING: No CUDA-enabled GPU found. This process will be extremely slow.")
        print("         It is highly recommended to run this on a machine with a GPU.")
    else:
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")

    dataset_path = Path(DATASET_FILE)
    if not dataset_path.exists():
        print(f"ERROR: Dataset file not found at '{DATASET_FILE}'.")
        print("Please run 'create_l104_finetune_dataset.py' first to generate it.")
        return

    # --- Step 2: Load and Prepare Data ---
    print(f"Loading dataset from '{DATASET_FILE}'...")
    dataset = load_dataset('json', data_files=str(dataset_path))['train']

    print("Loading tokenizer...")
    # The 'trust_remote_code=True' is often needed for newer models.
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    # A pad token is required for batching. We can use the EOS token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print("Formatting and tokenizing dataset...")
    formatted_dataset = dataset.map(lambda x: {"text": format_dataset_entry(x)})
    tokenized_dataset = formatted_dataset.map(
        lambda examples: tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512),
        batched=True
    )

    # --- Step 3: Load the Base Model ---
    print(f"Loading base model '{BASE_MODEL_NAME}'...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16, # Use bfloat16 for efficiency
        device_map="auto" # Automatically use GPU if available
    )

    # --- Step 4: Configure and Run Training ---
    print("Configuring training arguments...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_MODEL_DIR,
        num_train_epochs=3,  # 3 epochs is a good starting point
        per_device_train_batch_size=2, # Adjust based on your VRAM
        gradient_accumulation_steps=4, # Increase effective batch size
        learning_rate=2e-5,
        logging_steps=10,
        save_steps=50,
        fp16=False, # We use bf16 from the model loading
        bf16=True,
        optim="adamw_torch",
        report_to="none" # Disable reporting to services like W&B
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print("\n--- Starting Fine-Tuning ---")
    trainer.train()
    print("--- Fine-Tuning Complete ---")

    # --- Step 5: Save the Final Model ---
    print(f"Saving final model to '{OUTPUT_MODEL_DIR}'...")
    trainer.save_model(OUTPUT_MODEL_DIR)
    tokenizer.save_pretrained(OUTPUT_MODEL_DIR)
    
    print("\n--- Process Complete ---")
    print(f"Your fine-tuned L104 model is ready at: {OUTPUT_MODEL_DIR}")
    print("Next step: Convert the model to GGUF format using 'convert_to_gguf.sh'.")


if __name__ == "__main__":
    main()
