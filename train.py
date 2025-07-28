# train.py
"""
Main training script for the language model.
Orchestrates loading data, tokenizer, model, and initiating the training process.
"""

import os
import torch
from src.config import *  # Import configuration (including DEVICE)
from src.tokenizer import SimpleTokenizer
from src.dataset import LanguageModelDataset
from src.model import SimpleTransformerLM
from src.train_utils import train  # Import the training function

def main():
    """
    Main function to execute the training pipeline.
    """
    print("--- Starting Training Pipeline ---")

    # --- 1. Load or Build Tokenizer ---
    tokenizer = None
    try:
        # Try loading an existing tokenizer
        tokenizer = SimpleTokenizer.from_file(VOCAB_PATH)
        print(f"Loaded existing tokenizer with {tokenizer.vocab_size} tokens.")
    except FileNotFoundError:
        print("Existing tokenizer not found. Attempting to build from data...")
        # If vocab file doesn't exist, try building it from the training data
        if os.path.exists(DATA_PATH):
            print(f"Found training data at {DATA_PATH}. Building vocabulary...")
            with open(DATA_PATH, 'r', encoding='utf-8') as f:
                data_lines = f.readlines()
            # Create a temporary tokenizer instance to build vocab
            temp_tokenizer = SimpleTokenizer({})
            temp_tokenizer.build_from_text(data_lines, vocab_path=VOCAB_PATH)
            # Reload the tokenizer we just saved
            tokenizer = SimpleTokenizer.from_file(VOCAB_PATH)
            print(f"Built and loaded new tokenizer with {tokenizer.vocab_size} tokens.")
        else:
            print(f"Error: Training data file not found at {DATA_PATH}. "
                  f"Please create the file or check the path in src/config.py.")
            return # Exit if no data or tokenizer

    # Update VOCAB_SIZE in config if needed by other parts (though using tokenizer.vocab_size is preferred)
    # For now, we'll just ensure the tokenizer is loaded correctly.

    # --- 2. Load Training Dataset ---
    if not os.path.exists(DATA_PATH):
        print(f"Error: Training data file not found at {DATA_PATH} for dataset creation.")
        return

    print(f"Loading training data from {DATA_PATH}...")
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data_lines = f.readlines()

    print("Creating training dataset...")
    dataset = LanguageModelDataset(data_lines, tokenizer, context_length=SEQ_LEN)
    print(f"Dataset created with {len(dataset)} samples.")

    # --- 3. Initialize or Load Model ---
    print("Initializing model...")
    # Pass parameters explicitly based on your SimpleTransformerLM definition
    # Ensure these match the __init__ signature in src/model.py
    model = SimpleTransformerLM(
        vocab_size=tokenizer.vocab_size, # Use vocab size from tokenizer
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        num_layers=NUM_LAYERS
        # dropout=0.1 # Add if your model definition includes it and you want to set it
    )
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters.")

    # Optional: Load weights from a previous checkpoint if it exists
    # This allows resuming training. We'll load the latest epoch checkpoint.
    start_epoch = 0
    if os.path.exists(CHECKPOINT_DIR):
        checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("model_epoch_") and f.endswith(".pth")]
        if checkpoint_files:
            # Sort by epoch number to find the latest
            checkpoint_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
            latest_checkpoint = checkpoint_files[-1]
            checkpoint_path = os.path.join(CHECKPOINT_DIR, latest_checkpoint)
            try:
                # Load state dict strictly, assuming structure matches
                model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
                start_epoch = int(latest_checkpoint.split("_")[-1].split(".")[0])
                print(f"Resumed training from checkpoint: {checkpoint_path} (Epoch {start_epoch})")
            except Exception as e:
                print(f"Warning: Could not load checkpoint {checkpoint_path}. Starting fresh. Error: {e}")
                start_epoch = 0 # Reset if loading fails
        else:
            print("No existing checkpoints found. Starting training from scratch.")
    else:
         print(f"Checkpoint directory {CHECKPOINT_DIR} does not exist. Will create during training.")


    # --- 4. Start Training ---
    print("Starting training loop...")
    train(
        model=model,
        dataset=dataset,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        device=DEVICE, # Use device from config
        save_path=CHECKPOINT_DIR, # Save checkpoints to the specified directory
        start_epoch=start_epoch # Pass the starting epoch for checkpointing
    )
    print("Training loop completed.")

    # --- 5. Save Final Model ---
    print(f"Saving final model to {MODEL_SAVE_PATH}...")
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH) # Save only the state dict
    print("Final model saved successfully.")

    print("--- Training Pipeline Completed ---")


if __name__ == "__main__":
    main()

#EOF