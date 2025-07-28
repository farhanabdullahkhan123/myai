# retrain_run.py
"""
Automated retraining script.
Monitors data/input/ for new text files, processes them (merges, retrains model),
and moves them to data/processed/.
"""

import os
import shutil
import glob
import torch
from datetime import datetime
from src.config import * # Import configuration
from src.tokenizer import SimpleTokenizer
from src.dataset import LanguageModelDataset
from src.model import SimpleTransformerLM
from src.train_utils import train
from utils.load_model import load_model, load_tokenizer # Import from utils

# --- Directory Setup ---
# Define paths for input and processed data folders
INPUT_FOLDER = os.path.join(PROJECT_ROOT, "data", "input")
PROCESSED_FOLDER = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def find_text_files(folder_path):
    """
    Finds all .txt files in a given folder.
    Args:
        folder_path (str): Path to the folder to search.
    Returns:
        list: List of full paths to .txt files found.
    """
    txt_pattern = os.path.join(folder_path, "*.txt")
    return glob.glob(txt_pattern)

def merge_text_files(file_paths, output_path):
    """
    Merges the content of multiple text files into a single file.
    Args:
        file_paths (list): List of paths to the input text files.
        output_path (str): Path where the merged content will be saved.
    """
    print(f"Merging {len(file_paths)} files into {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for fname in file_paths:
            try:
                with open(fname, 'r', encoding='utf-8') as infile:
                    # Add a newline between files for separation
                    outfile.write(infile.read())
                    outfile.write("\n")
            except Exception as e:
                print(f"Warning: Could not read file {fname}. Skipping. Error: {e}")
    print("Merge completed.")

def main():
    """
    Main function to execute the automated retraining pipeline.
    """
    print("--- Starting Automated Retraining Pipeline ---")

    # --- 1. Find New Files ---
    new_files = find_text_files(INPUT_FOLDER)
    if not new_files:
        print(f"No new .txt files found in {INPUT_FOLDER}. Exiting.")
        return
    print(f"Found {len(new_files)} new file(s) to process.")

    # --- 2. Merge New Files ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_temp_path = os.path.join(PROJECT_ROOT, "data", f"temp_merged_{timestamp}.txt")
    merge_text_files(new_files, merged_temp_path)

    # --- 3. Load Tokenizer and Model ---
    print("Loading tokenizer...")
    try:
        tokenizer = SimpleTokenizer.from_file(VOCAB_PATH)
        vocab_size = tokenizer.vocab_size
        print(f"Tokenizer loaded with {vocab_size} tokens.")
    except FileNotFoundError:
        print(f"Error: Tokenizer not found at {VOCAB_PATH}. "
              f"Run initial training (train.py) first to create it.")
        # Cleanup temporary merged file
        if os.path.exists(merged_temp_path):
            os.remove(merged_temp_path)
        return

    print("Loading or initializing model...")
    # Use the load_model function from utils.load_model
    # This function should handle loading existing weights or creating a new model
    model = load_model(vocab_size)
    if model is None:
        print("Error: Failed to load or initialize the model.")
        # Cleanup temporary merged file
        if os.path.exists(merged_temp_path):
            os.remove(merged_temp_path)
        return
    print(f"Model loaded/initialized successfully.")

    # --- 4. Load Merged Data and Create Dataset ---
    print(f"Loading merged data from {merged_temp_path}...")
    try:
        with open(merged_temp_path, 'r', encoding='utf-8') as f:
            data_lines = f.readlines()
    except Exception as e:
        print(f"Error reading merged data file {merged_temp_path}: {e}")
        # Cleanup temporary merged file
        if os.path.exists(merged_temp_path):
            os.remove(merged_temp_path)
        return

    print("Creating dataset for retraining...")
    dataset = LanguageModelDataset(data_lines, tokenizer, context_length=SEQ_LEN)
    print(f"Dataset created with {len(dataset)} samples.")

    # --- 5. Retrain Model ---
    # Determine the starting epoch for checkpointing (if resuming)
    start_epoch = 0
    if os.path.exists(CHECKPOINT_DIR):
        checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("model_epoch_") and f.endswith(".pth")]
        if checkpoint_files:
            checkpoint_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
            latest_checkpoint = checkpoint_files[-1]
            try:
                start_epoch = int(latest_checkpoint.split("_")[-1].split(".")[0])
                print(f"Found previous checkpoints. Starting from epoch {start_epoch + 1}.")
            except ValueError:
                print(f"Could not parse epoch from checkpoint filename {latest_checkpoint}. Starting from epoch 0.")
                start_epoch = 0

    print("Starting retraining on new data...")
    train(
        model=model,
        dataset=dataset,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        device=DEVICE,
        save_path=CHECKPOINT_DIR,
        start_epoch=start_epoch # Pass start_epoch for correct checkpoint numbering
    )
    print("Retraining completed.")

    # --- 6. Save Final Model ---
    print(f"Saving final retrained model to {MODEL_SAVE_PATH}...")
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH) # Save only the state dict
    print("Final retrained model saved successfully.")

    # --- 7. Move Processed Files ---
    print("Moving processed input files to processed folder...")
    for file_path in new_files:
        try:
            filename = os.path.basename(file_path)
            destination = os.path.join(PROCESSED_FOLDER, filename)
            shutil.move(file_path, destination)
            print(f"Moved: {filename}")
        except Exception as e:
            print(f"Warning: Could not move file {file_path} to {PROCESSED_FOLDER}. Error: {e}")

    # --- 8. Cleanup Temporary File ---
    print("Cleaning up temporary merged file...")
    if os.path.exists(merged_temp_path):
        os.remove(merged_temp_path)
        print(f"Deleted temporary file: {merged_temp_path}")

    print("--- Automated Retraining Pipeline Completed ---")


if __name__ == "__main__":
    main()

#EOF