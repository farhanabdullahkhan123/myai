# download_data.py
"""
Script to download a subset of the TinyStories dataset and save it as plain text.
This file can then be used as training data for your language model.
"""

from datasets import load_dataset
import os

def main():
    """
    Main function to download and process the dataset.
    """
    print("Loading TinyStories dataset...")
    # Load the TinyStories dataset from Hugging Face
    # We'll use the 'default' configuration which is the full English version
    dataset = load_dataset("roneneldan/TinyStories", streaming=False) # streaming=False loads the full dataset into memory (be careful with large subsets)

    # Specify the subset you want to use for training
    # Options are typically 'train' and 'validation'
    split_to_use = "train" # You can change this to "validation" for a smaller set

    # Define how many examples (stories) you want to use
    # The full 'train' set is very large. Let's start with a manageable subset.
    # Adjust this number based on your needs and system resources.
    # A few thousand stories should be a good starting point.
    num_examples_to_save = 5000 # You can increase this later

    print(f"Selecting {num_examples_to_save} examples from the '{split_to_use}' split...")

    # Get the iterator for the selected split
    data_iter = iter(dataset[split_to_use])

    # Define the output path (matching your existing data path)
    output_file_path = os.path.join("data", "greetings.txt")
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True) # Ensure data/ directory exists

    print(f"Writing data to {output_file_path}...")

    story_count = 0
    with open(output_file_path, 'w', encoding='utf-8') as f_out:
        try:
            for example in data_iter:
                if story_count >= num_examples_to_save:
                    break
                # Each example in TinyStories is a dictionary, usually with a 'text' key
                story_text = example['text']
                # Write the story text to the file, followed by a newline for separation
                f_out.write(story_text.strip() + "\n")
                # Add an extra newline between stories for clarity (optional)
                f_out.write("\n")
                story_count += 1
                # Print progress every 500 stories
                if story_count % 500 == 0:
                    print(f"Written {story_count} stories...")
        except Exception as e:
            print(f"An error occurred during download/write: {e}")
            return

    print(f"Finished writing {story_count} stories to {output_file_path}")
    print("You can now use this file for training your model.")
    print("Remember to delete old checkpoints and the final model before retraining:")
    print(" - Delete contents of: checkpoints/")
    print(" - Delete file: models/final_model.pth")
    print("Then run: python train.py")

if __name__ == "__main__":
    main()

#EOF