import os
import json
from pathlib import Path
import urllib.request

import numpy as np

from custom_tokenizers.char_tokenizer import CharacterTokenizer

TINYSHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

def build_tokenizer(text, model_max_length=int(1e9)):
    # Extract unique characters from the text and sort them
    characters = sorted(list(set(text)))
    # Create and return an instance of CharacterTokenizer
    tokenizer = CharacterTokenizer(characters=characters, model_max_length=model_max_length)
    return tokenizer

def save_tokenizer(tokenizer, save_directory):
    os.makedirs(save_directory, exist_ok=True)
    tokenizer.save_pretrained(save_directory)
    print(f"Tokenizer saved to {save_directory}")

def main():
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    input_file_path = script_dir / "input.txt"

    if input_file_path.exists():
        with open(input_file_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        print(f"{input_file_path} not found, downloading from {TINYSHAKESPEARE_URL}")
        with urllib.request.urlopen(TINYSHAKESPEARE_URL) as resp:
            text = resp.read().decode("utf-8")
        input_file_path.write_text(text, encoding="utf-8")
        print(f"Saved to {input_file_path}")

    print(f"Length of text: {len(text)} characters")

    # Build the tokenizer using the full text
    tokenizer = build_tokenizer(text)

    # Save the tokenizer configuration
    save_directory = script_dir
    save_tokenizer(tokenizer, save_directory)

    # Split the text into training and validation sets (90/10 split)
    n = len(text)
    train_text = text[: int(n * 0.9)]
    val_text = text[int(n * 0.9):]

    # Tokenize each split using the tokenizer's encode method
    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)

    # Convert the lists of token IDs to numpy arrays (using uint16)
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)

    # Save the numpy arrays to binary files in the same save directory
    train_bin_path = save_directory / "train.bin"
    val_bin_path = save_directory / "val.bin"

    def save_with_header(filename, ids):
        header = np.zeros(256, dtype=np.int32)
        header[0] = 20240520  # Magic number
        header[1] = 1         # Version
        header[2] = len(ids)  # Number of tokens
        with open(filename, "wb") as f:
            f.write(header.tobytes())  # Write the header (256 * 4 bytes)
            f.write(ids.tobytes())     # Write the token IDs as uint16

    save_with_header(train_bin_path, train_ids)
    save_with_header(val_bin_path, val_ids)

    print(f"Train and validation data saved as:\n  {train_bin_path}\n  {val_bin_path}")

if __name__ == "__main__":
    main()

