import os
import requests
from datasets import load_dataset
import logging

# Enable basic logging
logging.basicConfig(level=logging.INFO)

# --- Step 1: Manually download the SINGLE, CORRECT file ---

# This is the correct and exact URL for the training data.
# There is no test split in this dataset.
url = "https://huggingface.co/datasets/Heralax/us-army-fm-instruct/blob/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
local_path = "train.parquet"

print("--- Checking and downloading data file ---")
if not os.path.exists(local_path):
    print(f"Downloading {local_path}...")
    try:
        # Use requests to download the file
        response = requests.get(url)
        response.raise_for_status()  # This will raise an error for bad responses (4xx or 5xx)
        with open(local_path, "wb") as f:
            f.write(response.content)
        print(f"Successfully downloaded {local_path}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url}. Error: {e}")
        exit()  # Stop the script if download fails
else:
    print(f"{local_path} already exists. Skipping download.")


# --- Step 2: Load the dataset from the now local file ---
print("\n--- Attempting to load dataset from local file ---")
try:
    # We point to the specific local file for the 'train' split.
    army_dataset = load_dataset("parquet", data_files={"train": local_path})

    print("\n--- Dataset loaded successfully! ---")
    print(army_dataset)

    print("\n--- First Example (Train Split) ---")
    print(army_dataset['train'][0])

except Exception as e:
    print(f"\nAn error occurred during dataset generation: {e}")
