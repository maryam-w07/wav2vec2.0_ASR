from datasets import load_dataset, DatasetDict

# Enable streaming and load datasets
dataset_train = load_dataset("librispeech_asr", "clean", split="train.100", streaming=True)
dataset_val = load_dataset("librispeech_asr", "clean", split="validation", streaming=True)
dataset_test = load_dataset("librispeech_asr", "clean", split="test", streaming=True)

small_train = list(dataset_train.take(1000))
small_val = list(dataset_val.take(100))
small_test = list(dataset_test.take(2))
# Create a new dataset dictionary
small_dataset = DatasetDict({
    "train": small_train,
    "validation": small_val,
    "test": small_test
})
print(small_dataset)

import re

chars_to_ignore_regex = r'[\,\?\.\!\-\;\:\"]'

def clean_text(example):
    """Removes special characters and lowercases the text."""
    example["text"] = re.sub(chars_to_ignore_regex, '', example["text"]).lower()
    return example

# Apply the transformation
small_dataset = {
    split: [clean_text(sample) for sample in small_dataset[split]]  # Process each element in the list
    for split in ["train", "validation", "test"]
}
