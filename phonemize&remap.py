from phonemizer import phonemize

# Phonemize each text entry
phonemized_dataset = {
    split: [
        {**sample, "phonemes": phonemize(sample["text"], language="en-us", backend="espeak")}
        for sample in small_dataset[split]
    ]
    for split in ["train", "validation", "test"]
}
from collections import Counter

# Extract individual phonemes
phoneme_counter = Counter()

for split in ["train", "validation", "test"]:
    for sample in phonemized_dataset[split]:
        phonemes = list(sample["phonemes"].replace(" ", ""))  # Remove spaces and split into individual symbols
        phoneme_counter.update(phonemes)

# Get unique phonemes
unique_phonemes = sorted(phoneme_counter.keys())

# Print unique phonemes
print(unique_phonemes)
print(f"\nTotal unique phonemes: {len(unique_phonemes)}")
phoneme_remap = {
    "e": "ɛ",
    "ɐ": "ʌ",
    "ɚ": "ɝ",
    "ɜ": "ɝ",
    "ɡ": "g",
    "ɾ": "t",
    "ʔ": "",
    "ᵻ": "ɪ",
    "ː": "",
    "̩": ""
}
def process_word(word, mapping):
    # Remove the unwanted symbols "ː" (length marker) and "̩" (syllabic marker)
    word = word.replace("ː", "").replace("̩", "")
    # Apply the remapping for extra phonemes
    for key, value in mapping.items():
        word = word.replace(key, value)
    return word
# Process each word in the phonemized text for all splits
for split in ["train", "validation", "test"]:
    for sample in phonemized_dataset[split]:
        # Split the phonemized string into words (each word is a complete phonemic transcription)
        words = sample["phonemes"].split()
        # Process each word using the process_word function
        processed_words = [process_word(word, phoneme_remap) for word in words]
        # Update the phonemes field with the processed words
        sample["phonemes"] = " ".join(processed_words)
