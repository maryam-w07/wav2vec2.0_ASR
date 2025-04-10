def prepare_sample(batch):
    inputs = processor.feature_extractor(batch["audio"]["array"],
                                          sampling_rate=batch["audio"]["sampling_rate"])
    batch["input_values"] = inputs.input_values[0]

    # Process labels: Use the 'text' argument in __call__
    batch["labels"] = processor(text=batch["phonemes"]).input_ids  # Update to use 'text' argument
    return batch

import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from transformers import Wav2Vec2Processor

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """
        Custom Data Collator for Phoneme-based CTC Training.
        Handles input audio and phoneme labels for batch processing.
        """
        # Debug: Verify that each feature has a valid audio array
        for idx, feature in enumerate(features):
            # Try to use "input_ids" first, else "input_values"
            input_val = feature.get("input_ids", feature.get("input_values"))
            if input_val is None:
                raise ValueError(f"Feature at index {idx} is missing both 'input_ids' and 'input_values': {feature}")
            # Optionally: check that the input array is not empty
            if hasattr(input_val, "__len__") and len(input_val) == 0:
                raise ValueError(f"Feature at index {idx} has an empty audio array: {feature}")

        # Prepare input features: use "input_values" field for audio data
        input_features = [
            {"input_values": feature.get("input_ids", feature.get("input_values"))}
            for feature in features
        ]

        # Debug: Verify that each feature has labels
        for idx, feature in enumerate(features):
            if "labels" not in feature:
                raise KeyError(f"Feature at index {idx} is missing the 'labels' key: {feature}")
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Pad audio inputs using the processor
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Pad phoneme labels using the target processor
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # Replace padding tokens in labels with -100 so they're ignored during loss computation.
        # NOTE: Ensure that your custom tokenizer has the correct pad_token_id for your 42-phoneme vocabulary.
        labels = labels_batch["input_ids"]
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        batch["labels"] = labels
        return batch
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
