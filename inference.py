from transformers import Wav2Vec2Processor

model.save_pretrained("./local_model")
processor.save_pretrained("./local_model")
sample = processed_dataset["validation"][1] 
input_values = torch.tensor(sample["input_values"]).unsqueeze(0) 

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
# Load the processor and model from your checkpoint
processor = Wav2Vec2Processor.from_pretrained("./local_model")  # Path where model was saved
model = Wav2Vec2ForCTC.from_pretrained("./local_model")
# Set model to evaluation mode
model.eval()
with torch.no_grad():  # No gradients needed for inference
    logits = model(input_values).logits 
predicted_ids = torch.argmax(logits, dim=-1)

# Decode token IDs into text
transcription = processor.batch_decode(predicted_ids)[0]

print("Predicted Transcription:", transcription)
