# wav2vec2.0_ASR

##  Wav2Vec2 Phoneme-Based ASR Pipeline for Viseme Generation Overview:

This project implements an end-to-end Automatic Speech Recognition pipeline that outputs phonetic transcriptions (IPA phonemes) instead of text. By converting speech to phonemes (and ultimately visemes), the system can drive avatar lip-sync or other applications where the articulation of speech sounds is needed. It leverages Facebook AI’s Wav2Vec2-Base model – a pretrained wav2vec 2.0 model – and fine-tunes it on a phoneme recognition task using the LibriSpeech dataset. Key steps include streaming dataset loading, text phonemization, custom phoneme-to-viseme vocabulary creation, and model fine-tuning with Hugging Face’s Trainer. 

### Dataset Streaming
We use the LibriSpeech ASR corpus (specifically the train.clean.100 subset) as the audio source. To handle the large dataset efficiently, the Hugging Face datasets library’s streaming mode is used.

#### Phonemizer
Text to Phonemes Conversion:

Transcript Processing: Phonemizer takes the text transcripts associated with the audio files in the dataset (e.g., LibriSpeech) and converts each transcript into a sequence of phonemes using a predefined phonetic alphabet (usually IPA).

Backend Selection: The eSpeak backend is used for phonemization, which supports multiple languages and dialects, ensuring that the phonetic transcription is accurate and consistent with the phonological rules of English.

##### Resources Used
https://learn.microsoft.com/en-us/azure/ai-services/speech-service/how-to-speech-synthesis-viseme?tabs=visemeid&pivots=programming-language-csharp#map-phonemes-to-visemes
