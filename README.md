# Audio QA System Documentation

This project provides an end-to-end system for **Audio Question Answering**, allowing you to ask questions based on the content of an audio file.

## 1. Project Overview
The system works by converting audio to text (ASR) and then performing Question Answering (QA) on the transcript.

**Pipeline:**
`Audio File` -> **Whisper (ASR)** -> `Text Context` -> **QA Model** -> `Answer`

## 2. Directory Structure
- `audio_qa.py`: Main class and CLI for inference.
- `prepare_training_data.py`: Script to generate training data from audio files.
- `train_model.py`: Script to fine-tune the QA model.
- `evaluate_on_dataset.py`: Script to evaluate model performance.
- `requirements.txt`: Python dependencies.

## 3. Usage

### A. Inference (Asking Questions)
To use the pre-trained model to answer questions about an audio file:
```powershell
python audio_qa.py --audio "path/to/audio.wav" --question "Who is the CEO?"
```

### B. Training (Fine-tuning)
If you want to train the model on the sAQA dataset:

**Step 1: Prepare Data**
This converts audio files to text and formats them for the QA model.
```powershell
# Processes 100 random samples
python prepare_training_data.py --samples 100 --output d:/data/sAQA_training_data.json
```

**Step 2: Train Model**
This fine-tunes a DistilBERT model on the prepared data.
```powershell
python train_model.py --data d:/data/sAQA_training_data.json --output_dir d:/data/finetuned_qa_model
```

### C. Using the Fine-tuned Model
You can update `audio_qa.py` to point to your new model directory:
```python
model = AudioQAModel(qa_model="d:/data/finetuned_qa_model")
```

## 4. Technical Details

### Preprocessing
- **Audio Loading**: Uses `librosa` to load and resample audio to 16kHz.
- **ASR**: Uses `openai/whisper-tiny` to generate transcripts.

### Training Logic
- **Data Generation**: We infer the answer position in the transcript by text matching.
- **Model Architecture**: We use `distilbert-base-cased-distilled-squad` as a base and fine-tune it for Extractive QA.
- **Optimization**: The training script uses `TrainingArguments` from Hugging Face for efficient training logic.

## 5. Dependencies
- transformers
- torch
- librosa
- soundfile
- datasets
- scikit-learn
