import torch
from transformers import pipeline
import argparse
import logging
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

class AudioQAModel:
    def __init__(self, asr_model="openai/whisper-tiny", qa_model="deepset/roberta-base-squad2"):
        print(f"Loading ASR model: {asr_model}...")
        self.asr_pipeline = pipeline("automatic-speech-recognition", model=asr_model)
        
        print(f"Loading QA model: {qa_model}...")
        # Check if local path exists, otherwise use string as model name
        self.qa_pipeline = pipeline("question-answering", model=qa_model)
        print("Models loaded successfully.")

    def transcribe(self, audio_path):
        """Converts audio to text."""
        try:
            # Load audio using librosa (uses soundfile for wav, which doesn't need system ffmpeg)
            import librosa
            # pipeline expects (array, sampling_rate) or just array if sampling_rate is correct?
            # Transformers pipeline expects: str, or bytes, or dict like {"array": np.array, "sampling_rate": int}
            
            # Load with default sr=None to keep original or 16000 for whisper?
            # Whisper usually expects 16kHz
            audio_array, sampling_rate = librosa.load(audio_path, sr=16000)
            
            result = self.asr_pipeline({"array": audio_array, "sampling_rate": sampling_rate})
            return result["text"]
        except Exception as e:
            print(f"Error during transcription: {e}")
            return ""

    def answer(self, context, question):
        """Answers a question based on context text."""
        try:
            result = self.qa_pipeline(question=question, context=context)
            return result["answer"], result["score"]
        except Exception as e:
            print(f"Error during QA inference: {e}")
            return "Error", 0.0

    def predict(self, audio_path, question):
        """End-to-end prediction from audio to answer."""
        transcript = self.transcribe(audio_path)
        if not transcript:
            return "Could not transcribe audio.", 0.0
        
        # print(f"Transcript: {transcript}")
        answer, score = self.answer(transcript, question)
        return answer, score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio QA System")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    parser.add_argument("--question", type=str, required=True, help="Question to ask")
    parser.add_argument("--model", type=str, default="deepset/roberta-base-squad2", help="Path to QA model (local or huggingface)")
    
    args = parser.parse_args()
    
    model = AudioQAModel(qa_model=args.model)
    answer, score = model.predict(args.audio, args.question)
    
    print("-" * 30)
    print(f"Question: {args.question}")
    print(f"Answer: {answer}")
    print(f"Confidence: {score:.4f}")
    print("-" * 30)
