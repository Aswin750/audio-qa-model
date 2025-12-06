from audio_qa import AudioQAModel
import os

# ==============================================================================
# CONFIGURATION
# Edit these variables to change the input
# ==============================================================================

# Path to the audio file you want to analyze
AUDIO_FILE = "d:/data/sAQA/sqa/aud_files/sentence_2366.wav"

# The question you want to ask
QUESTION = "Is this audio mentioning brazil?"

# Path to the model. 
# Options: 
#   1. "deepset/roberta-base-squad2" (Default pre-trained)
#   2. "d:/data/finetuned_qa_model"  (Your trained model)
MODEL_PATH = "d:/data/finetuned_qa_model"

# ==============================================================================

def main():
    if not os.path.exists(AUDIO_FILE):
        print(f"Error: Audio file not found at {AUDIO_FILE}")
        return

    print(f"Initializing model from: {MODEL_PATH}")
    # Initialize the model (this might take a few seconds)
    model = AudioQAModel(qa_model=MODEL_PATH)
    
    print(f"Transcribing and Analyzing: {AUDIO_FILE}...")
    
    # Run prediction
    answer, score = model.predict(AUDIO_FILE, QUESTION)
    
    # Print results
    print("\n" + "="*40)
    print(f"Question:   {QUESTION}")
    print(f"Answer:     {answer}")
    print(f"Confidence: {score:.4f}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
