import json
import os
import random
from audio_qa import AudioQAModel
from tqdm import tqdm

def prepare_data(json_path, audio_dir, output_path, num_samples=50):
    print(f"Loading original dataset from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Select a subset for demonstration purposes
    # processing 70k files would take days on a single machine without massive parallelization
    selected_data = random.sample(data, min(len(data), num_samples))
    print(f"Selected {len(selected_data)} samples for training data generation.")
    
    model = AudioQAModel()  # Uses Whisper-tiny by default
    
    training_data = []
    
    print("Starting transcription...")
    success_count = 0
    
    for item in tqdm(selected_data):
        audio_rel_path = item['name']
        audio_path = os.path.join(audio_dir, os.path.basename(audio_rel_path))
        
        if not os.path.exists(audio_path):
            continue
            
        # Transcribe audio to get "Context"
        context_text = model.transcribe(audio_path)
        
        if not context_text:
            continue
            
        question = item['question']
        answer_text = item['answer']
        
        # SQuAD format requires finding the start character of the answer in the context
        # Since we are generating the context via ASR, the exact answer string might not be present perfectly.
        # We will try to find it.
        
        start_idx = context_text.lower().find(answer_text.lower())
        
        if start_idx == -1:
            # If exact match not found, we skip this sample for training stability
            # or we could use fuzzy matching, but for now strict is safer for these models
            continue
            
        # Recover true case from context if possible, or just use what we fouund
        # We need the answer text exactly as it appears in context for SQuAD training usually
        matched_text = context_text[start_idx : start_idx + len(answer_text)]
        
        entry = {
            "id": str(item['id']),
            "context": context_text,
            "question": question,
            "answers": {
                "text": [matched_text],
                "answer_start": [start_idx]
            }
        }
        training_data.append(entry)
        success_count += 1

    print(f"Successfully prepared {success_count} samples.")
    print(f"Saving to {output_path}...")
    
    with open(output_path, 'w') as f:
        json.dump(training_data, f, indent=4)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, default="d:/data/Audiopedia/sAQA_release_qa.json")
    parser.add_argument("--audio_dir", type=str, default="d:/data/sAQA/sqa/aud_files")
    parser.add_argument("--output", type=str, default="d:/data/sAQA_training_data.json")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to process")
    
    args = parser.parse_args()
    prepare_data(args.json, args.audio_dir, args.output, args.samples)
