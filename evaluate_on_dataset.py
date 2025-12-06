import json
import random
import os
from audio_qa import AudioQAModel
import argparse

def load_dataset(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def evaluate(json_path, audio_dir, num_samples=5):
    print(f"Loading dataset from {json_path}...")
    dataset = load_dataset(json_path)
    
    model = AudioQAModel()
    
    samples = random.sample(dataset, min(len(dataset), num_samples))
    
    print(f"\nEvaluating on {len(samples)} random samples from {json_path}...")
    print("=" * 50)
    
    for i, item in enumerate(samples):
        # Audio path in JSON is relative, e.g., "sqa/aud_files/sentence_1.wav"
        # We need to construct the full path: d:/data/sAQA/sqa/aud_files/...
        # The JSON 'name' field seems to be "sqa/aud_files/X.wav"
        
        rel_path = item['name']
        # Extract filename from relative path (e.g. sentence_1.wav)
        filename = os.path.basename(rel_path)
        
        # Construct absolute path expected by our local setup
        audio_path = os.path.join(audio_dir, filename)
        
        question = item['question']
        ground_truth = item['answer']
        
        print(f"Sample {i+1}:")
        print(f"Audio: {audio_path}")
        if not os.path.exists(audio_path):
            print("  [WARNING] Audio file not found!")
            continue

        pred_answer, score = model.predict(audio_path, question)
        
        print(f"Question: {question}")
        print(f"Predicted: {pred_answer} (Conf: {score:.2f})")
        print(f"Ground Truth: {ground_truth}")
        print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, default="d:/data/Audiopedia/sAQA_release_qa.json", help="Path to QA JSON")
    parser.add_argument("--audio_dir", type=str, default="d:/data/sAQA/sqa/aud_files", help="Directory containing audio files")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples to test")
    
    args = parser.parse_args()
    
    evaluate(args.json, args.audio_dir, args.samples)
