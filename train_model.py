import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
from transformers import DefaultDataCollator

def train_qa_model(train_file, output_dir="d:/data/finetuned_qa_model"):
    print(f"Loading training data from {train_file}...")
    with open(train_file, 'r') as f:
        data = json.load(f)
        
    if not data:
        print("No training data found!")
        return

    # Convert to Hugging Face Dataset
    hf_dataset = Dataset.from_list(data)
    
    model_checkpoint = "distilbert-base-cased-distilled-squad"
    print(f"Loading model: {model_checkpoint}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
    
    def preprocess_function(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=384,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            if len(answer["answer_start"]) == 0:
                start_positions.append(0)
                end_positions.append(0)
                continue
                
            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label it (0, 0)
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    print("Preprocessing dataset...")
    tokenized_dataset = hf_dataset.map(preprocess_function, batched=True, remove_columns=hf_dataset.column_names)
    
    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="no",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4, # Small batch for compatibility
        num_train_epochs=3,
        weight_decay=0.01,
        push_to_hub=False,
        use_cpu=True # Force CPU if no GPU available/configured, safe default
    )
    
    data_collator = DefaultDataCollator()
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Saving model to {output_dir}...")
    trainer.save_model(output_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="d:/data/sAQA_training_data.json")
    parser.add_argument("--output_dir", type=str, default="d:/data/finetuned_qa_model")
    
    args = parser.parse_args()
    train_qa_model(args.data, args.output_dir)
