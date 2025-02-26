import json
import torch
from datasets import Dataset
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq

# Load and prepare dataset
def prepare_dataset(file_path):
    # Load the JSON data
    with open(file_path, 'r', encoding='utf-8') as f:
        data: dict = json.load(f)
    
    # Convert to format for datasets library
    dataset_dict = {
        "hebrew_word": [],
        "phoneme": []
    }
    
    for hebrew, phoneme in data.items():
        dataset_dict["hebrew_word"].append(hebrew)
        dataset_dict["phoneme"].append(phoneme)
    
    # Create dataset
    dataset = Dataset.from_dict(dataset_dict)
    
    # Split dataset
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    
    return dataset

# Tokenize the dataset
def tokenize_data(examples, tokenizer: MT5Tokenizer, max_length=128):
    # Tokenize Hebrew words (inputs)
    inputs = tokenizer(
        examples["hebrew_word"], 
        max_length=max_length,
        padding="max_length",
        truncation=True
    )
    
    # Tokenize phonemes (targets)
    with tokenizer.as_target_tokenizer():
        outputs = tokenizer(
            examples["phoneme"],
            max_length=max_length,
            padding="max_length",
            truncation=True
        )
    
    inputs["labels"] = outputs["input_ids"]
    return inputs

# Compute metrics for evaluation
def compute_metrics(eval_pred, tokenizer: MT5Tokenizer):
    predictions, labels = eval_pred
    # Decode generated tokens
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 with pad token id to decode properly
    labels = [[l if l != -100 else tokenizer.pad_token_id for l in label] for label in labels]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Calculate accuracy (exact match)
    correct = sum(pred == label for pred, label in zip(decoded_preds, decoded_labels))
    accuracy = correct / len(decoded_preds)
    
    # Print a few examples for manual inspection (direct to console)
    for i in range(min(3, len(decoded_preds))):  # Print up to 3 examples
        print(f"Predicted: {decoded_preds[i]}")
        print(f"Expected: {decoded_labels[i]}")
        print("-" * 30)
    
    return {"accuracy": accuracy}

# Main fine-tuning function
def fine_tune_mt5(dataset_path, model_name="google/mt5-small", output_dir="./save"):
    # Load dataset
    dataset = prepare_dataset(dataset_path)
    
    # Load tokenizer and model
    tokenizer: MT5Tokenizer = MT5Tokenizer.from_pretrained(model_name)
    model = MT5ForConditionalGeneration.from_pretrained(model_name)
    
    # Tokenize datasets
    tokenized_datasets = dataset.map(
        lambda examples: tokenize_data(examples, tokenizer),
        batched=True,
        remove_columns=["hebrew_word", "phoneme"]
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    
    # Create a function that wraps compute_metrics with tokenizer
    def metric_fn(eval_pred):
        return compute_metrics(eval_pred, tokenizer)
    
    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        logging_steps=1,
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=5,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        report_to="none",  # Disable wandb, etc.
        load_best_model_at_end=True,
    )
    
    # Create Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=metric_fn,
    )
    
    # Start training
    trainer.train()
    
    # Save model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return model, tokenizer

# Function to use the model for inference
def convert_hebrew_to_phoneme(hebrew_word, model: MT5ForConditionalGeneration, tokenizer: MT5Tokenizer):
    inputs = tokenizer(hebrew_word, return_tensors="pt", padding=True)
    
    # Move inputs to the same device as model
    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate prediction
    outputs = model.generate(**inputs, max_length=128)
    
    # Decode the prediction
    phoneme = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return phoneme

if __name__ == "__main__":
    # Fine-tune the model
    model, tokenizer = fine_tune_mt5("dataset.json")
    
    # Example usage
    test_word = "שלום"  # 'hello' in Hebrew
    phoneme = convert_hebrew_to_phoneme(test_word, model, tokenizer)
    print(f"Hebrew: {test_word}, Phoneme: {phoneme}")
    
    # Load a few test examples
    with open("dataset.json", 'r', encoding='utf-8') as f:
        data: dict = json.load(f)
    
    # Select a few samples for testing
    import random
    test_samples = random.sample(list(data.items()), 5)
    
    # Print results
    print("\nTest Results:")
    for hebrew, expected in test_samples:
        predicted = convert_hebrew_to_phoneme(hebrew, model, tokenizer)
        print(f"Hebrew: {hebrew}")
        print(f"Expected: {expected}")
        print(f"Predicted: {predicted}")
        print("-" * 30)