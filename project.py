from transformers import  AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset, DatasetDict
import numpy as np
import pandas as pd
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

# Create model
# access_token = ""
# model = AutoModelForSequenceClassification.from_pretrained("google/gemma-2b", token=access_token)
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

# Create tokenizer
# tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b", token=access_token)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Load the IMDb dataset
dataset = load_dataset('imdb')

# Split the original training data into new training data and validation data
train_test_split = dataset['train'].train_test_split(test_size=0.1)

dataset = DatasetDict({
    'train': train_test_split['train'],
    'validation': train_test_split['test'],
    'test': dataset['test']
})

print(dataset)

# Tokenize the whole dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

tokenized_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
tokenized_test_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
tokenized_validation_dataset = tokenized_datasets["validation"].shuffle(seed=42)

print(tokenized_train_dataset)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": (predictions == labels).mean()}


# Training and evaluation
training_args = TrainingArguments(
    output_dir="./data/training",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],  # Use 'test' split as validation
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model("./data/model")

pretrained_model = AutoModelForSequenceClassification.from_pretrained("./data/model")

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=16,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_lin", "k_lin", "v_lin", "out_lin"],
    bias="none"
)

peft_model = get_peft_model(pretrained_model, peft_config)
peft_model.print_trainable_parameters()


peft_training_args = TrainingArguments(
    output_dir="./data/peft_training",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
    load_best_model_at_end=True
)

peft_trainer = Trainer(
    model=peft_model,
    args=peft_training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],  # Use 'test' split as validation
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

peft_trainer.train()
peft_trainer.save_model("./data/peft_model")

# Alternative PEFT configuration
peft_config2 = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=16,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_lin", "k_lin", "v_lin", "out_lin"],
    bias="all",
    use_rslora=True
)


peft_model2 = get_peft_model(pretrained_model, peft_config2)

peft_training_args2 = TrainingArguments(
    output_dir="./data/peft_training2",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
    load_best_model_at_end=True
)

peft_trainer2 = Trainer(
    model=peft_model2,
    args=peft_training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],  # Use 'test' split as validation
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

peft_trainer2.train()
peft_trainer2.save_model("./data/peft_model2")

# Compare the trained peft-model
def compare_model_evaluations(trainer, peft_trainer, peft_trainer2, eval_dataset):
    # Evaluate the base model
    results = trainer.evaluate(eval_dataset)
    print("Base Model Evaluation Results:")
    for key, value in results.items():
        print(f"{key}: {value}")

    # Evaluate the PEFT model
    peft_results = peft_trainer.evaluate(eval_dataset)
    print("\nPEFT Model Evaluation Results:")
    for key, value in peft_results.items():
        print(f"{key}: {value}")

    # Evaluate the PEFT model2
    peft_results2 = peft_trainer2.evaluate(eval_dataset)
    print("\nPEFT Model 2 Evaluation Results:")
    for key, value in peft_results.items():
        print(f"{key}: {value}")

    # Compare the results
    print("\nComparison Base - PEFT:")
    for key in results.keys():
        val = results.get(key, 0)
        peft_val = peft_results.get(key, 0)
        diff = peft_val - val
        print(f"{key} Difference (PEFT - Base): {diff}")

    # Compare the results
    print("\nComparison PEFT 2 - PEFT 1:")
    for key in peft_results.keys():
        peft_val = peft_results.get(key, 0)
        peft_val2 = peft_results2.get(key, 0)
        diff = peft_val2 - peft_val
        print(f"{key} Difference (PEFT 2 - PEFT 1): {diff}")

# Call comparison
compare_model_evaluations(trainer, peft_trainer, peft_trainer2, tokenized_validation_dataset)
