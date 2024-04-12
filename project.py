from transformers import  AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset, DatasetDict
import numpy as np
import pandas as pd
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, AutoPeftModelForSequenceClassification
import torch
from torch.utils.data import DataLoader
import evaluate
from tqdm import tqdm

# Base model
base_model = "distilbert-base-uncased"
local_model = "distilbert-base-local"
modified_model = 'distilbert-base-modified'
peft_model_name = 'distilbert-base-peft'

# Load the IMDb dataset
dataset = load_dataset('imdb')

# Create tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

# Tokenize the dataset
def tokenize_function(examples):
    tokenized = tokenizer(examples['text'], truncation=True, padding=True)
    return tokenized

tokenized_dataset = dataset.map(tokenize_function, batched=True,  remove_columns=["text"])
train_dataset=tokenized_dataset['train']
eval_dataset=tokenized_dataset['test'].shard(num_shards=2, index=0)
test_dataset=tokenized_dataset['test'].shard(num_shards=2, index=1)

# Extract labels
names = dataset["train"].features["label"].names
num_classes = dataset['train'].features['label'].num_classes

# id2label mapping
id2label = {i: label for i, label in enumerate(names)}

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")


from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Define computation function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {
      'accuracy': acc,
      'f1': f1,
      'precision': precision,
      'recall': recall
    }

# Train the base model

# Training and evaluation
training_args = TrainingArguments(
    output_dir="./data/training",
    evaluation_strategy="steps",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    load_best_model_at_end=True
)

model = AutoModelForSequenceClassification.from_pretrained(base_model, id2label=id2label)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.save_model(local_model)


# Create LoRA PEFT Model

# Base model
model = AutoModelForSequenceClassification.from_pretrained(base_model, id2label=id2label)

# Create peft config
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_lin", "k_lin", "v_lin", "out_lin"],
    bias="none"
)

# Create peft model
peft_model = get_peft_model(model, peft_config)

peft_model.print_trainable_parameters()

# Training and evaluation
peft_training_args = TrainingArguments(
    output_dir="./data/peft_training",
    evaluation_strategy="steps",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    load_best_model_at_end=True
)

peft_trainer = Trainer(
    model=peft_model,
    args=peft_training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

peft_trainer.train()

tokenizer.save_pretrained(modified_model)
peft_trainer.save_model(peft_model_name)



# Evaluate base and PEFT model
metric = evaluate.load('accuracy')

def evaluate_model(inference_model, dataset):

    eval_dataloader = DataLoader(dataset.rename_column("label", "labels"), batch_size=8, collate_fn=data_collator)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inference_model.to(device)

    # Evaluation
    inference_model.eval()
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch.to(device)
        with torch.no_grad():
            outputs = inference_model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = predictions, batch["labels"]
        metric.add_batch(
            predictions=predictions,
            references=references,
        )

    eval_metric = metric.compute()
    return eval_metric 


# Evaluation of base model
print(f"Evaluation of base model ({local_model}): \n")
result_local = evaluate_model(AutoModelForSequenceClassification.from_pretrained(local_model, id2label=id2label), eval_dataset)
print(result_local)

# New line
print("\n\n")

# Load saved PEFT model
inference_model = AutoPeftModelForSequenceClassification.from_pretrained(peft_model_name, id2label=id2label)
tokenizer = AutoTokenizer.from_pretrained(modified_model)

# Evaluation of PEFT fine-tuned model
print(f"Evaluation of fine-tuned PEFT model: ({peft_model_name}) \n")
result_peft = evaluate_model(inference_model, eval_dataset)
print(result_peft)




