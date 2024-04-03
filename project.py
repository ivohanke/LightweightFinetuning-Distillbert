from transformers import  AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset, DatasetDict
import numpy as np
import pandas as pd
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, AutoPeftModelForSequenceClassification

# Create model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

# Create tokenizer
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

tokenized_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(100))
tokenized_test_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(100))
tokenized_validation_dataset = tokenized_datasets["validation"].shuffle(seed=42).select(range(100))

print(tokenized_train_dataset)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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


# Training and evaluation
training_args = TrainingArguments(
    output_dir="./data/training",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
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

# Create LoRA PEFT Model

pretrained_model = AutoModelForSequenceClassification.from_pretrained("./data/model")

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=1,
    lora_alpha=1,
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
    bias="all"
)


peft_model2 = get_peft_model(pretrained_model, peft_config2)

peft_training_args2 = TrainingArguments(
    output_dir="./data/peft_training2",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.5,
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

from peft import AutoPeftModelForSequenceClassification
from evaluate import evaluator

pretrained_model = AutoModelForSequenceClassification.from_pretrained("./data/model")
pretrained_tokenizer = AutoTokenizer.from_pretrained("./data/model")

# Doesn't work: AttributeError: 'PeftModelForSequenceClassification' object has no attribute 'task'
# inference_peft_model = AutoPeftModelForSequenceClassification.from_pretrained("./data/peft_model")


inference_peft_model = AutoModelForSequenceClassification.from_pretrained("./data/peft_model")
inference_peft_tokenizer = AutoTokenizer.from_pretrained("./data/peft_model")

# Doesn't work: AttributeError: 'PeftModelForSequenceClassification' object has no attribute 'task'
# inference_peft_model2 = AutoPeftModelForSequenceClassification.from_pretrained("./data/peft_model2")

inference_peft_model2 = AutoModelForSequenceClassification.from_pretrained("./data/peft_model2")
inference_peft_tokenizer2 = AutoTokenizer.from_pretrained("./data/peft_model2")


# Set evaluation mode
task_evaluator = evaluator("text-classification")


# Pretrained Model
eval_results_model = task_evaluator.compute(
    model_or_pipeline=pretrained_model,
    tokenizer=tokenizer,
    data=tokenized_validation_dataset,
    input_column="text",    #default
    label_column="label",   #default
    metric="accuracy",
    label_mapping={"LABEL_0": 0.0, "LABEL_1": 1.0},
    strategy="bootstrap",
    # n_resamples=10,
    # random_state=0
)

print(f"""Pretrained Model: {eval_results_model}""")

eval_results_peft_model = task_evaluator.compute(
    model_or_pipeline=inference_peft_model,
    tokenizer=tokenizer,
    data=tokenized_validation_dataset,
    input_column="text",    #default
    label_column="label",   #default
    metric="accuracy",      #default
    label_mapping={"LABEL_0": 0.0, "LABEL_1": 1.0},
    strategy="bootstrap",
    # n_resamples=10,
    # random_state=0
)

print(f"""PEFT Model 1: {eval_results_peft_model}""")

eval_results_peft_model2 = task_evaluator.compute(
    model_or_pipeline=inference_peft_model2,
    tokenizer=inference_peft_tokenizer2,
    data=tokenized_validation_dataset,
    input_column="text",    #default
    label_column="label",   #default
    metric="accuracy",
    label_mapping={"LABEL_0": 0.0, "LABEL_1": 1.0},
    strategy="bootstrap",
    # n_resamples=10,
    # random_state=0
)

print(f"""PEFT Model 2: {eval_results_peft_model2}""")

# Comparison

def compare_model_outputs(pretrained_model, peft_model_1, peft_model_2):

    # Create a dictionary with the relevant information
    data = {
        "Model": ["Pretrained Model", "PEFT Model 1", "PEFT Model 2"],
        "Accuracy Score": [pretrained_model['accuracy']['score'], peft_model_1['accuracy']['score'], peft_model_2['accuracy']['score']],
        "Confidence Interval Low": [pretrained_model['accuracy']['confidence_interval'][0], peft_model_1['accuracy']['confidence_interval'][0], peft_model_2['accuracy']['confidence_interval'][0]],
        "Confidence Interval High": [pretrained_model['accuracy']['confidence_interval'][1], peft_model_1['accuracy']['confidence_interval'][1], peft_model_2['accuracy']['confidence_interval'][1]],
        "Standard Error": [pretrained_model['accuracy']['standard_error'], peft_model_1['accuracy']['standard_error'], peft_model_2['accuracy']['standard_error']],
        "Total Time (s)": [pretrained_model['total_time_in_seconds'], peft_model_1['total_time_in_seconds'], peft_model_2['total_time_in_seconds']],
        "Samples per Second": [pretrained_model['samples_per_second'], peft_model_1['samples_per_second'], peft_model_2['samples_per_second']],
        "Latency (s)": [pretrained_model['latency_in_seconds'], peft_model_1['latency_in_seconds'], peft_model_2['latency_in_seconds']]
    }

    # Convert the dictionary into a DataFrame
    df = pd.DataFrame(data)

    # Display the DataFrame
    print(df)

compare_model_outputs(eval_results_model, eval_results_peft_model, eval_results_peft_model2)

# Output after running the code
""" 
Model  Accuracy Score  Confidence Interval Low  \
0  Pretrained Model            0.87                     0.79   
1      PEFT Model 1            0.88                     0.80   
2      PEFT Model 2            0.88                     0.80   

   Confidence Interval High  Standard Error  Total Time (s)  \
0                      0.93        0.033839        1.248748   
1                      0.93        0.032918        1.439280   
2                      0.93        0.032927        1.450926   

   Samples per Second  Latency (s)  
0           80.080180     0.012487  
1           69.479177     0.014393  
2           68.921491     0.014509  
 """