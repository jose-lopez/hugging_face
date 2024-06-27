from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
import numpy as np
import evaluate

def tokenize_function(example):
    return tokenizer(example["sentence"], truncation=True)


def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "sst2")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


raw_datasets = load_dataset("glue", "sst2")
print(f'raw_datasets: ')
print(raw_datasets)

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
print(f'tokenized_datasets: ')
print(tokenized_datasets)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments("test-trainer")

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

print(tokenized_datasets["train"])
training_dataset = tokenized_datasets["train"].select(range(100))
print(training_dataset)
validation_dataset = tokenized_datasets["validation"].select(range(100))
eval_dataset = tokenized_datasets["test"].select(range(100))

trainer = Trainer(
    model,
    training_args,
    train_dataset=training_dataset,
    eval_dataset=validation_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()

predictions = trainer.predict(validation_dataset)
print(predictions.predictions.shape, predictions.label_ids.shape)

preds = np.argmax(predictions.predictions, axis=-1)

metric = evaluate.load("glue", "sst2")
metric.compute(predictions=preds, references=predictions.label_ids)

training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

trainer = Trainer(
    model,
    training_args,
    train_dataset=training_dataset,
    eval_dataset=validation_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()