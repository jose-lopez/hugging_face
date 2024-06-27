from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
import numpy as np
import evaluate

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


raw_datasets = load_dataset("glue", "mrpc")
print(f'raw_datasets: ')
print(raw_datasets)

raw_train_dataset = raw_datasets["train"]
print(f'raw_train_dataset: {raw_train_dataset}')
print(f'raw_train_dataset.features: {raw_train_dataset.features}')
print(f'raw_train_dataset[0]: {raw_train_dataset[0]}')

for split, dataset in raw_datasets.items():
    dataset.to_json(f"mrpc-{split}.jsonl")

data_files = {
    "train": "mrpc-train.jsonl",
    "validation": "mrpc-validation.jsonl",
    "test": "mrpc-test.jsonl",
}
raw_datasets = load_dataset("json", data_files=data_files)

# "basic_sentiment holds values [-1,0,1]
new_labels = ClassLabel(num_classes = 2,names=['not_equivalent', 'equivalent'])
raw_datasets = raw_datasets.cast_column("label", new_labels)

raw_train_dataset = raw_datasets["train"]
print(f'raw_train_dataset: {raw_train_dataset}')
print(f'raw_train_dataset.features: {raw_train_dataset.features}')
print(f'raw_train_dataset[0]: {raw_train_dataset[0]}')

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

metric = evaluate.load("glue", "mrpc")
metrics = metric.compute(predictions=preds, references=predictions.label_ids)

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