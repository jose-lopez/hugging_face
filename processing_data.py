from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_datasets = load_dataset("glue", "mrpc")
print(f'raw_datasets: {raw_datasets}')

for split, dataset in raw_datasets.items():
    dataset.to_json(f"mrpc-{split}.jsonl")

raw_train_dataset = raw_datasets["train"]
print(f'raw_train_dataset: {raw_train_dataset}')
print(f'raw_train_dataset.features: {raw_train_dataset.features}')
print(f'raw_train_dataset[0]: {raw_train_dataset[0]}')

data_files = {
    "train": "mrpc-train.jsonl",
    "validation": "mrpc-validation.jsonl",
    "test": "mrpc-test.jsonl",
}
raw_datasets = load_dataset("json", data_files=data_files)

raw_train_dataset = raw_datasets["train"]
print(f'raw_train_dataset: {raw_train_dataset}')
print(f'raw_train_dataset.features: {raw_train_dataset.features}')
print(f'raw_train_dataset[0]: {raw_train_dataset[0]}')

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
print(tokenized_datasets)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

samples = tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
print([len(x) for x in samples["input_ids"]])

batch = data_collator(samples)
print({k: v.shape for k, v in batch.items()})



