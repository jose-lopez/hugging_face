from datasets import load_dataset
import html
from transformers import AutoTokenizer
from datasets import Dataset


def tokenize_function(examples):
    return tokenizer(examples["review"], truncation=True)

def lowercase_condition(example):
    return {"condition": example["condition"].lower()}

def tokenize_and_split(examples):
    return tokenizer(
        examples["review"],
        truncation=True,
        max_length=128,
        return_overflowing_tokens=True,
    )

def tokenize_and_split2(examples):
    result = tokenizer(
        examples["review"],
        truncation=True,
        max_length=128,
        return_overflowing_tokens=True,
    )
    # Extract mapping between new and old indices
    sample_map = result.pop("overflow_to_sample_mapping")
    print("sample_map: ",  sample_map)
    for key, values in examples.items():
        result[key] = [values[i] for i in sample_map]
    return result


def compute_review_length(example):
    return {"review_length": len(example["review"].split())}

data_files = {"train": "data/drugsComTrain_raw.tsv", "test": "data/drugsComTest_raw.tsv"}
# \t is the tab character in Python
drug_dataset = load_dataset("csv", data_files=data_files, delimiter="\t")

drug_sample = drug_dataset["train"].shuffle(seed=42).select(range(1000))
# Peek at the first few examples
print(drug_sample[:3])

for split in drug_dataset.keys():
    assert len(drug_dataset[split]) == len(drug_dataset[split].unique("Unnamed: 0"))

drug_dataset = drug_dataset.rename_column(
    original_column_name="Unnamed: 0", new_column_name="patient_id"
)
print(drug_dataset)

drug_dataset = drug_dataset.filter(lambda x: x["condition"] is not None)

drug_dataset = drug_dataset.map(lowercase_condition)
# Check that lowercasing worked
print(drug_dataset["train"]["condition"][:3])

drug_dataset = drug_dataset.map(compute_review_length)
# Inspect the first training example
print(drug_dataset["train"][0])

print(drug_dataset["train"].sort("review_length")[:3])

drug_dataset = drug_dataset.filter(lambda x: x["review_length"] > 30)
print(drug_dataset.num_rows)

# drug_dataset = drug_dataset.map(lambda x: {"review": html.unescape(x["review"])})

drug_dataset = drug_dataset.map(
    lambda x: {"review": [html.unescape(o) for o in x["review"]]}, batched=True
)

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# tokenized_dataset = drug_dataset.map(tokenize_function, batched=True)
tokenized_dataset = drug_dataset.map(tokenize_and_split2, batched=True)
print(tokenized_dataset)

print(drug_dataset["train"][0])

result = tokenize_and_split(drug_dataset["train"][0])
[len(inp) for inp in result["input_ids"]]

"""
tokenized_dataset = drug_dataset.map(
    tokenize_and_split2, batched=True, remove_columns=drug_dataset["train"].column_names
)
"""
print(tokenized_dataset['train'])

print(len(tokenized_dataset["train"]), len(drug_dataset["train"]))

drug_dataset.set_format("pandas")

print(drug_dataset)

train_df = drug_dataset["train"][:]

print(train_df)

frequencies = (
    train_df["condition"]
    .value_counts()
    .to_frame()
    .reset_index()
    .rename(columns={"index": "condition", "condition": "frequency"})
)
print(frequencies.head())

conditions = train_df["condition"]
frequencies1 = conditions.value_counts()
frequencies2 = frequencies1.to_frame()
frequencies3 = frequencies2.reset_index()
frequencies4 = frequencies3.rename(columns={"count": "frequency"})

conditions = frequencies1.keys()
conditions = list(frequencies1.to_dict().keys())
print(conditions)

print(conditions)

print(frequencies4.head())

drug_dataset.reset_format()

freq_dataset = Dataset.from_pandas(frequencies)
print(freq_dataset)

drug_dataset_clean = drug_dataset["train"].train_test_split(train_size=0.8, seed=42)
# Rename the default "test" split to "validation"
drug_dataset_clean["validation"] = drug_dataset_clean.pop("test")
# Add the "test" set to our `DatasetDict`
drug_dataset_clean["test"] = drug_dataset["test"]
print(drug_dataset_clean)

drug_dataset_clean.save_to_disk("drug-reviews")

from datasets import load_from_disk

drug_dataset_reloaded = load_from_disk("drug-reviews")
print(drug_dataset_reloaded)

for split, dataset in drug_dataset_clean.items():
    dataset.to_json(f"drug-reviews-{split}.jsonl")

data_files = {
    "train": "drug-reviews-train.jsonl",
    "validation": "drug-reviews-validation.jsonl",
    "test": "drug-reviews-test.jsonl",
}
drug_dataset_reloaded = load_dataset("json", data_files=data_files)

raw_train_dataset = drug_dataset_reloaded["train"]
print(f'raw_train_dataset: {raw_train_dataset}')
print(f'raw_train_dataset.features: {raw_train_dataset.features}')
print(f'raw_train_dataset[0]: {raw_train_dataset[0]}')

print(drug_dataset_reloaded)





























