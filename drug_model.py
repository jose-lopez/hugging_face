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
drug_dataset = drug_dataset.filter(lambda x: x["condition"].find('span') == -1)

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
test_df = drug_dataset["test"][:]

print(train_df)
print(test_df)

conditions_train = train_df["condition"].value_counts().to_frame().reset_index().rename(columns={"count": "frequency"})
conditions_test = test_df["condition"].value_counts().to_frame().reset_index().rename(columns={"count": "frequency"})

conditions_from_train = conditions_train.to_dict()
conditions_from_test = conditions_test.to_dict()

train_conditions_codes = conditions_from_train['condition']
test_conditions_codes = conditions_from_test['condition']

conditions_train_keys = list(train_conditions_codes.values())
conditions_test_keys = list(test_conditions_codes.values())
not_train_conditions = []
for condition in conditions_test_keys:
    if condition not in conditions_train_keys:
        not_train_conditions.append(condition)

condition_code = {}
code = 0
for condition in conditions_from_train.keys():
    condition_code[condition] = code
    code += 1

drug_dataset.reset_format()

features = drug_dataset.features.copy()
features["basic_sentiment"] = ClassLabel(names=["negative", "neutral", "positive"])
def adjust_labels(batch):
    batch["basic_sentiment"] = [sentiment + 1 for sentiment in batch["basic_sentiment"]]
    return batch
dataset = dataset.map(adjust_labels, batched=True, features=features)



