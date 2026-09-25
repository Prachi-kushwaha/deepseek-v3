import torch
import hashlib
import json
from pathlib import Path

from datasets import load_dataset
from torch.utils.data import Dataset, Dataloader, random_split

from config  import get_config

def load_raw_dataset(ds, split):
    dataset = load_dataset(ds, split=split)
    return dataset

# validate data
def validate_data(dataset):

    if "messages" not in dataset:
        return False

    dataset = dataset["messages"]

    if not isinstance(dataset, list):
        return False

    if len(dataset) == 0:
        return False

    for message in dataset:
        if not isinstance(message, dict):
            return False
        if "role" not in message:
            return False
        if "content" not in message:
            return False
        if message["role"] not in valid_roles:
            return False
        if not isinstance(message["content"], str):
            return False

        if not message["content"].strip():
            return False

    return True

# my_data = dataset.filter(validate_data)
# print(len(dataset))

def clean_example(my_data):

    messages = []
    for message in my_data["messages"]:
      messages.append({
            "role": message["role"],
            "content": message["content"]
        })

    return {
        "messages": messages
    }

def conversation_hash(my_data):

    serialized = json.dumps()(
        my_data["messages"],
        ensure_ascii=False,
        sort_keys=False
    )

    return hashlib.sha256(
        serialized.encode('utf-8')
    ).hexdigest()


def deduplicate(dataset, key):
    seen = set()
    keep_indices = []

    for i, example in enumerate(dataset):
        value = example[key]

        if value not in seen:
            seen.add(value)
            keep_indices.append(i)

    return dataset.select(keep_indices)


def format_conversion(example):

    output = ["[BOS]"]

    for message in example["messages"]:
        role = message["role"]
        content = message["content"]

        output.append(
            f"<|{role}|>\n{content}"
        )

    output.append("[EOS]")

    return {
        "text": "\n".join(output)
    }


def deduplicate(dataset):
    seen = set()
    keep_indices = []

    for idx, example in enumerate(dataset):
        h = example["conversation_hash"]

        if h not in seen:
            seen.add(h)
            keep_indices.append(idx)

    return dataset.select(keep_indices)