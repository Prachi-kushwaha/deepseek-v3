import torch
from torch.utils.data import random_split, DataLoader
from datasets import load_dataset

from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

dataset_name = "roneneldan/TinyStories"

def load_data(dataset_name):
    dataset = load_data("dataset_name", split="train")
    dataset = dataset.select(range(50000))
    dataset_split = dataset.train_test_split(test_size = 0.1,seed=42)
    train_dataset = dataset_split["train"]
    val_dataset = dataset_split["test"]
    return train_dataset, val_dataset

train_dataset, val_dataset = load_data(dataset_name)
# dataset = load_dataset(
#     "roneneldan/TinyStories",
#     split="train"
# )
# dataset = dataset.select(range(50000))

# dataset_split = dataset.train_test_split(
#     test_size = 0.1,
#     seed=42
# )
# train_dataset = dataset_split["train"]
# test_dataset = dataset_split["test"]


def get_data(data_set):
  for data in data_set:
    yield data["text"]


def tokenize():
  tokenizer = Tokenizer(
      WordLevel(unk_token="[UNK]")
  )

  tokenizer.pre_tokenizer = Whitespace()
  trainer = WordLevelTrainer(
      special_tokens = [
          "[UNK]",
          "[PAD]",
          "[BOS]",
          "[EOS]"
      ],
      min_frequency = 1,
      vocab_size = 10000
  )

  tokenizer.train_from_iterator(
      get_data(train_dataset),
      trainer= trainer
  )

  return tokenizer

tokenizer = tokenize()

print(f"vocabulary size {tokenizer.get_vocab_size()}")



def encode_data(data):
  output = tokenizer.encode(
      data["text"]
  )
  return {
      "input_ids" : output.ids
  }


train_tokenized_dataset = train_dataset.map(
    encode_data
)

val_tokenized_dataset = val_dataset.map(
    encode_data
)

train_tokenized_dataset = train_tokenized_dataset.filter(
    lambda data: 100 <= len(data["input_ids"]) <= 500
)

val_tokenized_dataset = val_tokenized_dataset.filter(
    lambda data: 100 <= len(data["input_ids"]) <= 500
)

max_seq_len = 500

def collate_fn(batch):
  input_batch = []

  for data in batch:
    ids = data["input_ids"]

    # truncate
    ids = ids[:max_seq_len+1]

    if len(ids) < 2:
      continue

    if len(ids) < max_seq_len + 1:

      padding_length = (
          max_seq_len + 1 - len(ids)
      )

      ids = ids + [
          tokenizer.token_to_id("[PAD]")
      ] * padding_length

    input_batch.append(ids)


  input_ids = torch.tensor(
       input_batch,
       dtype = torch.long
   )

  input = input_ids[:,:-1]
  labels = input_ids[:, 1:]

  return input, labels

batch_size = 8

train_loader = DataLoader(
    train_tokenized_dataset,
    batch_size = batch_size,
    shuffle=True,
    collate_fn = collate_fn
)


val_loader = DataLoader(
    val_tokenized_dataset,
    batch_size = batch_size,
    shuffle=True,
    collate_fn = collate_fn
)

