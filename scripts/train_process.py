from transformers import BertTokenizer
import re
from datasets import load_dataset

class DatasetProcessor:
    def __init__(self, path, tokenizer_name="bert-base-uncased"):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.path = path
        self.train_dataset = None
        self.val_dataset = None

    def modify_labels(self, example):
        if example['label'] < 2:
            example['label'] = 0
        elif example['label'] > 2:
            example['label'] = 1
        return example

    def filter_labels(self, example):
        return example['label'] != 2

    def remove_punctuation(self, example):
        example["text"] = re.sub(r'[^\w\s]', '', example["text"])
        return example

    def tokenize_function(self, examples):
        return self.tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512,
            add_special_tokens=True
        )

    def prepare_datasets(self):
        dataset = load_dataset("json", data_files=self.path)

        dataset = dataset.map(self.modify_labels)
        dataset = dataset.filter(self.filter_labels)
        dataset = dataset.map(self.remove_punctuation)
        tokenized_dataset = dataset.map(self.tokenize_function, batched=True, num_proc=4)

        split = tokenized_dataset["train"].train_test_split(test_size=0.1)
        self.train_dataset = split["train"]
        self.val_dataset = split["test"]
        return self.train_dataset, self.val_dataset