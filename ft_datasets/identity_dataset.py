# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

import copy
import json
import os
import torch

from sentencepiece import SentencePieceProcessor
from torch.utils.data import Dataset
from typing import List

TEMPLATE = {
    "prefix": "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.",
    "prompt": "USER: {query} ASSISTANT: ",
    "sep": "\n"
}

class IdentityDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_words=128):
        self.ann = json.load(open(dataset_config.data_path))

        if partition == "train":
            self.ann = self.ann
        else:
            self.ann = self.ann[:50]

        self.max_words = max_words
        # tokenizer = Tokenizer(model_path=model_path + "./tokenizer.model")
        self.tokenizer = tokenizer
        # self.tokenizer1 = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]

        query = TEMPLATE['prefix'] + TEMPLATE['sep'] + TEMPLATE['prompt'].format(query=ann["instruction"])
        response = ann["output"]
        dialog = [query, response]

        for turn, (user, bot) in enumerate(ann["history"]):
            dialog.append(TEMPLATE['prompt'].format(query=user))
            dialog.append(bot)
        
        input_ids, labels = [], []
        
        for i in range(len(dialog)//2):
            source_ids = self.tokenizer.encode(text=dialog[2*i], add_special_tokens=False)
            target_ids = self.tokenizer.encode(text=dialog[2*i+1], add_special_tokens=False)
            input_ids += source_ids + [self.tokenizer.bos_token_id] + target_ids + [self.tokenizer.eos_token_id]
            labels += [-100] * (len(source_ids) + 1) + target_ids + [self.tokenizer.eos_token_id]

        example = torch.tensor(
            input_ids, dtype=torch.int64
        )
        labels = torch.tensor(
            labels, dtype=torch.int64
        )

        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
            labels = torch.cat((labels, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[: self.max_words]
            labels = labels[: self.max_words]

        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = 0
        example_mask = example_mask.float()
        label_mask = label_mask.float()

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask":example_mask,
        }
