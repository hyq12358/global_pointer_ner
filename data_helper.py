#!/usr/bin/env python3
import torch
import json
import numpy as np
from utils import Preprocessor
from torch.utils.data import Dataset


def load_data(data_path, data_type="train"):
    if data_type == "train" or data_type == "valid":
        datas = []
        with open(data_path, encoding="utf8") as f:
            for line in f.readlines():
                line = json.loads(line)
                item = dict()
                item["text"] = line["text"]
                item["entity_list"] = []
                for k, v in line["label"].items():
                    for spans in v.values():
                        for start, end in spans:
                            item["entity_list"].append((start, end, k))
                print(item)
                datas.append(item)
        return datas
    else:
        return json.load(open(data_path, encoding="utf8"))


class MyDataset(Dataset):
    def __init__(self, data) -> None:
        self.data = data
        self.length = len(data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.length


class DataMaker(object):
    def __init__(self, tokenizer, add_special_tokens=True):
        super().__init__()
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens
        self.preprocessor = Preprocessor(tokenizer, add_special_tokens)

    def generate_inputs(self, datas, max_seq_len, ent2id, data_type="train"):
        ent_type_size = len(ent2id)
        all_inputs = []
        for sample in datas:
            inputs = self.tokenizer(
                sample["text"],
                max_length=max_seq_len,
                truncation=True,
                padding="max_length",
            )
            labels = None
            if data_type != "test":
                ent2token_spans = self.preprocessor.get_ent2token_spans(
                    sample["text"], sample["entity_list"]
                )
                labels = np.zeros((ent_type_size, max_seq_len, max_seq_len))
                for start, end, label in ent2token_spans:
                    labels[ent2id[label], start, end] = 1
            inputs["labels"] = labels

            input_ids = torch.tensor(inputs["input_ids"]).long()
            attention_mask = torch.tensor(inputs["attention_mask"]).long()
            token_type_ids = torch.tensor(inputs["token_type_ids"]).long()
            if labels is not None:
                labels = torch.tensor(inputs["labels"]).long()
            sample_input = (sample, input_ids, attention_mask, token_type_ids, labels)
            all_inputs.append(sample_input)
        return all_inputs

    def generate_batch(self, batch_data, max_seq_len, ent2id, data_type="train"):
        batch_data = self.generate_inputs(batch_data, max_seq_len, ent2id, data_type)
        sample_list = []
        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = []
        labels_list = []
        for sample in batch_data:
            sample_list.append(sample[0])
            input_ids_list.append(sample[1])
            attention_mask_list.append(sample[2])
            token_type_ids_list.append(sample[3])
            if data_type != "test":
                labels_list.append(sample[4])

        batch_input_ids = torch.stack(input_ids_list, dim=0)
        batch_attention_mask = torch.stack(attention_mask_list, dim=0)
        batch_token_type_ids = torch.stack(token_type_ids_list, dim=0)
        batch_labels = torch.stack(labels_list, dim=0) if data_type != "test" else None
        return (
            sample_list,
            batch_input_ids,
            batch_attention_mask,
            batch_token_type_ids,
            batch_labels,
        )


if __name__ == "__main__":
    load_data("./data/train.json")
