from torch.utils.data import Dataset
import torch
import json

def load_data(file_path):
    text_list = []
    label_list = []
    file = open(file_path, "r", encoding="utf8")
    data = [json.loads(line) for line in file.readlines()]
    for line in data:
        text_list.append(line["article"])
        label_list.append(line["label"])

    return text_list, label_list

class MyDataset(Dataset):
    def __init__(self, data, tokenizer, max_length, is_test, label_to_id):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = data[0]
        self.labels = data[1]
        self.is_test = is_test
        self.label_to_id = label_to_id
            
    def __len__(self):
        """returns the length of dataframe"""
        return len(self.texts)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""
        text = str(self.texts[index])
        source = self.tokenizer.batch_encode_plus(
            [text],
            max_length=self.max_length,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        data_sample = {
            "input_ids": source_ids,
            "attention_mask": source_mask,
        }
        if not self.is_test:
            label_name = self.labels[index]
            label = self.label_to_id[label_name]
            target_ids = torch.tensor(label).squeeze()
            data_sample["labels"] = target_ids
        
        return data_sample