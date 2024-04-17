from torch.utils.data import Dataset
import torch
import json

def load_data(file_path, max_length=512, max_nodes=10, max_edges=30):
    text_list = []
    label_list = []
    kw_pos_list = []
    edges_list = []
    file = open(file_path, "r", encoding="utf8")
    data = [json.loads(line) for line in file.readlines()]
    for line in data:
        text_list.append(line["article"])
        label_list.append(line["label"])

        kw_pos = []
        edges = line["information"]["graph"]["edges"]
        edges = [[x[0], x[1]] for x in edges]
        for node in line["information"]["graph"]["nodes"]:
            kw_pos.append(node["spans"])
        # 去除结束位置超过max_length的节点，去除多于max_nodes数量的节点
        nodes_to_remove = [i for i, pos in enumerate(kw_pos) if pos[1] > max_length-2 or i >= max_nodes or pos[0] < 0 or pos[1] < 0]
        if len(nodes_to_remove) > 0:
            # 去除对应边，并对剩余节点重新编号
            new_edges = [[src, dst] for src, dst in edges if src not in nodes_to_remove and dst not in nodes_to_remove]
            new_node_mapping = {}
            new_index = 0
            for node_index, pos in enumerate(kw_pos):
                if node_index not in nodes_to_remove:
                    new_node_mapping[node_index] = new_index
                    new_index += 1
            updated_kw_pos = [pos for i, pos in enumerate(kw_pos) if i not in nodes_to_remove]
            updated_edges = [[new_node_mapping[src], new_node_mapping[dst]] for src, dst in new_edges]
        else:
            updated_kw_pos = kw_pos
            updated_edges = edges

        # 如果 updated_kw_pos 的长度小于max_nodes，则补充至 max_nodes 个元素，每个元素为 [0, 0]
        if len(updated_kw_pos) < max_nodes:
            updated_kw_pos += [[0, 0]] * (max_nodes - len(updated_kw_pos))
        # 如果 updated_edges 的长度大于 max_edges ，则进行截断
        if len(updated_edges) >= max_edges:
            updated_edges = updated_edges[:max_edges]
        # 如果 updated_edges 的长度小于 max_edges ，则补充至 max_edges 个元素，每个元素为 [0, 0]
        elif len(updated_edges) < max_edges:
            updated_edges += [[0, 0]] * (max_edges - len(updated_edges))
        
        kw_pos_list.append(updated_kw_pos)
        edges_list.append(updated_edges)

    return text_list, label_list, kw_pos_list, edges_list

class MyDataset(Dataset):
    def __init__(self, data, tokenizer, max_length, is_test, label_to_id):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = data[0]
        self.labels = data[1]
        self.kw_pos_list = data[2]
        self.edges = data[3]
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
        data_sample["nodes"] = self.kw_pos_list[index]
        data_sample["edges"] = self.edges[index]
        return data_sample