import json
import random
import os

dataset_name = 'baike'

def read_data(fp):
    if fp.endswith(".jsonl"):
        file = open(fp, "r", encoding="utf8")
        data = [json.loads(line) for line in file.readlines()]
    return data

dataset = read_data(f'HC3-Chinese/{dataset_name}.jsonl')
with open(f"HC3-Chinese/chatgpt.jsonl", "w", encoding="utf8") as f:
    for data in dataset:
        text = data['chatgpt_answers'][0]
        label = "machine"
        new_data = {"article": text, "label": label}
        json.dump(new_data, f, ensure_ascii=False)
        f.write("\n")

with open(f"HC3-Chinese/human.jsonl", "w", encoding="utf8") as f:
    for data in dataset:
        if not data['human_answers'][0]:
            continue
        text = data['human_answers'][0]
        label = "human"
        new_data = {"article": text, "label": label}
        json.dump(new_data, f, ensure_ascii=False)
        f.write("\n")

# os.makedirs(f"/data2/donglz/codespace/AIGTDet/data/zh_{dataset_name}", exist_ok=True)

# # 读取按标签整理的文件
# machine_data = read_data('HC3-Chinese/chatgpt.jsonl')
# human_data = read_data('HC3-Chinese/human.jsonl')

# machine_train_data = random.sample(machine_data, int(0.6 * len(machine_data)))
# human_train_data = random.sample(human_data, int(0.6 * len(human_data)))

# # 将训练集写入文件
# with open(f"/data2/donglz/codespace/AIGTDet/data/zh_{dataset_name}/train.jsonl", "w", encoding="utf-8") as train_file:
#     for data in machine_train_data + human_train_data:
#         text = data['article']
#         label = data['label']
#         new_data = {"article": text, "label": label}
#         json.dump(new_data, train_file, ensure_ascii=False)
#         train_file.write("\n")

# # 将剩余的数据作为测试集写入文件
# with open(f"/data2/donglz/codespace/AIGTDet/data/zh_{dataset_name}/test.jsonl", "w", encoding="utf-8") as test_file:
#     for data in machine_data + human_data:
#         if data not in machine_train_data and data not in human_train_data:
#             text = data['article']
#             label = data['label']
#             new_data = {"article": text, "label": label}
#             json.dump(new_data, test_file, ensure_ascii=False)
#             test_file.write("\n")