import json
import os

def read_data(fp):
    if fp.endswith(".jsonl"):
        file = open(fp, "r", encoding="utf8")
        data = [json.loads(line) for line in file.readlines()]
    return data

def construct_data(MGT_name = "ChatGPT", split_name = "train", limit_num = 5000):
    MGT_dir = f"OpenLLMText/{MGT_name}/{split_name}-dirty.jsonl"
    HWT_dir = f"OpenLLMText/Human/{split_name}-dirty.jsonl"
    output_dir = f"OpenLLMText/Human_{MGT_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    MGTdataset = read_data(MGT_dir)
    HWTdataset = read_data(HWT_dir)
    with open(f"{output_dir}/{split_name}.jsonl", "w") as f:
        count = 0
        for data in MGTdataset:
            if count >= limit_num:
                break
            text = data['text']
            label = data['extra']['source']
            new_data = {"article": text, "label": label}
            json.dump(new_data, f)
            f.write("\n")
            count += 1
        count = 0
        for data in HWTdataset:
            if count >= limit_num:
                break
            text = data['text']
            label = data['extra']['source']
            new_data = {"article": text, "label": label}
            json.dump(new_data, f)
            f.write("\n")
            count += 1

def construct_data_all(split_name = "train", limit_num = 2000):
    MGT_list = ["ChatGPT", "GPT2", "Human", "LLaMA", "PaLM"]
    output_dir = f"OpenLLMText/Mixed"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(f"{output_dir}/{split_name}.jsonl", "w") as f:
        for MGT_name in MGT_list:
            MGT_dir = f"OpenLLMText/{MGT_name}/{split_name}-dirty.jsonl"
            MGTdataset = read_data(MGT_dir)
            count = 0
            for data in MGTdataset:
                if count >= limit_num:
                    break
                text = data['text']
                label = data['extra']['source']
                new_data = {"article": text, "label": label}
                json.dump(new_data, f)
                f.write("\n")
                count += 1

if __name__ == "__main__":
    construct_data("GPT2", "train", 500)
    construct_data("GPT2", "valid", 1000)
    # construct_data_all("train", 200)
    # construct_data_all("valid", 50)