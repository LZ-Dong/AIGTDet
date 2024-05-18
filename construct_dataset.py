import json
from transformers import RobertaTokenizer

def read_data(fp):
    if fp.endswith(".jsonl"):
        file = open(fp, "r", encoding="utf8")
        data = [json.loads(line) for line in file.readlines()]
    return data

if __name__ == "__main__":
    max_len = 500
    dataset = read_data('OpenLLMText/Human_LLaMA/test.jsonl')
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base", do_lower_case=False)
    truncated_data = {"human": [], "machine": []}
    for data in dataset:
        tokens = tokenizer.tokenize(data["article"])
        num_tokens = len(tokens)
        if num_tokens > 500:
            truncated_article = tokenizer.convert_tokens_to_string(tokens[:max_len])
            truncated_data[data["label"]].append({"article": truncated_article, "label": data["label"]})
    print("count_human: ", len(truncated_data["human"]))
    print("count_machine: ", len(truncated_data["machine"]))

    truncated_data["human"] = truncated_data["human"][:3000]
    truncated_data["machine"] = truncated_data["machine"][:3000]

    with open("data/LLaMA/test.jsonl", "w", encoding="utf8") as outfile:
        for label_data in truncated_data.values():
            for entry in label_data:
                json.dump(entry, outfile)
                outfile.write("\n")