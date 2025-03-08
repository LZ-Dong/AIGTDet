import sys
import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
os.environ['CORENLP_HOME'] = "/data2/donglz/codespace/stanford-corenlp-4.5.7"
from stanza.server import CoreNLPClient
sys.path.append(parent_dir)
import json
import re
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse


def read_data(fp):
    if fp.endswith(".jsonl"):
        file = open(fp, "r", encoding="utf8")
        data = [json.loads(line) for line in file.readlines()]
    return data


def cal_pos(cleaned_tokens, text):
    pos = []
    max_start, max_length = 0, 0
    single = []
    for idx, token in enumerate(cleaned_tokens):
        p = text.find(token)
        if p != -1 and token != "":
            if len(pos) == 0:
                pos = [idx]
            else:
                if abs(idx - pos[-1]) == 1 or (
                    abs(idx - pos[-1]) == 2 and cleaned_tokens[idx - 1] == ""
                ):
                    pos.append(idx)
                else:
                    single.append(pos)
                    pos = [idx]
                if (pos[-1] - pos[0] + 1) > max_length:
                    max_start = pos[0]
                    max_length = pos[-1] - pos[0] + 1
        else:
            if len(pos) > 0:
                single.append(pos)
                if (pos[-1] - pos[0] + 1) > max_length:
                    max_start = pos[0]
                    max_length = pos[-1] - pos[0] + 1
                pos = []

    if len(single) > 1:
        min_dis, final_span = 999999, []
        for span in single:
            span_text = "".join([cleaned_tokens[s] for s in span])
            dis = abs(len(text) - len(span_text))
            if dis < min_dis:
                min_dis = dis
                final_span = span
        return final_span[0], final_span[-1] - final_span[0] + 1

    if max_length == 0 and not pos:
        return -1, -1

    return max_start, max_length if max_length > 0 else pos[-1] - pos[0] + 1


def first_index_list(cleaned_tokens, text):
    start, length = cal_pos(cleaned_tokens, text)
    return start, length


def build_graph_from_clusters(ann):
    nodes = []
    edges = []
    last_sen_cnt = 0
    sens = [''.join([x.word for x in sent.token]) for sent in ann.sentence]

    for cluster in ann.corefChain:
        kws_cnt = 0
        # Append nodes
        for mention in cluster.mention:
            words_list = ann.sentence[mention.sentenceIndex].token[mention.beginIndex:mention.endIndex]
            #build a string out of the words of this mention
            ment_word = ''.join([x.word for x in words_list])
            nodes.append({"text": ment_word, "sentence_id": mention.sentenceIndex})
            kws_cnt += 1
        # Append edges
        for i in range(kws_cnt - 1):
            for j in range(i + 1, kws_cnt):
                if nodes[last_sen_cnt + i]['sentence_id'] == nodes[last_sen_cnt + j]['sentence_id']:
                    edges += [
                        tuple([last_sen_cnt + i, last_sen_cnt + j, "inner"])
                    ]
                else:
                    edges += [
                        tuple([last_sen_cnt + i, last_sen_cnt + j, "inter"])
                    ]
        last_sen_cnt += kws_cnt
    return nodes, list(set(edges)), sens


# 去除占位符，将中文中混杂的英文转小写
def clean_string(string):
    return re.sub(r"#", "", string).lower()


def generate_rep_mask_based_on_graph(ent_nodes, sens, tokenizer):
    sen_start_idx = [0]
    sen_idx_pair, sen_tokens, all_tokens, drop_nodes = [], [], [], []
    for sen in sens:
        sen_token = tokenizer.tokenize(sen)
        cleaned_sen_token = [clean_string(token) for token in sen_token]
        sen_tokens.append(cleaned_sen_token)
        sen_idx_pair.append(
            tuple([sen_start_idx[-1], sen_start_idx[-1] + len(sen_token)])
        )
        sen_start_idx.append(sen_start_idx[-1] + len(sen_token))
        all_tokens += sen_token

    for nidx, node in enumerate(ent_nodes):
        node_text = node["text"]
        start_pos, node_len = first_index_list(
            sen_tokens[node["sentence_id"]], clean_string(node_text)
        )
        if start_pos != -1:
            final_start_pos = sen_start_idx[node["sentence_id"]] + start_pos
            max_pos = final_start_pos + node_len
            ent_nodes[nidx]["spans"] = tuple([final_start_pos, max_pos])
        else:
            ent_nodes[nidx]["spans"] = tuple([-1, -1])
        if ent_nodes[nidx]["spans"][0] == -1:
            drop_nodes.append(nidx)
        else:
            ent_nodes[nidx]["spans_check"] = all_tokens[final_start_pos:max_pos]

    return ent_nodes, all_tokens, drop_nodes, sen_idx_pair


parser = argparse.ArgumentParser()
parser.add_argument("--raw_dir", type=str, required=True, help="The path of the input dataset.")
# parser.add_argument("--raw_dir", type=str, default="/data2/donglz/codespace/AIGTDet/data/zh_finance/train.jsonl", help="The path of the input dataset.")
parser.add_argument("--output_dir", type=str, default="", help="The path to the output dataset with graph.")
args = parser.parse_args()


if __name__ == "__main__":
    if args.output_dir == "":
        args.output_dir = args.raw_dir.replace(".jsonl", "_coref.jsonl")
    print("Loading Dataset ...")
    data = read_data(args.raw_dir)
    print("Loading Tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained('/data1/models/chinese-roberta-wwm-ext')
    max_seq_length = 512
    no_node = 0
    with open(args.output_dir, "w", encoding="utf8") as outf:
        with CoreNLPClient(
                annotators=['tokenize','ssplit','pos','lemma','ner', 'parse', 'depparse','coref'],
                endpoint="http://localhost:9091",
                properties='chinese',
                # threads=1,
                be_quiet=True,
                memory='16G') as client:
            for idx, line in tqdm(enumerate(data)):
                text = line["article"]
                ann = client.annotate(text)
                nodes, edges, sens = build_graph_from_clusters(ann)
                if not nodes:
                    no_node += 1
                (
                    nodes,
                    all_tokens,
                    drop_nodes,
                    sen_idx_pair
                ) = generate_rep_mask_based_on_graph(nodes, sens, tokenizer)

                line["information"] = {}
                line["information"]["graph"] = {
                    "nodes": nodes,
                    "edges": edges,
                    "all_tokens": all_tokens,
                }
                outf.write(json.dumps(line, ensure_ascii=False) + "\n")

    print("{} instances are too short that have no graph".format(no_node))
