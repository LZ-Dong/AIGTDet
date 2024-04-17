import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__)) 
parent_dir = os.path.dirname(script_dir)  
sys.path.append(parent_dir)
import json
import nltk
import re
from transformers import RobertaTokenizer
from tqdm import tqdm
import argparse
import spacy

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

def find_sentence_position(sent, sentence_positions):
    for i, pos in enumerate(sentence_positions):
        if sent.start == pos:
            return i
    return None

def build_graph_from_clusters(doc):
    nodes = []
    edges = []
    entity_occur = {}
    sentence_positions = []
    sen2node = []
    last_sen_cnt = 0

    for sent in doc.sents:
        sentence_positions.append(sent.start)
    sens = [sent.text.replace("\t", " ") for sent in doc.sents]
    clusters = [
        val for key, val in doc.spans.items() if key.startswith("coref_cluster")
    ]
    # Iterate through every found cluster
    for cluster in clusters:
        # Iterate through every span in the cluster
        kws_cnt = 0
        # Append nodes
        for span in list(cluster):
            kw = re.sub(r"[^a-zA-Z0-9,.\'\`!?]+", " ", span.text)
            words = [
                word
                for word in nltk.word_tokenize(kw)
            ]
            sen_idx = find_sentence_position(span.sent, sentence_positions)
            # sen_tmp_node.append(len(nodes))
            nodes.append({"text": kw, "words": words, "sentence_id": sen_idx})
            if kw not in entity_occur.keys():
                entity_occur[kw] = 0
            entity_occur[kw] += 1
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
    # Build sen2node
    for i in range(len(sentence_positions)):
        sen2node.append([])
    for idx, node in enumerate(nodes):
        sen2node[node['sentence_id']].append(idx)
    if not nodes:
        return [], [], [], [], []
    return nodes, list(set(edges)), entity_occur, sens, sen2node


def clean_string(string):
    return re.sub(r"[^a-zA-Z0-9,.\'!?]+", "", string)


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
parser.add_argument(
    "--raw_dir",
    type=str,
    required=True,
    help="The path of the input dataset.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="",
    help="The path to the output dataset with graph.",
)
args = parser.parse_args()

if __name__ == "__main__":
    if args.output_dir == "":
        args.output_dir = args.raw_dir.replace(".jsonl", "_coref.jsonl")
    print("Loading Dataset ...")
    data = read_data(args.raw_dir)
    print("Loading Tokenizer ...")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base", do_lower_case=False)
    max_seq_length = 512
    no_node = 0
    spacy.prefer_gpu(3)
    nlp = spacy.load("en_coreference_web_trf")
    with open(args.output_dir, "w", encoding="utf8") as outf:
        for idx, line in tqdm(enumerate(data)):
            text = line["article"]
            # 仅处理前max_seq_length个token的文本
            tokens = tokenizer.tokenize(text)[:max_seq_length-2]
            text = tokenizer.convert_tokens_to_string(tokens)
            doc = nlp(text)
            nodes, edges, entity_occur, sens, sen2node = build_graph_from_clusters(doc)
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
                "drop_nodes": drop_nodes,
                "sentence_to_node_id": sen2node,
                "sentence_start_end_idx_pair": sen_idx_pair,
            }
            outf.write(json.dumps(line) + "\n")

    print("{} instances are too short that have no graph".format(no_node))
