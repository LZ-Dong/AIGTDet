import json
import argparse

def read_data(fp):
    if fp.endswith(".jsonl"):
        file = open(fp, "r", encoding="utf8")
        data = [json.loads(line) for line in file.readlines()]
    return data

parser = argparse.ArgumentParser()
parser.add_argument("--raw_dir", default="data/GROVER500/test_coref.jsonl", type=str, help="The path of the input dataset.")
args = parser.parse_args()

if __name__ == "__main__":
    dataset = read_data(args.raw_dir)
    # Initialize counters for nodes and edges
    num_human_nodes = 0
    num_machine_nodes = 0
    num_human_edges = 0
    num_human_inner_edges = 0
    num_human_inter_edges = 0
    num_machine_edges = 0
    num_machine_inner_edges = 0
    num_machine_inter_edges = 0
    num_human_graphs = 0
    num_machine_graphs = 0
    
    for data in dataset:
        # Extract graph information
        graph_info = data['information']['graph']
        
        # Determine the label of the graph
        label = data['label']
        
        # Count the number of nodes and edges
        num_nodes = len(graph_info['nodes'])
        num_edges = len(graph_info['edges'])
        
        # Update counters based on the label
        if label == "human":
            num_human_nodes += num_nodes
            num_human_edges += num_edges
            num_human_graphs += 1
            for edge in graph_info['edges']:
                edge_type = edge[2]
                if edge_type == "inter":
                    num_human_inter_edges += 1
                elif edge_type == "inner":
                    num_human_inner_edges += 1
        elif label == "machine":
            num_machine_nodes += num_nodes
            num_machine_edges += num_edges
            num_machine_graphs += 1
            for edge in graph_info['edges']:
                edge_type = edge[2]
                if edge_type == "inter":
                    num_machine_inter_edges += 1
                elif edge_type == "inner":
                    num_machine_inner_edges += 1
    
    # Calculate average nodes and edges for human and machine graphs
    avg_human_nodes = num_human_nodes / num_human_graphs if num_human_graphs != 0 else 0
    avg_human_edges = num_human_edges / num_human_graphs if num_human_graphs != 0 else 0
    avg_human_inter_edges = num_human_inter_edges / num_human_graphs if num_human_graphs != 0 else 0
    avg_human_inner_edges = num_human_inner_edges / num_human_graphs if num_human_graphs != 0 else 0
    avg_machine_nodes = num_machine_nodes / num_machine_graphs if num_machine_graphs != 0 else 0
    avg_machine_edges = num_machine_edges / num_machine_graphs if num_machine_graphs != 0 else 0
    avg_machine_inter_edges = num_machine_inter_edges / num_machine_graphs if num_machine_graphs != 0 else 0
    avg_machine_inner_edges = num_machine_inner_edges / num_machine_graphs if num_machine_graphs != 0 else 0
    
    print("Human Graphs:")
    print("Average nodes:", avg_human_nodes)
    print("Average edges:", avg_human_edges)
    print("Average inter edges:", avg_human_inter_edges)
    print("Average inner edges:", avg_human_inner_edges)
    
    print("\nMachine Graphs:")
    print("Average nodes:", avg_machine_nodes)
    print("Average edges:", avg_machine_edges)
    print("Average inter edges:", avg_machine_inter_edges)
    print("Average inner edges:", avg_machine_inner_edges)