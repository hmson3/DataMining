import os
import time
import csv
import networkx as nx
import sys
sys.path.append("../src")
from leiden_lpa import leiden_lpa_hybrid
from evaluation import compute_modularity, compute_nmi

def run_leiden(G, seed=None):
    import igraph as ig
    import leidenalg
    G_ig = ig.Graph.TupleList(G.edges(), directed=False)
    if seed is not None:
        partition = leidenalg.find_partition(G_ig, leidenalg.ModularityVertexPartition, seed=seed)
    else:
        partition = leidenalg.find_partition(G_ig, leidenalg.ModularityVertexPartition)
    return {v["name"]: partition.membership[i] for i, v in enumerate(G_ig.vs)}

def load_graph_and_labels(dataset_folder):
    graph_path = os.path.join(dataset_folder, "graph.edgelist")
    label_path = os.path.join(dataset_folder, "labels.txt")

    G = nx.read_edgelist(graph_path, nodetype=str)
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    gt = {}
    with open(label_path, "r") as f:
        for line in f:
            node, label = line.strip().split()
            gt[node] = int(label)

    return G, gt

def run_experiment(dataset_base="../datasets", repeat=10, output_csv="results/results_ratio.csv"):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    fieldnames = ["Graph", "Repeat", "Algorithm", "Time (s)", "Modularity", "NMI"]

    append = os.path.exists(output_csv)
    with open(output_csv, 'a' if append else 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not append:
            writer.writeheader()

        for dataset_name in sorted(os.listdir(dataset_base)):
            folder = os.path.join(dataset_base, dataset_name)
            if not os.path.isdir(folder):
                continue

            print(f"[INFO] Running on {dataset_name}...")
            G, gt = load_graph_and_labels(folder)

            for core_ratio in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                algo_name = f"Leiden-LPA-coreratio_{core_ratio}"
                print(f"  [SUB] core_ratio = {core_ratio}...")

                for i in range(repeat):
                    seed = i + 42

                    start = time.time()
                    labels = leiden_lpa_hybrid(G, core_ratio=core_ratio, seed=seed)
                    t = time.time() - start
                    m = compute_modularity(G, labels)
                    n = compute_nmi(labels, gt)

                    writer.writerow({
                        "Graph": dataset_name,
                        "Repeat": i,
                        "Algorithm": algo_name,
                        "Time (s)": round(t, 4),
                        "Modularity": round(m, 4),
                        "NMI": round(n, 4)
                    })

if __name__ == "__main__":
    run_experiment()
