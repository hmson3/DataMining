import os
import time
import csv
import networkx as nx
import sys
import random
sys.path.append("./src")
from leiden_lpa import leiden_lpa_hybrid
from evaluation import compute_modularity, compute_nmi

# 버전 이름을 명확히 설정
ALGORITHM_VERSION = "Leiden-LPA-v6"

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

def run_experiment(dataset_base="realdata", repeat=1, output_csv=f"realresults/results_{ALGORITHM_VERSION}.csv"):
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

            for i in range(repeat):
                seed = i + 42

                # 개선 알고리즘 실행
                start = time.time()
                hybrid_labels = leiden_lpa_hybrid(G, seed=seed)
                t1 = time.time() - start
                m1 = compute_modularity(G, hybrid_labels)
                n1 = compute_nmi(hybrid_labels, gt)
                writer.writerow({
                    "Graph": dataset_name,
                    "Repeat": i,
                    "Algorithm": ALGORITHM_VERSION,
                    "Time (s)": round(t1, 4),
                    "Modularity": round(m1, 4),
                    "NMI": round(n1, 4)
                })

                # Leiden 원형 알고리즘 실행
                start = time.time()
                leiden_labels = run_leiden(G, seed=seed)
                t2 = time.time() - start
                m2 = compute_modularity(G, leiden_labels)
                n2 = compute_nmi(leiden_labels, gt)
                writer.writerow({
                    "Graph": dataset_name,
                    "Repeat": i,
                    "Algorithm": "Leiden",
                    "Time (s)": round(t2, 4),
                    "Modularity": round(m2, 4),
                    "NMI": round(n2, 4)
                })

if __name__ == "__main__":
    run_experiment()
