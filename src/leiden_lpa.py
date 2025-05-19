import networkx as nx
from collections import Counter
from leidenalg import find_partition, ModularityVertexPartition
import igraph as ig

def leiden_lpa_hybrid(G_nx, core_ratio=0.2, max_iter=10, seed=None):
    pagerank = nx.pagerank(G_nx, alpha=0.85)
    sorted_nodes = sorted(pagerank, key=pagerank.get, reverse=True)
    num_core = int(len(sorted_nodes) * core_ratio)
    V_core = sorted_nodes[:num_core]
    V_periphery = sorted_nodes[num_core:]

    if core_ratio >= 1.0 or len(V_periphery) == 0:
        G_ig = ig.Graph.TupleList(G_nx.edges(), directed=False)
        if seed is not None:
            part = find_partition(G_ig, ModularityVertexPartition, seed=seed)
        else:
            part = find_partition(G_ig, ModularityVertexPartition)
        return {v["name"]: part.membership[i] for i, v in enumerate(G_ig.vs)}

    G_core_nx = G_nx.subgraph(V_core).copy()
    G_core_ig = ig.Graph.TupleList(G_core_nx.edges(), directed=False)
    if seed is not None:
        part = find_partition(G_core_ig, ModularityVertexPartition, seed=seed)
    else:
        part = find_partition(G_core_ig, ModularityVertexPartition)
    C_core = {v["name"]: part.membership[i] for i, v in enumerate(G_core_ig.vs)}

    labels = {v: C_core[v] if v in C_core else None for v in G_nx.nodes()}

    for _ in range(max_iter):
        updated = False
        for v in V_periphery:
            neighbor_labels = [labels[n] for n in G_nx.neighbors(v) if labels[n] is not None]
            if neighbor_labels:
                most_common = Counter(neighbor_labels).most_common(1)[0][0]
                if labels[v] != most_common:
                    labels[v] = most_common
                    updated = True
        if not updated:
            break

    # 전체 그래프에 대해 1회 LPA refinement
    all_nodes = list(G_nx.nodes())
    for _ in range(1):
        updated = False
        for v in all_nodes:
            neighbor_labels = [labels[n] for n in G_nx.neighbors(v) if labels[n] is not None]
            if neighbor_labels:
                most_common = Counter(neighbor_labels).most_common(1)[0][0]
                if labels[v] != most_common:
                    labels[v] = most_common
                    updated = True
        if not updated:
            break

    return labels
