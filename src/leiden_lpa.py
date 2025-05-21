import networkx as nx
from collections import Counter
from leidenalg import find_partition, ModularityVertexPartition
import igraph as ig

def leiden_lpa_hybrid(G_nx, core_ratio=0.4, seed=None, max_iter=10):

    # 코어 비율 0.0 LPA만 수행행
    if core_ratio <= 0.0:
        labels = {v: i for i, v in enumerate(G_nx.nodes())}  # 초기 라벨
        updated = True
        while updated:
            updated = False
            for v in G_nx.nodes():
                neighbor_labels = [labels[n] for n in G_nx.neighbors(v)]
                if neighbor_labels:
                    most_common = Counter(neighbor_labels).most_common(1)[0][0]
                    if labels[v] != most_common:
                        labels[v] = most_common
                        updated = True
        return labels

    # 코어 비율 1.0 → Leiden만 수행
    if core_ratio >= 1.0:
        G_ig = ig.Graph.TupleList(G_nx.edges(), directed=False)
        part = find_partition(G_ig, ModularityVertexPartition, seed=seed) if seed else find_partition(G_ig, ModularityVertexPartition)
        return {v["name"]: part.membership[i] for i, v in enumerate(G_ig.vs)}


    # pagerank에 따라 중심 노드 선별
    pagerank = nx.pagerank(G_nx, alpha=0.85)
    sorted_nodes = sorted(pagerank, key=pagerank.get, reverse=True)
    num_core = int(len(sorted_nodes) * core_ratio)
    V_core = sorted_nodes[:num_core]
    V_periphery = sorted_nodes[num_core:]

    # 코어 → Leiden
    G_core_nx = G_nx.subgraph(V_core).copy()
    G_core_ig = ig.Graph.TupleList(G_core_nx.edges(), directed=False)
    part = find_partition(G_core_ig, ModularityVertexPartition, seed=seed) if seed else find_partition(G_core_ig, ModularityVertexPartition)
    core_labels = {v["name"]: part.membership[i] for i, v in enumerate(G_core_ig.vs)}

    # 전체 라벨 초기화
    labels = {v: core_labels[v] if v in core_labels else None for v in G_nx.nodes()}

    # 비코어 노드 lpa
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

    return labels
