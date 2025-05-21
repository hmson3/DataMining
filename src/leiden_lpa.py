import networkx as nx
from collections import Counter
from leidenalg import find_partition, ModularityVertexPartition
import igraph as ig

def leiden_lpa_hybrid(G_nx, core_ratio=0.4, seed=None):
    # 1. PageRank 계산
    pagerank = nx.pagerank(G_nx, alpha=0.85)
    
    # 2. 상위 core_ratio% 코어 선정
    sorted_nodes = sorted(pagerank, key=pagerank.get, reverse=True)
    num_core = int(len(sorted_nodes) * core_ratio)
    V_core = sorted_nodes[:num_core]
    V_periphery = sorted_nodes[num_core:]

    # 3. 코어 그래프 Leiden 클러스터링
    G_core_nx = G_nx.subgraph(V_core).copy()
    G_core_ig = ig.Graph.TupleList(G_core_nx.edges(), directed=False)
    part = find_partition(G_core_ig, ModularityVertexPartition, seed=seed) if seed else find_partition(G_core_ig, ModularityVertexPartition)
    core_labels = {v["name"]: part.membership[i] for i, v in enumerate(G_core_ig.vs)}

    # 4. 전체 노드 라벨 초기화
    labels = {v: core_labels[v] if v in core_labels else None for v in G_nx.nodes()}

    # 5. 비코어 노드 1회 라벨 전파 (LPA 없음, 단순 최대 빈도)
    for v in V_periphery:
        neighbor_labels = [labels[n] for n in G_nx.neighbors(v) if labels[n] is not None]
        if neighbor_labels:
            most_common = Counter(neighbor_labels).most_common(1)[0][0]
            labels[v] = most_common

    return labels
