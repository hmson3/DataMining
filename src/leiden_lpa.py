import networkx as nx
from collections import defaultdict
from leidenalg import find_partition, ModularityVertexPartition
import igraph as ig

def neighborhood_overlap(G, v, u):
    """v와 u 사이의 이웃 유사도(공통 이웃 비율)"""
    neighbors_v = set(G.neighbors(v))
    neighbors_u = set(G.neighbors(u))
    if not neighbors_v or not neighbors_u:
        return 0.0
    intersection = len(neighbors_v & neighbors_u)
    union = len(neighbors_v | neighbors_u)
    return intersection / union if union > 0 else 0.0

def leiden_lpa_hybrid(G_nx, core_ratio=0.2, max_iter=10, alpha=0.5, beta=0.5, seed=None):
    pagerank = nx.pagerank(G_nx, alpha=0.85)
    degree = dict(G_nx.degree())

    # Step 1. Core 노드 선정 (PageRank 기준)
    sorted_nodes = sorted(pagerank, key=pagerank.get, reverse=True)
    num_core = int(len(sorted_nodes) * core_ratio)
    V_core = sorted_nodes[:num_core]
    V_periphery = sorted_nodes[num_core:]

    # Step 2. Core 부분 그래프 Leiden 클러스터링
    G_core_nx = G_nx.subgraph(V_core).copy()
    G_core_ig = ig.Graph.TupleList(G_core_nx.edges(), directed=False)
    part = find_partition(G_core_ig, ModularityVertexPartition, seed=seed) if seed else find_partition(G_core_ig, ModularityVertexPartition)
    C_core = {v["name"]: part.membership[i] for i, v in enumerate(G_core_ig.vs)}

    # Step 3. 라벨 초기화
    labels = {v: C_core[v] if v in C_core else None for v in G_nx.nodes()}

    # Step 4. 비코어 노드 업데이트 (soft-weighted voting)
    for _ in range(max_iter):
        updated = False
        for v in V_periphery:
            scores = defaultdict(float)
            for n in G_nx.neighbors(v):
                if labels[n] is not None:
                    trust = alpha * pagerank[n] + (1 - alpha) * degree[n]
                    sim = neighborhood_overlap(G_nx, v, n)
                    scores[labels[n]] += trust * (1 + beta * sim)
            if scores:
                best_label = max(scores.items(), key=lambda x: x[1])[0]
                if labels[v] != best_label:
                    labels[v] = best_label
                    updated = True
        if not updated:
            break

    # Step 5. 전체 노드에 대해 1회 soft LPA refinement
    for _ in range(1):
        updated = False
        for v in G_nx.nodes():
            scores = defaultdict(float)
            for n in G_nx.neighbors(v):
                if labels[n] is not None:
                    trust = alpha * pagerank[n] + (1 - alpha) * degree[n]
                    sim = neighborhood_overlap(G_nx, v, n)
                    scores[labels[n]] += trust * (1 + beta * sim)
            if scores:
                best_label = max(scores.items(), key=lambda x: x[1])[0]
                if labels[v] != best_label:
                    labels[v] = best_label
                    updated = True
        if not updated:
            break

    return labels
