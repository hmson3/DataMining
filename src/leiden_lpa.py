import networkx as nx
from collections import Counter, defaultdict
from leidenalg import find_partition, ModularityVertexPartition
import igraph as ig

def leiden_lpa_hybrid(G_nx, core_ratio=0.2, max_iter=10, alpha=0.5, seed=None):
    pagerank = nx.pagerank(G_nx, alpha=0.85)
    degree = dict(G_nx.degree())
    
    # Step 2. Core 노드 선정 (PageRank 기준)
    sorted_nodes = sorted(pagerank, key=pagerank.get, reverse=True)
    num_core = int(len(sorted_nodes) * core_ratio)
    V_core = sorted_nodes[:num_core]
    V_periphery = sorted_nodes[num_core:]
    
    # Step 3. Core 서브그래프에 Leiden 적용
    G_core_nx = G_nx.subgraph(V_core).copy()
    G_core_ig = ig.Graph.TupleList(G_core_nx.edges(), directed=False)
    if seed is not None:
        part = find_partition(G_core_ig, ModularityVertexPartition, seed=seed)
    else:
        part = find_partition(G_core_ig, ModularityVertexPartition)
    C_core = {v["name"]: part.membership[i] for i, v in enumerate(G_core_ig.vs)}
    
    # Step 4. 초기 라벨 설정
    labels = {v: C_core[v] if v in C_core else None for v in G_nx.nodes()}
    
    # Step 5. 비코어 노드 라벨 업데이트
    for _ in range(max_iter):
        updated = False
        for v in V_periphery:
            scores = defaultdict(float)
            for n in G_nx.neighbors(v):
                if labels[n] is not None:
                    score = alpha * pagerank[n] + (1 - alpha) * degree[n]
                    scores[labels[n]] += score
            if scores:
                best_label = max(scores.items(), key=lambda x: x[1])[0]
                if labels[v] != best_label:
                    labels[v] = best_label
                    updated = True
        if not updated:
            break
    
    # Step 6. 전체 그래프 1회 LPA refinement
    for _ in range(1):
        updated = False
        for v in G_nx.nodes():
            scores = defaultdict(float)
            for n in G_nx.neighbors(v):
                if labels[n] is not None:
                    score = alpha * pagerank[n] + (1 - alpha) * degree[n]
                    scores[labels[n]] += score
            if scores:
                best_label = max(scores.items(), key=lambda x: x[1])[0]
                if labels[v] != best_label:
                    labels[v] = best_label
                    updated = True
        if not updated:
            break
    
    return labels

