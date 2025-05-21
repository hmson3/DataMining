import networkx as nx
from collections import Counter
from leidenalg import find_partition, ModularityVertexPartition
import igraph as ig

def adaptive_core_ratio(G):
    avg_deg = sum(dict(G.degree()).values()) / G.number_of_nodes()
    if avg_deg < 3:
        return 0.1
    elif avg_deg < 5:
        return 0.2
    else:
        return 0.3

def leiden_lpa_hybrid(G_nx, max_iter=10, max_pass=3, entropy_threshold=0.01, seed=None):
    pagerank = nx.pagerank(G_nx, alpha=0.85)
    degree = dict(G_nx.degree())
    labels = {v: None for v in G_nx.nodes()}
    prev_modularity = -1.0

    for iteration in range(max_pass):
        # Step 1. Adaptive Core Ratio
        core_ratio = adaptive_core_ratio(G_nx)
        sorted_nodes = sorted(pagerank, key=pagerank.get, reverse=True)
        num_core = int(len(sorted_nodes) * core_ratio)
        V_core = sorted_nodes[:num_core]
        V_periphery = sorted_nodes[num_core:]

        # Step 2. Leiden clustering on core
        G_core_nx = G_nx.subgraph(V_core).copy()
        G_core_ig = ig.Graph.TupleList(G_core_nx.edges(), directed=False)
        part = find_partition(G_core_ig, ModularityVertexPartition, seed=seed) if seed else find_partition(G_core_ig, ModularityVertexPartition)
        core_labels = {v["name"]: part.membership[i] for i, v in enumerate(G_core_ig.vs)}

        # Step 3. Label Initialization
        for v in G_nx.nodes():
            labels[v] = core_labels[v] if v in core_labels else None

        # Step 4. LPA refinement on periphery
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

        # Step 5. Modularity Approximation (using edge agreement)
        intra_edges = 0
        for u, v in G_nx.edges():
            if labels[u] == labels[v]:
                intra_edges += 1
        mod_approx = intra_edges / G_nx.number_of_edges()

        # Early stop if modularity converges
        if abs(mod_approx - prev_modularity) < entropy_threshold:
            break
        prev_modularity = mod_approx

    return labels