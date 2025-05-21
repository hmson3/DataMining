import os
import random
import numpy as np
import networkx as nx

# 파라미터 설정
node_sizes = [120, 500, 1000, 5000, 10000]
mu_values = [0.1, 0.3, 0.5]
avg_degree = 10
tau1 = 3
tau2 = 1.5
base_dir = "../datasets"

for n in node_sizes:
    min_community = 5 if n <= 500 else 10 if n <= 1000 else 20

    for mu in mu_values:
        name = f"lfr-n{n}-mu{int(mu * 10)}"
        folder = os.path.join(base_dir, name)
        os.makedirs(folder, exist_ok=True)

        print(f"[INFO] Generating {name}...")

        success = False
        for attempt in range(5):
            adjusted_min_c = max(1, min_community - attempt)
            adjusted_avg_deg = max(5, avg_degree - attempt)
            seed_val = int(n + mu * 1000 + attempt)

            random.seed(seed_val)
            np.random.seed(seed_val)

            try:
                g = nx.generators.community.LFR_benchmark_graph(
                    n=n,
                    tau1=tau1,
                    tau2=tau2,
                    mu=mu,
                    average_degree=adjusted_avg_deg,
                    min_community=adjusted_min_c,
                    max_iters=500,
                    seed=seed_val
                )
                success = True
                break
            except Exception as e:
                print(f"[WARN] attempt {attempt+1} failed: {e}")

        if not success:
            print(f"[ERROR] Failed to generate {name}")
            continue

        g = nx.convert_node_labels_to_integers(g, first_label=1)

        # graph.edgelist 저장
        with open(os.path.join(folder, "graph.edgelist"), "w") as f:
            for u, v in sorted(g.edges()):
                f.write(f"{u}\t{v}\n")

        # labels.txt 저장
        community_attr = nx.get_node_attributes(g, "community")
        comm_dict = {}
        cluster_id = 1
        seen = set()
        for comm in community_attr.values():
            frozen = frozenset(comm)
            if frozen in seen:
                continue
            seen.add(frozen)
            for node in comm:
                comm_dict[int(node)] = cluster_id
            cluster_id += 1

        with open(os.path.join(folder, "labels.txt"), "w") as f:
            for node in sorted(comm_dict.keys()):
                f.write(f"{node+1}\t{comm_dict[node]}\n")

        print(f"[SAVED] {name}")
