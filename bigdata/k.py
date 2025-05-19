def convert_dblp_to_edgelist_and_labels(edge_path, comm_path, out_edge_path, out_label_path):
    import networkx as nx

    print("[1] Reading edge list...")
    G = nx.read_edgelist(edge_path, comments='#', nodetype=int)
    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

    print("[2] Saving graph.edgelist...")
    with open(out_edge_path, 'w') as f:
        for u, v in G.edges():
            f.write(f"{u} {v}\n")

    print("[3] Reading community labels...")
    node_to_label = {}
    with open(comm_path, 'r') as f:
        for cid, line in enumerate(f):
            nodes = list(map(int, line.strip().split()))
            for node in nodes:
                # 중복 노드가 여러 커뮤니티에 들어있는 경우 한 곳에만 배정
                if node not in node_to_label:
                    node_to_label[node] = cid

    print(f"Total labeled nodes: {len(node_to_label)}")

    print("[4] Saving labels.txt...")
    with open(out_label_path, 'w') as f:
        for node, label in sorted(node_to_label.items()):
            f.write(f"{node} {label}\n")

    print("[Done] Files written:")
    print(f" - {out_edge_path}")
    print(f" - {out_label_path}")

convert_dblp_to_edgelist_and_labels(
    edge_path="com-youtube.ungraph.txt",
    comm_path="com-youtube.all.cmty.txt",
    out_edge_path="graph.edgelist",
    out_label_path="labels.txt"
)