import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import networkx as nx

from src.preprocessing.build_hierarchy import load_edges, build_graph, compute_tree_distances, save_graph, save_distances
from src.training.trainer import prepare_training_data
from src.utils.config import RAW_DATA_DIR


def _bfs_order(G: nx.DiGraph, start_node, max_depth: int, max_nodes: int | None):
    visited = {start_node}
    order = [start_node]
    frontier = [(start_node, 0)]

    while frontier:
        node, depth = frontier.pop(0)
        if depth >= max_depth:
            continue

        for nbr in G.neighbors(node):
            if nbr in visited:
                continue
            visited.add(nbr)
            order.append(nbr)
            frontier.append((nbr, depth + 1))
            if max_nodes is not None and len(order) >= max_nodes:
                return order

    return order


def _pick_root(G_child_parent: nx.DiGraph):
    roots = [n for n in G_child_parent.nodes() if G_child_parent.out_degree(n) == 0]
    if not roots:
        return None

    G_topdown = G_child_parent.reverse(copy=False)

    def score(r):
        lengths = nx.single_source_shortest_path_length(G_topdown, r, cutoff=8)
        max_depth = max(lengths.values()) if lengths else 0
        count = len(lengths)
        return (max_depth, count)

    return max(roots, key=score)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-edges', default='wordnet_edges.txt')
    parser.add_argument('--dataset-prefix', default='subtree')
    parser.add_argument('--root', default=None)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--max-nodes', type=int, default=3000)
    parser.add_argument('--test-split', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    edges = load_edges(args.input_edges)
    G = build_graph(edges)  # child -> parent

    root = args.root
    if root is None:
        root = _pick_root(G)
        if root is None:
            raise ValueError('Could not infer a root; please pass --root')

    if root not in G:
        raise ValueError(f"Root node '{root}' not found in graph")

    max_nodes = None if args.max_nodes == 0 else args.max_nodes

    G_topdown = G.reverse(copy=False)  # parent -> child
    order = _bfs_order(G_topdown, root, max_depth=args.depth, max_nodes=max_nodes)
    nodes = set(order)

    G_sub_topdown = G_topdown.subgraph(nodes).copy()
    G_sub = G_sub_topdown.reverse(copy=True)  # back to child -> parent

    subtree_edges = list(G_sub.edges())

    out_edges_filename = f"{args.dataset_prefix}_edges.txt"
    out_edges_path = RAW_DATA_DIR / out_edges_filename

    with open(out_edges_path, 'w', encoding='utf-8') as f:
        for child, parent in subtree_edges:
            f.write(f"{child}\t{parent}\n")

    print(f"Saved subtree edges to: {out_edges_path}")
    print(f"Subtree nodes: {G_sub.number_of_nodes()}  edges: {G_sub.number_of_edges()}  root: {root}")

    graph_pkl = f"{args.dataset_prefix}_graph.pkl"
    dist_pkl = f"{args.dataset_prefix}_distances.pkl"

    save_graph(G_sub, filename=graph_pkl)
    distances = compute_tree_distances(G_sub)
    save_distances(distances, filename=dist_pkl)

    prepare_training_data(
        edges_filename=out_edges_filename,
        test_split=args.test_split,
        seed=args.seed,
        dataset_prefix=args.dataset_prefix,
    )


if __name__ == '__main__':
    main()
