import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx

from src.preprocessing.dataset_utils import load_graph
from src.utils.config import FIGURES_DIR


def _get_roots(G: nx.DiGraph):
    # Edges are stored as child -> parent, so a "root" (top-most concept) has no parent.
    # That means out_degree == 0 in the stored orientation.
    roots = [node for node in G.nodes() if G.out_degree(node) == 0]
    return roots


def _pick_default_root(G: nx.DiGraph):
    roots = _get_roots(G)
    if not roots:
        return None

    # Prefer a root that yields a deeper/larger visible subtree.
    # Compute reachability in top-down orientation (parent -> child) with a small cutoff.
    G_topdown = G.reverse(copy=False)

    def score(r):
        lengths = nx.single_source_shortest_path_length(G_topdown, r, cutoff=6)
        max_depth = max(lengths.values()) if lengths else 0
        count = len(lengths)
        # prioritize depth first, then breadth
        return (max_depth, count)

    return max(roots, key=score)


def _bfs_nodes(G: nx.DiGraph, start_node, max_depth: int):
    visited = {start_node}
    frontier = [(start_node, 0)]

    while frontier:
        node, depth = frontier.pop(0)
        if depth >= max_depth:
            continue

        for nbr in G.neighbors(node):
            if nbr not in visited:
                visited.add(nbr)
                frontier.append((nbr, depth + 1))

    return visited


def _bfs_order_and_depths(G: nx.DiGraph, start_node, max_depth: int):
    visited = {start_node}
    order = [start_node]
    depths = {start_node: 0}
    frontier = [(start_node, 0)]

    while frontier:
        node, depth = frontier.pop(0)
        if depth >= max_depth:
            continue

        for nbr in G.neighbors(node):
            if nbr not in visited:
                visited.add(nbr)
                depths[nbr] = depth + 1
                order.append(nbr)
                frontier.append((nbr, depth + 1))

    return order, depths


def _bfs_depths(G: nx.DiGraph, start_node, max_depth: int):
    depths = {start_node: 0}
    frontier = [(start_node, 0)]

    while frontier:
        node, depth = frontier.pop(0)
        if depth >= max_depth:
            continue

        for nbr in G.neighbors(node):
            if nbr not in depths:
                depths[nbr] = depth + 1
                frontier.append((nbr, depth + 1))

    return depths


def extract_subtree(
    G: nx.DiGraph,
    root=None,
    depth: int = 3,
    max_nodes: int | None = 300,
):
    if root is None:
        root = _pick_default_root(G)
        if root is None:
            raise ValueError("Could not infer a root node (no nodes with out_degree == 0). Please pass --root.")

    if root not in G:
        raise ValueError(f"Root node '{root}' not found in graph")

    # Edges are stored as child -> parent. For a top-down tree view we want parent -> child.
    G_topdown = G.reverse(copy=False)
    order, depths = _bfs_order_and_depths(G_topdown, root, max_depth=depth)

    if max_nodes is not None and len(order) > max_nodes:
        order = order[:max_nodes]

    nodes = set(order)
    G_sub = G_topdown.subgraph(nodes).copy()

    # Ensure the layered layout can still compute depth even if some deeper nodes were truncated.
    nx.set_node_attributes(G_sub, {n: depths.get(n, 0) for n in G_sub.nodes()}, 'layer')

    return G_sub, root


def _hierarchy_layout(G: nx.DiGraph, root, depth_limit: int):
    depths = _bfs_depths(G, root, max_depth=depth_limit)
    for n, d in depths.items():
        G.nodes[n]['layer'] = d

    pos = nx.multipartite_layout(G, subset_key='layer', align='vertical')
    return pos, depths


def _graph_layout(G: nx.DiGraph, root, depth_limit: int):
    try:
        from networkx.drawing.nx_pydot import graphviz_layout

        return graphviz_layout(G, prog='dot')
    except Exception:
        pos, _ = _hierarchy_layout(G, root=root, depth_limit=depth_limit)
        return pos


def save_tree_plot(
    G_sub: nx.DiGraph,
    output_path: Path,
    root,
    depth_limit: int,
    title: str | None = None,
    node_size: int = 300,
    font_size: int = 7,
):
    pos, depths = _hierarchy_layout(G_sub, root=root, depth_limit=depth_limit)
    try:
        from networkx.drawing.nx_pydot import graphviz_layout

        pos = graphviz_layout(G_sub, prog='dot')
    except Exception:
        pass

    fig_w = max(10, min(30, 0.06 * G_sub.number_of_nodes()))
    fig_h = max(8, min(24, 0.05 * G_sub.number_of_nodes()))

    plt.figure(figsize=(fig_w, fig_h))

    node_colors = [depths.get(n, 0) for n in G_sub.nodes()]

    nx.draw_networkx_edges(G_sub, pos, arrows=True, arrowstyle='-|>', arrowsize=10, alpha=0.35, width=1.0)
    nx.draw_networkx_nodes(
        G_sub,
        pos,
        node_size=node_size,
        node_color=node_colors,
        cmap='viridis',
        alpha=0.95,
    )
    nx.draw_networkx_labels(G_sub, pos, font_size=font_size)

    if title is not None:
        plt.title(title)

    plt.axis('off')
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize the WordNet hierarchy graph (subtree).')
    parser.add_argument('--graph-pkl', default='wordnet_graph.pkl', help='Pickle filename in data/processed')
    parser.add_argument('--root', default=None, help='Root node to visualize (default: inferred root)')
    parser.add_argument('--depth', type=int, default=3, help='BFS depth from the root')
    parser.add_argument('--max-nodes', type=int, default=300, help='Cap number of nodes drawn (0 to disable)')
    parser.add_argument('--out', default=None, help='Output PNG path (default: results/figures/hierarchy_subtree.png)')
    parser.add_argument('--print-stats', action='store_true', help='Print depth distribution stats for the extracted subtree')

    args = parser.parse_args()

    G = load_graph(args.graph_pkl)

    max_nodes = None if args.max_nodes == 0 else args.max_nodes
    G_sub, root = extract_subtree(G, root=args.root, depth=args.depth, max_nodes=max_nodes)

    output_path = Path(args.out) if args.out is not None else (FIGURES_DIR / 'hierarchy_subtree.png')
    title = f"Hierarchy subtree (root={root}, depth={args.depth}, nodes={G_sub.number_of_nodes()})"

    if args.print_stats:
        layers = nx.get_node_attributes(G_sub, 'layer')
        counts = {}
        for d in layers.values():
            counts[d] = counts.get(d, 0) + 1
        depth_counts = ', '.join([f"{k}:{counts[k]}" for k in sorted(counts.keys())])
        print(f"Depth counts (depth:count): {depth_counts}")

    save_tree_plot(G_sub, output_path, root=root, depth_limit=args.depth, title=title)

    print(f"Saved hierarchy visualization to: {output_path}")


if __name__ == '__main__':
    main()
