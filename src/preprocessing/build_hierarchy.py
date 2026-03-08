import networkx as nx
import pickle
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

def load_edges(filename="wordnet_edges.txt"):
    """
    Load edges from text file.
    
    Args:
        filename: input filename
    
    Returns:
        list of (child, parent) tuples
    """
    filepath = RAW_DATA_DIR / filename
    
    edges = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) == 2:
                    edges.append((parts[0], parts[1]))
    
    print(f"Loaded {len(edges)} edges from {filepath}")
    return edges

def build_graph(edges):
    """
    Build a directed graph from edge list.
    
    Args:
        edges: list of (child, parent) tuples
    
    Returns:
        networkx.DiGraph
    """
    G = nx.DiGraph()
    G.add_edges_from(edges)
    
    print(f"Graph statistics:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Connected components: {nx.number_weakly_connected_components(G)}")
    
    if nx.is_directed_acyclic_graph(G):
        print("  Graph is a DAG (Directed Acyclic Graph)")
    else:
        print("  Warning: Graph contains cycles")
    
    return G

def compute_tree_distances(G):
    """
    Compute shortest path distances between all node pairs.
    
    Args:
        G: networkx graph
    
    Returns:
        dict mapping (node1, node2) to distance
    """
    print("Computing tree distances...")
    
    distances = {}
    
    G_undirected = G.to_undirected()
    
    all_pairs = dict(nx.all_pairs_shortest_path_length(G_undirected))
    
    for source in all_pairs:
        for target, dist in all_pairs[source].items():
            distances[(source, target)] = dist
    
    print(f"Computed {len(distances)} pairwise distances")
    return distances

def save_graph(G, filename="wordnet_graph.pkl"):
    """
    Save graph to pickle file.
    
    Args:
        G: networkx graph
        filename: output filename
    """
    output_path = PROCESSED_DATA_DIR / filename
    
    with open(output_path, 'wb') as f:
        pickle.dump(G, f)
    
    print(f"Saved graph to {output_path}")
    return output_path

def save_distances(distances, filename="tree_distances.pkl"):
    """
    Save distance dictionary to pickle file.
    
    Args:
        distances: dict of distances
        filename: output filename
    """
    output_path = PROCESSED_DATA_DIR / filename
    
    with open(output_path, 'wb') as f:
        pickle.dump(distances, f)
    
    print(f"Saved distances to {output_path}")
    return output_path

def build_and_save_hierarchy(edges_filename="wordnet_edges.txt"):
    """
    Build hierarchy graph and compute distances.
    
    Args:
        edges_filename: input edges file
    
    Returns:
        tuple of (graph, distances)
    """
    edges = load_edges(edges_filename)
    
    G = build_graph(edges)
    
    save_graph(G)
    
    distances = compute_tree_distances(G)
    save_distances(distances)
    
    return G, distances

if __name__ == "__main__":
    build_and_save_hierarchy()
