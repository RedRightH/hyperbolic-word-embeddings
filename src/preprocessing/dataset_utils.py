import numpy as np
import pickle
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import PROCESSED_DATA_DIR, EVALUATION_CONFIG

def load_graph(filename="wordnet_graph.pkl"):
    """
    Load graph from pickle file.
    
    Args:
        filename: input filename
    
    Returns:
        networkx graph
    """
    filepath = PROCESSED_DATA_DIR / filename
    
    with open(filepath, 'rb') as f:
        G = pickle.load(f)
    
    print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G

def load_distances(filename="tree_distances.pkl"):
    """
    Load distance dictionary from pickle file.
    
    Args:
        filename: input filename
    
    Returns:
        dict of distances
    """
    filepath = PROCESSED_DATA_DIR / filename
    
    with open(filepath, 'rb') as f:
        distances = pickle.load(f)
    
    print(f"Loaded {len(distances)} pairwise distances")
    return distances

def create_train_test_split(edges, test_split=None, seed=None):
    """
    Split edges into training and testing sets.
    
    Args:
        edges: list of (child, parent) tuples
        test_split: fraction of edges for testing
        seed: random seed
    
    Returns:
        tuple of (train_edges, test_edges)
    """
    if test_split is None:
        test_split = EVALUATION_CONFIG['test_split']
    if seed is None:
        seed = EVALUATION_CONFIG['seed']
    
    np.random.seed(seed)
    
    edges = list(edges)
    np.random.shuffle(edges)
    
    n_test = int(len(edges) * test_split)
    test_edges = edges[:n_test]
    train_edges = edges[n_test:]
    
    print(f"Split: {len(train_edges)} train edges, {len(test_edges)} test edges")
    
    return train_edges, test_edges

def save_split(train_edges, test_edges, train_file="train_edges.pkl", test_file="test_edges.pkl"):
    """
    Save train/test split to files.
    
    Args:
        train_edges: list of training edges
        test_edges: list of testing edges
        train_file: output filename for training edges
        test_file: output filename for testing edges
    """
    train_path = PROCESSED_DATA_DIR / train_file
    test_path = PROCESSED_DATA_DIR / test_file
    
    with open(train_path, 'wb') as f:
        pickle.dump(train_edges, f)
    
    with open(test_path, 'wb') as f:
        pickle.dump(test_edges, f)
    
    print(f"Saved train edges to {train_path}")
    print(f"Saved test edges to {test_path}")

def load_split(train_file="train_edges.pkl", test_file="test_edges.pkl"):
    """
    Load train/test split from files.
    
    Args:
        train_file: input filename for training edges
        test_file: input filename for testing edges
    
    Returns:
        tuple of (train_edges, test_edges)
    """
    train_path = PROCESSED_DATA_DIR / train_file
    test_path = PROCESSED_DATA_DIR / test_file
    
    with open(train_path, 'rb') as f:
        train_edges = pickle.load(f)
    
    with open(test_path, 'rb') as f:
        test_edges = pickle.load(f)
    
    print(f"Loaded {len(train_edges)} train edges and {len(test_edges)} test edges")
    
    return train_edges, test_edges

def create_node_to_id_mapping(edges):
    """
    Create mapping from node names to integer IDs.
    
    Args:
        edges: list of (child, parent) tuples
    
    Returns:
        tuple of (node2id dict, id2node dict)
    """
    nodes = set()
    for child, parent in edges:
        nodes.add(child)
        nodes.add(parent)
    
    nodes = sorted(list(nodes))
    
    node2id = {node: i for i, node in enumerate(nodes)}
    id2node = {i: node for i, node in enumerate(nodes)}
    
    print(f"Created mapping for {len(nodes)} unique nodes")
    
    return node2id, id2node

def save_mapping(node2id, id2node, node2id_file="node2id.pkl", id2node_file="id2node.pkl"):
    """
    Save node mappings to files.
    
    Args:
        node2id: dict mapping node names to IDs
        id2node: dict mapping IDs to node names
        node2id_file: output filename for node2id
        id2node_file: output filename for id2node
    """
    node2id_path = PROCESSED_DATA_DIR / node2id_file
    id2node_path = PROCESSED_DATA_DIR / id2node_file
    
    with open(node2id_path, 'wb') as f:
        pickle.dump(node2id, f)
    
    with open(id2node_path, 'wb') as f:
        pickle.dump(id2node, f)
    
    print(f"Saved mappings to {node2id_path} and {id2node_path}")

def load_mapping(node2id_file="node2id.pkl", id2node_file="id2node.pkl"):
    """
    Load node mappings from files.
    
    Args:
        node2id_file: input filename for node2id
        id2node_file: input filename for id2node
    
    Returns:
        tuple of (node2id dict, id2node dict)
    """
    node2id_path = PROCESSED_DATA_DIR / node2id_file
    id2node_path = PROCESSED_DATA_DIR / id2node_file
    
    with open(node2id_path, 'rb') as f:
        node2id = pickle.load(f)
    
    with open(id2node_path, 'rb') as f:
        id2node = pickle.load(f)
    
    print(f"Loaded mappings for {len(node2id)} nodes")
    
    return node2id, id2node
