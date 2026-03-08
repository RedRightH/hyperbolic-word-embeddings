import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.preprocessing.dataset_utils import load_graph, create_train_test_split, create_node_to_id_mapping, save_split, save_mapping

def prepare_training_data(edges_filename="wordnet_edges.txt", test_split=0.2, seed=42):
    """
    Prepare training data by loading graph and creating train/test split.
    
    Args:
        edges_filename: input edges file
        test_split: fraction of edges for testing
        seed: random seed
    
    Returns:
        tuple of (train_edges, test_edges, node2id, id2node)
    """
    from src.utils.config import RAW_DATA_DIR
    
    edges_path = RAW_DATA_DIR / edges_filename
    
    if not edges_path.exists():
        print(f"Edges file not found: {edges_path}")
        print("Please run extract_wordnet.py first")
        return None
    
    edges = []
    with open(edges_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) == 2:
                    edges.append((parts[0], parts[1]))
    
    print(f"Loaded {len(edges)} edges")
    
    train_edges, test_edges = create_train_test_split(edges, test_split=test_split, seed=seed)
    
    all_edges = train_edges + test_edges
    node2id, id2node = create_node_to_id_mapping(all_edges)
    
    save_split(train_edges, test_edges)
    save_mapping(node2id, id2node)
    
    return train_edges, test_edges, node2id, id2node

if __name__ == "__main__":
    prepare_training_data()
