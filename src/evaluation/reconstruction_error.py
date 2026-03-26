import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import numpy as np
import pickle
import json
from src.preprocessing.dataset_utils import load_graph, load_mapping
from src.utils.config import MODELS_DIR, TABLES_DIR
from src.utils.distance_metrics import compute_distance_batch
from src.evaluation.metrics import compute_reconstruction_error
import argparse

def evaluate_reconstruction(model_path, model_type='euclidean', graph_filename=None):
    """
    Evaluate reconstruction error for a trained model.
    
    Args:
        model_path: path to trained model
        model_type: 'euclidean' or 'poincare'
    
    Returns:
        dict of reconstruction metrics
    """
    print(f"Evaluating reconstruction error for {model_type} model...")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    embeddings = model_data['embeddings']
    node2id = model_data['node2id']
    id2node = model_data['id2node']
    
    print(f"Loaded embeddings: {embeddings.shape}")
    
    # Load the graph to compute actual tree distances
    try:
        G = load_graph(graph_filename) if graph_filename else load_graph()
    except FileNotFoundError:
        print("Graph not found. Building from edges...")
        from src.preprocessing.build_hierarchy import build_and_save_hierarchy
        G, _ = build_and_save_hierarchy()
    
    tree_dist_list = []
    emb_dist_list = []
    
    metric = 'poincare' if model_type == 'poincare' else 'euclidean'
    
    print("Computing pairwise distances from graph...")
    
    # Get nodes that are in both the graph and the model
    common_nodes = [node for node in node2id.keys() if node in G.nodes()]
    
    if len(common_nodes) < 10:
        print(f"Warning: Only {len(common_nodes)} common nodes found")
        return None
    
    # Sample pairs of nodes and compute both tree and embedding distances
    import networkx as nx
    G_undirected = G.to_undirected()
    
    # Sample node pairs
    sample_size = min(1000, len(common_nodes) * 10)
    pairs_checked = 0
    
    for i, node1 in enumerate(common_nodes[:100]):  # Sample from first 100 nodes
        for node2 in common_nodes[i+1:min(i+50, len(common_nodes))]:  # Compare with next nodes
            if pairs_checked >= sample_size:
                break
            
            # Compute tree distance (shortest path in graph)
            try:
                tree_dist = nx.shortest_path_length(G_undirected, node1, node2)
            except nx.NetworkXNoPath:
                continue  # Skip disconnected nodes
            
            # Compute embedding distance
            id1 = node2id[node1]
            id2 = node2id[node2]
            
            emb1 = embeddings[id1:id1+1]
            emb2 = embeddings[id2:id2+1]
            
            emb_dist = compute_distance_batch(emb1, emb2, metric=metric)[0, 0]
            
            tree_dist_list.append(tree_dist)
            emb_dist_list.append(emb_dist)
            pairs_checked += 1
        
        if pairs_checked >= sample_size:
            break
    
    tree_dist_array = np.array(tree_dist_list)
    emb_dist_array = np.array(emb_dist_list)
    
    print(f"Evaluated {len(tree_dist_list)} node pairs")
    
    if len(tree_dist_list) == 0:
        print("Warning: No valid node pairs found for evaluation")
        return {
            'mse': 0.0,
            'mae': 0.0,
            'rmse': 0.0,
            'correlation': 0.0
        }
    
    metrics = compute_reconstruction_error(tree_dist_array, emb_dist_array)
    
    print("\nReconstruction Error Metrics:")
    print(f"  MSE:         {metrics['mse']:.4f}")
    print(f"  MAE:         {metrics['mae']:.4f}")
    print(f"  RMSE:        {metrics['rmse']:.4f}")
    print(f"  Correlation: {metrics['correlation']:.4f}")
    
    return metrics

def compare_reconstruction_errors(dataset_prefix=None):
    """
    Compare reconstruction errors between Euclidean and Poincaré models.
    """
    print("=" * 60)
    print("Comparing Reconstruction Errors")
    print("=" * 60)
    
    euclidean_filename = f"{dataset_prefix}_euclidean_embeddings.pkl" if dataset_prefix else "euclidean_embeddings.pkl"
    poincare_filename = f"{dataset_prefix}_poincare_embeddings.pkl" if dataset_prefix else "poincare_embeddings.pkl"

    euclidean_path = MODELS_DIR / euclidean_filename
    poincare_path = MODELS_DIR / poincare_filename

    graph_filename = f"{dataset_prefix}_graph.pkl" if dataset_prefix else None
    
    results = {}
    
    if euclidean_path.exists():
        euclidean_metrics = evaluate_reconstruction(euclidean_path, 'euclidean', graph_filename=graph_filename)
        results['euclidean'] = euclidean_metrics
    else:
        print(f"Euclidean model not found: {euclidean_path}")
    
    print("\n" + "-" * 60 + "\n")
    
    if poincare_path.exists():
        poincare_metrics = evaluate_reconstruction(poincare_path, 'poincare', graph_filename=graph_filename)
        results['poincare'] = poincare_metrics
    else:
        print(f"Poincaré model not found: {poincare_path}")
    
    output_filename = f"{dataset_prefix}_reconstruction_error.json" if dataset_prefix else "reconstruction_error.json"
    output_path = TABLES_DIR / output_filename
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    if 'euclidean' in results and 'poincare' in results:
        print("\n" + "=" * 60)
        print("Comparison Summary")
        print("=" * 60)
        print(f"{'Metric':<15} {'Euclidean':<15} {'Poincaré':<15} {'Winner':<15}")
        print("-" * 60)
        
        for metric in ['mse', 'mae', 'rmse']:
            euc_val = results['euclidean'][metric]
            poi_val = results['poincare'][metric]
            winner = 'Poincaré' if poi_val < euc_val else 'Euclidean'
            print(f"{metric.upper():<15} {euc_val:<15.4f} {poi_val:<15.4f} {winner:<15}")
        
        metric = 'correlation'
        euc_val = results['euclidean'][metric]
        poi_val = results['poincare'][metric]
        winner = 'Poincaré' if poi_val > euc_val else 'Euclidean'
        print(f"{metric.capitalize():<15} {euc_val:<15.4f} {poi_val:<15.4f} {winner:<15}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare reconstruction errors')
    parser.add_argument('--dataset-prefix', type=str, default=None, help='Dataset prefix (uses prefixed processed artifacts and models)')
    args = parser.parse_args()

    compare_reconstruction_errors(dataset_prefix=args.dataset_prefix)
