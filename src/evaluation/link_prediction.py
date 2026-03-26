import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import numpy as np
import pickle
import json
from src.preprocessing.dataset_utils import load_split, load_mapping
from src.utils.config import MODELS_DIR, TABLES_DIR
from src.utils.distance_metrics import compute_distance_batch
from src.evaluation.metrics import compute_rank_metrics
from tqdm import tqdm
import argparse

def evaluate_link_prediction(model_path, test_edges, node2id, model_type='euclidean', max_test=1000):
    """
    Evaluate link prediction performance.
    
    Args:
        model_path: path to trained model
        test_edges: list of test edges
        node2id: dict mapping node names to IDs
        model_type: 'euclidean' or 'poincare'
        max_test: maximum number of test edges to evaluate
    
    Returns:
        dict of link prediction metrics
    """
    print(f"Evaluating link prediction for {model_type} model...")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    embeddings = model_data['embeddings']
    
    print(f"Loaded embeddings: {embeddings.shape}")
    
    metric = 'poincare' if model_type == 'poincare' else 'euclidean'
    
    valid_test_edges = []
    for child, parent in test_edges:
        if child in node2id and parent in node2id:
            valid_test_edges.append((child, parent))
    
    if len(valid_test_edges) == 0:
        print("Warning: No valid test edges found")
        return {
            'mean_rank': 0.0,
            'median_rank': 0.0,
            'hits@1': 0.0,
            'hits@5': 0.0,
            'hits@10': 0.0
        }
    
    if len(valid_test_edges) > max_test:
        np.random.seed(42)
        indices = np.random.choice(len(valid_test_edges), max_test, replace=False)
        valid_test_edges = [valid_test_edges[i] for i in indices]
    
    print(f"Evaluating {len(valid_test_edges)} test edges...")
    
    all_distances = []
    true_indices = []
    
    for child, parent in tqdm(valid_test_edges, desc="Computing distances"):
        child_id = node2id[child]
        parent_id = node2id[parent]
        
        child_emb = embeddings[child_id:child_id+1]
        
        distances = compute_distance_batch(child_emb, embeddings, metric=metric)[0]
        
        all_distances.append(distances)
        true_indices.append(parent_id)
    
    all_distances = np.array(all_distances)
    true_indices = np.array(true_indices)
    
    metrics = compute_rank_metrics(all_distances, true_indices, k_values=[1, 5, 10])
    
    print("\nLink Prediction Metrics:")
    print(f"  Mean Rank:   {metrics['mean_rank']:.2f}")
    print(f"  Median Rank: {metrics['median_rank']:.2f}")
    print(f"  Hits@1:      {metrics['hits@1']:.4f}")
    print(f"  Hits@5:      {metrics['hits@5']:.4f}")
    print(f"  Hits@10:     {metrics['hits@10']:.4f}")
    
    return metrics

def compare_link_prediction(dataset_prefix=None):
    """
    Compare link prediction performance between Euclidean and Poincaré models.
    """
    print("=" * 60)
    print("Comparing Link Prediction Performance")
    print("=" * 60)
    
    try:
        if dataset_prefix:
            train_edges, test_edges = load_split(
                train_file=f"{dataset_prefix}_train_edges.pkl",
                test_file=f"{dataset_prefix}_test_edges.pkl",
            )
            node2id, id2node = load_mapping(
                node2id_file=f"{dataset_prefix}_node2id.pkl",
                id2node_file=f"{dataset_prefix}_id2node.pkl",
            )
        else:
            train_edges, test_edges = load_split()
            node2id, id2node = load_mapping()
    except FileNotFoundError:
        print("Training data not found. Please run training first.")
        return

    euclidean_filename = f"{dataset_prefix}_euclidean_embeddings.pkl" if dataset_prefix else "euclidean_embeddings.pkl"
    poincare_filename = f"{dataset_prefix}_poincare_embeddings.pkl" if dataset_prefix else "poincare_embeddings.pkl"
    euclidean_path = MODELS_DIR / euclidean_filename
    poincare_path = MODELS_DIR / poincare_filename
    
    results = {}
    
    if euclidean_path.exists():
        euclidean_metrics = evaluate_link_prediction(
            euclidean_path, test_edges, node2id, 'euclidean'
        )
        results['euclidean'] = euclidean_metrics
    else:
        print(f"Euclidean model not found: {euclidean_path}")
    
    print("\n" + "-" * 60 + "\n")
    
    if poincare_path.exists():
        poincare_metrics = evaluate_link_prediction(
            poincare_path, test_edges, node2id, 'poincare'
        )
        results['poincare'] = poincare_metrics
    else:
        print(f"Poincaré model not found: {poincare_path}")
    
    output_filename = f"{dataset_prefix}_link_prediction.json" if dataset_prefix else "link_prediction.json"
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
        
        for metric in ['mean_rank', 'median_rank']:
            euc_val = results['euclidean'][metric]
            poi_val = results['poincare'][metric]
            winner = 'Poincaré' if poi_val < euc_val else 'Euclidean'
            print(f"{metric.replace('_', ' ').title():<15} {euc_val:<15.2f} {poi_val:<15.2f} {winner:<15}")
        
        for metric in ['hits@1', 'hits@5', 'hits@10']:
            euc_val = results['euclidean'][metric]
            poi_val = results['poincare'][metric]
            winner = 'Poincaré' if poi_val > euc_val else 'Euclidean'
            print(f"{metric.upper():<15} {euc_val:<15.4f} {poi_val:<15.4f} {winner:<15}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare link prediction performance')
    parser.add_argument('--dataset-prefix', type=str, default=None, help='Dataset prefix (uses prefixed processed artifacts and models)')
    args = parser.parse_args()

    compare_link_prediction(dataset_prefix=args.dataset_prefix)
