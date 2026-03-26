import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import pickle

from src.preprocessing.dataset_utils import load_graph
from src.utils.config import MODELS_DIR, TABLES_DIR
from src.utils.distance_metrics import compute_distance_batch


def _load_model(model_path: Path):
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def _compute_separation_and_overlap(model_path: Path, model_type: str, graph_filename: str | None):
    model_data = _load_model(model_path)
    embeddings = model_data['embeddings']
    node2id = model_data['node2id']

    G = load_graph(graph_filename) if graph_filename else load_graph()

    metric = 'poincare' if model_type == 'poincare' else 'euclidean'

    parent_child_dists = []
    edges = list(G.edges())[:200]
    for child, parent in edges:
        if child in node2id and parent in node2id:
            id1 = node2id[child]
            id2 = node2id[parent]
            emb1 = embeddings[id1:id1 + 1]
            emb2 = embeddings[id2:id2 + 1]
            dist = compute_distance_batch(emb1, emb2, metric=metric)[0, 0]
            parent_child_dists.append(dist)

    nodes = list(node2id.keys())[:100]
    non_connected_dists = []
    np.random.seed(42)
    for _ in range(200):
        node1, node2 = np.random.choice(nodes, 2, replace=False)
        if not G.has_edge(node1, node2) and not G.has_edge(node2, node1):
            if node1 in node2id and node2 in node2id:
                id1 = node2id[node1]
                id2 = node2id[node2]
                emb1 = embeddings[id1:id1 + 1]
                emb2 = embeddings[id2:id2 + 1]
                dist = compute_distance_batch(emb1, emb2, metric=metric)[0, 0]
                non_connected_dists.append(dist)

    if len(parent_child_dists) == 0 or len(non_connected_dists) == 0:
        return {
            'separation_ratio': 0.0,
            'distribution_overlap': 1.0,
            'pc_mean': 0.0,
            'nc_mean': 0.0,
        }

    pc_mean = float(np.mean(parent_child_dists))
    nc_mean = float(np.mean(non_connected_dists))
    separation_ratio = float((nc_mean - pc_mean) / pc_mean) if pc_mean > 0 else 0.0

    # overlap fraction: connected pairs that are farther than the mean non-connected distance
    distribution_overlap = float(len([d for d in parent_child_dists if d >= nc_mean]) / len(parent_child_dists))

    return {
        'separation_ratio': separation_ratio,
        'distribution_overlap': distribution_overlap,
        'pc_mean': pc_mean,
        'nc_mean': nc_mean,
    }


def _safe_get(d: dict, key: str, default=None):
    val = d.get(key, default)
    return default if val is None else val


def compare_models(dataset_prefix: str | None):
    prefix = dataset_prefix

    graph_filename = f"{prefix}_graph.pkl" if prefix else None

    euc_model_name = f"{prefix}_euclidean_embeddings.pkl" if prefix else "euclidean_embeddings.pkl"
    poi_model_name = f"{prefix}_poincare_embeddings.pkl" if prefix else "poincare_embeddings.pkl"

    euc_path = MODELS_DIR / euc_model_name
    poi_path = MODELS_DIR / poi_model_name

    if not euc_path.exists():
        raise FileNotFoundError(f"Euclidean model not found: {euc_path}")
    if not poi_path.exists():
        raise FileNotFoundError(f"Poincaré model not found: {poi_path}")

    from src.evaluation.reconstruction_error import evaluate_reconstruction
    from src.evaluation.link_prediction import evaluate_link_prediction
    from src.preprocessing.dataset_utils import load_split, load_mapping

    if prefix:
        _, test_edges = load_split(
            train_file=f"{prefix}_train_edges.pkl",
            test_file=f"{prefix}_test_edges.pkl",
        )
        node2id, _ = load_mapping(
            node2id_file=f"{prefix}_node2id.pkl",
            id2node_file=f"{prefix}_id2node.pkl",
        )
    else:
        _, test_edges = load_split()
        node2id, _ = load_mapping()

    euc_recon = evaluate_reconstruction(euc_path, 'euclidean', graph_filename=graph_filename)
    poi_recon = evaluate_reconstruction(poi_path, 'poincare', graph_filename=graph_filename)

    euc_lp = evaluate_link_prediction(euc_path, test_edges, node2id, 'euclidean')
    poi_lp = evaluate_link_prediction(poi_path, test_edges, node2id, 'poincare')

    euc_sep = _compute_separation_and_overlap(euc_path, 'euclidean', graph_filename)
    poi_sep = _compute_separation_and_overlap(poi_path, 'poincare', graph_filename)

    euc_dim = int(_load_model(euc_path)['embeddings'].shape[1])
    poi_dim = int(_load_model(poi_path)['embeddings'].shape[1])

    results = {
        'dataset_prefix': prefix,
        'euclidean': {
            'dimensions': euc_dim,
            'reconstruction_mse': float(_safe_get(euc_recon, 'mse', 0.0)),
            'link_prediction_mean_rank': float(_safe_get(euc_lp, 'mean_rank', 0.0)),
            'separation_ratio': float(_safe_get(euc_sep, 'separation_ratio', 0.0)),
            'distribution_overlap': float(_safe_get(euc_sep, 'distribution_overlap', 1.0)),
        },
        'poincare': {
            'dimensions': poi_dim,
            'reconstruction_mse': float(_safe_get(poi_recon, 'mse', 0.0)),
            'link_prediction_mean_rank': float(_safe_get(poi_lp, 'mean_rank', 0.0)),
            'separation_ratio': float(_safe_get(poi_sep, 'separation_ratio', 0.0)),
            'distribution_overlap': float(_safe_get(poi_sep, 'distribution_overlap', 1.0)),
        },
    }

    return results


def _print_table(results: dict):
    e = results['euclidean']
    p = results['poincare']

    rows = [
        ("Reconstruction MSE", e['reconstruction_mse'], p['reconstruction_mse'], "lower"),
        ("Link Prediction Mean Rank", e['link_prediction_mean_rank'], p['link_prediction_mean_rank'], "lower"),
        ("Separation Ratio", e['separation_ratio'], p['separation_ratio'], "higher"),
        ("Distribution Overlap", e['distribution_overlap'], p['distribution_overlap'], "lower"),
        ("Dimensions", e['dimensions'], p['dimensions'], "lower"),
    ]

    print("\n" + "=" * 80)
    print("EUCLIDEAN vs POINCARE - COMPARISON")
    if results.get('dataset_prefix'):
        print(f"Dataset prefix: {results['dataset_prefix']}")
    print("=" * 80)

    header = f"{'Metric':<32} {'Euclidean':>16} {'Poincaré':>16} {'Winner':>12}"
    print(header)
    print("-" * len(header))

    def winner(e_val, p_val, rule):
        if rule == 'lower':
            return 'Poincaré' if p_val < e_val else 'Euclidean'
        return 'Poincaré' if p_val > e_val else 'Euclidean'

    for name, e_val, p_val, rule in rows:
        w = winner(e_val, p_val, rule)
        if isinstance(e_val, float) or isinstance(p_val, float):
            print(f"{name:<32} {e_val:>16.4f} {p_val:>16.4f} {w:>12}")
        else:
            print(f"{name:<32} {str(e_val):>16} {str(p_val):>16} {w:>12}")


def main():
    parser = argparse.ArgumentParser(description='Compare Euclidean vs Poincaré models on key metrics')
    parser.add_argument('--dataset-prefix', type=str, default=None, help='Dataset prefix (uses prefixed artifacts/models)')
    parser.add_argument('--out', type=str, default=None, help='Optional output JSON path')
    args = parser.parse_args()

    results = compare_models(args.dataset_prefix)
    _print_table(results)

    default_name = f"{args.dataset_prefix}_model_comparison.json" if args.dataset_prefix else "model_comparison.json"
    out_path = Path(args.out) if args.out else (TABLES_DIR / default_name)

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved comparison JSON to: {out_path}")


if __name__ == '__main__':
    main()
