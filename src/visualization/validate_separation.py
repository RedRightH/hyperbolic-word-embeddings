"""
Create normalized validation showing separation quality.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pickle
import networkx as nx
from src.utils.config import MODELS_DIR, FIGURES_DIR
from src.preprocessing.dataset_utils import load_graph
from src.utils.distance_metrics import compute_distance_batch

def compute_separation_metrics(model_path, model_type):
    """
    Compute normalized separation metrics.
    """
    # Load model
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    embeddings = model_data['embeddings']
    node2id = model_data['node2id']
    
    # Load graph
    G = load_graph()
    
    metric = 'poincare' if model_type == 'poincare' else 'euclidean'
    
    # Get parent-child distances
    parent_child_dists = []
    edges = list(G.edges())[:200]
    
    for child, parent in edges:
        if child in node2id and parent in node2id:
            id1 = node2id[child]
            id2 = node2id[parent]
            emb1 = embeddings[id1:id1+1]
            emb2 = embeddings[id2:id2+1]
            dist = compute_distance_batch(emb1, emb2, metric=metric)[0, 0]
            parent_child_dists.append(dist)
    
    # Sample non-connected pairs
    nodes = list(node2id.keys())[:100]
    non_connected_dists = []
    np.random.seed(42)
    for _ in range(200):
        node1, node2 = np.random.choice(nodes, 2, replace=False)
        if not G.has_edge(node1, node2) and not G.has_edge(node2, node1):
            if node1 in node2id and node2 in node2id:
                id1 = node2id[node1]
                id2 = node2id[node2]
                emb1 = embeddings[id1:id1+1]
                emb2 = embeddings[id2:id2+1]
                dist = compute_distance_batch(emb1, emb2, metric=metric)[0, 0]
                non_connected_dists.append(dist)
    
    pc_mean = np.mean(parent_child_dists)
    nc_mean = np.mean(non_connected_dists)
    
    # Compute normalized metrics
    separation_ratio = (nc_mean - pc_mean) / pc_mean if pc_mean > 0 else 0
    overlap = len([d for d in parent_child_dists if d >= nc_mean]) / len(parent_child_dists)
    
    return {
        'pc_mean': pc_mean,
        'nc_mean': nc_mean,
        'pc_std': np.std(parent_child_dists),
        'nc_std': np.std(non_connected_dists),
        'separation_ratio': separation_ratio,
        'overlap': overlap,
        'pc_dists': parent_child_dists,
        'nc_dists': non_connected_dists
    }

def plot_normalized_comparison(euclidean_path, poincare_path, output_path):
    """
    Create normalized comparison showing separation quality.
    """
    print("\nCreating normalized separation comparison...")
    
    euc_metrics = compute_separation_metrics(euclidean_path, 'euclidean')
    poi_metrics = compute_separation_metrics(poincare_path, 'poincare')
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Hierarchical Structure Validation: Normalized Comparison', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Separation Ratio (higher is better)
    ax = axes[0, 0]
    models = ['Euclidean', 'Poincaré']
    ratios = [euc_metrics['separation_ratio'], poi_metrics['separation_ratio']]
    colors = ['#e74c3c' if r < 0 else '#2ecc71' for r in ratios]
    bars = ax.bar(models, ratios, color=colors, edgecolor='black', linewidth=2)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_ylabel('Separation Ratio\n(Non-Connected - Connected) / Connected', 
                  fontsize=11, fontweight='bold')
    ax.set_title('Separation Quality (Higher = Better)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, ratio in zip(bars, ratios):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{ratio:.2f}',
                ha='center', va='bottom' if ratio > 0 else 'top', 
                fontsize=14, fontweight='bold')
    
    # Plot 2: Overlap Percentage (lower is better)
    ax = axes[0, 1]
    overlaps = [euc_metrics['overlap'] * 100, poi_metrics['overlap'] * 100]
    colors = ['#e74c3c' if o > 50 else '#2ecc71' for o in overlaps]
    bars = ax.bar(models, overlaps, color=colors, edgecolor='black', linewidth=2)
    ax.set_ylabel('Overlap %\n(Connected pairs that look non-connected)', 
                  fontsize=11, fontweight='bold')
    ax.set_title('Distribution Overlap (Lower = Better)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 100)
    
    for bar, overlap in zip(bars, overlaps):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{overlap:.1f}%',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Plot 3: Normalized distributions for Euclidean
    ax = axes[1, 0]
    
    # Normalize to 0-1 scale
    euc_pc_norm = (np.array(euc_metrics['pc_dists']) - min(euc_metrics['pc_dists'])) / \
                  (max(euc_metrics['nc_dists']) - min(euc_metrics['pc_dists']) + 1e-10)
    euc_nc_norm = (np.array(euc_metrics['nc_dists']) - min(euc_metrics['pc_dists'])) / \
                  (max(euc_metrics['nc_dists']) - min(euc_metrics['pc_dists']) + 1e-10)
    
    ax.hist(euc_pc_norm, bins=30, alpha=0.7, label='Parent-Child', color='blue', edgecolor='black')
    ax.hist(euc_nc_norm, bins=30, alpha=0.7, label='Non-Connected', color='red', edgecolor='black')
    ax.axvline(np.mean(euc_pc_norm), color='blue', linestyle='--', linewidth=2)
    ax.axvline(np.mean(euc_nc_norm), color='red', linestyle='--', linewidth=2)
    
    ax.set_xlabel('Normalized Distance (0-1)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title(f'Euclidean: Normalized Distributions\nSeparation Ratio: {euc_metrics["separation_ratio"]:.2f}', 
                 fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Normalized distributions for Poincaré
    ax = axes[1, 1]
    
    # Normalize to 0-1 scale
    poi_pc_norm = (np.array(poi_metrics['pc_dists']) - min(poi_metrics['pc_dists'])) / \
                  (max(poi_metrics['nc_dists']) - min(poi_metrics['pc_dists']) + 1e-10)
    poi_nc_norm = (np.array(poi_metrics['nc_dists']) - min(poi_metrics['pc_dists'])) / \
                  (max(poi_metrics['nc_dists']) - min(poi_metrics['pc_dists']) + 1e-10)
    
    ax.hist(poi_pc_norm, bins=30, alpha=0.7, label='Parent-Child', color='blue', edgecolor='black')
    ax.hist(poi_nc_norm, bins=30, alpha=0.7, label='Non-Connected', color='red', edgecolor='black')
    ax.axvline(np.mean(poi_pc_norm), color='blue', linestyle='--', linewidth=2)
    ax.axvline(np.mean(poi_nc_norm), color='red', linestyle='--', linewidth=2)
    
    ax.set_xlabel('Normalized Distance (0-1)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title(f'Poincaré: Normalized Distributions\nSeparation Ratio: {poi_metrics["separation_ratio"]:.2f}', 
                 fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {output_path}")
    plt.close()
    
    return euc_metrics, poi_metrics

def main():
    """Run normalized validation."""
    print("=" * 70)
    print("NORMALIZED SEPARATION ANALYSIS")
    print("=" * 70)
    
    euclidean_path = MODELS_DIR / "euclidean_embeddings.pkl"
    poincare_path = MODELS_DIR / "poincare_embeddings.pkl"
    output_path = FIGURES_DIR / "validation_normalized_separation.png"
    
    euc_metrics, poi_metrics = plot_normalized_comparison(euclidean_path, poincare_path, output_path)
    
    print("\n" + "=" * 70)
    print("RESULTS (What Really Matters)")
    print("=" * 70)
    print("\n1. SEPARATION RATIO (Higher = Better)")
    print(f"   Euclidean: {euc_metrics['separation_ratio']:.3f}")
    print(f"   Poincaré:  {poi_metrics['separation_ratio']:.3f}")
    if poi_metrics['separation_ratio'] > euc_metrics['separation_ratio']:
        print(f"   ✓ Poincaré wins by {poi_metrics['separation_ratio'] - euc_metrics['separation_ratio']:.3f}")
    else:
        print(f"   ✓ Euclidean wins by {euc_metrics['separation_ratio'] - poi_metrics['separation_ratio']:.3f}")
    
    print("\n2. OVERLAP (Lower = Better)")
    print(f"   Euclidean: {euc_metrics['overlap']*100:.1f}%")
    print(f"   Poincaré:  {poi_metrics['overlap']*100:.1f}%")
    if poi_metrics['overlap'] < euc_metrics['overlap']:
        print(f"   ✓ Poincaré has {(euc_metrics['overlap'] - poi_metrics['overlap'])*100:.1f}% less overlap")
    
    print("\n3. RAW DISTANCES (For Reference)")
    print(f"   Euclidean - Parent-Child: {euc_metrics['pc_mean']:.3f}, Non-Connected: {euc_metrics['nc_mean']:.3f}")
    print(f"   Poincaré  - Parent-Child: {poi_metrics['pc_mean']:.3f}, Non-Connected: {poi_metrics['nc_mean']:.3f}")
    print(f"   (Note: Different scales - use separation ratio for comparison)")
    
    print("\n" + "=" * 70)
    print("INTERPRETATION:")
    print("=" * 70)
    print("Separation Ratio = (Non-Connected - Connected) / Connected")
    print("  • Positive = Good separation (connected pairs are closer)")
    print("  • Negative = Bad separation (no clear distinction)")
    print("  • Higher absolute value = Better structure preservation")
    print("\nOverlap % = How many connected pairs are farther than average non-connected")
    print("  • Lower = Better (less confusion between connected/non-connected)")
    print("=" * 70)

if __name__ == "__main__":
    main()
