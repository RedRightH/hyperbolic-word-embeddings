"""
Quick demo script with small dataset for fast testing.

This script runs a minimal version of the pipeline with:
- Small subset of WordNet (1000 edges)
- Reduced training epochs
- Quick evaluation
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

def quick_demo():
    """Run a quick demo with minimal data."""
    
    print("=" * 80)
    print("QUICK DEMO: Hyperbolic vs Euclidean Embeddings")
    print("=" * 80)
    print("\nThis demo uses a small dataset for fast testing.")
    print("For full results, run: python run_full_pipeline.py")
    print()
    
    from src.preprocessing.extract_wordnet import extract_and_save
    from src.preprocessing.build_hierarchy import build_and_save_hierarchy
    from src.training.trainer import prepare_training_data
    from src.models.euclidean_embeddings import EuclideanEmbeddings
    from src.models.poincare_embeddings import PoincareEmbeddings
    from src.utils.config import MODELS_DIR
    
    print("[1/6] Extracting small WordNet subset...")
    extract_and_save(pos_filter='n', max_depth=5, limit=1000, filename="wordnet_edges_demo.txt")
    
    print("\n[2/6] Building hierarchy graph...")
    build_and_save_hierarchy(edges_filename="wordnet_edges_demo.txt")
    
    print("\n[3/6] Preparing training data...")
    train_edges, test_edges, node2id, id2node = prepare_training_data(
        edges_filename="wordnet_edges_demo.txt"
    )
    
    print("\n[4/6] Training Euclidean embeddings (quick)...")
    euc_model = EuclideanEmbeddings(embedding_dim=50)
    euc_model.train(train_edges, node2id, id2node, epochs=10)
    euc_path = MODELS_DIR / "euclidean_demo.pkl"
    euc_model.save(euc_path)
    print(f"Saved to {euc_path}")
    
    print("\n[5/6] Training Poincaré embeddings (quick)...")
    poi_model = PoincareEmbeddings(embedding_dim=5)
    poi_model.train(train_edges, node2id, id2node, epochs=50, batch_size=16)
    poi_path = MODELS_DIR / "poincare_demo.pkl"
    poi_model.save(poi_path)
    print(f"Saved to {poi_path}")
    
    print("\n[6/6] Quick evaluation...")
    from src.utils.distance_metrics import compute_distance_batch
    import numpy as np
    
    sample_size = min(100, len(test_edges))
    sample_edges = test_edges[:sample_size]
    
    euc_errors = []
    poi_errors = []
    
    for child, parent in sample_edges:
        if child in node2id and parent in node2id:
            child_id = node2id[child]
            parent_id = node2id[parent]
            
            euc_dist = compute_distance_batch(
                euc_model.embeddings[child_id:child_id+1],
                euc_model.embeddings[parent_id:parent_id+1],
                metric='euclidean'
            )[0, 0]
            
            poi_dist = compute_distance_batch(
                poi_model.embeddings[child_id:child_id+1],
                poi_model.embeddings[parent_id:parent_id+1],
                metric='poincare'
            )[0, 0]
            
            euc_errors.append(euc_dist)
            poi_errors.append(poi_dist)
    
    print("\n" + "=" * 80)
    print("DEMO RESULTS")
    print("=" * 80)
    print(f"\nEuclidean embeddings (dim={euc_model.embedding_dim}):")
    print(f"  Mean distance to parent: {np.mean(euc_errors):.4f}")
    print(f"  Std distance to parent:  {np.std(euc_errors):.4f}")
    
    print(f"\nPoincaré embeddings (dim={poi_model.embedding_dim}):")
    print(f"  Mean distance to parent: {np.mean(poi_errors):.4f}")
    print(f"  Std distance to parent:  {np.std(poi_errors):.4f}")
    
    print(f"\nDimension efficiency: {euc_model.embedding_dim / poi_model.embedding_dim:.1f}x")
    print("\nNote: This is a quick demo with limited data.")
    print("For comprehensive results, run the full pipeline.")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    try:
        quick_demo()
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
