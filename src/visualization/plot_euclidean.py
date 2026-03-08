import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import pickle
import numpy as np
import matplotlib.pyplot as plt
from src.visualization.embedding_visualizer import reduce_dimensions, plot_embeddings_2d, visualize_hierarchy_levels
from src.preprocessing.dataset_utils import load_graph
from src.utils.config import MODELS_DIR, FIGURES_DIR

def plot_euclidean_embeddings(max_nodes=500):
    """
    Visualize Euclidean embeddings using PCA and t-SNE.
    
    Args:
        max_nodes: maximum number of nodes to visualize
    """
    print("=" * 60)
    print("Visualizing Euclidean Embeddings")
    print("=" * 60)
    
    model_path = MODELS_DIR / "euclidean_embeddings.pkl"
    
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Please train the model first.")
        return
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    embeddings = model_data['embeddings']
    node2id = model_data['node2id']
    id2node = model_data['id2node']
    
    print(f"Loaded embeddings: {embeddings.shape}")
    
    if len(embeddings) > max_nodes:
        indices = np.random.choice(len(embeddings), max_nodes, replace=False)
        embeddings_subset = embeddings[indices]
        labels = [id2node[i] for i in indices]
    else:
        embeddings_subset = embeddings
        labels = [id2node[i] for i in range(len(embeddings))]
    
    print(f"Visualizing {len(embeddings_subset)} nodes")
    
    print("Reducing dimensions with PCA...")
    embeddings_pca = reduce_dimensions(embeddings_subset, method='pca', n_components=2)
    
    pca_path = FIGURES_DIR / "euclidean_pca.png"
    plot_embeddings_2d(
        embeddings_pca, 
        labels=None,
        title="Euclidean Embeddings (PCA)", 
        save_path=pca_path
    )
    
    print("Reducing dimensions with t-SNE...")
    embeddings_tsne = reduce_dimensions(
        embeddings_subset, 
        method='tsne', 
        n_components=2,
        random_state=42,
        perplexity=min(30, len(embeddings_subset) - 1)
    )
    
    tsne_path = FIGURES_DIR / "euclidean_tsne.png"
    plot_embeddings_2d(
        embeddings_tsne, 
        labels=None,
        title="Euclidean Embeddings (t-SNE)", 
        save_path=tsne_path
    )
    
    try:
        graph = load_graph()
        
        print("Creating hierarchy visualization...")
        fig = visualize_hierarchy_levels(embeddings, node2id, graph, 
                                         method='pca', max_nodes=max_nodes)
        hierarchy_path = FIGURES_DIR / "euclidean_hierarchy.png"
        plt.savefig(hierarchy_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {hierarchy_path}")
        plt.close()
    except Exception as e:
        print(f"Could not create hierarchy visualization: {e}")
    
    print("\n" + "=" * 60)
    print("Visualization Complete!")
    print(f"Figures saved to: {FIGURES_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    plot_euclidean_embeddings()
