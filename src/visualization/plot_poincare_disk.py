import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from src.visualization.embedding_visualizer import reduce_dimensions, visualize_hierarchy_levels
from src.preprocessing.dataset_utils import load_graph
from src.utils.config import MODELS_DIR, FIGURES_DIR

def plot_poincare_disk(embeddings, labels=None, title="Poincaré Disk Embeddings",
                       save_path=None, figsize=(12, 12), alpha=0.6, max_labels=30):
    """
    Plot embeddings in the Poincaré disk.
    
    Args:
        embeddings: numpy array of shape (n_samples, 2)
        labels: optional list of labels for each point
        title: plot title
        save_path: path to save figure
        figsize: figure size
        alpha: point transparency
        max_labels: maximum number of labels to show
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    circle = Circle((0, 0), 1, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(circle)
    
    norms = np.linalg.norm(embeddings, axis=1)
    
    scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], 
                        c=norms, cmap='plasma', s=50, alpha=alpha,
                        vmin=0, vmax=1)
    
    if labels is not None and len(labels) <= max_labels:
        for i, label in enumerate(labels):
            ax.annotate(label, (embeddings[i, 0], embeddings[i, 1]), 
                       fontsize=8, alpha=0.7)
    
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Dimension 1', fontsize=12)
    ax.set_ylabel('Dimension 2', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Distance from Origin', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def plot_poincare_embeddings(max_nodes=500):
    """
    Visualize Poincaré embeddings in the Poincaré disk.
    
    Args:
        max_nodes: maximum number of nodes to visualize
    """
    print("=" * 60)
    print("Visualizing Poincaré Embeddings")
    print("=" * 60)
    
    model_path = MODELS_DIR / "poincare_embeddings.pkl"
    
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
    
    if embeddings_subset.shape[1] > 2:
        print("Reducing to 2D for visualization...")
        embeddings_2d = reduce_dimensions(embeddings_subset, method='pca', n_components=2)
        
        norms = np.linalg.norm(embeddings_2d, axis=1, keepdims=True)
        embeddings_2d = embeddings_2d / (norms + 1e-5) * 0.95
    else:
        embeddings_2d = embeddings_subset
    
    disk_path = FIGURES_DIR / "poincare_disk.png"
    plot_poincare_disk(
        embeddings_2d,
        labels=None,
        title="Poincaré Disk Embeddings",
        save_path=disk_path
    )
    
    try:
        graph = load_graph()
        
        print("Creating hierarchy visualization...")
        
        import networkx as nx
        
        node_depths = {}
        for node in graph.nodes():
            if node in node2id:
                try:
                    ancestors = nx.ancestors(graph, node)
                    depth = len(ancestors)
                except:
                    depth = 0
                node_depths[node] = depth
        
        nodes_to_plot = list(node2id.keys())[:max_nodes]
        depths = [node_depths.get(node, 0) for node in nodes_to_plot]
        indices = [node2id[node] for node in nodes_to_plot]
        
        emb_subset = embeddings[indices]
        
        if emb_subset.shape[1] > 2:
            emb_subset = reduce_dimensions(emb_subset, method='pca', n_components=2)
            norms = np.linalg.norm(emb_subset, axis=1, keepdims=True)
            emb_subset = emb_subset / (norms + 1e-5) * 0.95
        
        fig, ax = plt.subplots(figsize=(12, 12))
        
        circle = Circle((0, 0), 1, fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        
        scatter = ax.scatter(emb_subset[:, 0], emb_subset[:, 1], 
                            c=depths, cmap='coolwarm', s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
        
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect('equal')
        ax.set_xlabel('Dimension 1', fontsize=12)
        ax.set_ylabel('Dimension 2', fontsize=12)
        ax.set_title('Poincaré Embeddings Colored by Hierarchy Depth', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Hierarchy Depth', fontsize=12)
        
        plt.tight_layout()
        
        hierarchy_path = FIGURES_DIR / "poincare_hierarchy.png"
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
    plot_poincare_embeddings()
