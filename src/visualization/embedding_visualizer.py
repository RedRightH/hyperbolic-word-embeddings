import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pickle
import networkx as nx

def reduce_dimensions(embeddings, method='pca', n_components=2, **kwargs):
    """
    Reduce embedding dimensions for visualization.
    
    Args:
        embeddings: numpy array of shape (n_samples, n_features)
        method: 'pca' or 'tsne'
        n_components: number of output dimensions
        **kwargs: additional parameters for the method
    
    Returns:
        numpy array of shape (n_samples, n_components)
    """
    if method == 'pca':
        reducer = PCA(n_components=n_components, **kwargs)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    reduced = reducer.fit_transform(embeddings)
    
    return reduced

def plot_embeddings_2d(embeddings, labels=None, title="Embeddings Visualization", 
                       save_path=None, figsize=(12, 10), alpha=0.6):
    """
    Plot 2D embeddings.
    
    Args:
        embeddings: numpy array of shape (n_samples, 2)
        labels: optional list of labels for each point
        title: plot title
        save_path: path to save figure
        figsize: figure size
        alpha: point transparency
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], 
                        alpha=alpha, s=30, c=range(len(embeddings)), 
                        cmap='viridis')
    
    if labels is not None and len(labels) <= 50:
        for i, label in enumerate(labels):
            ax.annotate(label, (embeddings[i, 0], embeddings[i, 1]), 
                       fontsize=8, alpha=0.7)
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Dimension 1', fontsize=12)
    ax.set_ylabel('Dimension 2', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.colorbar(scatter, ax=ax, label='Node Index')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def visualize_hierarchy_levels(embeddings, node2id, G, method='pca', max_nodes=500):
    """
    Visualize embeddings colored by hierarchy level.
    
    Args:
        embeddings: embedding matrix
        node2id: node to ID mapping
        G: NetworkX graph
        method: dimensionality reduction method
        max_nodes: maximum nodes to visualize
    
    Returns:
        matplotlib figure
    """
    # Get common nodes
    common_nodes = [node for node in node2id.keys() if node in G.nodes()]
    
    if len(common_nodes) > max_nodes:
        np.random.seed(42)
        common_nodes = np.random.choice(common_nodes, max_nodes, replace=False).tolist()
    
    # Get embeddings for these nodes
    indices = [node2id[node] for node in common_nodes]
    emb_subset = embeddings[indices]
    
    # Reduce dimensionality
    if emb_subset.shape[1] > 2:
        emb_subset = reduce_dimensions(emb_subset, method=method, n_components=2)
    
    # Find root nodes (nodes with no incoming edges or in-degree = 0)
    root_nodes = [node for node in G.nodes() if G.in_degree(node) == 0]
    
    if len(root_nodes) == 0:
        # If no clear roots, use nodes with lowest out-degree as approximation
        root_nodes = sorted(G.nodes(), key=lambda n: G.out_degree(n), reverse=True)[:5]
    
    # Compute hierarchy depth from root nodes
    depths = []
    for node in common_nodes:
        min_depth = float('inf')
        for root in root_nodes:
            try:
                if nx.has_path(G, root, node):
                    depth = nx.shortest_path_length(G, root, node)
                    min_depth = min(min_depth, depth)
            except:
                pass
        
        # If no path from any root, use out-degree as proxy
        if min_depth == float('inf'):
            min_depth = G.out_degree(node)
        
        depths.append(min_depth)
    
    depths = np.array(depths)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    scatter = ax.scatter(emb_subset[:, 0], emb_subset[:, 1], 
                        c=depths, cmap='coolwarm', s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    ax.set_title('Embeddings Colored by Hierarchy Depth', fontsize=16, fontweight='bold')
    ax.set_xlabel('Dimension 1', fontsize=12)
    ax.set_ylabel('Dimension 2', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Hierarchy Depth', fontsize=12)
    
    plt.tight_layout()
    
    return fig
