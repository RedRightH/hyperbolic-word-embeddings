import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def compute_reconstruction_error(tree_distances, embedding_distances):
    """
    Compute reconstruction error between tree distances and embedding distances.
    
    Args:
        tree_distances: array of tree distances
        embedding_distances: array of embedding distances
    
    Returns:
        dict of error metrics
    """
    mse = mean_squared_error(tree_distances, embedding_distances)
    mae = mean_absolute_error(tree_distances, embedding_distances)
    rmse = np.sqrt(mse)
    
    # Compute correlation with error handling
    try:
        with np.errstate(invalid='ignore', divide='ignore'):
            correlation = np.corrcoef(tree_distances, embedding_distances)[0, 1]
            # Handle NaN or inf values
            if not np.isfinite(correlation):
                correlation = 0.0
    except:
        correlation = 0.0
    
    return {
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(rmse),
        'correlation': float(correlation)
    }

def compute_rank_metrics(distances, true_indices, k_values=[1, 5, 10]):
    """
    Compute ranking metrics (Hits@K and Mean Rank).
    
    Args:
        distances: array of distances, shape (n_queries, n_candidates)
        true_indices: array of true target indices for each query
        k_values: list of K values for Hits@K
    
    Returns:
        dict of ranking metrics
    """
    n_queries = distances.shape[0]
    
    ranks = []
    for i in range(n_queries):
        true_idx = true_indices[i]
        
        sorted_indices = np.argsort(distances[i])
        
        rank = np.where(sorted_indices == true_idx)[0][0] + 1
        ranks.append(rank)
    
    ranks = np.array(ranks)
    
    mean_rank = np.mean(ranks)
    median_rank = np.median(ranks)
    
    hits_at_k = {}
    for k in k_values:
        hits = np.sum(ranks <= k) / n_queries
        hits_at_k[f'hits@{k}'] = hits
    
    return {
        'mean_rank': mean_rank,
        'median_rank': median_rank,
        **hits_at_k
    }

def compute_mrr(ranks):
    """
    Compute Mean Reciprocal Rank.
    
    Args:
        ranks: array of ranks
    
    Returns:
        MRR score
    """
    reciprocal_ranks = 1.0 / ranks
    return np.mean(reciprocal_ranks)
