import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from .hyperbolic_math import poincare_distance_numpy

def compute_euclidean_distance(embeddings1, embeddings2):
    """
    Compute Euclidean distance between embeddings.
    
    Args:
        embeddings1: numpy array of shape (n, dim)
        embeddings2: numpy array of shape (m, dim)
    
    Returns:
        distance matrix of shape (n, m)
    """
    return euclidean_distances(embeddings1, embeddings2)

def compute_cosine_distance(embeddings1, embeddings2):
    """
    Compute cosine distance (1 - cosine similarity) between embeddings.
    
    Args:
        embeddings1: numpy array of shape (n, dim)
        embeddings2: numpy array of shape (m, dim)
    
    Returns:
        distance matrix of shape (n, m)
    """
    similarity = cosine_similarity(embeddings1, embeddings2)
    return 1 - similarity

def compute_poincare_distance_matrix(embeddings1, embeddings2):
    """
    Compute pairwise Poincaré distances between two sets of embeddings.
    
    Args:
        embeddings1: numpy array of shape (n, dim)
        embeddings2: numpy array of shape (m, dim)
    
    Returns:
        distance matrix of shape (n, m)
    """
    n = embeddings1.shape[0]
    m = embeddings2.shape[0]
    distances = np.zeros((n, m))
    
    for i in range(n):
        for j in range(m):
            dist = poincare_distance_numpy(
                embeddings1[i:i+1], 
                embeddings2[j:j+1]
            )
            # Handle both scalar and array returns
            distances[i, j] = float(dist) if np.isscalar(dist) else float(dist.item())
    
    return distances

def compute_distance_batch(embeddings1, embeddings2, metric='euclidean'):
    """
    Compute distances between embeddings using specified metric.
    
    Args:
        embeddings1: numpy array of shape (n, dim)
        embeddings2: numpy array of shape (m, dim)
        metric: 'euclidean', 'cosine', or 'poincare'
    
    Returns:
        distance matrix of shape (n, m)
    """
    if metric == 'euclidean':
        return compute_euclidean_distance(embeddings1, embeddings2)
    elif metric == 'cosine':
        return compute_cosine_distance(embeddings1, embeddings2)
    elif metric == 'poincare':
        return compute_poincare_distance_matrix(embeddings1, embeddings2)
    else:
        raise ValueError(f"Unknown metric: {metric}")
