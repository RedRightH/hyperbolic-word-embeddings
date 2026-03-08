from abc import ABC, abstractmethod
import numpy as np

class BaseEmbeddingModel(ABC):
    """
    Abstract base class for embedding models.
    """
    
    def __init__(self, embedding_dim, **kwargs):
        """
        Initialize the embedding model.
        
        Args:
            embedding_dim: dimension of embeddings
            **kwargs: additional model-specific parameters
        """
        self.embedding_dim = embedding_dim
        self.embeddings = None
        self.node2id = None
        self.id2node = None
    
    @abstractmethod
    def train(self, edges, node2id, id2node, **kwargs):
        """
        Train the embedding model on edge data.
        
        Args:
            edges: list of (child, parent) tuples
            node2id: dict mapping node names to IDs
            id2node: dict mapping IDs to node names
            **kwargs: additional training parameters
        """
        pass
    
    @abstractmethod
    def get_embedding(self, node):
        """
        Get embedding for a single node.
        
        Args:
            node: node name or ID
        
        Returns:
            numpy array of shape (embedding_dim,)
        """
        pass
    
    def get_embeddings(self, nodes):
        """
        Get embeddings for multiple nodes.
        
        Args:
            nodes: list of node names or IDs
        
        Returns:
            numpy array of shape (len(nodes), embedding_dim)
        """
        return np.array([self.get_embedding(node) for node in nodes])
    
    @abstractmethod
    def save(self, filepath):
        """
        Save model to file.
        
        Args:
            filepath: path to save model
        """
        pass
    
    @abstractmethod
    def load(self, filepath):
        """
        Load model from file.
        
        Args:
            filepath: path to load model from
        """
        pass
    
    def get_all_embeddings(self):
        """
        Get all embeddings as a matrix.
        
        Returns:
            numpy array of shape (num_nodes, embedding_dim)
        """
        return self.embeddings
