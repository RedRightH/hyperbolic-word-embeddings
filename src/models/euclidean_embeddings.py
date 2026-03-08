import numpy as np
from gensim.models import Word2Vec
import pickle
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.base_model import BaseEmbeddingModel
from src.utils.config import EUCLIDEAN_CONFIG

class EuclideanEmbeddings(BaseEmbeddingModel):
    """
    Euclidean embeddings using Word2Vec (Skip-gram).
    """
    
    def __init__(self, embedding_dim=None, **kwargs):
        """
        Initialize Euclidean embedding model.
        
        Args:
            embedding_dim: dimension of embeddings
            **kwargs: additional Word2Vec parameters
        """
        if embedding_dim is None:
            embedding_dim = EUCLIDEAN_CONFIG['embedding_dim']
        
        super().__init__(embedding_dim)
        self.model = None
        self.config = {**EUCLIDEAN_CONFIG, **kwargs}
        if embedding_dim:
            self.config['embedding_dim'] = embedding_dim
    
    def train(self, edges, node2id, id2node, **kwargs):
        """
        Train Word2Vec embeddings on edge data.
        
        Args:
            edges: list of (child, parent) tuples
            node2id: dict mapping node names to IDs
            id2node: dict mapping IDs to node names
            **kwargs: additional training parameters
        """
        self.node2id = node2id
        self.id2node = id2node
        
        config = {**self.config, **kwargs}
        
        sentences = []
        for child, parent in edges:
            sentences.append([child, parent])
        
        print(f"Training Word2Vec with {len(sentences)} sentences...")
        
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=config['embedding_dim'],
            window=config['window_size'],
            min_count=config['min_count'],
            workers=config['workers'],
            epochs=config['epochs'],
            sg=1,
            seed=config['seed']
        )
        
        print("Training complete!")
        
        self.embeddings = np.zeros((len(node2id), self.embedding_dim))
        for node, idx in node2id.items():
            if node in self.model.wv:
                self.embeddings[idx] = self.model.wv[node]
            else:
                self.embeddings[idx] = np.random.randn(self.embedding_dim) * 0.01
        
        print(f"Embeddings shape: {self.embeddings.shape}")
    
    def get_embedding(self, node):
        """
        Get embedding for a single node.
        
        Args:
            node: node name or ID
        
        Returns:
            numpy array of shape (embedding_dim,)
        """
        if isinstance(node, str):
            if node not in self.node2id:
                return np.zeros(self.embedding_dim)
            node_id = self.node2id[node]
        else:
            node_id = node
        
        return self.embeddings[node_id]
    
    def save(self, filepath):
        """
        Save model to file.
        
        Args:
            filepath: path to save model
        """
        data = {
            'embeddings': self.embeddings,
            'node2id': self.node2id,
            'id2node': self.id2node,
            'embedding_dim': self.embedding_dim,
            'config': self.config
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """
        Load model from file.
        
        Args:
            filepath: path to load model from
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.embeddings = data['embeddings']
        self.node2id = data['node2id']
        self.id2node = data['id2node']
        self.embedding_dim = data['embedding_dim']
        self.config = data['config']
        
        print(f"Model loaded from {filepath}")
        print(f"Embeddings shape: {self.embeddings.shape}")
